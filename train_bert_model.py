

from datasets import load_from_disk
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import EarlyStoppingCallback
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import numpy as np
import torch
import argparse
from collections import Counter

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average="macro")
    acc = accuracy_score(labels, predictions)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}


def run_classification_report(trainer, dataset, id2label, labels):

    results = trainer.predict(dataset)

    label_ids = results.label_ids.reshape(-1)
    predictions = np.argmax(results.predictions, axis=1).reshape(-1)
    
    report = classification_report(label_ids, predictions, labels=sorted(id2label.keys()), target_names=labels, zero_division=0.0)

    print(report)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dataset',type=str,required=True,help='Input dataset of preprepared sentences with relations and labels')
    parser.add_argument('--output_model',type=str,required=True,help='Where to save the resulting model')
    args = parser.parse_args()

    dataset = load_from_disk(args.input_dataset)

    labels = sorted(set(dataset['train']['label']))
    id2label = { idx:label for idx,label in enumerate(labels) }
    label2id = { label:idx for idx,label in id2label.items() }

    
    def preprocess_labels(example):
        example["label"] = label2id[example["label"]]
        return example
    dataset = dataset.map(preprocess_labels)

    model_name = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    tokenizer.add_tokens(["[E1]","[/E1]","[E2]","[/E2]"])
    
    # Deal with the model not have the max_length saved (which gives a warning)
    model_config = AutoConfig.from_pretrained(model_name)
    tokenizer.model_max_length = model_config.max_position_embeddings

    def tokenize_function(example):
        return tokenizer(example["text"], padding="max_length", truncation=True, max_length=512)
    
    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(model_name, id2label=id2label)
    model.resize_token_embeddings(len(tokenizer))
    for param in model.parameters(): param.data = param.data.contiguous() # To fix strange non-contiguous error
    
    # Compute class weights to address class imbalance
    label_list = tokenized_datasets["train"]["label"]
    label_counts = Counter(label_list)
    total = sum(label_counts.values())
    weights = [(total+1) / (label_counts[label]+1) for label in labels ]
    class_weights = torch.tensor(weights, dtype=torch.float)
    
    # Wrap model with weighted loss
    class WeightedLossModel(torch.nn.Module):
        def __init__(self, model, weights):
            super().__init__()
            self.model = model
            self.loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
    
        def forward(self, input_ids=None, attention_mask=None, labels=None):
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            loss = self.loss_fn(logits, labels) if labels is not None else None
            return {"loss": loss, "logits": logits}
    
    weighted_model = WeightedLossModel(model, class_weights)

    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=3,
        early_stopping_threshold=0.001
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=1e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        num_train_epochs=16,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
    )
    
    # Trainer
    trainer = Trainer(
        model=weighted_model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        compute_metrics=compute_metrics,
        callbacks=[early_stopping_callback]
    )

    trainer.train()

    tokenizer.save_pretrained(args.output_model)
    model.save_pretrained(args.output_model)

    run_classification_report(trainer, tokenized_datasets["test"], id2label, labels)

    print("Done.")

if __name__ == '__main__':
    main()



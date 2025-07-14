# Biomedical relation extraction trained on synthetic labels

This project uses a larger LLM (Llama 3.3 70B) to annotate a large dataset of biomedical sentences to create synthetic labels which are then used to train a smaller BERT model for classifying relations. This is effectively distilling the Llama model down to a BERT model.

## 🚀 Example Usage

The model can classify the relationship between two entities into one of X labels. The labels are: 

To use the model, take the input text and wrap the first entity in [E1][/E1] tags and second entity in [E2][/E2] tags as in the example below. The classifier then outputs the predicted relation label with an associated score.

```python
from transformers import pipeline

classifier = pipeline("text-classification", model="Glasgow-AI4BioMed/synthetic_relex")

classifier("[E1]Paclitaxel[/E1] is a common chemotherapy used for [E2]lung cancer[/E2].")

# Output:
# [{'label': 'treats', 'score': 0.99671870470047}]
```

## 📝 Dataset

The model is trained from relation annotations that were created using a Llama3.3 70B model. No humans were involved in the annotation process so there will be mistakes. The dataset contains sentences and entity annotations from [PubTator Central](https://www.ncbi.nlm.nih.gov/research/pubtator3/).

The dataset can be accessed through the HuggingFace datasets repo [Glasgow-AI4BioMed/synthetic_relex](https://huggingface.co/datasets/Glasgow-AI4BioMed/synthetic_relex). It can be loaded using the [datasets library](https://pypi.org/project/datasets/) as below:

```python
from datasets import load_dataset

dataset = load_dataset('Glasgow-AI4BioMed/synthetic_relex')

# To see a single sample of the dataset:
dataset['train'][0]
```

## 🛠️ Building the dataset and model

There are a few prerequisities including transformers and spaCy. They can be installed with the following:

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm # Installs the English model for spaCy
```

The text to be annotated is from [PubTator Central](https://www.ncbi.nlm.nih.gov/research/pubtator3/). The first step is to download a large archive of pre-annotated text that contains annotations of Genes, Chemicals, Diseases, etc:

```bash
curl -o BioCXML.0.tar.gz https://ftp.ncbi.nlm.nih.gov/pub/lu/PubTator3/BioCXML.0.tar.gz
```

Now we run a large Llama model on sentences from this large archive. The relation_specs.json file outlines the names of relations (and their argument types) to be extracted. A zero-shot prompting method is applied using a quantized version. This requires a larger GPU.

```bash
python create_synthetic_labels.py --input_archive BioCXML.0.tar.gz --relation_specs relation_specs.json --output_sentences sentences.jsonl.gz --target_sentence_count 250000
```

The annotated sentences are then reformatted into a HuggingFace dataset with appropriate training/validation/test split.

```bash
python prepare_dataset.py --input_sentences sentences.jsonl.gz --relation_specs relation_specs.json --min_sample_count 500 --output_dataset synthetic_relex_dataset
```

A BERT-based classifier, using [microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext](https://huggingface.co/microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext) as the base model, is then trained with the dataset with a Sequence Classification objective:

```bash
python train_bert_model.py --input_dataset synthetic_relex_dataset --output_model synthetic_relex_model
```

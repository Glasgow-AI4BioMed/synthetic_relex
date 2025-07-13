---
task: sequence-classification
tags:
- biomedical
- bionlp
- relation extraction
license: mit
base_model: microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext
---

# synthetic_relex model for biomedical relation extraction

This is a relation extraction model that is distilled from Llama 3.3 70B down to a BERT model. It is a [microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext](https://huggingface.co/microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext) model that has been fine-tuned on synthetic labels created with Llama 3.3 70B when prompted with sentences from [PubTator Central](https://www.ncbi.nlm.nih.gov/research/pubtator3/). The dataset is available [here](https://huggingface.co/datasets/Glasgow-AI4BioMed/synthetic_relex).

**Note:** No humans were involved in annotating the dataset used, so there may be erroneous annotations. Detailed evaluation by human experts would be needed to gain an accurate view of the model's accuracy. The dataset and model offer a starting point for understanding and development of biomedical relation extraction models.

More information about the model and dataset can be found at the project repo: https://github.com/Glasgow-AI4BioMed/synthetic_relex

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

## 📈 Performance

Todo

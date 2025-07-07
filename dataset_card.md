---
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
  - split: validation
    path: data/validation-*
  - split: test
    path: data/test-*
license: mit
task_categories:
- text-classification
language:
- en
tags:
- medical
- biology
---

# synthetic_relex dataset for biomedical relation extraction

This is a relation extraction dataset containing synthetic labels created with Llama 3.3 70B when prompted with sentences from [PubTator Central](https://www.ncbi.nlm.nih.gov/research/pubtator3/). It has been used to train a BERT-based relation classifier, enabling the distillation of a larger Llama model to a BERT model.

**Note:** No humans were involved in annotating this dataset, so there may be erroneous annotations. Detailed evaluation by human experts would be needed to gain an accurate view of the dataset's accuracy. This model offers a starting point for understanding and development of biomedical relation extraction models.

More information about the model and dataset can be found at the project repo: https://github.com/Glasgow-AI4BioMed/synthetic_relex

## 📝 Getting access

The dataset can be loaded using the HuggingFace [datasets library](https://pypi.org/project/datasets/) as below:

```python
from datasets import load_dataset

dataset = load_dataset('Glasgow-AI4BioMed/synthetic_relex')

# To see a single sample of the dataset:
dataset['train'][0]
```

## ⬇️ Format

The dataset contains three columns:

- doc_id: This is a combination of the PubMed ID, PMC ID and DOI for a document (pipe-delimited)
- text: This is a sentence from the document with two entities identified with [E1][/E1] and [E2][/E2] tags around them.
- label: This is the label for the relation between the two entities in the text. Note that it is directional (so [E1] -> [E2] does not mean that [E2] -> [E1]).
- entity1: The text of the first entity
- type1: The type of the first entity
- entity2: The text of the second entity
- type2: The type of the second entity

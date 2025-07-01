# synthetic_relex
Distilling a larger LLMs relation extraction abilities down to a BERT model

```
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

```
curl -o BioCXML.0.tar.gz https://ftp.ncbi.nlm.nih.gov/pub/lu/PubTator3/BioCXML.0.tar.gz
```

```
python create_synthetic_labels.py --input_archive BioCXML.0.tar.gz --relation_specs relation_specs.json --output_sentences relex.jsonl.gz --target_sentence_count 10000
```

```
python prepare_dataset.py --input_sentences relex.jsonl.gz --relation_specs relation_specs.json --min_sample_count 10 --output_dataset relex_dataset
```


```
python train_bert_model.py --input_dataset relex_dataset --output_model relex_model
```

import argparse
import json
import jsonlines
import gzip
import itertools
import random
from collections import defaultdict, Counter
from datasets import Dataset, DatasetDict

def split_dataset(data, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, seed=42):
    # Group samples by doc_id
    doc_id_to_samples = defaultdict(list)
    for sample in data:
        doc_id_to_samples[sample['doc_id']].append(sample)

    # Get list of unique doc_ids and shuffle
    doc_ids = list(doc_id_to_samples.keys())
    random.seed(seed)
    random.shuffle(doc_ids)

    # Compute split sizes
    total = len(doc_ids)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    # Split doc_ids
    train_ids = set(doc_ids[:train_end])
    val_ids = set(doc_ids[train_end:val_end])

    # Assign samples to splits
    train_set, val_set, test_set = [], [], []
    for doc_id, samples in doc_id_to_samples.items():
        if doc_id in train_ids:
            train_set.extend(samples)
        elif doc_id in val_ids:
            val_set.extend(samples)
        else:
            test_set.extend(samples)

    return train_set, val_set, test_set


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_sentences',type=str,required=True,help='Input dataset of sentences labelled with relations')
    parser.add_argument('--relation_specs',type=str,required=True,help='JSON file with information about relations to extract')
    parser.add_argument('--min_sample_count',type=int,required=True,help='Minimum number of sentences containing relation to include it')
    parser.add_argument('--output_dataset',type=str,required=True,help='Output dataset')
    args = parser.parse_args()


    with gzip.open(args.input_sentences,'rt') as f:
        reader = jsonlines.Reader(f)
        sentence_data = [ x for x in reader ]
    print(f"Loaded {len(sentence_data)} sentences")

    with open(args.relation_specs) as f:
        relation_specs = json.load(f)
        relation_specs = { tuple(k.split('|')):v for k,v in relation_specs.items() }
    
    dataset = []
    for sentence, entities, rels, doc_id in sentence_data:
        labeled_pairs = { (head,tail):reltype for reltype,head,tail in rels }
    
        for (a_name,a_type),(b_name,b_type) in itertools.product(entities, entities):
            
            if a_name != b_name and (a_type,b_type) in relation_specs:
                tmp_sentence = sentence.replace(a_name,f'[E1]{a_name}[/E1]').replace(b_name,f'[E2]{b_name}[/E2]')
                dataset.append( {'doc_id':doc_id, 'text':tmp_sentence, 'label':labeled_pairs.get((a_name,b_name),'none')} )

    sample_counts = Counter( x['label'] for x in dataset )
    print(f"{sample_counts}")

    rels_to_keep = set( reltype for reltype,count in sample_counts.items() if count >= args.min_sample_count )
    print(f"Only keeping following relations: {rels_to_keep}")

    dataset = [ x for x in dataset if x['label'] in rels_to_keep ]
                
    print(f"Created dataset with {len(dataset)} samples")

    train_set, val_test, test_set = split_dataset(dataset, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, seed=42)
    print(f"Split: {len(train_set)=} {len(val_test)=} {len(test_set)=}")

    final_dataset = DatasetDict({
            "train": Dataset.from_list(train_set),
            "validation": Dataset.from_list(val_test),
            "test": Dataset.from_list(test_set),
        })

    print("Saving...")
    final_dataset.save_to_disk(args.output_dataset)

    print("Done.")

if __name__ == '__main__':
    main()




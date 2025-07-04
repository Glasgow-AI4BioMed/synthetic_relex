
from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams
from tqdm.auto import tqdm
import bioc
import os

import tarfile
from io import BytesIO
import jsonlines
import gzip
import argparse

def make_id(article_info):
    id_keys = [ 'article-id_pmid', 'article-id_pmc', 'article-id_doi' ]
    if all ( not id_key in article_info for id_key in id_keys ):
        return None

    return "|".join( article_info.get(id_key,'') for id_key in id_keys )

def mark_sentences(nlp, doc):
    for passage in doc.passages:
        passage.sentences = []

        parsed = nlp(passage.text)
        for sent in parsed.sents:
            start = sent[0].idx
            end = sent[-1].idx + len(sent[-1].text)

            sentence = bioc.BioCSentence()
            sentence.offset = passage.offset+start
            sentence.infons['length'] = (end-start)
            passage.add_sentence(sentence)

            
def get_sentence_annotations(passage, sentence):
    sentence_start = sentence.offset
    sentence_end = sentence.offset+int(sentence.infons['length'])

    annotations = [ anno for anno in passage.annotations if anno.total_span.offset >= sentence_start and anno.total_span.end <= sentence_end ]

    return annotations

from more_itertools import chunked
import json

triple_schema = {
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "List of String Triples",
  "type": "array",
  "items": {
    "type": "array",
    "items": {
      "type": "string"
    },
    "minItems": 3,
    "maxItems": 3
  }
}

def do_openie(llm_model, nlp, doc_id, doc, relnames):
    mark_sentences(nlp, doc)

    sentences_with_entities = []
    for passage in doc.passages:
        for sentence in passage.sentences:
            sentence_start = sentence.offset
            sentence_end = sentence.offset+int(sentence.infons['length'])

            sentence_text = passage.text[ (sentence_start-passage.offset):(sentence_end-passage.offset) ] 
            sentence_annotations = get_sentence_annotations(passage, sentence)

            entities_with_types = sorted(set( (anno.text, anno.infons['type']) for anno in sentence_annotations))

            sentences_with_entities.append( (sentence_text,entities_with_types)  )

    questions = []
    for sentence_text, entities_with_types in sentences_with_entities:
        entities_as_json = json.dumps( [ entity_name for entity_name,entity_type in entities_with_types ] )
        relnames_as_json = json.dumps(relnames)

        question = """
        <text>{sentence_text}</text>
        <entities>{entities_as_json}</entities>
        <relations>{relnames_as_json}</relations>

        Output all relations that are explicitly stated in the text above (inside the <text> tags). They should be triples with the first being the relation type from the options in <relations> above (e.g. 'activates'), the second element being head entity (from the <entities> list),  and the third being the tail entity (from the <entities> list). Output as a JSON list of lists. Both the head and tail entity must come from the list of entities provided. If no explicit relations are described, return an empty list.
        """.format(sentence_text=sentence_text, entities_as_json=entities_as_json, relnames_as_json=relnames_as_json)

        questions.append( question )


    system_prompt = "You are a knowledgable biomedical assistant"
    chat_template = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"

    prompts = [ chat_template.format(system_prompt=system_prompt, question=question) for question in questions ]

    guided_decoding_params = GuidedDecodingParams(json=triple_schema)
        
    results = []
    for chunk in chunked(prompts, 100):
        results += llm_model.generate(chunk, SamplingParams(max_tokens=1024, top_k=1, guided_decoding=guided_decoding_params), use_tqdm=False)

    relnames_set = set(relnames)
    final_output = []
    for (sentence_text, entities_with_types), result in zip(sentences_with_entities, results):
        output = result.outputs[0].text
        entity_names = set([ entity_name for entity_name,entity_type in entities_with_types ])

        try:
            triples = json.loads(output)
            triples = [ (rel,head,tail) for rel,head,tail in triples if head in entity_names and tail in entity_names and rel in relnames_set ]
            #if triples:
            final_output.append( (sentence_text, entities_with_types, triples, doc_id) )
        except json.JSONDecodeError:
            continue

    return final_output



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name',type=str,required=False,default="casperhansen/llama-3.3-70b-instruct-awq")
    parser.add_argument('--max_model_len',type=int,required=False,default=4020)
    parser.add_argument('--gpu_count',type=int,required=False,default=2)
    parser.add_argument('--input_archive',type=str,required=True,help='Input tar.gz archive containing BioC XML file')
    parser.add_argument('--output_sentences',type=str,required=True,help='Output filename')
    parser.add_argument('--target_sentence_count',type=int,required=True,help='Number of sentences to aim for')
    parser.add_argument('--relation_specs',type=str,required=True,help='JSON file with information about relations to extract')
    parser.add_argument('--resume',action='store_true',help='Continue with a previous file and skip already processed documents')
    args = parser.parse_args()

    llm_model = LLM(model=args.model_name, 
                    max_model_len=args.max_model_len, 
                    dtype='auto', 
                    gpu_memory_utilization=0.92,
                    quantization="awq", 
                    tensor_parallel_size=args.gpu_count,
                    enforce_eager=True,
                    disable_custom_all_reduce=True,
                   )

    import spacy
    nlp = spacy.load("en_core_web_sm")

    sentence_count, doc_count = 0, 0
    previous_doc_ids = set()
    if args.resume:
        with gzip.open(args.output_sentences,'rt') as f:
            reader = jsonlines.Reader(f)
            previous_doc_ids = [ doc_id for _,_,_,doc_id in reader ]
            sentence_count = len(previous_doc_ids)
            previous_doc_ids = set(previous_doc_ids)
            doc_count = len(previous_doc_ids)
            
        print(f"Found {sentence_count} sentences from {doc_count} previously processed documents to be skipped")
    else:
        assert not os.path.isfile(args.output_sentences), "Output file already exists. Use --resume or delete the file"

    with open(args.relation_specs) as f:
        relation_specs = json.load(f)
    relnames = sorted(set( relation for args,relations in relation_specs.items() for relation in relations ))
    print(f"Extracting {len(relnames)} relation types: {relnames}")

    
    with tqdm(total=args.target_sentence_count) as pbar:
        with tarfile.open(args.input_archive, 'r:gz') as tar, gzip.open(args.output_sentences,'at') as out_f:
            writer = jsonlines.Writer(out_f)
            pbar.update(sentence_count)
        
            for member in tar:
                if member.name.lower().endswith('.xml') and member.isfile():
        
                    file_obj = tar.extractfile(member)
        
                    xml_content = file_obj.read()
        
                    xml_io = BytesIO(xml_content)
        
                    collection = bioc.biocxml.load(xml_io)
                    
                    pbar.set_description(f"Processing: {member.name} - {len(collection.documents)} docs")

                    for doc in collection.documents:

                        # Decide whether to skip this document
                        article_info = doc.passages[0].infons
                        doc_id = make_id(article_info)
                        if doc_id is None or doc_id in previous_doc_ids:
                            continue
                        
                        rels = do_openie(llm_model, nlp, doc_id, doc, relnames)
        
                        writer.write_all(rels)
                        out_f.flush()
                        
                        pbar.update(len(rels))
                        sentence_count += len(rels)
                        doc_count += 1
                        
                    if sentence_count >= args.target_sentence_count:
                        break

    print(f"Processed {sentence_count} sentences from {doc_count} documents")
    print("Done.")

if __name__ == '__main__':
    main()


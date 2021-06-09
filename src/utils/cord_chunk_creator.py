
import os
import sys
sys.path.append('.')
import argparse
import json
import csv
import time
import re
from src.utils.doc_sliding_windower import doc_chunks_creator

def main(dataset_dir):

    ''' Takes all the json files from document_parses/pdf_json/ 
    and returns one tsv file with 5 sentence chunks'''
    start = time.time()

    output_file = 'data/cord_document_chunks.tsv'
    if os.path.exists(output_file):
        os.remove(output_file)

    total_chunks = 0
    
    for file_idx, json_file in enumerate(os.listdir(dataset_dir)):

        with open(os.path.join(dataset_dir, json_file), encoding = 'utf-8') as file:
                json_text = json.load(file)

                title = json_text["metadata"]["title"]
                abstract = ''
                for entry in json_text['abstract']:
                    abstract += entry['text']

                body_text = ""
                for entry in json_text['body_text']:
                    body_text += entry['text']

                # delimiter = '\n\n'
                delimiter = ' '
                text = delimiter.join([title, abstract, body_text])

                # Removing paper references like [77, 85, 86], [12, 15], [101]
                text = re.sub(r'\[[0-9]{1,3}(,\s[0-9]{1,3})*\]', '', text)
                text = ' '.join(text.split())

                file_name = os.path.splitext(json_file)[0]
                # file_name = json_file

                # Spacy sentence segmenter for 212k documents is very slow - Hence chunking by tokens
                # text_chunks = doc_chunks_creator(text, 5, 5)
                text_split = text.split()
                chunk_token_size = 60
                stride = 15
                text_len = len(text_split)
                # num_chunks = math.ceil(text_len / chunk_token_size)
                text_chunks = [' '.join(text_split[i:i+chunk_token_size]) for i in range(0, text_len, chunk_token_size-stride)]
                
                chunk_ids = [file_name + '_' + str(i+1) for i in range(len(text_chunks))]
                total_chunks += len(chunk_ids)

                with open(output_file, 'a', encoding = 'utf-8', newline='') as fp:
                    tsv_writer = csv.writer(fp, delimiter='\t')
                    tsv_writer.writerows(list(zip(chunk_ids, text_chunks)))
        
        if (file_idx+1) % 1000 == 0:
            print(f'Processed {file_idx+1} files, Created {total_chunks} passages, Time elapsed: {time.time() - start}')
            # break
    
    print(f'Processed {file_idx+1} files, Created {total_chunks} passages, Time elapsed: {time.time() - start}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str,
                    default="data/document_parses/pdf_json/",
                    help="Data directory to read the json files from")

    args = parser.parse_args()
    main(args.data_dir)
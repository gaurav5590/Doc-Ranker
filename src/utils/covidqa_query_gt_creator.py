
import json
import csv
import os

data_file_path = 'data/covidqa.json'
query_output_file_path = 'data/covidqa_queries.tsv'
gt_output_file_path = 'data/covidqa_references.json'

with open(data_file_path) as f:
    covidqa = json.load(f)

# Storing the queries in tsv file to use the QueryReader
if os.path.exists(query_output_file_path):
    os.remove(query_output_file_path)

for data in covidqa['data']:
    for para in data['paragraphs']:
        # print(qas)
        for qa in para['qas']:
            query = qa['question'].strip()
            qid = qa['id']
            # print(qid, query)
            with open(query_output_file_path, 'a', encoding='utf-8') as tsv_file:
                writer = csv.writer(tsv_file, delimiter='\t', lineterminator='\n')
                writer.writerow([qid, query])


# Storing the ground truth answers in json file to use the MSMarco QA Evaluator
if os.path.exists(gt_output_file_path):
    os.remove(gt_output_file_path)

for data in covidqa['data']:
    for para in data['paragraphs']:
        # print(qas)
        for qa in para['qas']:
            row_dict = {'query_id':qa['id'], 'answers':[qa['answers'][0]['text'].strip()]}
            query = qa['question'].strip()
            qid = qa['id']
            # print(row_dict)
            with open(gt_output_file_path, 'a', newline='\n', encoding='utf-8') as fp:
                json.dump(row_dict, fp)
                fp.write('\n')
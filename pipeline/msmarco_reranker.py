"""
Script to index MAMARCO documents
"""
import argparse
import os

import yaml
import torch
from forte.common.configuration import Config
from forte.data.data_pack import DataPack
from forte.data.multi_pack import MultiPack
from forte.pipeline import Pipeline
from forte.processors.ir import (ElasticSearchQueryCreator, ElasticSearchProcessor)
from forte.data.readers import MSMarcoPassageReader
from query_file_reader import EvalReader
from ms_marco_evaluator import MSMarcoEvaluator
from transformers import AutoTokenizer
from model import TransformerModel
from ms_marco_eval import compute_metrics_from_files
import utils

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", default="./config.yml",
                        help="Config YAML filepath")
    # parser.add_argument("--input_file",
    #                     default="./data/collectionandqueries/query_doc_id.tsv",
    #                     help="Input query filepath")
    args = parser.parse_args()

    # loading config
    config = yaml.safe_load(open(args.config_file, "r"))
    config = Config(config, default_hparams=None)

    # reading query input file
    input_file = config.evaluator.input_file

    pipeline = Pipeline[MultiPack]()
    pipeline.set_reader(EvalReader(), config = config.reader)
    pipeline.add(ElasticSearchQueryCreator(), config=config.query_creator)
    pipeline.add(ElasticSearchProcessor(), config=config.full_ranker)
    pipeline.add(MSMarcoEvaluator(), config = config.evaluator)
    pipeline.initialize()

    ## Full ranking using elastic search
    for idx, m_pack in enumerate(pipeline.process_dataset(input_file)):
        if (idx + 1) % 50 == 0:
            print(f"Processed {idx + 1} examples")



    ### Re ranking starts from here
    max_seq_length = config.reranker.max_seq_length
    curr_dir = os.path.dirname(__file__)
    output_file = os.path.join(curr_dir, config.evaluator.output_file)
    gt_file = os.path.join(curr_dir, config.evaluator.ground_truth_file)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    f = open(output_file, "w")

    ### Create a dictionary to store fullraraking output
    query_all_ids, query_all_text, doc_all_ids, doc_all_text = utils.vectorize_fullranking_data(config, pipeline)
    
    ## BERT Inference
    tokenizer = AutoTokenizer.from_pretrained(config.reranker.model_name)
    model = TransformerModel(config.reranker.model_name)
    model.eval()
    encodings = tokenizer(query_all_text, doc_all_text,padding = True, max_length=max_seq_length, return_tensors= 'pt')
    with torch.no_grad():
        scores = model(encodings, train=False)


    ## Output post processing
    doc_scores = list(zip(query_all_ids, doc_all_ids, scores))
    doc_scores = sorted(doc_scores, key = lambda x: (x[0],x[2]), reverse=True)
    doc_ranks = []
    rank = 1
    prev_qid = None
    for elem in doc_scores:
        q_id, doc_id, score = elem
        if q_id!= prev_qid:
            rank = 1
            prev_qid = q_id
        doc_ranks.append([q_id, doc_id, str(rank)])
        rank+=1
    
    [f.write('\t'.join(result) + '\n') for result in doc_ranks]
    f.close()

    ## Final MS marco evaluation scores
    scores = compute_metrics_from_files(gt_file, output_file)
    print(scores)








    




    

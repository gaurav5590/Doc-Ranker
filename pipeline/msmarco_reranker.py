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
    for idx, m_pack in enumerate(pipeline.process_dataset(input_file)):
        if (idx + 1) % 50 == 0:
            print(f"Processed {idx + 1} examples")

    query_passage_dict = {}
    max_rank = config.reranker.size
    max_seq_length = config.reranker.max_seq_length
    for elem in pipeline.components[-1].predicted_text:
        query_id = elem[0]
        query_text = elem[1]
        passage_id = elem[2]
        passage_text = elem[3]
        rank = int(elem[4])

        ## Format of the dictionary
        ## key (query_id): value-[query text, list of all doc ids, list of all doc text]
        if query_id not in query_passage_dict.keys():
            query_passage_dict[query_id] = [query_text,['0'] * (max_rank), ['0'] * (max_rank)]
        query_passage_dict[query_id][1][rank-1] = passage_id
        query_passage_dict[query_id][2][rank-1] = passage_text


    for query_id in query_passage_dict.keys():
        docs_id = query_passage_dict[query_id][1]
        docs_content = query_passage_dict[query_id][2]
        query_text = query_passage_dict[query_id][0]
        ## Tokenization

        tokenizer = AutoTokenizer.from_pretrained(config.reranker.model_name)
        encodings = tokenizer([query_text] * len(docs_content), docs_content,padding = True, max_length=max_seq_length, return_tensors= 'pt')

        model = TransformerModel(config.reranker.model_name)
        model.eval()

        with torch.no_grad():
            scores = model(encodings, train=False)
        doc_scores = list(zip(docs_id, scores))
        doc_scores = sorted(doc_scores, key = lambda x: x[1], reverse=True)
        doc_ranks = [[query_id, row[0], idx+1] for idx, row in enumerate(doc_scores)]
        print(doc_ranks)







    




    

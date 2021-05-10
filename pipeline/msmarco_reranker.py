"""
Script to index MAMARCO documents
"""
import argparse
import os

import yaml

from forte.common.configuration import Config
from forte.data.data_pack import DataPack
from forte.data.multi_pack import MultiPack
from forte.pipeline import Pipeline
from forte.processors.ir import (ElasticSearchQueryCreator, ElasticSearchProcessor)
from forte.data.readers import MSMarcoPassageReader
from query_file_reader import EvalReader
from ms_marco_evaluator import MSMarcoEvaluator
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
        if (idx + 1) % 5 == 0:
            print(f"Processed {idx + 1} examples")

    query_passage_dict = {}
    max_rank = config.reranker.size
    for elem in pipeline.components[-1].predicted_results:
        query_id = elem[0]
        passage_id = elem[1]
        rank = int(elem[2])
        if query_id not in query_passage_dict.keys():
            query_passage_dict[query_id] = ['0'] * max_rank
        query_passage_dict[query_id][rank - 1] = passage_id




    

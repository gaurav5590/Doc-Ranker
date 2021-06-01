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

from src.readers.query_file_reader import EvalReader
from src.evaluators.ms_marco_evaluator import MSMarcoEvaluator
from src.processors.bert_reranking_processor import BertRerankingProcessor
from src.processors.qa_processor import QAProcessor
from src.evaluators.qa_evaluator import QAEvaluator
#import utils

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
    #pipeline.add(MSMarcoEvaluator(), config = config.evaluator)
    pipeline.add(BertRerankingProcessor(), config=config.reranker)
    pipeline.add(QAProcessor(), config = config.qa_system)
    pipeline.add(QAEvaluator(), config = config.qa_evaluator)
    #pipeline.add(MSMarcoEvaluator(), config = config.evaluator)
    pipeline.initialize()

    ## Full ranking using elastic search
    for idx, m_pack in enumerate(pipeline.process_dataset(input_file)):
        if (idx + 1) % 2 == 0:
            print(f"Processed {idx + 1} examples")
        if idx == 5:
            break

    score = pipeline.components[-1].get_result()
    print("Score after QA:", score)
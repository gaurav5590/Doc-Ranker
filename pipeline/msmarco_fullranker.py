"""
Script to index MAMARCO documents
"""
import argparse
import os

import yaml

from forte.common.configuration import Config
from forte.data.data_pack import DataPack
from forte.pipeline import Pipeline

# from composable_source.readers import CORDReader
from forte.processors.ir import (ElasticSearchQueryCreator, ElasticSearchProcessor)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--config_file", default="./config.yml",
                        help="Config YAML filepath")
    args = parser.parse_args()

    # loading config
    config = yaml.safe_load(open(args.config_file, "r"))
    config = Config(config, default_hparams=None)

    # reading query input file
    parser.add_argument("--input_file",
                        default="./data/collectionandqueries/query_doc_id.tsv",
                        help="Input query filepath")

    input_file = config.evaluator.input_file

    pipeline = Pipeline[DataPack]()

    pipeline.set_reader(MSMarcoPassageReader())
    pipeline.add(ElasticSearchPackIndexProcessor(), config=config.create_index)

    pipeline.run(dataset_dir)

    args = parser.parse_args()
    main(args.data_dir)


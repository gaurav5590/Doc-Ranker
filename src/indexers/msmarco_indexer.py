"""
Script to index MAMARCO documents
"""
import argparse
import os

import yaml

from forte.common.configuration import Config
from forte.data.data_pack import DataPack
from forte.pipeline import Pipeline
from src.processors.elastic_search_index_processor import ElasticSearchTextIndexProcessor
# from forte.data.readers import MSMarcoPassageReader
from src.readers.ms_marco_passage_reader import MSMarcoPassageReader



def main(args):
    """
    Build a pipeline to process MS_MARCO dataset using
    MSMarcoPassageReader and build elastic indexer.
    """
    # config_file = os.path.join(os.path.dirname(__file__), 'config.yml')
    config_file = args.config_file
    config = yaml.safe_load(open(config_file, "r"))
    config = Config(config, default_hparams=None)

    pipeline = Pipeline[DataPack]()

    pipeline.set_reader(MSMarcoPassageReader())
    pipeline.add(ElasticSearchTextIndexProcessor(), config=config.create_index)
    pipeline.run(args.data_dir)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str,
                    default="data/collectionandqueries/",
                    help="Data directory to read the collections tsv file from")
    parser.add_argument("--config_file", type=str,
                    default="config.yml",
                    help="Config file for indexer")

    args = parser.parse_args()
    main(args)


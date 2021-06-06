"""
Script to index MAMARCO documents
"""
import argparse
import os
import sys
sys.path.append('.')

import yaml

from forte.common.configuration import Config
from forte.data.data_pack import DataPack
from forte.pipeline import Pipeline
from src.processors.elastic_search_index_processor import ElasticSearchTextIndexProcessor
# from forte.processors.ir import ElasticSearchPackIndexProcessor
# from forte.data.readers import MSMarcoPassageReader
# from src.readers.ms_marco_passage_reader import MSMarcoPassageReader
from src.readers.cord_reader import CORDReader



def main(dataset_dir: str):
    """
    Build a pipeline to process MS_MARCO dataset using
    MSMarcoPassageReader and build elastic indexer.
    """
    # config_file = os.path.join(os.path.dirname(__file__), 'config_cord.yml')
    config_file = 'config_cord.yml'
    config = yaml.safe_load(open(config_file, "r"))
    config = Config(config, default_hparams=None)

    pipeline = Pipeline[DataPack]()

    pipeline.set_reader(CORDReader())
    pipeline.add(ElasticSearchTextIndexProcessor(), config=config.create_index)
    # pipeline.add(ElasticSearchPackIndexProcessor(), config=config.create_index)
    pipeline.run(dataset_dir)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str,
                    default="data/document_parses/pdf_json/",
                    help="Data directory to read the collections tsv file from")

    args = parser.parse_args()
    main(args.data_dir)

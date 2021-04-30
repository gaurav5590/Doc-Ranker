"""
Script for MSMARCO Reader.
"""
import argparse

from forte.data.data_pack import DataPack
from forte.pipeline import Pipeline
from forte.processors.writers import PackNameJsonPackWriter

# from composable_source.readers import CORDReader
from forte.data.readers import MSMarcoPassageReader

def main(dataset_dir: str, output_dir: str):
    """
    Build an NLP pipeline to process CORD_NER dataset using
    CORDReader, then write the processed dataset out.
    """
    pipeline = Pipeline[DataPack]()
    pipeline.set_reader(MSMarcoPassageReader())

    pipeline.add(
        PackNameJsonPackWriter(),
        {
            'output_dir': output_dir,
            'indent': 2,
            'overwrite': True,
            'drop_record': True
        }
    )
    pipeline.run(dataset_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str,
                    default="sample_data/cord_paper/",
                    help="Data directory to read the text files from.")
    parser.add_argument("--output-dir", type=str,
                    default='./',
                    help="Output dir to save the processed datapack.")
    args = parser.parse_args()
    main(args.data_dir, args.output_dir)



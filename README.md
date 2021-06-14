# Composable NLP Workflows using Forte for BERT-based Ranking and QA System

End-to-end Ranking and Question-Answering (QA) system using [Forte](https://github.com/asyml), a toolkit that makes composable NLP pipelines.

You can read the full report [here](https://gaurav5590.github.io/data/UCSD_CASL_Research_Gaurav_Murali.pdf)

## Tasks
* Build an end-to-end QA system with following components
    * **Full-ranker** using ElasticSearch indexer and BM25 algorithm
    * **Re-ranker** using BERT
    * **Question-Answering** using BERT

![](https://i.imgur.com/W3OhFXQ.png =300x)

## Datasets
The task has been implemented using two sets of datasets:
* **MS-MARCO QA:** [MS-MARCO](https://microsoft.github.io/msmarco/) passage ranking dataset and QA dataset
* **Covid QA:** [CORD-19](https://allenai.org/data/cord-19) dataset and [Covid-QA](https://www.aclweb.org/anthology/2020.nlpcovid19-acl.18/) dataset


## How to run the Pipeline
* Make sure the input and reference file paths are properly set in `config.yml`
* Run the followng commands:
    * On Linux/Mac Run `export PYTHONPATH="$(pwd):$PYTHONPATH`
    * On windows add the code to the file to be run`import sys; sys.path.append('.')`
* Create elastic search index 
    * Modify `config.yml` with proper datasets and elastic search index name
    * Run `python src/indexers/msmarco_indexer.py --config_file config.yml` 

* Run pipeline
    * Modify `config.yml` with proper dataset name, filenames, re-ranking size.
    * Run `pipeline/msmarco_reranker_qa_pipeline.py --config_file config.yml`

* Results are saved in `output` folder
* **Changes to be done for Covid QA:** Change `config_cord.yml`. Run the `cord_indexer.py` to index the cord-19 documents and `cord_reranker_qa_pipeline.py` to get the answers

## Pipeline Information Flow

![](https://i.imgur.com/pJ0CHEq.png)


## Experiment Results

### MS-MARCO

#### Result on 1000 queries from dev.small with multiple re-ranking sizes

|                   |             | Full Ranking |         |           |            | Reranker |         |           |            | QA     |        |        |        |         |           |        |      |              |
|-------------------|-------------|--------------|---------|-----------|------------|----------|---------|-----------|------------|--------|--------|--------|--------|---------|-----------|--------|------|--------------|
| Re-Ranking Size | Time per Query(s) | MRR@10       | MRR@100 | Recall@10 | Recall@100 | MRR@10   | MRR@100 | Recall@10 | Recall@100 | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | ROUGE-L | PRECISION | RECALL | F1   | Semantic Sim |
| 1                 | 0.49        | 0.09         | 0.09    | 0.09      | 0.09       | 0.09     | 0.09    | 0.09      | 0.09       | 0.24   | 0.15   | 0.11   | 0.09   | 0.22    | 0.20      | 0.23   | 0.21 | 0.75         |
| 10                | 0.56        | 0.16         | 0.16    | 0.34      | 0.34       | 0.23     | 0.23    | 0.34      | 0.34       | 0.30   | 0.21   | 0.17   | 0.15   | 0.29    | 0.26      | 0.32   | 0.29 | 0.79         |
| 50                | 0.85        | 0.16         | 0.17    | 0.34      | 0.50       | 0.28     | 0.28    | 0.45      | 0.50       | 0.31   | 0.23   | 0.19   | 0.17   | 0.31    | 0.27      | 0.34   | 0.30 | 0.79         |
| 100               | 1.24        | 0.16         | 0.17    | 0.34      | 0.59       | 0.30     | 0.30    | 0.50      | 0.59       | 0.31   | 0.24   | 0.20   | 0.18   | 0.32    | 0.28      | 0.35   | 0.31 | 0.80         |
| 500               | 4.63        | 0.16         | 0.17    | 0.34      | 0.59       | 0.33     | 0.33    | 0.56      | 0.73       | 0.32   | 0.24   | 0.21   | 0.19   | 0.32    | 0.29      | 0.36   | 0.32 | 0.80         |
| 1000              | 8.80        | 0.16         | 0.17    | 0.34      | 0.59       | 0.34     | 0.35    | 0.58      | 0.77       | 0.32   | 0.25   | 0.21   | 0.19   | 0.32    | 0.29      | 0.36   | 0.32 | 0.80         |




### Covid-19


#### Results on all covid-qa queries with multiple re-ranking sizes

|                   |                    | QA     |        |        |        |         |           |        |      |              |
|-------------------|--------------------|--------|--------|--------|--------|---------|-----------|--------|------|--------------|
| Re-Ranking Size | Time per Query(s) | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | ROUGE-L | PRECISION | RECALL | F1   | Semantic Sim |
| 100               | 1.21               | 0.20   | 0.15   | 0.13   | 0.12   | 0.22    | 0.18      | 0.29   | 0.22 | 0.71         |
| 1000              | 6.64               | 0.20   | 0.15   | 0.13   | 0.12   | 0.22    | 0.18      | 0.29   | 0.22 | 0.71         |

## Acknowledgements
We would like to thank Professor Dr. Zhiting Hu and Dr. Zhengzhong (Hector) Liu for guiding us throughout the project. We also extend our gratitude towards Petuum Inc. for providing us the computing support needed to run our pipeline on GPU.

## Contact
Murali Mohana Krishna Dandu - `mdandu@ucsd.edu` [![Linkedin](https://i.stack.imgur.com/gVE0j.png) LinkedIn](https://www.linkedin.com/in/muralimohanakrishna/)
&nbsp;
Gaurav Kumar - `gkumar@ucsd.edu` [![Linkedin](https://i.stack.imgur.com/gVE0j.png) LinkedIn](https://www.linkedin.com/in/gauravkumar5590/)
&nbsp;

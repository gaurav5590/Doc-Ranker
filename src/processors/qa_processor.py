# Copyright 2019 The Forte Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: disable=attribute-defined-outside-init
import os
from typing import Dict, Any

import torch

from texar.torch.data.tokenizers.bert_tokenizer import BERTTokenizer

from forte.common.configuration import Config
from forte.common.resources import Resources
from forte.data.multi_pack import MultiPack
from forte.data.ontology import Query
from forte.processors.base import MultiPackProcessor
from transformers import pipeline
import transformers
transformers.logging.set_verbosity_error()
# import warnings
# warnings.simplefilter("ignore")

import spacy
spacy_nlp = spacy.load("en_core_web_sm")

__all__ = [
    "BertRerankingProcessor"
]


class QAProcessor(MultiPackProcessor):

    def initialize(self, resources: Resources, configs: Config):
        self.resources = resources
        self.config = Config(configs, self.default_configs())

        self.device = torch.device('cuda:0') \
            if torch.cuda.is_available() else torch.device('cpu')

        # self.model = AutoModelForSequenceClassification.from_pretrained(self.config.model_name).to(self.device)
        # self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        self.qa_pipeline = pipeline(self.config.task_name, model=self.config.model_name, tokenizer=self.config.model_name)
        

    @classmethod
    def default_configs(cls) -> Dict[str, Any]:
        configs = super().default_configs()
        model_name = 'deepset/roberta-base-squad2-covid'
        task_name = 'question-answering'
        configs.update({
            "size": 5,
            "query_pack_name": "query",
            "field": "content",
            "model_dir": os.path.join(os.path.dirname(__file__), "models"),
            "max_seq_length": 512,
            "model_name": model_name,
            "task_name": task_name
        })
        return configs

    def _process(self, input_pack: MultiPack):
        max_len = self.config.max_seq_length
        query_pack_name = self.config.query_pack_name

        query_pack = input_pack.get_pack(self.config.query_pack_name)
        query_entry = list(query_pack.get(Query))[0]
        query_text = query_pack.text
        doc_score_dict = query_entry.results
        # print('Query: ', query_text, '\n')
        # print('doc_score_dict: ', doc_score_dict, '\n')
        best_doc_id = max(doc_score_dict, key = lambda x: doc_score_dict[x])
        packs = {}

        for doc_id in input_pack.pack_names:
            if doc_id == query_pack_name:
                continue
            pack = input_pack.get_pack(doc_id)
            doc_id_final = pack.pack_name
            if(doc_id_final!=best_doc_id):
                query_entry.update_results({doc_id_final: ""})
                continue
            query_doc_input = {'question':query_text, 'context': pack.text}
            result = self.qa_pipeline(query_doc_input)
            
            # answer = result['answer']
            
            # Changing answer phrase to the whole sentence where it is present
            # print("====Full Passage Text: ", pack.text)
            # print("====Answer Phrase: ", result['answer'])
            answer_phrase = result['answer']
            ans_sents = [sent.text for sent in spacy_nlp(pack.text).sents]
            # print("====Passage Sentences: ", ans_sents)
            answer = None
            for sent in ans_sents:
                if answer_phrase in sent:
                    answer = sent
                    break
            if not answer:
                answer = answer_phrase

            # print("====Final Answer Sentence: ", answer)
            query_entry.update_results({doc_id_final: answer})
            packs[doc_id] = pack
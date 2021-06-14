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
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import transformers, logging
transformers.logging.set_verbosity_error()
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
import warnings
warnings.simplefilter("ignore")


__all__ = [
    "BertRerankingProcessor"
]


class BertRerankingProcessor(MultiPackProcessor):

    def initialize(self, resources: Resources, configs: Config):
        self.resources = resources
        self.config = Config(configs, self.default_configs())

        #print(self.config)
        self.device = torch.device('cuda:0') \
            if torch.cuda.is_available() else torch.device('cpu')

        self.model = AutoModelForSequenceClassification.from_pretrained(self.config.model_name).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        

    @classmethod
    def default_configs(cls) -> Dict[str, Any]:
        configs = super().default_configs()
        model_name = 'amberoad/bert-multilingual-passage-reranking-msmarco'
        configs.update({
            "size": 5,
            "query_pack_name": "query",
            "field": "content",
            "model_dir": os.path.join(os.path.dirname(__file__), "models"),
            "max_seq_length": 512,
            "model_name": model_name,
            "batch_size" : 32
        })
        return configs

    def _process(self, input_pack: MultiPack):
        max_len = self.config.max_seq_length
        query_pack_name = self.config.query_pack_name
        batch_size = self.config.batch_size

        query_pack = input_pack.get_pack(self.config.query_pack_name)
        query_entry = list(query_pack.get(Query))[0]
        query_text = query_pack.text
        #print(input_pack.pack_ids)
        #print(type(list(query_entry.results.values())[0]))
        #print(query_entry, 'Here', query_pack.get(Query))
        #print(query_entry, "Before")
        doc_text_list = []
        doc_id_list = []
        
        for doc_id in input_pack.pack_names:
            if doc_id == query_pack_name:
                continue
            pack = input_pack.get_pack(doc_id)
            document_text = pack.text
            doc_id_final = pack.pack_name
            doc_text_list.append(document_text)
            doc_id_list.append(doc_id_final)
        query_text_list = [query_text] * len(doc_text_list)   
        # ## BERT Inference

        num_batches = int(len(doc_text_list)/batch_size) + (len(doc_text_list) % batch_size > 0)
        score_list = []
        for i in range(0, num_batches):
            start = i * batch_size
            end = (i+1) * batch_size
            if(end > len(doc_text_list)):
                end = len(doc_text_list)
            encodings = self.tokenizer(query_text_list[start:end], doc_text_list[start:end], padding = True, 
                                        truncation=True, max_length=max_len, return_tensors= 'pt').to(self.device)      
            self.model.eval()
            with torch.no_grad():
                logits = self.model(**encodings)
            pt_predictions = torch.nn.functional.softmax(logits[0], dim=1)
            scores = pt_predictions[:,1].tolist()
            score_list+=scores
        
        torch.cuda.empty_cache()

        for doc_id_final, score in zip(doc_id_list, score_list):
            query_entry.update_results({doc_id_final: score})
        #torch.cuda.empty_cache()
        #print(query_entry, "After")
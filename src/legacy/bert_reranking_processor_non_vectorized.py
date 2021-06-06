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
            "batch_size": 256
        })
        return configs

    def _process(self, input_pack: MultiPack):
        max_len = self.config.max_seq_length
        query_pack_name = self.config.query_pack_name

        query_pack = input_pack.get_pack(self.config.query_pack_name)
        query_entry = list(query_pack.get(Query))[0]
        query_text = query_pack.text
        #print(input_pack.pack_ids)
        #print(type(list(query_entry.results.values())[0]))
        print(" I am here finally", self.device)
        packs = {}
        #print(query_entry, 'Here', query_pack.get(Query))
        #print(query_entry, "Before")
        for doc_id in input_pack.pack_names:

            if doc_id == query_pack_name:
                continue
            pack = input_pack.get_pack(doc_id)
            document_text = pack.text
            doc_id_final = pack.pack_name
             # ## BERT Inference
            encodings = self.tokenizer(query_text, document_text, padding = True, model_max_length=max_len, return_tensors= 'pt').to(self.device)
            # model.eval()
            with torch.no_grad():
                logits = self.model(**encodings)
            pt_predictions = torch.nn.functional.softmax(logits[0], dim=1)
            score = pt_predictions.tolist()[0][1]

            query_entry.update_results({doc_id_final: score})
            packs[doc_id] = pack
        #print(query_entry, "After")
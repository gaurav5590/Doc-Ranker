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

import os, json

from typing import List, Optional, Tuple

from forte.common import ProcessExecutionException
from forte.data.data_pack import DataPack
from forte.evaluation.base import Evaluator
from forte.data.multi_pack import MultiPack
from forte.data.ontology import Query
from src.evaluators.eval_utils.ms_marco_eval_qa import compute_metrics_from_files



class QAEvaluator(Evaluator[MultiPack]):
    def __init__(self):
        super().__init__()
        self.predicted_results: List[Tuple[str, str, str, str, str]] = []
        self._score: Optional[float] = None

    def consume_next(self, pred_pack: MultiPack, _):
        #print(self.configs.pack_name)
        query_pack: DataPack = pred_pack.get_pack(self.configs.pack_name)
        query = list(query_pack.get(Query))[0]
        query_text = query_pack.text
        query_id = query_pack.pack_name
        #print(pred_pack.get_pack('passage_6').text)
        qa_results_dict = query.results
        # print("Printing")
        # for elem in self.predicted_results:
        #     print(elem)
        for p_name in pred_pack.pack_names:
            if p_name!=self.configs.pack_name:
                passage_id = pred_pack.get_pack(p_name).pack_name
                if qa_results_dict[passage_id]:
                    passage_text = pred_pack.get_pack(p_name).text
                    answer_text = qa_results_dict[passage_id]
                    self.predicted_results.append((query_id, query_text, passage_id, passage_text,answer_text))

    def get_result(self):
        # curr_dir = os.path.dirname(__file__)
        # output_file = os.path.join(curr_dir, self.configs.output_file)
        # gt_file = os.path.join(curr_dir, self.configs.ground_truth_file)
        # os.makedirs(os.path.dirname(output_file), exist_ok=True)

        # if self._score is None:
        #     with open(output_file, "w") as f:
        #         for result in self.predicted_results:
        #             f.write('\t'.join(result) + '\n')

        #     self._score = compute_metrics_from_files(gt_file, output_file)
        # return self._score
        # curr_dir = os.path.dirname(__file__)
        # output_file = os.path.join(curr_dir, self.configs.output_file)
        # gt_file = os.path.join(curr_dir, self.configs.ground_truth_file)
        # filtered_gt_file = os.path.join(curr_dir, self.configs.filtered_gt_file)
        output_file = self.configs.output_file
        gt_file = self.configs.ground_truth_file
        filtered_gt_file = self.configs.filtered_gt_file
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        if os.path.exists(output_file):
            os.remove(output_file)
        if os.path.exists(filtered_gt_file):
            os.remove(filtered_gt_file)

        # res = [
        #     (300674, 'this is query', 101, 'this is passage text', 'this is answer'),
        #     (125705, 'this is query', 102, 'this is passage text', 'this is answer'),
        #     (320792, 'this is query', 103, 'this is passage text', 'this is answer'),
        #     (89786, 'this is query', 104, 'this is passage text', 'this is answer'),
        # ]

        # Convert the self.predicted results to json style
        for row in self.predicted_results:
            row_dict = {'query_id':int(row[0]), 'answers':[row[-1]]}
            # print(row_dict)
            with open(output_file, 'a', newline='\n') as fp:
                json.dump(row_dict, fp)
                fp.write('\n')

        ## Filter the gt file to only self.predicted results query set
        q_ids = list(zip(*self.predicted_results))[0]
        q_ids = [int(x) for x in q_ids]
        with open(gt_file,'r') as f:
            for line in f:
                ans = eval(line)
                if ans['query_id'] in q_ids:
                    with open(filtered_gt_file, 'a') as fp:
                        # print(filtered_gt_file)
                        # print(ans)
                        json.dump(ans, fp)
                        fp.write('\n')

        self._score = compute_metrics_from_files(filtered_gt_file, output_file, self.configs.max_bleu_order)
    
        # if self._score is None:
        #     with open(output_file, "w") as f:
        #         for result in self.predicted_results:
        #             f.write('\t'.join(result) + '\n')

        #     self._score = compute_metrics_from_files(gt_file, output_file)
        return self._score

    @classmethod
    def default_configs(cls):
        return {
            'pack_name': None,
            'output_file': None,
            'ground_truth_file': None,
            'input_file': None,
            'max_bleu_order': None,
            'filtered_gt_file' : None,
        }
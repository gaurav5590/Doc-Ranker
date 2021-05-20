import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline

class MSMarcoTransformerModel(nn.Module):

    def __init__(self, model_name):
        super(MSMarcoTransformerModel, self).__init__()
        self.encoder = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.softmax = nn.Softmax(dim = 1)

    def forward(self, query_doc_vectors, train = False):
        logits = self.encoder(**query_doc_vectors)
        #predictions = self.softmax(logits)
        predictions = F.softmax(logits[0], dim=1)
        if not train:
            return predictions[:,1]
        else:
            return None  ## Include only if training the model


class QAModel(nn.Module):

    def __init__(self, task_name, model_name):
        super(QAModel, self).__init__()
        self.qa_pipeline = pipeline(task_name, model=model_name, tokenizer=model_name)
    def forward(self, query_doc_input, train = False):
        result = self.qa_pipeline(query_doc_input)
        if not train:
            return result
        else:
<<<<<<< HEAD
            return None ## Include only if training the model
=======
            return None ## Include only if training the model
>>>>>>> 0a424ae7cd8e163eddb21a35eab9e1f414b0089d

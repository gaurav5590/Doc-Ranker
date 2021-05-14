
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class TransformerModel(nn.Module):

    def __init__(self, model_name):
        super(TransformerModel, self).__init__()
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

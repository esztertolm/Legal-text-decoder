import torch
import torch.nn as nn
from transformers import AutoModel
from config import MODEL_NAME
from transformers.modeling_outputs import SequenceClassifierOutput


class LegalBERT(nn.Module):
    def __init__(self, num_labels):
        super(LegalBERT, self).__init__()
        
        self.bert = AutoModel.from_pretrained(MODEL_NAME)

        for param in self.bert.embeddings.parameters():
            param.requires_grad = False

        for i in range(7):
            for param in self.bert.encoder.layer[i].parameters():
                param.requires_grad = False

        self.hidden_size = self.bert.config.hidden_size
        
        self.classifier = nn.Sequential(
                nn.Linear(self.hidden_size, 256), # 768 -> 256
                nn.ReLU(),                        
                nn.Dropout(0.1),                  
                nn.Linear(256, num_labels)        # 256 -> 5
            )

    def forward(self, input_ids, attention_mask, token_type_ids=None, labels=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        cls_token = outputs.last_hidden_state[:, 0, :] 
        logits = self.classifier(cls_token)
            
        return SequenceClassifierOutput(
            loss=None,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
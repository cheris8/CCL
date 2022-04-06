import torch
from torch import nn

from transformers import RobertaModel


class Multiple_Choice_Model(nn.Module):
    def __init__(self, roberta_model: RobertaModel, dropout: float = None):
          super(Multiple_Choice_Model, self).__init__()
          self.roberta = roberta_model
          self.dropout = nn.Dropout(self.roberta.config.hidden_dropout_prob)
          self.classifier = nn.Linear(self.roberta.config.hidden_size, 1)
   
    def forward(self, input_ids: torch.tensor, attention_mask: torch.tensor, labels=None):
        num_choices = input_ids.shape[1]
          
        flat_input_ids = input_ids.view(-1, input_ids.size(-1)) # size : [batch_size*num_choices, seq_len]
        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1)) 

        outputs = self.roberta(
            input_ids = flat_input_ids, 
            attention_mask = flat_attention_mask,
            )

        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.view(-1, num_choices)

        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(reshaped_logits, labels)

        return loss, reshaped_logits

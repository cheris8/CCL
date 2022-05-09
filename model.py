import torch
from torch import nn
import transformers
from transformers import RobertaModel
import torch.nn.functional as F

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


class MultitaskModel(transformers.PreTrainedModel):
	def __init__(self, encoder, taskmodels_dict):
		"""
		Setting MultitaskModel up as a PretrainedModel allows us
		to take better advantage of Trainer features
		"""
		super().__init__(transformers.PretrainedConfig())

		self.encoder = encoder
		self.taskmodels_dict = nn.ModuleDict(taskmodels_dict)

	@classmethod
	def create(cls, model_name, model_type_dict, model_config_dict):
		"""
		This creates a MultitaskModel using the model class and config objects
		from single-task models.

		We do this by creating each single-task model, and having them share
		the same encoder transformer.
		"""
		shared_encoder = None
		taskmodels_dict = {}
		for task_name, model_type in model_type_dict.items():
			model = model_type.from_pretrained(
				model_name,
				config=model_config_dict[task_name],
			)
			if shared_encoder is None:
				shared_encoder = getattr(
					model, cls.get_encoder_attr_name(model))
			else:
				setattr(model, cls.get_encoder_attr_name(
					model), shared_encoder)
			taskmodels_dict[task_name] = model
		return cls(encoder=shared_encoder, taskmodels_dict=taskmodels_dict)

	@classmethod
	def get_encoder_attr_name(cls, model):
		"""
		The encoder transformer is named differently in each model "architecture".
		This method lets us get the name of the encoder attribute
		"""
		model_class_name = model.__class__.__name__
		if model_class_name.startswith("Bert"):
			return "bert"
		elif model_class_name.startswith("Roberta"):
			return "roberta"
		elif model_class_name.startswith("Albert"):
			return "albert"
		else:
			raise KeyError(f"Add support for new model {model_class_name}")

	def forward(self, task_name, **kwargs):
		return self.taskmodels_dict[task_name](**kwargs)

class BiEncoder(nn.Module):
	def __init__(self, context_bert_model, response_bert_model):
		super(BiEncoder, self).__init__()
		self.context_bert = context_bert_model
		self.response_bert = response_bert_model
		
	def forward(self, context_input_ids, context_input_masks,
							responses_input_ids, responses_input_masks):

		context = self.context_bert(input_ids = context_input_ids, attention_mask = context_input_masks)

		# print(output[1])

		response = self.response_bert(input_ids = responses_input_ids, attention_mask = responses_input_masks)

		context_vector = context[1]
		response_vector = response[1]
		dot_product = torch.matmul(context_vector, response_vector.t())  # [bs, bs]
		mask = torch.eye(context_input_ids.size(0)).to(context_input_ids.device)
		loss = F.log_softmax(dot_product, dim=-1) * mask
		loss = (-loss.sum(dim=1)).mean()
		return loss

	def encode_response(self, input_ids, input_masks):
		encoded = self.response_bert(input_ids=input_ids, attention_mask=input_masks)
		encoded_vector = encoded[1]
		return encoded_vector

	def encode_context(self, input_ids, input_masks):
		encoded = self.context_bert(input_ids=input_ids, attention_mask=input_masks)
		encoded_vector = encoded[1]
		return encoded_vector


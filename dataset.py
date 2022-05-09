import os
import pandas as pd

import torch
from torch.utils.data import Dataset


def load_atomic(root_dir):
    file_train = os.path.join(root_dir, 'data', 'atomic', 'train_adv-answer.jsonl')
    file_dev = os.path.join(root_dir, 'data', 'atomic', 'dev_adv-answer.jsonl')
    json_train = pd.read_json(path_or_buf=file_train, lines=True)
    json_dev = pd.read_json(path_or_buf=file_dev, lines=True)
    train_data = []
    for context, candidates in zip(json_train['context'].tolist(), json_train['candidates'].tolist()):
        train_data.append((context, candidates[0], candidates[1], candidates[2]))
    dev_data = []
    for context, candidates in zip(json_dev['context'].tolist(), json_dev['candidates'].tolist()):
        dev_data.append((context, candidates[0], candidates[1], candidates[2]))
    train_labels = json_train['correct'].tolist()
    dev_labels = json_dev['correct'].tolist()
    return train_data, train_labels, dev_data, dev_labels


def load_cwwv(root_dir):
    file_train = os.path.join(root_dir, 'data', 'cwwv', 'train_adv-answer.jsonl')
    file_dev = os.path.join(root_dir, 'data', 'cwwv', 'dev_adv-answer.jsonl')
    json_train = pd.read_json(path_or_buf=file_train, lines=True)
    json_dev = pd.read_json(path_or_buf=file_dev, lines=True)
    train_data = []
    for sample in json_train['question']:
        question = sample['stem']
        answer_a = sample['choices'][0]['text']
        answer_b = sample['choices'][1]['text']
        answer_c = sample['choices'][2]['text']
        train_data.append((question, answer_a, answer_b, answer_c))
    dev_data = []
    for sample in json_dev['question']:
        question = sample['stem']
        answer_a = sample['choices'][0]['text']
        answer_b = sample['choices'][1]['text']
        answer_c = sample['choices'][2]['text']
        dev_data.append((question, answer_a, answer_b, answer_c))
    train_answerkeys = json_train['answerKey'].tolist()
    dev_answerkeys = json_dev['answerKey'].tolist()
    answerkey_to_label = {'A':0, 'B':1, 'C':2}
    train_labels = [answerkey_to_label[answerkey] for answerkey in train_answerkeys]
    dev_labels = [answerkey_to_label[answerkey] for answerkey in dev_answerkeys]
    return train_data, train_labels, dev_data, dev_labels


def load_siqa(root_dir, mode):
    if mode == 'train':
        file_path = os.path.join(root_dir, 'data', 'siqa', "train-predictions.jsonl")
    elif mode == 'dev':
        file_path = os.path.join(root_dir, 'data', 'siqa', "dev-predictions.jsonl")
    json_file = pd.read_json(path_or_buf=file_path, lines=True)
    data = [elem for elem in zip(json_file['context'].tolist(), json_file['question'].tolist(), json_file['answerA'].tolist(), json_file['answerB'].tolist(), json_file['answerC'].tolist())]
    corrects = json_file['correct'].tolist()
    correct_to_label = {'A':0, 'B':1, 'C':2}
    labels = [correct_to_label[correct] for correct in corrects]
    
    return data, labels


def load_csqa(root_dir, mode):
    if mode == 'train':
        file_path = os.path.join(root_dir, 'data', 'csqa', 'train.jsonl')
    elif mode == 'dev':
        file_path = os.path.join(root_dir, 'data', 'csqa', 'dev.jsonl')
    json_file = pd.read_json(path_or_buf=file_path, lines=True)
    data = []
    for sample in json_file['question']:
        question = sample['stem']
        answer_a = sample['choices'][0]['text']
        answer_b = sample['choices'][1]['text']
        answer_c = sample['choices'][2]['text']
        answer_d = sample['choices'][3]['text']
        answer_e = sample['choices'][4]['text']
        data.append((question, answer_a, answer_b, answer_c, answer_d, answer_e))
    answerkeys = json_file['answerKey'].tolist()
    answerkey_to_label = {'A':0, 'B':1, 'C':2, 'D':3, 'E':4}
    labels = [answerkey_to_label[answerkey] for answerkey in answerkeys]
    return data, labels


def load_cmqa(root_dir, mode):
    if mode == 'train':
        file_path = os.path.join(root_dir, 'data', 'cmqa', "train.jsonl")
    elif mode == 'dev':
        file_path = os.path.join(root_dir, 'data', 'cmqa', "dev.jsonl")
    json_file = pd.read_json(path_or_buf=file_path, lines=True)
    data = [elem for elem in zip(json_file['context'].tolist(), json_file['question'].tolist(), json_file['answer0'].tolist(), json_file['answer1'].tolist(), json_file['answer2'].tolist(), json_file['answer3'].tolist())]
    labels = json_file['label'].tolist()
    return data, labels


def load_piqa(root_dir, mode):
    if mode == 'train':
        file_path = os.path.join(root_dir, 'data', 'piqa', 'train.jsonl')
    elif mode == 'dev':
        file_path = os.path.join(root_dir, 'data', 'piqa', 'dev.jsonl')
    json_file = pd.read_json(path_or_buf=file_path, lines=True)
    data = [elem for elem in zip(json_file['goal'].tolist(), json_file['sol1'].tolist(), json_file['sol2'].tolist())]
    labels = json_file['label'].tolist()
    return data, labels



def load_siqa_answersheet(root_dir):
    file_path = os.path.join(root_dir, 'data', 'siqa', "train-predictions.jsonl")
    json_file = pd.read_json(path_or_buf=file_path, lines=True)
    context = [elem for elem in zip(json_file['context'].tolist(), json_file['question'].tolist())]
    answer_candidate = [cand for cand in zip(json_file['answerA'].tolist(), json_file['answerB'].tolist(), json_file['answerC'].tolist())]
    corrects = json_file['correct'].tolist()
    correct_to_label = {'A':0, 'B':1, 'C':2}
    labels = [correct_to_label[correct] for correct in corrects]
    train_data=[]
    assert len(context) == len(answer_candidate) == len(labels)
    for k in range(len(context)):
        answer = answer_candidate[k][labels[k]]
        train_data.append((context[k], answer))

    
    return train_data

def load_csqa_answersheet(root_dir):

    file_path = os.path.join(root_dir, 'data', 'csqa', 'train.jsonl')
    json_file = pd.read_json(path_or_buf=file_path, lines=True)
    data = []
    answerkeys = json_file['answerKey'].tolist()
    print(answerkeys)
    answerkey_to_label = {'A':0, 'B':1, 'C':2, 'D':3, 'E':4}
    labels = [answerkey_to_label[answerkey] for answerkey in answerkeys]
    questions = json_file['question'].tolist()
    assert len(questions) == len(labels)
    idx=0
    for sample in json_file['question']:
        question = sample['stem']
        get_idx = labels[idx]
        answer = sample['choices'][get_idx]['text']
        data.append((question, answer))
        idx+=1

    return data

def load_cmqa_answersheet(root_dir):
    file_path = os.path.join(root_dir, 'data', 'cmqa', "train.jsonl")
    json_file = pd.read_json(path_or_buf=file_path, lines=True)
    context = [elem for elem in zip(json_file['context'].tolist(), json_file['question'].tolist())]
    answer_candidate = [cand for cand in zip(json_file['answer0'].tolist(), json_file['answer1'].tolist(), json_file['answer2'].tolist(), json_file['answer3'].tolist())]
    labels = json_file['label'].tolist()
    assert len(context) == len(answer_candidate) == len(labels)
    train_data=[]
    for k in range(len(context)):
        answer = answer_candidate[k][labels[k]]
        train_data.append((context[k], answer))        
    return train_data

def load_piqa_answersheet(root_dir):
    file_path = os.path.join(root_dir, 'data', 'piqa', 'train.jsonl')
    json_file = pd.read_json(path_or_buf=file_path, lines=True)
    context = json_file['goal'].tolist()
    answer_candidate = [cand for cand in zip(json_file['sol1'].tolist(), json_file['sol2'].tolist())]
    labels = json_file['label'].tolist()
    assert len(context) == len(answer_candidate) == len(labels)
    train_data=[]
    for k in range(len(context)):
        answer = answer_candidate[k][labels[k]]
        train_data.append((context[k], answer))        
    return train_data

class BiEncoderDataset(Dataset):
    def __init__(self, tokenizer, input):
        self.roberta_tokenizer = tokenizer
        self.input = input
        self.context_tokenized=[]
        self.answer_tokenized=[]
        for idx in self.input:
            context = idx[0]
            answer = idx[1]
            encoded_context = self.roberta_tokenizer(context, padding=True, truncation=True, return_tensors='pt')
            encoded_answer = self.roberta_tokenizer(answer, padding=True, truncation=True, return_tensors='pt')
            self.context_tokenized.append(encoded_context)
            self.answer_tokenized.append(encoded_answer)

    def __getitem__(self, idx):
        return (self.context_tokenized[idx], self.answer_tokenized[idx])

    def __len__(self):
        return len(self.input)

class SingleEncoderDataset(Dataset):
    def __init__(self, tokenizer, input):
        self.roberta_tokenizer = tokenizer
        self.input = input
        self.context_tokenized=[]
        self.answer_tokenized=[]
        for idx in self.input:
            context = idx
            encoded_context = self.roberta_tokenizer(context, padding=True, truncation=True, return_tensors='pt')
            self.context_tokenized.append(encoded_context)

    def __getitem__(self, idx):
        return (self.context_tokenized[idx], 0)

    def __len__(self):
        return len(self.input)

        
class AtomicDataset(Dataset):
    def __init__(self, tokenizer, x, y):
        # x: list of tuples containing (context, answer1, answer2, answer3)
        # y: list of indices of the correct answer
        self.roberta_tokenizer = tokenizer
        self.x = x
        self.y = y
        self.x_tokenized = []
        for point in self.x:
            input_answers = [point[1], point[2], point[3]]
            num_choices = len(input_answers)
            input_context_question = [point[0]]*num_choices
            encoded_text_train = self.roberta_tokenizer(input_context_question, input_answers, return_tensors='pt', padding=True)
            self.x_tokenized.append(encoded_text_train)

    def __getitem__(self, idx):
        return (self.x_tokenized[idx], self.y[idx])

    def __len__(self):
        return len(self.x)


class CwwvDataset(Dataset):
    def __init__(self, tokenizer, x, y):
        # x: list of tuples containing (question, answer_a, answer_b, answer_c)
        # y: list of indices of the correct answer
        self.roberta_tokenizer = tokenizer
        self.x = x
        self.y = y
        self.x_tokenized = []
        for point in self.x:
            input_answers = [point[1], point[2], point[3]]
            num_choices = len(input_answers)
            input_context_question = [point[0]]*num_choices
            encoded_text_train = self.roberta_tokenizer(input_context_question, input_answers, return_tensors='pt', padding=True)
            self.x_tokenized.append(encoded_text_train)

    def __getitem__(self, idx):
        return (self.x_tokenized[idx], self.y[idx])

    def __len__(self):
        return len(self.x)


class SocialiqaDataset(Dataset):
    def __init__(self, tokenizer, x, y):
        # x: list of tuples containing (context, question, answer1, answer2, answer3)
        # y: list of indices of the correct answer
        self.roberta_tokenizer = tokenizer
        self.x = x
        self.y = y
        self.x_tokenized = []
        for point in self.x:
            input_answers = [point[2], point[3], point[4]]
            num_choices = len(input_answers)
            input_context_question = [point[0] + self.roberta_tokenizer.sep_token + point[1]] * num_choices
            encoded_text_train = self.roberta_tokenizer(input_context_question, input_answers, padding=True, return_tensors='pt')
            self.x_tokenized.append(encoded_text_train)

    def __getitem__(self, idx):
        return (self.x_tokenized[idx], self.y[idx])

    def __len__(self):
        return len(self.x)


class CommonsenseqaDataset(Dataset):
    def __init__(self, tokenizer, x, y):
        # x: list of tuples containing (question, answer_a, answer_b, answer_c, answer_d, answer_e)
        # y: list of indices of the correct answer
        self.roberta_tokenizer = tokenizer
        self.x = x
        self.y = y
        self.x_tokenized = []
        for point in self.x:
            input_answers = [point[1], point[2], point[3], point[4], point[5]]
            num_choices = len(input_answers)
            input_context_question = [point[0]] * num_choices
            encoded_text_train = self.roberta_tokenizer(input_context_question, input_answers, padding=True, return_tensors='pt')
            self.x_tokenized.append(encoded_text_train)

    def __getitem__(self, idx):
        return (self.x_tokenized[idx], self.y[idx])

    def __len__(self):
        return len(self.x)


class CosmosqaDataset(Dataset):
    def __init__(self, tokenizer, x, y):
        # x: list of tuples containing (context, question, answer1, answer2, answer3, answer4)
        # y: list of indices of the correct answer
        self.roberta_tokenizer = tokenizer
        self.x = x
        self.y = y
        self.x_tokenized = []
        for point in self.x:
            input_answers = [point[2], point[3], point[4], point[5]]
            num_choices = len(input_answers)
            input_context_question = [point[0] + self.roberta_tokenizer.sep_token + point[1]]*num_choices
            encoded_text_train = self.roberta_tokenizer(input_context_question, input_answers, padding=True, return_tensors='pt')
            self.x_tokenized.append(encoded_text_train)

    def __getitem__(self, idx):
        return (self.x_tokenized[idx], self.y[idx])

    def __len__(self):
        return len(self.x)


class PhysicaliqaDataset(Dataset):
    def __init__(self, tokenizer, x, y):
        # x: list of tuples containing (goal, sol1, sol2)
        # y: list of indices of the correct answer
        self.roberta_tokenizer = tokenizer
        self.x = x
        self.y = y
        self.x_tokenized = []
        for point in self.x:
            input_answers = [point[1], point[2]]
            num_choices = len(input_answers)
            input_context_question = [point[0]]*num_choices
            input_answers = [point[1], point[2]]
            encoded_text_train = self.roberta_tokenizer(input_context_question, input_answers, padding=True, truncation=True, return_tensors='pt')
            self.x_tokenized.append(encoded_text_train)

    def __getitem__(self, idx):
        return (self.x_tokenized[idx], self.y[idx])

    def __len__(self):
        return len(self.x)


class SocialiqaDatasetForKD(Dataset):
    def __init__(self, tokenizer, x, y, pseudo_y):
        self.roberta_tokenizer = tokenizer
        self.x = x
        self.y = y
        self.pseudo_y = pseudo_y
        self.x_tokenized = []
        for point in self.x:
            input_answers = [point[2], point[3], point[4]]
            num_choices = len(input_answers)
            input_context_question = [point[0] + self.roberta_tokenizer.sep_token + point[1]] * num_choices
            encoded_text_train = self.roberta_tokenizer(input_context_question, input_answers, padding=True, return_tensors='pt')
            self.x_tokenized.append(encoded_text_train)

    def __getitem__(self, idx):
        return (self.x_tokenized[idx], self.y[idx], self.pseudo_y[idx])

    def __len__(self):
        return len(self.x)


class CommonsenseqaDatasetForKD(Dataset):
    def __init__(self, tokenizer, x, y, pseudo_y):
        # x: list of tuples containing (question, answer_a, answer_b, answer_c, answer_d, answer_e)
        # y: list of indices of the correct answer
        self.roberta_tokenizer = tokenizer
        self.x = x
        self.y = y
        self.pseudo_y = pseudo_y
        self.x_tokenized = []
        for point in self.x:
            input_answers = [point[1], point[2], point[3], point[4], point[5]]
            num_choices = len(input_answers)
            input_context_question = [point[0]]*num_choices
            encoded_text_train = self.roberta_tokenizer(input_context_question, input_answers, padding=True, return_tensors='pt')
            self.x_tokenized.append(encoded_text_train)

    def __getitem__(self, idx):
        return (self.x_tokenized[idx], self.y[idx], self.pseudo_y[idx])

    def __len__(self):
        return len(self.x)


class CosmosqaDatasetForKD(Dataset):
    def __init__(self, tokenizer, x, y, pseudo_y):
        # x: list of tuples containing (context, question, answer1, answer2, answer3, answer4)
        # y: list of indices of the correct answer
        self.roberta_tokenizer = tokenizer
        self.x = x
        self.y = y
        self.pseudo_y = pseudo_y
        self.x_tokenized = []
        for point in self.x:
            input_answers = [point[2], point[3], point[4], point[5]]
            num_choices = len(input_answers)
            input_context_question = [point[0] + self.roberta_tokenizer.sep_token + point[1]]*num_choices
            encoded_text_train = self.roberta_tokenizer(input_context_question, input_answers, padding=True, return_tensors='pt')
            self.x_tokenized.append(encoded_text_train)

    def __getitem__(self, idx):
        return (self.x_tokenized[idx], self.y[idx], self.pseudo_y[idx])

    def __len__(self):
        return len(self.x)


class PhysicaliqaDatasetForKD(Dataset):
    def __init__(self, tokenizer, x, y, pseudo_y):
        # x: list of tuples containing (goal, sol1, sol2)
        # y: list of indices of the correct answer
        self.roberta_tokenizer = tokenizer
        self.x = x
        self.y = y
        self.pseudo_y = pseudo_y
        self.x_tokenized = []
        for point in self.x:
            input_answers = [point[1], point[2]]
            num_choices = len(input_answers)
            input_context_question = [point[0]]*num_choices
            input_answers = [point[1], point[2]]
            encoded_text_train = self.roberta_tokenizer(input_context_question, input_answers, padding=True, truncation=True, return_tensors='pt')
            self.x_tokenized.append(encoded_text_train)

    def __getitem__(self, idx):
        return (self.x_tokenized[idx], self.y[idx], self.pseudo_y[idx])

    def __len__(self):
        return len(self.x)


def prepare_batch(batch):
    """
    This collate function will pad the batch to be the same length. This requires
    flattening, then unflattening for the multiple choice format.
    One example will be a list of length 'num choices', each element being a list
    of (encoded) tokens representing qustion/answer [sep] choicex
    """
    # batch: [batch_size, (text, label)]
    batch_size = len(batch)

    features, labels = zip(*batch)
    # features: tuple of length batch_size, 
    #        each element is a dict with keys = ["input_ids", "attention_mask"]
    # labels: tuple of int indicies length batch_size
    num_choices = len(features[0]["input_ids"])
    
    # flatten
    input_ids_features = []
    attention_mask_features = []
    max_len = 0
    for feature in features:
        for i in range(num_choices):
            input_ids_features.append(feature['input_ids'][i])
            attention_mask_features.append(feature['attention_mask'][i])
            if feature['input_ids'][i].shape[0] > max_len:
                max_len = feature['input_ids'][i].shape[0]
    # flattened_features list length num_choices*batch_size

    # padding
    padded_input_ids_features = []
    padded_attention_mask_features = []
    for input_ids, attention_mask in zip(input_ids_features, attention_mask_features):
        pad_len = max_len - input_ids.shape[0]
        if pad_len > 0:
            padded_input_ids = torch.cat([input_ids, torch.LongTensor([0] * pad_len)])
            padded_attention_mask = torch.cat([attention_mask, torch.LongTensor([0] * pad_len)])
            padded_input_ids_features.append(padded_input_ids)
            padded_attention_mask_features.append(padded_attention_mask)
        else:
            padded_input_ids_features.append(input_ids)
            padded_attention_mask_features.append(attention_mask)

    # un-flatten
    texts = {}
    texts['input_ids'] = torch.stack(padded_input_ids_features).view(batch_size, num_choices, -1)
    texts['attention_mask'] = torch.stack(padded_attention_mask_features).view(batch_size, num_choices, -1)

    batch = (texts, torch.LongTensor(labels))
    return batch


def prepare_batch_KD(batch):
    # batch: [batch_size, (text, label)]
    batch_size = len(batch)

    features, labels, pseudo_labels = zip(*batch)
    # features: tuple of length batch_size, 
    #        each element is a dict with keys = ["input_ids", "attention_mask"]
    # labels: tuple of int indicies length batch_size
    num_choices = len(features[0]["input_ids"])
    
    # flatten
    input_ids_features = []
    attention_mask_features = []
    max_len = 0
    for feature in features:
        for i in range(num_choices):
            input_ids_features.append(feature['input_ids'][i])
            attention_mask_features.append(feature['attention_mask'][i])
            if feature['input_ids'][i].shape[0] > max_len:
                max_len = feature['input_ids'][i].shape[0]
    # flattened_features list length num_choices*batch_size

    # padding
    padded_input_ids_features = []
    padded_attention_mask_features = []
    for input_ids, attention_mask in zip(input_ids_features, attention_mask_features):
        pad_len = max_len - input_ids.shape[0]
        if pad_len > 0:
            padded_input_ids = torch.cat([input_ids, torch.LongTensor([0] * pad_len)])
            padded_attention_mask = torch.cat([attention_mask, torch.LongTensor([0] * pad_len)])
            padded_input_ids_features.append(padded_input_ids)
            padded_attention_mask_features.append(padded_attention_mask)
        else:
            padded_input_ids_features.append(input_ids)
            padded_attention_mask_features.append(attention_mask)

    # un-flatten
    texts = {}
    texts['input_ids'] = torch.stack(padded_input_ids_features).view(batch_size, num_choices, -1)
    texts['attention_mask'] = torch.stack(padded_attention_mask_features).view(batch_size, num_choices, -1)
    
    flatten_pseudo_labels = sum(pseudo_labels, [])
    tensor_pseudo_labels = torch.tensor(flatten_pseudo_labels).view(batch_size, num_choices)
    tensor_labels = torch.LongTensor(labels)

    batch = (texts, tensor_labels, tensor_pseudo_labels)
    return batch


def biencoder_batch(batch):
    
    batch_size = len(batch)
    contexts, answers = zip(*batch)

    input_ids_context=[]
    attention_mask_context=[]
    max_len_for_context=0
    for context in contexts:
        input_ids_context.append(context['input_ids'][0])
        attention_mask_context.append(context['attention_mask'][0])
        if context['input_ids'][0].shape[0] > max_len_for_context:
            max_len_for_context = context['input_ids'][0].shape[0]

    input_ids_answer=[]
    attention_mask_answer=[]
    max_len_for_answer=0
    for answer in answers:
        input_ids_answer.append(answer['input_ids'][0])
        attention_mask_answer.append(answer['attention_mask'][0])
        if answer['input_ids'][0].shape[0] > max_len_for_answer:
            max_len_for_answer = answer['input_ids'][0].shape[0]

    padded_input_ids_context=[]
    padded_input_ids_answer=[]
    padded_attention_mask_context=[]
    padded_attention_mask_answer=[]
    for input_ids, attention_mask in zip(input_ids_context, attention_mask_context):
        padding_len = max_len_for_context - input_ids.shape[0]
        if padding_len > 0:
            padded_input_ids = torch.cat([input_ids, torch.LongTensor([0] * padding_len)])
            padded_attention_mask = torch.cat([attention_mask, torch.LongTensor([0] * padding_len)])
            padded_input_ids_context.append(padded_input_ids)
            padded_attention_mask_context.append(padded_attention_mask)            
        else:
            padded_input_ids_context.append(input_ids)
            padded_attention_mask_context.append(attention_mask)

    for input_ids, attention_mask in zip(input_ids_answer, attention_mask_answer):
        padding_len = max_len_for_answer - input_ids.shape[0]
        if padding_len > 0:
            padded_input_ids = torch.cat([input_ids, torch.LongTensor([0] * padding_len)])
            padded_attention_mask = torch.cat([attention_mask, torch.LongTensor([0] * padding_len)])
            padded_input_ids_answer.append(padded_input_ids)
            padded_attention_mask_answer.append(padded_attention_mask)            
        else:
            padded_input_ids_answer.append(input_ids)
            padded_attention_mask_answer.append(attention_mask)
            
    

    batch = (torch.stack(padded_input_ids_context), torch.stack(padded_input_ids_answer), torch.stack(padded_attention_mask_context), torch.stack(padded_attention_mask_answer))
    return batch


def singleencoder_batch(batch):
    
    batch_size = len(batch)
    contexts, _ = zip(*batch)

    input_ids_context=[]
    attention_mask_context=[]
    max_len_for_context=0
    for context in contexts:
        input_ids_context.append(context['input_ids'][0])
        attention_mask_context.append(context['attention_mask'][0])
        if context['input_ids'][0].shape[0] > max_len_for_context:
            max_len_for_context = context['input_ids'][0].shape[0]


    padded_input_ids_context=[]
    padded_attention_mask_context=[]
    for input_ids, attention_mask in zip(input_ids_context, attention_mask_context):
        padding_len = max_len_for_context - input_ids.shape[0]
        if padding_len > 0:
            padded_input_ids = torch.cat([input_ids, torch.LongTensor([0] * padding_len)])
            padded_attention_mask = torch.cat([attention_mask, torch.LongTensor([0] * padding_len)])
            padded_input_ids_context.append(padded_input_ids)
            padded_attention_mask_context.append(padded_attention_mask)            
        else:
            padded_input_ids_context.append(input_ids)
            padded_attention_mask_context.append(attention_mask)
            
    

    batch = (torch.stack(padded_input_ids_context), torch.stack(padded_attention_mask_context))
    return batch
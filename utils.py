import os
import random
import json
import pathlib
import numpy as np
from numpy.lib.npyio import _savez_compressed_dispatcher
import pandas as pd
from tqdm.auto import tqdm
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from sentence_transformers import SentenceTransformer, util
from transformers import RobertaTokenizer, RobertaModel, AdamW
from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report, accuracy_score, f1_score

from dataset import load_siqa, load_csqa, load_cmqa, load_piqa, prepare_batch_KD, prepare_batch
from dataset import SocialiqaDataset, CommonsenseqaDataset, CosmosqaDataset, PhysicaliqaDataset
from dataset import SocialiqaDatasetForKD, CommonsenseqaDatasetForKD, CosmosqaDatasetForKD, PhysicaliqaDatasetForKD

import warnings
warnings.filterwarnings(action='ignore')
random.seed(42)
torch.manual_seed(42)
np.random.seed(42)

TASK_CONFIG = {
    'siqa':{'load':load_siqa, 'dataset':SocialiqaDataset, 'dataset-kd':SocialiqaDatasetForKD, 'context':True, 'question':True, 'num_choices':3, 'target_names':['Answer A', 'Answer B', 'Answer C'] },
    'csqa':{'load':load_csqa, 'dataset':CommonsenseqaDataset, 'dataset-kd':CommonsenseqaDatasetForKD, 'context':False, 'question':True, 'num_choices':5, 'target_names':['Answer A', 'Answer B', 'Answer C', 'Answer D', 'Answer E']},
    'cmqa':{'load':load_cmqa, 'dataset':CosmosqaDataset, 'dataset-kd':CosmosqaDatasetForKD, 'context':True, 'question':True, 'num_choices':4, 'target_names':['Answer A', 'Answer B', 'Answer C', 'Answer D']},
    'piqa':{'load':load_piqa, 'dataset':PhysicaliqaDataset, 'dataset-kd':PhysicaliqaDatasetForKD, 'context':False, 'question':True, 'num_choices':2, 'target_names':['Answer A', 'Answer B']}
}


def KLDivergence(A, B):
    return np.sum([v for v in A * np.log2(A/B)])

def JSDivergence(P, Q):
    """Compute the Jensen-Shannon divergence between two probability distributions.
    Input
    -----
    P, Q : array-like Probability distributions of equal length that sum to 1
    """
    P = np.array(P)
    Q = np.array(Q)
    M = 0.5 * (P + Q)
    return 0.5 * (KLDivergence(P, M) + KLDivergence(Q, M))

def make_dir(d):
    pathlib.Path(d).mkdir(parents=True, exist_ok=True)

def save_samples(args, texts, labels, logits):
    file = {'texts':texts, 'labels':labels, 'logits':logits}
    with open(os.path.join(args.base_dir, 'additional_samples.json'), 'w') as f:
        json.dump(file, f)

def sort_samples_by_similarity(args, texts, labels, logits, sampling_type):
    # reference : uniform, skewed, wrong, reversed
    num_choices = TASK_CONFIG[args.cur_task]['num_choices']
    scores = []
    if sampling_type == 'uniform':
        reference_list = [1/num_choices]*num_choices
        for label, logit in zip(labels, logits):
            logit_softmaxed = nn.Softmax(dim=1)(torch.tensor(logit)).squeeze(0)
            score = JSDivergence(reference_list, logit_softmaxed)
            scores.append(score)
    elif sampling_type == 'skewed':
        for label, logit in zip(labels, logits):
            logit_softmaxed = nn.Softmax(dim=1)(torch.tensor(logit)).squeeze(0)
            reference_list = [0]*num_choices
            reference_list[label] = 1
            score = JSDivergence(reference_list, logit_softmaxed)
            scores.append(score)
    elif sampling_type == 'reversed':
        for label, logit in zip(labels, logits):
            logit_softmaxed = nn.Softmax(dim=1)(torch.tensor(logit)).squeeze(0)
            scores_per_reference_list = []
            for i in range(num_choices):
                if i == label:
                    pass
                else:
                    reference_list = [0]*num_choices
                    reference_list[i] = 1
                    reference_list = np.clip(reference_list, 1e-10, 1 - 1e-10) 
                    score = JSDivergence(reference_list, logit_softmaxed)
                    scores_per_reference_list.append(score)
            scores.append(min(scores_per_reference_list))
    assert len(texts) == len(labels) == len(logits) == len(scores)
    sorted_texts = [text for _, text in sorted(zip(scores, texts))]
    sorted_labels = [label for _, label in sorted(zip(scores, labels))]
    sorted_logits = [logit for _, logit in sorted(zip(scores, logits))]
    return sorted_texts, sorted_labels, sorted_logits


def get_best_model_path(path):
    all_files = os.listdir(path)
    files = [file for file in all_files if file.endswith('.pt')]
    accuracies = [float(file[-9:-3]) for file in files]
    assert len(files) == len(accuracies)
    acc_to_file = {}
    for file, acc in zip(files, accuracies):
        acc_to_file[acc] = file
    best_acc = max(acc_to_file.keys())
    return os.path.join(path, acc_to_file.get(best_acc))

def get_best_loss_path(path):
    all_files = os.listdir(path)
    files = [file for file in all_files if file.endswith('.pt')]
    losses = [float(file[-9:-3]) for file in files]
    assert len(files) == len(losses)
    loss_to_file = {}
    for file, loss in zip(files, losses):
        loss_to_file[loss] = file
    best_loss = min(loss_to_file.keys())
    return os.path.join(path, loss_to_file.get(best_loss))

def get_predictions_from_model(args, model, tokenizer, texts, labels):
    dataset = TASK_CONFIG[args.cur_task]['dataset'](tokenizer, texts, labels)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)

    model = model.to(args.device)
    model.eval()

    epoch_logits = []
    with torch.no_grad():
        for batch in tqdm(data_loader):
            input_ids = batch[0]['input_ids'].to(args.device)
            attention_mask = batch[0]['attention_mask'].to(args.device)
            labels = batch[1].to(args.device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            _, logits = outputs[0], outputs[1]
            epoch_logits.append(logits.to('cpu').tolist())
            torch.cuda.empty_cache()
    return epoch_logits


def right_wrong_split(texts, labels, logits):
    right_texts = [] ; wrong_texts = []
    right_labels = [] ; wrong_labels = []
    right_logits = [] ; wrong_logits = []
    for text, label, logit in zip(texts, labels, logits):
        # TODO np softmax
        if np.argmax(logit) == label:
            right_texts.append(text)
            right_labels.append(label)
            right_logits.append(logit)
        elif np.argmax(logit) != label:
            wrong_texts.append(text)
            wrong_labels.append(label)
            wrong_logits.append(logit)
    return right_texts, wrong_texts, right_labels, wrong_labels, right_logits, wrong_logits


def get_answer_options_pool(texts, num_choices):
    # set the pool of answer options for replacement
    answer_options_pool = []
    for text in texts:
        for i in range(1, num_choices+1):
            answer_options_pool.append(text[-i])
    answer_options_pool = list(set(answer_options_pool))
    return answer_options_pool


def create_data_per_sample(num_choices, start_idx, target_text, target_label, answer_options_pool):
    target_text = list(target_text)
    text_per_sample = [] ; label_per_sample = []
    for i in range(0, len(answer_options_pool)-num_choices):
        for j, k in zip(range(num_choices), range(i, i+num_choices)):
            if j == target_label:
                pass
            else:
                target_text[start_idx+j] = answer_options_pool[k]
        text_per_sample.append(tuple(target_text))
        label_per_sample.append(target_label)
    return text_per_sample, label_per_sample


def modify_wrong_to_right_samples_by_sbert(args, model, tokenizer, wrong_texts, wrong_labels, answer_options_pool):
    
    print('Number of incorrectly predicted samples:', len(wrong_labels))

    num_choices = TASK_CONFIG[args.cur_task]['num_choices']
    start_idx = TASK_CONFIG[args.cur_task]['context'] + TASK_CONFIG[args.cur_task]['question']

    model.to(args.device)
    model.eval()

    sbert = SentenceTransformer('all-MiniLM-L6-v2')
    answer_options_embeddings = sbert.encode(answer_options_pool)

    # find and save correctly predicted samples
    right_texts = []; right_labels = []; right_logits = []
    done_cnt = 0
    for wrong_text, wrong_label in zip(wrong_texts, wrong_labels):
        print(f'Trying {done_cnt}-th sample of {len(wrong_labels)} samples ...')
        anchor_embedding = sbert.encode(wrong_text[start_idx+wrong_label]) # ground-truth embedding
        cos_sim = util.cos_sim(anchor_embedding, answer_options_embeddings).squeeze(0).tolist()
        answer_options_sorted = [answer_option for sim, answer_option in sorted(zip(cos_sim, answer_options_pool), reverse=True) if (sim < 0.9)]

        text_per_sample, label_per_sample = create_data_per_sample(num_choices, start_idx, wrong_text, wrong_label, answer_options_sorted)
        dataset = TASK_CONFIG[args.cur_task]['dataset'](tokenizer, text_per_sample, label_per_sample)
        data_loader = DataLoader(dataset, batch_size=1, shuffle=False)
        torch.cuda.empty_cache()
        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                input_ids = batch[0]['input_ids'].to(args.device)
                attention_mask = batch[0]['attention_mask'].to(args.device)
                label = batch[1].to(args.device)
                output = model(input_ids=input_ids, attention_mask=attention_mask, labels=label)
                _, logit = output[0], output[1]
                pred = torch.argmax(nn.Softmax(dim=1)(logit), dim=1)
                if pred == label:
                    print(f'Select {batch_idx} for {done_cnt}-th sample ...')
                    right_texts.append(text_per_sample[batch_idx])
                    right_labels.append(label_per_sample[batch_idx])
                    right_logits.append(logit.to('cpu').tolist())
                    done_cnt += 1
                    torch.cuda.empty_cache()
                    break
                torch.cuda.empty_cache()
    torch.cuda.empty_cache()
    save_samples(args, right_texts, right_labels, right_logits)
    return right_texts, right_labels, right_logits

def modify_wrong_to_right_samples_by_biencoder(args, model, tokenizer, biencoder_model, encoded_answer_candidate_dir, wrong_texts, wrong_labels, answer_options_pool):
    
    print('Number of incorrectly predicted samples:', len(wrong_labels))

    num_choices = TASK_CONFIG[args.cur_task]['num_choices']
    start_idx = TASK_CONFIG[args.cur_task]['context'] + TASK_CONFIG[args.cur_task]['question']

    model.to(args.device)
    model.eval()

    biencoder_model.to(args.device)
    biencoder_model.eval()

    with open(encoded_answer_candidate_dir, 'rb') as f:
        answer_candidate_embedding = pickle.load(f)
    answer_option_embedding = [answer_candidate_embedding[answer] for answer in answer_candidate_embedding]
    answer_options = [answer for answer in answer_candidate_embedding]


    # find and save correctly predicted samples
    right_texts = []; right_labels = []; right_logits = []
    done_cnt = 0
    for wrong_text, wrong_label in zip(wrong_texts, wrong_labels):
        print(f'Trying {done_cnt}-th sample of {len(wrong_labels)} samples ...')
        get_text = wrong_text[start_idx+wrong_label]
        encoded_text = tokenizer(get_text, padding=True, truncation=True, return_tensors='pt')
        anchor_embedding = biencoder_model.encode_context(input_ids=encoded_text['input_ids'].to(args.device), input_masks=encoded_text['attention_mask'].to(args.device)).detach().cpu()
        cos_similiarity = nn.CosineSimilarity(dim=1)
        cos_sim = [cos_similiarity(anchor_embedding, answer_candidate) for answer_candidate in answer_option_embedding]
        answer_options_sorted = [answer_option for sim, answer_option in sorted(zip(cos_sim, answer_options), reverse=True)]
        text_per_sample, label_per_sample = create_data_per_sample(num_choices, start_idx, wrong_text, wrong_label, answer_options_sorted)
        dataset = TASK_CONFIG[args.cur_task]['dataset'](tokenizer, text_per_sample, label_per_sample)
        data_loader = DataLoader(dataset, batch_size=1, shuffle=False)
        torch.cuda.empty_cache()
        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                input_ids = batch[0]['input_ids'].to(args.device)
                attention_mask = batch[0]['attention_mask'].to(args.device)
                label = batch[1].to(args.device)
                output = model(input_ids=input_ids, attention_mask=attention_mask, labels=label)
                _, logit = output[0], output[1]
                pred = torch.argmax(nn.Softmax(dim=1)(logit), dim=1)
                if pred == label:
                    print(f'Select {batch_idx} for {done_cnt}-th sample ...')
                    right_texts.append(text_per_sample[batch_idx])
                    right_labels.append(label_per_sample[batch_idx])
                    right_logits.append(logit.to('cpu').tolist())
                    done_cnt += 1
                    torch.cuda.empty_cache()
                    break
                torch.cuda.empty_cache()
    torch.cuda.empty_cache()
    save_samples(args, right_texts, right_labels, right_logits)
    return right_texts, right_labels, right_logits


def modify_wrong_to_right_samples(args, model, tokenizer, wrong_texts, wrong_labels, answer_options_pool):
    print('Number of incorrectly predicted samples:', len(wrong_labels))

    num_choices = TASK_CONFIG[args.cur_task]['num_choices']
    start_idx = TASK_CONFIG[args.cur_task]['context'] + TASK_CONFIG[args.cur_task]['question']

    model.to(args.device)
    model.eval()

    # find and save correctly predicted samples
    right_texts = []; right_labels = []; right_logits = []
    done_cnt = 0
    for wrong_text, wrong_label in zip(wrong_texts, wrong_labels):
        # targeting one incorrect sample
        print(f'Trying {done_cnt}-th sample of {len(wrong_labels)} samples ...')
        text_per_sample, label_per_sample = create_data_per_sample(num_choices, start_idx, wrong_text, wrong_label, answer_options_pool)
        dataset = TASK_CONFIG[args.cur_task]['dataset'](tokenizer, text_per_sample, label_per_sample)
        data_loader = DataLoader(dataset, batch_size=1, shuffle=True)
        torch.cuda.empty_cache()
        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                input_ids = batch[0]['input_ids'].to(args.device)
                attention_mask = batch[0]['attention_mask'].to(args.device)
                label = batch[1].to(args.device)
                output = model(input_ids=input_ids, attention_mask=attention_mask, labels=label)
                _, logit = output[0], output[1]
                pred = torch.argmax(nn.Softmax(dim=1)(logit), dim=1)
                if pred == label:
                    print(f'Select {batch_idx} for {done_cnt}-th sample ...')
                    right_texts.append(text_per_sample[batch_idx])
                    right_labels.append(label_per_sample[batch_idx])
                    right_logits.append(logit.to('cpu').tolist())
                    done_cnt += 1
                    torch.cuda.empty_cache()
                    break
                torch.cuda.empty_cache()
    torch.cuda.empty_cache()
    save_samples(args, right_texts, right_labels, right_logits)
    return right_texts, right_labels, right_logits


def test(model, loader, device):
    """
    Evaluate the model on a validation set.
    Only do batch size = 1.
    """
    epoch_true_labels = []
    epoch_preds = []
    epoch_loss = 0

    model = model.to(device)
    model.eval()
    with torch.no_grad(): 
        for batch in tqdm(loader):
            input_ids = batch[0]['input_ids'].to(device)
            attention_mask = batch[0]['attention_mask'].to(device)
            labels = batch[1].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss, logits = outputs[0], outputs[1]
            preds = torch.argmax(nn.Softmax(dim=1)(logits), dim=1)
                
            epoch_true_labels.extend(labels.tolist())
            epoch_preds.extend(preds.tolist())
            epoch_loss += loss.item()

            torch.cuda.empty_cache()
    return epoch_loss/len(loader), epoch_true_labels, epoch_preds


class EarlyStopping(object):
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=10, verbose=True, delta=0, trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved. Default: 10
            verbose (bool): If True, prints a message for each validation loss improvement. Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement. Default: 0
            trace_func (function): trace print function. Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.trace_func = trace_func
    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0
    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path)
        self.val_loss_min = val_loss


class CriterionForKD(object):
    def __init__(self, alpha=0.5, T=1):
        self.criterion = self.knowledge_distillation_loss
        self.alpha = alpha
        self.T = T
    def __call__(self, student_logits, teacher_logits, teacher_labels, labels, loss):
        return self.criterion(student_logits, teacher_logits, teacher_labels, labels, loss)
    
    def knowledge_distillation_loss(self, student_logits, teacher_logits, teacher_labels, labels, loss):
        if teacher_logits == None:
            return loss
        else:
            kld_loss = nn.KLDivLoss(reduction='batchmean')(F.softmax(student_logits/self.T, dim=1), F.softmax((teacher_logits)/self.T, dim=1)) * (self.T * self.T)
            total_loss =  (1.-self.alpha)*loss + self.alpha*kld_loss
            return total_loss


class CriterionForSKD(object):
    def __init__(self, alpha=0.5, T=1):
        self.criterion = self.knowledge_distillation_loss
        self.alpha = alpha
        self.T = T
    def __call__(self, student_logits, teacher_logits, teacher_labels, labels, loss):
        return self.criterion(student_logits, teacher_logits, teacher_labels, labels, loss) 

    def knowledge_distillation_loss(self, student_logits, teacher_logits, teacher_labels, labels, loss):
        total_loss = 0
        kld_loss = nn.KLDivLoss(reduction='none')(F.softmax(student_logits/self.T, dim=1), F.softmax((teacher_logits)/self.T, dim=1)) * (self.T * self.T)
        kld_loss = kld_loss.sum(axis=1)/len(student_logits)
        for i in range(len(student_logits)):
            if teacher_labels[i] == labels[i]:
                total_loss += (1.-self.alpha)*loss + self.alpha*kld_loss[i]
            else:
                total_loss += (1.-self.alpha)*loss + self.alpha*loss
        return total_loss


class Trainer(object):
    """
    Trainer for training a multiple choice classification model.
    For FT, ST.
    """
    def __init__(self, model, optimizer, device='cpu'):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device

    def _print_summary(self):
        print(self.model)
        print(self.optimizer)

    def train(self, loader):
        """
        Run a single epoch of training
        """
        self.model.train() # Run model in training mode

        epoch_true_labels = []
        epoch_preds = []
        epoch_loss = 0
        for batch in tqdm(loader):
            # clear gradient
            self.optimizer.zero_grad()
            # input_ids shape: (batch_size, num_choices, sequence_length)
            input_ids = batch[0]['input_ids'].to(self.device)
            # attention_mask shape: (batch_size, num_choices, sequence_length)
            attention_mask = batch[0]['attention_mask'].to(self.device)
            # labels shape: (batch_size, )
            labels = batch[1].to(self.device)
            
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss, logits = outputs[0], outputs[1]
            preds = torch.argmax(nn.Softmax(dim=1)(logits), dim=1)
            
            epoch_true_labels.extend(labels.tolist())
            epoch_preds.extend(preds.tolist())
            epoch_loss += loss.item()
            
            # back propagation
            loss.backward()
            # do gradient descent
            self.optimizer.step()

            torch.cuda.empty_cache()
        return epoch_loss / len(loader), epoch_true_labels, epoch_preds

    def evaluate(self, loader):
        """
        Evaluate the model on a validation set.
        Only do batch size = 1.
        """
        self.model.eval() # Run model in eval mode (disables dropout layer)

        epoch_true_labels = []
        epoch_preds = []
        epoch_loss = 0
        with torch.no_grad(): # Disable gradient computation - required only during training
            for batch in tqdm(loader):
                # input_ids shape: (batch_size, num_choices, sequence_length)
                input_ids = batch[0]['input_ids'].to(self.device)
                # attention_mask shape: (batch_size, num_choices, sequence_length)
                attention_mask = batch[0]['attention_mask'].to(self.device)
                # labels shape: (batch_size, )
                labels = batch[1].to(self.device)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss, logits = outputs[0], outputs[1]
                preds = torch.argmax(nn.Softmax(dim=1)(logits), dim=1)
                
                epoch_true_labels.extend(labels.tolist())
                epoch_preds.extend(preds.tolist())
                epoch_loss += loss.item()

                torch.cuda.empty_cache()
        return epoch_loss/len(loader), epoch_true_labels, epoch_preds

    def get_model_dict(self):
        return self.model.state_dict()

    def run_training(self, args, train_loader, valid_loader, target_names):
        early_stopping = EarlyStopping(patience=5, verbose=True)

        for i in range(args.n_epoch):
            train_epoch_loss, train_labels, train_preds = self.train(train_loader)
            valid_epoch_loss, valid_labels, valid_preds = self.evaluate(valid_loader)
            
            print(f"Epoch {i}")
            print(f"Train loss: {train_epoch_loss}")
            print(f"Valid loss: {valid_epoch_loss}")
            print("Train eval")
            print(classification_report(train_labels, train_preds, target_names=target_names))
            print("Valid eval")
            print(classification_report(valid_labels, valid_preds, target_names=target_names))
            
            valid_acc = accuracy_score(valid_labels, valid_preds)
            model_name = 'bs{}-lr{}-epoch{}-acc{:.04f}.pt'.format(args.batch_size, args.lr, i+1, valid_acc)
            model_path = os.path.join(args.base_dir, model_name)

            early_stopping(valid_epoch_loss, self.model, model_path)
            if early_stopping.early_stop:
                print("Early stopping")              
                break

            torch.cuda.empty_cache()


class TrainerForKD(object):
    """
    For FT with KD, ST with KD.
    """
    def __init__(self, model, optimizer, criterion, device='cpu'):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device

    def _print_summary(self):
        print(self.model)
        print(self.optimizer)

    def get_model_dict(self):
        return self.model.state_dict()

    def train(self, loader):
        epoch_true_labels = []
        epoch_preds = []
        epoch_loss = 0

        self.model.train()
        for batch in tqdm(loader):
            # clear gradient
            self.optimizer.zero_grad()
            # input_ids shape: (batch_size, num_choices, sequence_length)
            input_ids = batch[0]['input_ids'].to(self.device)
            # attention_mask shape: (batch_size, num_choices, sequence_length)
            attention_mask = batch[0]['attention_mask'].to(self.device)
            # labels shape: (batch_size, )
            labels = batch[1].to(self.device)
            # teacher_logits shape: (batch_size, num_choices)
            teacher_logits = batch[2].to(self.device)
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss, logits = outputs[0], outputs[1]
            preds = torch.argmax(nn.Softmax(dim=1)(logits), dim=1)
            teacher_preds = torch.argmax(nn.Softmax(dim=1)(teacher_logits), dim=1)
            
            total_loss = self.criterion(logits, teacher_logits, teacher_preds, labels, loss)

            epoch_true_labels.extend(labels.tolist())
            epoch_preds.extend(preds.tolist())
            epoch_loss += total_loss.item()
            
            total_loss.backward()
            self.optimizer.step()

            torch.cuda.empty_cache()
        return epoch_loss / len(loader), epoch_true_labels, epoch_preds

    def evaluate(self, loader):
        epoch_true_labels = []
        epoch_preds = []
        epoch_loss = 0

        self.model.eval()
        with torch.no_grad():
            for batch in tqdm(loader):
                # input_ids shape: (batch_size, num_choices, sequence_length)
                input_ids = batch[0]['input_ids'].to(self.device)
                # attention_mask shape: (batch_size, num_choices, sequence_length)
                attention_mask = batch[0]['attention_mask'].to(self.device)
                # labels shape: (batch_size, )
                labels = batch[1].to(self.device)
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss, logits = outputs[0], outputs[1]
                preds = torch.argmax(nn.Softmax(dim=1)(logits), dim=1)
                
                epoch_true_labels.extend(labels.tolist())
                epoch_preds.extend(preds.tolist())
                epoch_loss += loss.item()

                torch.cuda.empty_cache()
        return epoch_loss/len(loader), epoch_true_labels, epoch_preds

    def run_training(self, args, train_loader, valid_loader, target_names):
        early_stopping = EarlyStopping(patience=5, verbose=True)

        for i in range(args.n_epoch):
            train_epoch_loss, train_labels, train_preds = self.train(train_loader)
            valid_epoch_loss, valid_labels, valid_preds = self.evaluate(valid_loader)
            
            print(f"Epoch {i}")
            print(f"Train loss: {train_epoch_loss}")
            print(f"Valid loss: {valid_epoch_loss}")
            print("Train eval")
            print(classification_report(train_labels, train_preds, target_names=target_names))
            print("Valid eval")
            print(classification_report(valid_labels, valid_preds, target_names=target_names))

            valid_acc = accuracy_score(valid_labels, valid_preds)

            model_name = 'bs{}-lr{}-epoch{}-acc{:.04f}.pt'.format(args.batch_size, args.lr, i+1, valid_acc) 
            # model_name = 'bs{}-lr{}-alpha{}-epoch{}-acc{:.04f}.pt'.format(args.batch_size, args.lr, args.alpha, i+1, valid_acc)     
            model_path = os.path.join(args.base_dir, model_name)

            early_stopping(valid_epoch_loss, self.model, model_path)
            if early_stopping.early_stop:
                print("Early stopping")              
                break
            
            torch.cuda.empty_cache()

class TrainerForEncoder(object):
    """
    Trainer for training a multiple choice classification model.
    For FT, ST.
    """
    def __init__(self, model, optimizer, device='cpu'):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device

    def _print_summary(self):
        print(self.model)
        print(self.optimizer)

    def train(self, loader):
        """
        Run a single epoch of training
        """
        self.model.train() # Run model in training mode

        epoch_loss = 0
        for batch in tqdm(loader):
            # clear gradient
            self.optimizer.zero_grad()
            input_ids_context, input_ids_answer, attention_mask_context, attention_mask_answer =batch
            loss = self.model(context_input_ids=input_ids_context.to(self.device), context_input_masks = attention_mask_context.to(self.device),
                            responses_input_ids=input_ids_answer.to(self.device), responses_input_masks=attention_mask_answer.to(self.device))
            
            epoch_loss += loss.item()
            
            # back propagation
            loss.backward()
            # do gradient descent
            self.optimizer.step()

            torch.cuda.empty_cache()
        return epoch_loss / len(loader)

    def evaluate(self, loader):
        """
        Evaluate the model on a validation set.
        Only do batch size = 1.
        """
        self.model.eval() # Run model in eval mode (disables dropout layer)

        epoch_true_labels = []
        epoch_preds = []
        epoch_loss = 0
        with torch.no_grad(): # Disable gradient computation - required only during training
            for batch in tqdm(loader):
                # input_ids shape: (batch_size, num_choices, sequence_length)
                input_ids_context, input_ids_answer, attention_mask_context, attention_mask_answer =batch
                loss = self.model(context_input_ids=input_ids_context.to(self.device), context_input_masks = attention_mask_context.to(self.device),
                                responses_input_ids=input_ids_answer.to(self.device), responses_input_masks=attention_mask_answer.to(self.device))
                
                epoch_loss += loss.item()

                torch.cuda.empty_cache()
        return epoch_loss/len(loader)

    def get_model_dict(self):
        return self.model.state_dict()

    def run_training(self, args, train_loader, valid_loader, base_dir):
        early_stopping = EarlyStopping(patience=5, verbose=True)

        for i in range(args.n_epoch):
            train_epoch_loss= self.train(train_loader)
            valid_epoch_loss = self.evaluate(valid_loader)
            
            print(f"Epoch {i}")
            print(f"Train loss: {train_epoch_loss}")
            print(f"Valid loss: {valid_epoch_loss}")
            model_name = 'bs{}-epoch{}-loss{:.04f}.pt'.format(args.batch_size, i+1, valid_epoch_loss)
            model_path = os.path.join(base_dir, model_name)

            early_stopping(valid_epoch_loss, self.model, model_path)
            if early_stopping.early_stop:
                print("Early stopping")              
                break

            torch.cuda.empty_cache()
import os
import json
import pathlib
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from transformers import RobertaTokenizer, RobertaModel, AdamW
from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report, accuracy_score, f1_score

from dataset import load_siqa, load_csqa, load_cmqa, load_piqa, prepare_batch_KD, prepare_batch
from dataset import SocialiqaDataset, CommonsenseqaDataset, CosmosqaDataset, PhysicaliqaDataset
from dataset import SocialiqaDatasetForKD, CommonsenseqaDatasetForKD, CosmosqaDatasetForKD, PhysicaliqaDatasetForKD


TASK_CONFIG = {
    'siqa':{'load':load_siqa, 'dataset':SocialiqaDataset, 'dataset-kd':SocialiqaDatasetForKD, 'context':True, 'question':True, 'num_choices':3, 'target_names':['Answer A', 'Answer B', 'Answer C'] },
    'csqa':{'load':load_csqa, 'dataset':CommonsenseqaDataset, 'dataset-kd':CommonsenseqaDatasetForKD, 'context':False, 'question':True, 'num_choices':5, 'target_names':['Answer A', 'Answer B', 'Answer C', 'Answer D', 'Answer E']},
    'cmqa':{'load':load_cmqa, 'dataset':CosmosqaDataset, 'dataset-kd':CosmosqaDatasetForKD, 'context':True, 'question':True, 'num_choices':4, 'target_names':['Answer A', 'Answer B', 'Answer C', 'Answer D']},
    'piqa':{'load':load_piqa, 'dataset':PhysicaliqaDataset, 'dataset-kd':PhysicaliqaDatasetForKD, 'context':False, 'question':True, 'num_choices':2, 'target_names':['Answer A', 'Answer B']}
}


def make_dir(d):
    pathlib.Path(d).mkdir(parents=True, exist_ok=True)

def save_samples(args, texts, labels, logits):
    file = {'texts':texts, 'labels':labels, 'logits':logits}
    with open(os.path.join(args.base_dir, 'additional_samples.json'), 'w') as f:
        json.dump(file, f)


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


def get_predictions_from_teacher_model(args, model, tokenizer, texts, labels, device):
    dataset = TASK_CONFIG[args.cur_task]['dataset'](tokenizer, texts, labels)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)

    model = model.to(device)
    model.eval()

    epoch_logits = []
    with torch.no_grad():
        for batch in tqdm(data_loader):
            input_ids = batch[0]['input_ids'].to(device)
            attention_mask = batch[0]['attention_mask'].to(device)
            labels = batch[1].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss, logits = outputs[0], outputs[1]
            epoch_logits.append(logits.to('cpu').tolist())
            torch.cuda.empty_cache()
    return epoch_logits

def create_dataloader_with_zeroshot(args, model, tokenizer, texts, labels, device):
    dataset = TASK_CONFIG[args.cur_task]['dataset'](tokenizer, texts, labels)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    model.to(device)
    model.eval()
    wrong_idxs = []
    wrong_preds = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(data_loader)):
            input_ids = batch[0]['input_ids'].to(device)
            attention_mask = batch[0]['attention_mask'].to(device)
            label = batch[1].to(device)
            output = model(input_ids=input_ids, attention_mask=attention_mask, labels=label)
            loss, logit = output[0], output[1]
            pred = torch.argmax(nn.Softmax(dim=1)(logit), dim=1)
            if pred != label:
                wrong_idxs.append(batch_idx)
                wrong_preds.append(int(pred.to('cpu')))
                torch.cuda.empty_cache()
            torch.cuda.empty_cache()
    print(f'Number of incorrectly predicted samples: {len(wrong_idxs)}/{batch_idx}')
    text_per_sample=[]
    label_per_sample=[]
    for wrong_idx, wrong_pred in zip(wrong_idxs, wrong_preds):
        wrong_text = list(texts[wrong_idx])
        wrong_label = labels[wrong_idx]
        text_per_sample.append(tuple(wrong_text))
        label_per_sample.append(wrong_label)
    dataset = TASK_CONFIG[args.cur_task]['dataset'](tokenizer, text_per_sample, label_per_sample)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=True)
    torch.cuda.empty_cache()
    return data_loader, dataset, len(wrong_idxs)


def replace_wrong_to_right_samples(args, model, tokenizer, texts, labels, device):
    dataset = TASK_CONFIG[args.cur_task]['dataset'](tokenizer, texts, labels)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)

    model.to(device)
    model.eval()

    # gather incorrectly predicted samples
    wrong_idxs = []
    wrong_preds = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(data_loader)):
            input_ids = batch[0]['input_ids'].to(device)
            attention_mask = batch[0]['attention_mask'].to(device)
            label = batch[1].to(device)
            output = model(input_ids=input_ids, attention_mask=attention_mask, labels=label)
            loss, logit = output[0], output[1]
            pred = torch.argmax(nn.Softmax(dim=1)(logit), dim=1)
            if pred != label:
                wrong_idxs.append(batch_idx)
                wrong_preds.append(int(pred.to('cpu')))
                torch.cuda.empty_cache()
            torch.cuda.empty_cache()
    print('Number of incorrectly predicted samples:', len(wrong_idxs))

    # set the pool of answer options for replacement
    num_choices = TASK_CONFIG[args.cur_task]['num_choices']
    answer_options_pool = []
    for text in texts:
        for i in range(1, num_choices+1):
            answer_options_pool.append(text[-i])
    answer_options_pool = list(set(answer_options_pool))

    # find and save correctly predicted samples
    right_texts = []; right_labels = []; right_logits = []
    done_cnt = 0
    for wrong_idx, wrong_pred in zip(wrong_idxs, wrong_preds): # targeting one incorrect sample
        print(f'Trying {wrong_idx}-th sample ... {done_cnt}/{len(wrong_idxs)}')
        text_per_sample = []
        label_per_sample = []
        wrong_text = list(texts[wrong_idx])
        wrong_label = labels[wrong_idx]
        start_idx = TASK_CONFIG[args.cur_task]['context'] + TASK_CONFIG[args.cur_task]['question']
        for i in range(0, len(answer_options_pool)-num_choices, num_choices):
            for j, k in zip(range(num_choices), range(i, i+num_choices)):
                if j == wrong_label:
                    pass
                else:
                    #print(len(wrong_text))
                    #print(wrong_text)
                    #print(start_idx, j)
                    #print(k, len(answer_options_pool))
                    wrong_text[start_idx+j] = answer_options_pool[k]
            text_per_sample.append(tuple(wrong_text))
            label_per_sample.append(wrong_label)
        dataset = TASK_CONFIG[args.cur_task]['dataset'](tokenizer, text_per_sample, label_per_sample)
        data_loader = DataLoader(dataset, batch_size=1, shuffle=True)
        torch.cuda.empty_cache()
        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                input_ids = batch[0]['input_ids'].to(device)
                attention_mask = batch[0]['attention_mask'].to(device)
                label = batch[1].to(device)
                output = model(input_ids=input_ids, attention_mask=attention_mask, labels=label)
                loss, logit = output[0], output[1]
                pred = torch.argmax(nn.Softmax(dim=1)(logit), dim=1)
                if pred == label:
                    print(f'Select {batch_idx} for {wrong_idx}-th sample ...')
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
            model_path = os.path.join(args.base_dir, model_name)

            early_stopping(valid_epoch_loss, self.model, model_path)
            if early_stopping.early_stop:
                print("Early stopping")              
                break
            
            torch.cuda.empty_cache()


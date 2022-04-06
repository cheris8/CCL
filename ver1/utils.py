import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from transformers import RobertaTokenizer, RobertaModel, AdamW

from tqdm.auto import tqdm
from sklearn.metrics import classification_report, accuracy_score

from model import Multiple_Choice_Model


def get_predictions_from_teacher_model(root_dir, dataset):
    file_train = os.path.join(root_dir, 'data', dataset, "train-predictions.jsonl")
    json_train = pd.read_json(path_or_buf=file_train, lines=True)
    train_predictions = json_train['prediction'].tolist()
    return train_predictions


def generate_predictions(model, loader, device):
    model = model.to(device)
    model.eval()

    epoch_true_labels = []
    epoch_preds = []
    epoch_logits = []
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
            epoch_logits.append(logits.tolist())

            torch.cuda.empty_cache()
    return epoch_true_labels, epoch_preds, epoch_logits


def get_predictions_from_pre_model(args, dataset, device):
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)

    if args.lm == 'roberta-large':
        model_name = args.lm
    elif args.lm == 'roberta-cskg':
        model_name = os.path.join(args.root_dir, args.lm)
    roberta_model = RobertaModel.from_pretrained(model_name)
    model = Multiple_Choice_Model(roberta_model)

    n_pre_task = int(len(args.pre_task.split('-'))/2)
    if n_pre_task >= 2:
        path = os.path.join(args.root_dir, 'models', 'STKD')
        condition = f'{args.lm}-kd-{args.pre_task}-bs8'
    else:
        path = os.path.join(args.root_dir, 'models', 'FT')
        condition = f'{args.lm}-{args.pre_task}-bs8'
    best_name, _ = get_best_model(path, condition)
    best_path = os.path.join(path, best_name)
    restore_dict = torch.load(best_path, map_location=device)
    model.load_state_dict(restore_dict)

    labels, preds, logits = generate_predictions(model, data_loader, device)
    return logits


def get_best_model(path, condition):
    all_files = os.listdir(path)
    files = [file for file in all_files if (condition in file)]
    accuracies = [float(file[-9:-3]) for file in files]
    assert len(files) == len(accuracies)
    acc_to_file = {}
    for file, acc in zip(files, accuracies):
        acc_to_file[acc] = file
    best_acc = max(acc_to_file.keys())
    return acc_to_file.get(best_acc), best_acc


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
    def __call__(self, student_logits, teacher_logits, loss):
        return self.criterion(student_logits, teacher_logits, loss) 

    def knowledge_distillation_loss(self, student_logits, teacher_logits, loss):
        if teacher_logits == None:
            return loss
        else:
            kld_loss = nn.KLDivLoss(reduction='batchmean')(F.softmax(student_logits/self.T, dim=1), F.softmax((teacher_logits)/self.T, dim=1)) * (self.T * self.T)
            total_loss =  (1.-self.alpha)*loss + self.alpha*kld_loss
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

    def run_training(self, args, train_loader, valid_loader, target_names, save_dir):
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
            if args.pre_task is None:
                model_name = '{}-{}-ts{}-bs{}-lr{}-epoch{}-acc{:.04f}.pt'.format(args.lm, args.cur_task, args.training_size, args.batch_size, args.lr, i+1, valid_acc)
            else:
                model_name = '{}-{}-{}-ts{}-bs{}-lr{}-epoch{}-acc{:.04f}.pt'.format(args.lm, args.pre_task, args.cur_task, args.training_size, args.batch_size, args.lr, i+1, valid_acc)       
            model_path = os.path.join(save_dir, model_name)

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
            # pseudo_labels shape: (batch_size, num_choices)
            pseudo_labels = batch[2].to(self.device)
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss, logits = outputs[0], outputs[1]
            preds = torch.argmax(nn.Softmax(dim=1)(logits), dim=1)
            
            total_loss = self.criterion(logits, pseudo_labels, loss)

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

    def run_training(self, args, train_loader, valid_loader, target_names, save_dir):
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
            if args.pre_task is None: # FT with KD
                model_name = '{}-kd-{}-ts{}-bs{}-lr{}-epoch{}-acc{:.04f}.pt'.format(args.lm, args.cur_task, args.training_size, args.batch_size, i+1, valid_acc)
            else: # ST with KD
                model_name = '{}-kd-{}-{}-ts{}-bs{}-lr{}-epoch{}-acc{:.04f}.pt'.format(args.lm, args.pre_task, args.cur_task, args.training_size, args.batch_size, args.lr, i+1, valid_acc)     
            model_path = os.path.join(save_dir, model_name)

            early_stopping(valid_epoch_loss, self.model, model_path)
            if early_stopping.early_stop:
                print("Early stopping")              
                break
            
            torch.cuda.empty_cache()
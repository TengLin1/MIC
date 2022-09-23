from visualizer import get_local
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from model import TransformerEncoder
from Focal_loss import FocalLoss
import pandas as pd


class Trainer:
    def __init__(self, args, train_loader, test_loader, tokenizer, all_labels, weights):
        self.args = args
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.vocab_size = tokenizer.vocab_size
        self.pad_id = tokenizer.pad_token_id
        self.device = 'cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu'
        self.all_labels = all_labels
        n_classes = []
        for i in range(0, len(self.all_labels)):
            n_class = len(self.all_labels[i])
            n_classes.append(n_class)
        self.n_classes = n_classes
        print(self.n_classes)
        self.model = TransformerEncoder(vocab_size=self.vocab_size,
                                        seq_len=args.max_seq_len,
                                        d_model=args.hidden,
                                        n_layers=args.n_layers,
                                        n_heads=args.n_attn_heads,
                                        p_drop=args.dropout,
                                        d_ff=args.ffn_hidden,
                                        pad_id=self.pad_id,
                                        n_classes=self.n_classes
                                        )
        if torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)
        self.model.to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr)  # , weight_decay=0.01
        # self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.9, last_epoch=-1)

        # self.criterion = nn.ModuleList([nn.CrossEntropyLoss(weight=torch.tensor(class_weights).to(self.device))
        #                                 for class_weights in weights])# , reduction='none'

        self.criterion_2 = nn.CrossEntropyLoss()  # , reduction='none'
        self.criterion_mse = nn.MSELoss()
        # print(self.n_classes[0])
        self.criterion_f = nn.ModuleList([FocalLoss(n_classes) for n_classes in self.n_classes])

    def train(self, epoch):
        losses = 0
        accs0_all = []
        accs_all = []
        pred = []
        ground = []
        n_batches, n_samples = len(self.train_loader), len(self.train_loader.dataset)

        self.model.train()
        for i, batch in enumerate(self.train_loader):
            inputs, labels = map(lambda x: x.to(self.device), batch)
            get_local.clear()
            outputs, attention_weights = self.model(inputs)
            loss_all = 0
            accs0 = []
            accs = []
            for j in range(0, labels.shape[1]):
                loss1 = self.criterion_2(outputs[j], labels[:, j]).detach()
                loss = loss1 * self.criterion_f[j](outputs[j], labels[:, j])
                loss_all += loss
                # loss.backward(retain_graph = True)
                acc0 = (outputs[j].argmax(dim=-1) == labels[:, j]).sum()
                output_list = [float(self.all_labels[j][n]) for n in outputs[j].argmax(dim=-1)]
                output_list1 = [float(n) for n in outputs[j].argmax(dim=-1)]
                output = torch.tensor(output_list1, requires_grad=True).to(self.device)
                label_list = [float(self.all_labels[j][n]) for n in labels[:, j]]
                label_list1 = [float(n) for n in labels[:, j]]
                label_t = torch.tensor(label_list1, requires_grad=True).to(self.device)
                acc3 = (output >= label_t / 2).sum()
                acc4 = (output <= label_t * 2).sum()
                if j == 0:
                    for p in output.cpu().detach().numpy():
                        pred.append(p)
                    for g in label_t.cpu().detach().numpy():
                        ground.append(g)

                acc_t = acc3 + acc4 - len(labels[:, j])
                accs0.append(acc0.item())
                accs.append(acc_t.item())

            self.optimizer.zero_grad()
            loss_all.backward()
            self.optimizer.step()
            losses += loss_all.item()
            accs0_all.append(accs0)
            accs_all.append(accs)

        print('Train Epoch: {}\t>\tLoss: {:.4f} '.format(epoch, losses / n_batches))
        ave = 0
        for x in zip(*accs_all):
            # print("acc: {:.4f} ".format(sum(x) / n_samples))
            ave += sum(x) / n_samples
        print('ave = ', ave/len(self.n_classes))

    def validate(self, epoch):
        losses = 0
        accs0_all = []
        accs_all = []
        accs1_all = []
        n_batches, n_samples = len(self.test_loader), len(self.test_loader.dataset)
        pred = []
        ground = []
        self.model.eval()
        for i, batch in enumerate(self.test_loader):
            inputs, labels = map(lambda x: x.to(self.device), batch)
            get_local.clear()
            outputs, attention_weights = self.model(inputs)
            loss_all = 0
            accs0 = []
            accs = []
            accs1 = []
            for j in range(0, labels.shape[1]):
                acc0 = (outputs[j].argmax(dim=-1) == labels[:, j]).sum()
                output_list = [float(self.all_labels[j][n]) for n in outputs[j].argmax(dim=-1)]
                output_list1 = [float(n) for n in outputs[j].argmax(dim=-1)]
                output = torch.tensor(output_list1, requires_grad=True).to(self.device)
                label_list = [float(self.all_labels[j][n]) for n in labels[:, j]]
                label_list1 = [float(n) for n in labels[:, j]]
                label_t = torch.tensor(label_list1, requires_grad=True).to(self.device)

                acc3 = (output >= label_t / 2).sum()
                acc4 = (output <= label_t * 2).sum()
                if j == 0:
                    for p in output.cpu().detach().numpy():
                        pred.append(p)
                    for g in label_t.cpu().detach().numpy():
                        ground.append(g)

                acc = 0
                for k in range(len(labels[:, j])):
                    if labels[:, j][k] + 2 >= outputs[j].argmax(dim=-1)[k] >= labels[:, j][k] - 2:
                        acc += 1

                # print(acc1)
                # print(acc2)
                acc_t = acc3 + acc4 - len(labels[:, j])

                accs0.append(acc0.item())
                accs.append(acc_t.item())
                # accs1.append(acc.item())
                accs1.append(acc)
                # print(outputs[i])
                loss1 = self.criterion_2(outputs[j], labels[:, j]).detach()
                loss = loss1 * self.criterion_f[j](outputs[j], labels[:, j])
                # loss = self.criterion_f[j](outputs[j], labels[:, j]) * self.criterion_2(outputs[j], labels[:, j])
                loss_all += loss

            losses += loss_all.item()
            accs0_all.append(accs0)
            accs_all.append(accs)
            accs1_all.append(accs1)
        print('Valid Epoch: {}\t>\tLoss: {:.4f} '.format(epoch, losses / n_batches))

        # print(accs_all)
        ave_row = 0
        for x in zip(*accs0_all):
            # print("acc: {:.4f} ".format(sum(x)/n_samples))
            ave_row += sum(x) / n_samples
        print('ave = ', ave_row / len(self.n_classes))

        ave = 0
        for x in zip(*accs_all):
            # print("acc: {:.4f} ".format(sum(x)/n_samples))
            ave += sum(x)/n_samples
        print('ave = ', ave/len(self.n_classes))

        ave1 = 0
        for x in zip(*accs1_all):
            # print("acc: {:.4f} ".format(sum(x)/n_samples))
            ave1 += sum(x) / n_samples
        print('ave = ', ave1 / len(self.n_classes))

        lr = self.optimizer.state_dict()['param_groups'][0]['lr']
        return lr, losses / n_batches, [(sum(x) / n_samples) for x in zip(*accs0_all)], [(sum(x) / n_samples) for x in
                                                                                     zip(*accs1_all)], ave_row / len(
            self.n_classes), ave1 / len(self.n_classes)

    def save(self, epoch, model_prefix='model', root='.model'):
        path = Path(root) / (model_prefix + '.ep%d' % epoch)
        if not path.parent.exists():
            path.parent.mkdir()
        torch.save(self.model, path)
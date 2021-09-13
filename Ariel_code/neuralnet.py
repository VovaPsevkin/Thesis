import nni
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch import optim

import numpy as np
from sklearn.metrics import roc_auc_score

class Dataset:
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i], self.label[i]


class NetworkClassifier(nn.Module):
    def __init__(self, _rec_size, first, second):
        super(NetworkClassifier, self).__init__()

        self.size = _rec_size
        self.fc0 = nn.Linear(_rec_size, first)
        self.fc1 = nn.Linear(first, second)
        self.fc2 = nn.Linear(second, 1)

    def forward(self, x):
        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return torch.sigmoid(x)


def train(n_epochs, model, train_dataloader, test_dataloader, optimizer):
    model.train()
    for epoch in range(n_epochs):
        for batch_idx, (receptors_batch, labels) in enumerate(train_dataloader):
            optimizer.zero_grad()
            output = model(receptors_batch)
            vertoutput = torch.reshape(output, (-1,))
            loss = F.binary_cross_entropy(vertoutput.type(torch.float), labels.type(torch.float))  # cross entropy
            loss.backward()
            optimizer.step()
        eval(model, train_dataloader, "Train")
        eval(model, test_dataloader, "Val")


def neural_network(train_data, train_labels, val_data, val_labels, test_data, test_labels, params):
    rec_size = len(train_data[0])
    model = NetworkClassifier(rec_size, params["hidden_size_01"], params["hidden_size_02"])
    train_dataset = Dataset(train_data, train_labels)
    val_dataset = Dataset(val_data, val_labels)
    test_dataset = Dataset(test_data, test_labels)
    batch_size = params["batch_size"]
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    lr = params["lr"]
    optimizer = optim.Adam(model.parameters(), lr=lr)
    train(params["n_epoch"], model, train_dataloader, val_dataloader, optimizer)
    auc = eval(model, test_dataloader, "Test")
    print(auc)
    return auc



def eval(model, test_loader, name):
    model.eval()
    test_loss = 0
    correct = 0
    outputfull = None
    targetfull = None
    with torch.no_grad():
        for data, target in test_loader:
            output = torch.reshape(model(data), (-1,))
            if outputfull is not None:
                outputfull = torch.cat((outputfull, output), -1)
                targetfull = torch.cat((targetfull, target), -1)
            else:
                outputfull = output
                targetfull = target
            test_loss += F.binary_cross_entropy(output.type(torch.float), target.type(torch.float)).item()
            # pred = output.max(1, keepdim=True)  # get the index of the max log-probability
            # correct += pred.eq(target.view_as(pred)).cpu().sum()
    test_loss /= len(test_loader.dataset)
    print(name + 'set: Average loss: {:.4f}\n'.format(test_loss,
             correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))
    return roc_auc_score(targetfull.tolist(), outputfull.tolist())


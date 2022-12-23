# Copyright 2021 The ODE-LSTM Authors. All Rights Reserved.

import os
import torch
import argparse
from irregular_sampled_datasets import PersonData, ETSMnistData, XORData
import torch.utils.data as data
from torch_node_cell import ODELSTM, IrregularSequenceLearner
import pytorch_lightning as pl

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="person")
parser.add_argument("--solver", default="fixed_rk4")
parser.add_argument("--size", default=64, type=int)
parser.add_argument("--epochs", default=100, type=int)
parser.add_argument("--lr", default=0.01, type=float)
parser.add_argument("--batch_size", default=64, type=int)
parser.add_argument("--gpus", default=1, type=int)
args = parser.parse_args()


def load_dataset(args):
    return_sequences = False
    dataset = PersonData(return_sequences)
    train_x = torch.Tensor(dataset.train_x)
    train_y = torch.LongTensor(dataset.train_y)
    train_ts = torch.Tensor(dataset.train_t)
    test_x = torch.Tensor(dataset.test_x)
    test_y = torch.LongTensor(dataset.test_y)
    test_ts = torch.Tensor(dataset.test_t)
    train = data.TensorDataset(train_x, train_ts, train_y)
    test = data.TensorDataset(test_x, test_ts, test_y)


    trainloader = data.DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=2)
    testloader = data.DataLoader(test, batch_size=args.batch_size, shuffle=False, num_workers=2)
    in_features = train_x.size(-1)
    num_classes = int(torch.max(train_y).item() + 1)
    return trainloader, testloader, in_features, num_classes, return_sequences


def run():
    trainloader, testloader, in_features, num_classes, return_sequences = load_dataset(args)
    print(in_features, num_classes, return_sequences)
    ode_lstm = ODELSTM(
        in_features,
        args.size,
        num_classes,
        return_sequences=return_sequences,
        solver_type=args.solver,
    )
    learn = IrregularSequenceLearner(ode_lstm, lr=args.lr)
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        gradient_clip_val=1,
        gpus=args.gpus,
    )
    trainer.fit(learn, train_dataloaders=trainloader, val_dataloaders=testloader)

    results = trainer.test(learn, testloader)
    base_path = "results/{}".format(args.dataset)
    os.makedirs(base_path, exist_ok=True)
    with open("{}/pt_ode_lstm_{}.csv".format(base_path, args.size), "a") as f:
        f.write("{:06f}\n".format(results[0]["val_acc"]))

if __name__ =="__main__":
    run()

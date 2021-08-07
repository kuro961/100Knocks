from prog70 import my_argparse
from prog71 import NeuralNetwork
from prog73 import NewsDataset
import torch
import torch.nn as nn
import torch.utils.data as data
import numpy as np
import matplotlib.pyplot as plt

def calculate_loss_accuracy(model, criterion, loader):
    model.eval()
    total = 0
    correct = 0
    loss = 0.
    with torch.no_grad():
        for inputs, labels in loader:
            outputs = model(inputs)
            loss += criterion(outputs, labels)
            pred = torch.argmax(outputs, dim=-1)
            total += len(inputs)
            correct += len([True for p, l in zip(pred, labels) if p == l])

    return loss / len(loader), correct / total

def plot_result(loss_acc_train, loss_acc_valid, num_epochs, figure_file):
    x_data = [i + 1 for i in range(num_epochs)]
    loss_acc_train = np.array(loss_acc_train).T
    loss_acc_valid = np.array(loss_acc_valid).T

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.plot(x_data, loss_acc_train[0], label='train')
    plt.plot(x_data, loss_acc_valid[0], label='valid')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(x_data, loss_acc_train[1], label='train')
    plt.plot(x_data, loss_acc_valid[1], label='valid')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend()

    plt.savefig(figure_file)
    plt.show()

def train_model(model, dataloader_train, dataloader_valid, num_epochs, optimizer, criterion, figure_file):
    loss_acc_train = []
    loss_acc_valid = []
    for epoch in range(num_epochs):
        model.train()
        for i, (inputs, labels) in enumerate(dataloader_train):
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        loss_train, acc_train = calculate_loss_accuracy(model, criterion, dataloader_train)
        loss_valid, acc_valid = calculate_loss_accuracy(model, criterion, dataloader_valid)

        loss_acc_train.append([loss_train, acc_train])
        loss_acc_valid.append([loss_valid, acc_valid])

        print(f'epoch: {epoch + 1}, '
              f'loss_train: {loss_train:.4f}, accuracy_train: {acc_train:.4f}, '
              f'loss_valid: {loss_valid:.4f}, accuracy_valid: {acc_valid:.4f}')

    plot_result(loss_acc_train, loss_acc_valid, num_epochs, figure_file)

def main():
    args = my_argparse()

    x_train = torch.load(args.x_train)
    y_train = torch.load(args.y_train)
    x_valid = torch.load(args.x_valid)
    y_valid = torch.load(args.y_valid)

    dataset_train = NewsDataset(x_train, y_train)
    dataset_valid = NewsDataset(x_valid, y_valid)

    dataloader_train = data.DataLoader(dataset_train, batch_size=1, shuffle=True)
    dataloader_valid = data.DataLoader(dataset_valid, batch_size=len(dataset_valid), shuffle=False)

    model = NeuralNetwork(300, 4)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    num_epochs = 10

    train_model(model, dataloader_train, dataloader_valid, num_epochs, optimizer, criterion, args.figure_file)

if __name__ == '__main__':
    main()
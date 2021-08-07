from prog70 import my_argparse
from prog71 import NeuralNetwork
from prog73 import NewsDataset
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import time

def calculate_loss_accuracy(model, criterion, loader, device=None):
    model.eval()
    total = 0
    correct = 0
    loss = 0.
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss += criterion(outputs, labels)
            pred = torch.argmax(outputs, dim=-1)
            total += len(inputs)
            correct += len([True for p, l in zip(pred, labels) if p == l])

    return loss / len(loader), correct / total

def plot_result(loss_acc_train, loss_acc_valid, num_epochs, figure_file):
    x_data = [i + 1 for i in range(num_epochs)]
    loss_acc_train = np.array(torch.tensor(loss_acc_train).cpu()).T
    loss_acc_valid = np.array(torch.tensor(loss_acc_valid).cpu()).T

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

def train_model(model, dataloader_train, dataloader_valid, num_epochs, optimizer, criterion, figure_file, device=None):
    loss_acc_train = []
    loss_acc_valid = []
    for epoch in range(num_epochs):
        model.train()
        for _, (inputs, labels) in enumerate(dataloader_train):
            optimizer.zero_grad()

            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        loss_train, acc_train = calculate_loss_accuracy(model, criterion, dataloader_train, device)
        loss_valid, acc_valid = calculate_loss_accuracy(model, criterion, dataloader_valid, device)

        loss_acc_train.append([loss_train, acc_train])
        loss_acc_valid.append([loss_valid, acc_valid])

        torch.save({'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()},
                   f'checkpoint{epoch + 1}.pt')

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

    dataloader_valid = DataLoader(dataset_valid, batch_size=len(dataset_valid), shuffle=False)

    criterion = nn.CrossEntropyLoss()

    num_epochs = 5
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    for batch_size in [2**i for i in range(9)]:
        dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
        model = NeuralNetwork(300, 4).to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        start = time.time()
        print(f'バッチサイズ : {batch_size}')
        train_model(model, dataloader_train, dataloader_valid, num_epochs, optimizer, criterion, args.figure_file, device)
        elapsed_time = time.time() - start
        print(f'学習時間(1エポック) : {elapsed_time/num_epochs:.4f} [sec]')

if __name__ == '__main__':
    main()
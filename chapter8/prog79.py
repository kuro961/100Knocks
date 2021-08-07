from prog70 import my_argparse
from prog73 import NewsDataset
import torch
import torch.nn as nn
import torch.utils.data as data
import numpy as np
import matplotlib.pyplot as plt
import time

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, mid_size, output_size):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_size, mid_size),
            nn.Dropout(0.2),
            nn.Linear(mid_size, mid_size),
            nn.Dropout(0.2),
            nn.Linear(mid_size, output_size)
        )

    def forward(self, x):
        x = self.linear_relu_stack(x)
        return x

def calculate_loss_accuracy(model, criterion, loader, d_type):
    model.eval()
    total = 0
    correct = 0
    loss = 0.
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(d_type)
            labels = labels.to(d_type)
            outputs = model(inputs)
            loss += criterion(outputs, labels)
            pred = torch.argmax(outputs, dim=-1)
            total += len(inputs)
            correct += len([True for p, l in zip(pred, labels) if p == l])

    return loss / len(loader), correct / total

def train_model(dataset_train, dataset_valid, dataset_test, model, criterion, optimizer, num_epochs, batch_size):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.to(device)

    dataloader_train = data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    dataloader_valid = data.DataLoader(dataset_valid, batch_size=len(dataset_valid), shuffle=False)
    dataloader_test = data.DataLoader(dataset_test, batch_size=len(dataset_test), shuffle=False)

    loss_acc_train = []
    loss_acc_valid = []
    loss_acc_test = []
    for epoch in range(num_epochs):
        model.train()
        for i, (inputs, labels) in enumerate(dataloader_train):
            optimizer.zero_grad()

            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        loss_train, acc_train = calculate_loss_accuracy(model, criterion, dataloader_train, device)
        loss_valid, acc_valid = calculate_loss_accuracy(model, criterion, dataloader_valid, device)
        loss_test, acc_test = calculate_loss_accuracy(model, criterion, dataloader_test, device)

        loss_acc_train.append([loss_train, acc_train])
        loss_acc_valid.append([loss_valid, acc_valid])
        loss_acc_test.append([loss_test, acc_test])

        '''
        torch.save({'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()},
                   f'checkpoint{epoch + 1}.pt')
        '''

        print(f'epoch: {epoch + 1}, '
              f'loss_train: {loss_train:.4f}, accuracy_train: {acc_train:.4f}, '
              f'loss_valid: {loss_valid:.4f}, accuracy_valid: {acc_valid:.4f}')

    return loss_acc_train, loss_acc_valid, loss_acc_test

def main():
    args = my_argparse()

    x_train = torch.load(args.x_train)
    y_train = torch.load(args.y_train)
    x_valid = torch.load(args.x_valid)
    y_valid = torch.load(args.y_valid)
    x_test = torch.load(args.x_test)
    y_test = torch.load(args.y_test)

    dataset_train = NewsDataset(x_train, y_train)
    dataset_valid = NewsDataset(x_valid, y_valid)
    dataset_test = NewsDataset(x_test, y_test)

    model = NeuralNetwork(300, 200, 4)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
    num_epochs = 40
    batch_size = 32
    
    start = time.time()
    print(f'バッチサイズ : {batch_size}')
    loss_acc_train, loss_acc_valid, loss_acc_test = train_model(dataset_train, dataset_valid, dataset_test, model, criterion, optimizer, num_epochs, batch_size)
    elapsed_time = time.time() - start
    print(f'学習時間(1エポック) : {elapsed_time/num_epochs:.4f} [sec]')

    #正解率のプロット
    x_data = [i + 1 for i in range(num_epochs)]
    loss_acc_train = np.array(torch.tensor(loss_acc_train).to('cpu')).T
    loss_acc_valid = np.array(torch.tensor(loss_acc_valid).to('cpu')).T
    loss_acc_test = np.array(torch.tensor(loss_acc_test).to('cpu')).T

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

    plt.savefig(args.figure_file)
    plt.show()

    print(f'正解率（学習データ）：{loss_acc_train[1][-1]:.3f}')
    print(f'正解率（評価データ）：{loss_acc_test[1][-1]:.3f}')

if __name__ == '__main__':
    main()
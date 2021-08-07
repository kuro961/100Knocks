from prog70 import my_argparse
from prog71 import NeuralNetwork
from prog73 import NewsDataset
from prog75 import calculate_loss_accuracy, plot_result
import torch
import torch.nn as nn
import torch.utils.data as data

def train_model(model, dataloader_train, dataloader_valid, num_epochs, optimizer, criterion, figure_file):
    loss_acc_train = []
    loss_acc_valid = []
    for epoch in range(num_epochs):
        model.train()
        for _, (inputs, labels) in enumerate(dataloader_train):
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        loss_train, acc_train = calculate_loss_accuracy(model, criterion, dataloader_train)
        loss_valid, acc_valid = calculate_loss_accuracy(model, criterion, dataloader_valid)

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

    dataloader_train = data.DataLoader(dataset_train, batch_size=1, shuffle=True)
    dataloader_valid = data.DataLoader(dataset_valid, batch_size=len(dataset_valid), shuffle=False)

    model = NeuralNetwork(300, 4)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    num_epochs = 10

    train_model(model, dataloader_train, dataloader_valid, num_epochs, optimizer, criterion, args.figure_file)

if __name__ == '__main__':
    main()
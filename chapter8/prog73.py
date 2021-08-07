from prog70 import my_argparse
from prog71 import NeuralNetwork
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class NewsDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return [self.x[idx], self.y[idx]]

def train_model(model, dataloader_train, dataloader_valid, num_epochs, optimizer, criterion):
    for epoch in range(num_epochs):
        model.train()
        loss_train = 0.
        for (inputs, labels) in dataloader_train:
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            loss_train += loss.item()

        loss_train = loss_train / len(dataloader_train)

        #検証データの損失
        model.eval()
        with torch.no_grad():
            for (inputs, labels) in dataloader_valid:
                outputs = model(inputs)
                loss_valid = criterion(outputs, labels)

        print(f'epoch: {epoch + 1}, '
              f'loss_train: {loss_train:.4f}, '
              f'loss_valid: {loss_valid:.4f}')

def main():
    args = my_argparse()

    x_train = torch.load(args.x_train)
    y_train = torch.load(args.y_train)
    x_valid = torch.load(args.x_valid)
    y_valid = torch.load(args.y_valid)

    dataset_train = NewsDataset(x_train, y_train)
    dataset_valid = NewsDataset(x_valid, y_valid)

    dataloader_train = DataLoader(dataset_train, batch_size=1, shuffle=True)
    dataloader_valid = DataLoader(dataset_valid, batch_size=len(dataset_valid), shuffle=False)

    model = NeuralNetwork(300, 4)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    num_epochs = 10

    train_model(model, dataloader_train, dataloader_valid, num_epochs, optimizer, criterion)

if __name__ == '__main__':
    main()
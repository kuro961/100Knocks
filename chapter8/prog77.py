from prog70 import my_argparse
from prog71 import NeuralNetwork
from prog73 import NewsDataset
from prog76 import train_model
import torch
import torch.nn as nn
import torch.utils.data as data
import time

def main():
    args = my_argparse()

    x_train = torch.load(args.x_train)
    y_train = torch.load(args.y_train)
    x_valid = torch.load(args.x_valid)
    y_valid = torch.load(args.y_valid)

    dataset_train = NewsDataset(x_train, y_train)
    dataset_valid = NewsDataset(x_valid, y_valid)

    dataloader_valid = data.DataLoader(dataset_valid, batch_size=len(dataset_valid), shuffle=False)

    model = NeuralNetwork(300, 4)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(model.parameters(), lr=1e-1)

    num_epochs = 5

    for batch_size in [2**i for i in range(9)]:
        dataloader_train = data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
        start = time.time()
        print(f'バッチサイズ : {batch_size}')
        train_model(model, dataloader_train, dataloader_valid, num_epochs, optimizer, criterion, args.figure_file)
        elapsed_time = time.time() - start
        print(f'学習時間(1エポック) : {elapsed_time/num_epochs:.4f} [sec]')

if __name__ == '__main__':
    main()
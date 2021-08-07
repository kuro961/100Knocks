from prog70 import my_argparse
from prog71 import NeuralNetwork
import torch
import torch.nn as nn

def main():
    args = my_argparse()

    model = NeuralNetwork(300, 4)
    x_train = torch.load(args.x_train)
    y_train = torch.load(args.y_train)

    criterion = nn.CrossEntropyLoss()

    loss = criterion(model(x_train[:1]), y_train[:1])
    model.zero_grad()
    loss.backward()
    print(f'損失: {loss:.4f}')
    print(f'勾配:\n{model.fc.weight.grad}')

    loss = criterion(model(x_train[:4]), y_train[:4])
    model.zero_grad()
    loss.backward()
    print(f'損失: {loss:.4f}')
    print(f'勾配:\n{model.fc.weight.grad}')

if __name__ == '__main__':
    main()
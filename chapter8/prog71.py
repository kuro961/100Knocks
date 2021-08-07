import torch
import torch.nn as nn

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.fc = nn.Linear(input_size, output_size, bias=False)
        #nn.init.xavier_normal_(self.fc.weight)

    def forward(self, x):
        x = self.fc(x)
        return x

def main():
    model = NeuralNetwork(300, 4)
    x_train = torch.load('x_train.pt')

    y_hat_1 = torch.softmax(model(x_train[0]), dim=-1)
    Y_hat = torch.softmax(model(x_train[:4]), dim=-1)
    print(y_hat_1)
    print(Y_hat)

if __name__ == '__main__':
    main()
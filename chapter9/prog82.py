from prog80 import my_argparse, get_word2id
from prog81 import NewsDataset, RNN
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import time

def calculate_loss_accuracy(model, dataset, criterion=None, device=None):
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    loss = 0.0
    total = 0
    correct = 0
    with torch.no_grad():
        for data in dataloader:
            inputs = data['inputs'].to(device)
            labels = data['labels'].to(device)
            outputs = model(inputs)
            if criterion is not None:
                loss += criterion(outputs, labels)
            pred = torch.argmax(outputs, dim=-1)
            total += len(inputs)
            correct += len([True for p, l in zip(pred, labels) if p == l])

    return loss / len(dataset), correct / total


def train_model(dataset_train, dataset_valid, batch_size, model, criterion, optimizer, num_epochs, collate_fn=None, device=None):
    model.to(device)

    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    for epoch in range(num_epochs):
        start = time.time()

        model.train()
        for data in dataloader_train:
            optimizer.zero_grad()
            inputs = data['inputs'].to(device)
            labels = data['labels'].to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # 損失と正解率の算出
        model.eval()
        loss_train, acc_train = calculate_loss_accuracy(model, dataset_train, criterion, device)
        loss_valid, acc_valid = calculate_loss_accuracy(model, dataset_valid, criterion, device)

        elapsed_time = time.time() - start

        print(f'epoch: {epoch + 1}, '
              f'loss_train: {loss_train:.4f}, accuracy_train: {acc_train:.4f}, '
              f'loss_valid: {loss_valid:.4f}, accuracy_valid: {acc_valid:.4f}, '
              f'{elapsed_time:.4f}sec')

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

    # 各種パラメータ
    word2id = get_word2id(args.src_word2id)
    VOCAB_SIZE = len(word2id) + 1
    EMB_SIZE = 300
    PADDING_IDX = len(word2id)
    OUTPUT_SIZE = 4
    HIDDEN_SIZE = 50
    BATCH_SIZE = 1
    NUM_EPOCHS = 10

    model = RNN(VOCAB_SIZE, EMB_SIZE, PADDING_IDX, OUTPUT_SIZE, HIDDEN_SIZE)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(model.parameters(), lr=0.02)

    train_model(dataset_train, dataset_valid, BATCH_SIZE, model, criterion, optimizer, NUM_EPOCHS)

    _, acc_train = calculate_loss_accuracy(model, dataset_train)
    _, acc_test = calculate_loss_accuracy(model, dataset_test)
    print(f'正解率（学習データ）：{acc_train:.3f}')
    print(f'正解率（評価データ）：{acc_test:.3f}')

if __name__ == '__main__':
    main()
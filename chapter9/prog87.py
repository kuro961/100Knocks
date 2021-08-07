from prog80 import my_argparse, get_word2id
from prog81 import NewsDataset
from prog82 import train_model, calculate_loss_accuracy
from prog83 import Padsequence
from prog84 import get_emb_weights
from prog86 import CNN
import torch
import torch.nn as nn

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
    VOCAB_SIZE = len(set(word2id.values())) + 1
    EMB_SIZE, emb_weights = get_emb_weights(word2id)
    PADDING_IDX = len(set(word2id.values()))
    OUTPUT_SIZE = 4
    OUT_CHANNELS = 100
    KERNEL_HEIGHTS = 3
    STRIDE = 1
    PADDING = 1
    LEARNING_RATE = 5e-2
    BATCH_SIZE = 64
    NUM_EPOCHS = 10
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = CNN(VOCAB_SIZE, EMB_SIZE, PADDING_IDX, OUTPUT_SIZE, OUT_CHANNELS, KERNEL_HEIGHTS, STRIDE, PADDING, emb_weights=emb_weights)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

    train_model(dataset_train, dataset_valid, BATCH_SIZE, model, criterion, optimizer, NUM_EPOCHS, Padsequence(PADDING_IDX), DEVICE)

    _, acc_train = calculate_loss_accuracy(model, dataset_train, device=DEVICE)
    _, acc_test = calculate_loss_accuracy(model, dataset_test, device=DEVICE)
    print(f'正解率（学習データ）：{acc_train:.3f}')
    print(f'正解率（評価データ）：{acc_test:.3f}')

if __name__ == '__main__':
    main()
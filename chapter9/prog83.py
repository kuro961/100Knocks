from prog80 import my_argparse, get_word2id
from prog81 import NewsDataset, RNN
from prog82 import train_model, calculate_loss_accuracy
import torch
import torch.nn as nn

class Padsequence:
    def __init__(self, padding_idx):
        self.padding_idx = padding_idx

    def __call__(self, batch):
        sequences = [x['inputs'] for x in batch]
        sequences_padded = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=self.padding_idx)
        labels = torch.tensor([x['labels'] for x in batch])

        return {'inputs': sequences_padded, 'labels': labels}

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

    word2id = get_word2id(args.src_word2id)
    VOCAB_SIZE = len(word2id) + 1
    EMB_SIZE = 300
    PADDING_IDX = len(word2id)
    OUTPUT_SIZE = 4
    HIDDEN_SIZE = 50

    BATCH_SIZE = 32
    NUM_EPOCHS = 10
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = RNN(VOCAB_SIZE, EMB_SIZE, PADDING_IDX, OUTPUT_SIZE, HIDDEN_SIZE)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(model.parameters(), lr=0.02)

    train_model(dataset_train, dataset_valid, BATCH_SIZE, model, criterion, optimizer, NUM_EPOCHS, Padsequence(PADDING_IDX), DEVICE)

    _, acc_train = calculate_loss_accuracy(model, dataset_train, device=DEVICE)
    _, acc_test = calculate_loss_accuracy(model, dataset_test, device=DEVICE)
    print(f'正解率（学習データ）：{acc_train:.3f}')
    print(f'正解率（評価データ）：{acc_test:.3f}')

if __name__ == '__main__':
    main()
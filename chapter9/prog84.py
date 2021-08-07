from prog80 import my_argparse, get_word2id
from prog81 import NewsDataset
from prog82 import train_model, calculate_loss_accuracy
from prog83 import Padsequence
import torch
import torch.nn as nn
from gensim.models import KeyedVectors

class RNN(nn.Module):
    def __init__(self, vocab_size, emb_size, padding_idx, output_size, hidden_size, emb_weights=None):
        super().__init__()
        self.hidden_size = hidden_size
        if emb_weights is None:
            self.emb = nn.Embedding(vocab_size, emb_size, padding_idx=padding_idx)
        else:
            self.emb = nn.Embedding.from_pretrained(emb_weights, padding_idx=padding_idx)
        self.rnn = nn.RNN(emb_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        device = x.device
        batch_size = x.size()[0]
        emb = self.emb(x)
        init_h = self.init_hidden(batch_size, device)
        out_rnn, hidden = self.rnn(emb, init_h)
        out = self.fc(out_rnn[:, -1, :])
        return out

    def init_hidden(self, batch_size, device):
        return torch.zeros(1, batch_size, self.hidden_size, device=device)

# 学習済み単語ベクトルの取得
def get_emb_weights(word2id):
    wv = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary=True)
    vocab_size = len(word2id) + 1
    emb_size = wv.vector_size
    emb_weights = torch.zeros(vocab_size, emb_size)
    for i, word in enumerate(word2id.keys()):
        try:
            emb_weights[i] = torch.tensor(wv[word])
        except KeyError:
            emb_weights[i] = torch.rand(emb_size)

    return emb_size, emb_weights

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
    EMB_SIZE, emb_weights = get_emb_weights(word2id)
    PADDING_IDX = len(word2id)
    OUTPUT_SIZE = 4
    HIDDEN_SIZE = 50
    BATCH_SIZE = 32
    NUM_EPOCHS = 10
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = RNN(VOCAB_SIZE, EMB_SIZE, PADDING_IDX, OUTPUT_SIZE, HIDDEN_SIZE, emb_weights)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(model.parameters(), lr=0.02)

    train_model(dataset_train, dataset_valid, BATCH_SIZE, model, criterion, optimizer, NUM_EPOCHS, Padsequence(PADDING_IDX), DEVICE)

    _, acc_train = calculate_loss_accuracy(model, dataset_train, device=DEVICE)
    _, acc_test = calculate_loss_accuracy(model, dataset_test, device=DEVICE)
    print(f'正解率（学習データ）：{acc_train:.3f}')
    print(f'正解率（評価データ）：{acc_test:.3f}')

if __name__ == '__main__':
    main()
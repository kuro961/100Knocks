from prog80 import my_argparse, get_word2id
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class RNN(nn.Module):
    def __init__(self, vocab_size, emb_size, padding_idx, output_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.emb = nn.Embedding(vocab_size, emb_size, padding_idx=padding_idx)
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

class NewsDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return {'inputs': self.x[idx],
                'labels': self.y[idx]}

def main():
    args = my_argparse()

    x_train = torch.load(args.x_train)
    y_train = torch.load(args.y_train)

    dataset_train = NewsDataset(x_train, y_train)

    dataloader_train = DataLoader(dataset_train, batch_size=1, shuffle=False)

    # 各種パラメータ
    word2id = get_word2id(args.src_word2id)
    VOCAB_SIZE = len(word2id) + 1
    EMB_SIZE = 300
    PADDING_IDX = len(word2id)
    OUTPUT_SIZE = 4
    HIDDEN_SIZE = 50

    model = RNN(VOCAB_SIZE, EMB_SIZE, PADDING_IDX, OUTPUT_SIZE, HIDDEN_SIZE)

    # 10件の予測値のみ表示
    for data, i in zip(dataloader_train, range(10)):
        print(torch.softmax(model(data['inputs']), dim=-1))
        if i > 10:
            break

if __name__ == '__main__':
    main()
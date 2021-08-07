from prog80 import my_argparse, get_word2id
from prog81 import NewsDataset
from prog84 import get_emb_weights
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn import functional as F

class CNN(nn.Module):
    def __init__(self, vocab_size, emb_size, padding_idx, output_size, out_channels, kernel_heights, stride, padding, emb_weights=None):
        super().__init__()
        if emb_weights is None:
            self.emb = nn.Embedding(vocab_size, emb_size, padding_idx=padding_idx)
        else:
            self.emb = nn.Embedding.from_pretrained(emb_weights, padding_idx=padding_idx)
        self.conv = nn.Conv2d(1, out_channels, (kernel_heights, emb_size), stride, (padding, 0))
        self.fc = nn.Linear(out_channels, output_size)

    def forward(self, x):
        emb = self.emb(x).unsqueeze(1)
        conv = self.conv(emb)
        act = F.relu(conv.squeeze(3))
        max_pool = F.max_pool1d(act, act.size()[2])
        out = self.fc(max_pool.squeeze(2))
        return out

def main():
    args = my_argparse()

    x_train = torch.load(args.x_train)
    y_train = torch.load(args.y_train)

    dataset_train = NewsDataset(x_train, y_train)

    dataloader_train = DataLoader(dataset_train, batch_size=1, shuffle=False)

    word2id = get_word2id(args.src_word2id)
    VOCAB_SIZE = len(word2id) + 1
    EMB_SIZE, emb_weights = get_emb_weights(word2id)
    PADDING_IDX = len(set(word2id.values()))
    OUTPUT_SIZE = 4
    OUT_CHANNELS = 100
    KERNEL_HEIGHTS = 3
    STRIDE = 1
    PADDING = 1

    model = CNN(VOCAB_SIZE, EMB_SIZE, PADDING_IDX, OUTPUT_SIZE, OUT_CHANNELS, KERNEL_HEIGHTS, STRIDE, PADDING, emb_weights=emb_weights)

    # 10件の予測値のみ表示
    for data, i in zip(dataloader_train, range(1)):
        print(torch.softmax(model(data['inputs']), dim=-1))

if __name__ == '__main__':
    main()
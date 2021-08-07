from prog80 import my_argparse, get_word2id
from prog81 import NewsDataset
from prog82 import train_model, calculate_loss_accuracy
from prog83 import Padsequence
from prog84 import get_emb_weights
from torch.nn import functional as F
import torch
import torch.nn as nn
import optuna

args = my_argparse()

x_train = torch.load('x_train.pt')
y_train = torch.load('y_train.pt')
x_valid = torch.load('x_valid.pt')
y_valid = torch.load('y_valid.pt')

dataset_train = NewsDataset(x_train, y_train)
dataset_valid = NewsDataset(x_valid, y_valid)

# 各種パラメータ(固定)
word2id = get_word2id(args.src_word2id)
VOCAB_SIZE = len(set(word2id.values())) + 1
EMB_SIZE, emb_weights = get_emb_weights(word2id)
PADDING_IDX = len(set(word2id.values()))
OUTPUT_SIZE = 4
KERNEL_HEIGHTS = 3
STRIDE = 1
PADDING = 1
LEARNING_RATE = 5e-2
NUM_EPOCHS = 10
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# prog86:CNNにdropを追加
class CNN(nn.Module):
    def __init__(self, vocab_size, emb_size, padding_idx, output_size, out_channels, kernel_heights, stride, padding, drop_rate, emb_weights=None):
        super().__init__()
        if emb_weights is None:
            self.emb = nn.Embedding(vocab_size, emb_size, padding_idx=padding_idx)
        else:
            self.emb = nn.Embedding.from_pretrained(emb_weights, padding_idx=padding_idx)
        self.conv = nn.Conv2d(1, out_channels, (kernel_heights, emb_size), stride, (padding, 0))
        self.drop = nn.Dropout(drop_rate)
        self.fc = nn.Linear(out_channels, output_size)

    def forward(self, x):
        emb = self.emb(x).unsqueeze(1)
        conv = self.conv(emb)
        act = F.relu(conv.squeeze(3))
        max_pool = F.max_pool1d(act, act.size()[2])
        out = self.fc(self.drop(max_pool.squeeze(2)))
        return out

def objective(trial):
    # 各種パラメータ(チューニング対象)
    out_channels = int(trial.suggest_discrete_uniform('out_channels', 50, 200, 50))
    drop_rate = trial.suggest_discrete_uniform('drop_rate', 0.0, 0.5, 0.1)
    learning_rate = trial.suggest_loguniform('learning_rate', 5e-4, 5e-2)
    momentum = trial.suggest_discrete_uniform('momentum', 0.5, 0.9, 0.1)
    batch_size = int(trial.suggest_discrete_uniform('batch_size', 16, 128, 16))

    model = CNN(VOCAB_SIZE, EMB_SIZE, PADDING_IDX, OUTPUT_SIZE, out_channels, KERNEL_HEIGHTS, STRIDE, PADDING, drop_rate, emb_weights=emb_weights)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

    train_model(dataset_train, dataset_valid, batch_size, model, criterion, optimizer, NUM_EPOCHS, collate_fn=Padsequence(PADDING_IDX), device=DEVICE)

    loss_valid, _ = calculate_loss_accuracy(model, dataset_valid, criterion, device=DEVICE)

    return loss_valid

def main():
    study = optuna.create_study()
    study.optimize(objective, n_trials=50)

    # 結果の表示
    best_trial = study.best_trial
    print('Best trial:')
    print('Value: {:.3f}'.format(best_trial.value))
    print('Params:')
    for key, value in best_trial.params.items():
        print('  {}: {}'.format(key, value))

if __name__ == '__main__':
    main()
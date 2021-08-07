import pandas as pd
import time
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel

class BERTClass(torch.nn.Module):
    def __init__(self, drop_rate, otuput_size):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.drop = torch.nn.Dropout(drop_rate)
        self.fc = torch.nn.Linear(768, otuput_size)  # BERTの出力に合わせて768次元を指定

    def forward(self, ids, mask):
        _, out = self.bert(ids, attention_mask=mask, return_dict=False)
        out = self.drop(out)
        out = self.fc(out)
        return out

class CreateDataset(Dataset):
    def __init__(self, x, y, tokenizer, max_len):
        self.x = x
        self.y = y
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        text = self.x[index]
        inputs = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']

        return {
            'ids': torch.LongTensor(ids),
            'mask': torch.LongTensor(mask),
            'labels': torch.Tensor(self.y[index])
        }

def calculate_loss_accuracy(model, dataloader, criterion=None, device=None):
    loss = 0.0
    total = 0
    correct = 0
    with torch.no_grad():
        for data in dataloader:
            ids = data['ids'].to(device)
            mask = data['mask'].to(device)
            labels = data['labels'].to(device)
            outputs = model(ids, mask)
            if criterion is not None:
                loss += criterion(outputs, labels).item()
            pred = torch.argmax(outputs, dim=-1).cpu().numpy()
            labels = torch.argmax(labels, dim=-1).cpu().numpy()
            total += len(labels)
            correct += len([True for p, l in zip(pred, labels) if p == l])

    return loss / len(dataloader), correct / total

def train_model(dataset_train, dataset_valid, batch_size, model, criterion, optimizer, num_epochs, device=None):
    model.to(device)

    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    dataloader_valid = DataLoader(dataset_valid, batch_size=len(dataset_valid), shuffle=False)

    for epoch in range(num_epochs):
        start = time.time()

        model.train()
        for data in dataloader_train:
            optimizer.zero_grad()
            ids = data['ids'].to(device)
            mask = data['mask'].to(device)
            labels = data['labels'].to(device)
            outputs = model(ids, mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        model.eval()
        loss_train, acc_train = calculate_loss_accuracy(model, dataloader_train, criterion, device)
        loss_valid, acc_valid = calculate_loss_accuracy(model, dataloader_valid, criterion, device)

        elapsed_time = time.time() - start

        print(f'epoch: {epoch + 1}, '
              f'loss_train: {loss_train:.4f}, accuracy_train: {acc_train:.4f}, '
              f'loss_valid: {loss_valid:.4f}, accuracy_valid: {acc_valid:.4f}, '
              f'{elapsed_time:.4f}sec')

def main():
    train = pd.read_table('train.txt')
    valid = pd.read_table('valid.txt')
    test = pd.read_table('test.txt')

    y_train = pd.get_dummies(train, columns=['CATEGORY'])[['CATEGORY_b', 'CATEGORY_e', 'CATEGORY_t', 'CATEGORY_m']].values
    y_valid = pd.get_dummies(valid, columns=['CATEGORY'])[['CATEGORY_b', 'CATEGORY_e', 'CATEGORY_t', 'CATEGORY_m']].values
    y_test = pd.get_dummies(test, columns=['CATEGORY'])[['CATEGORY_b', 'CATEGORY_e', 'CATEGORY_t', 'CATEGORY_m']].values

    max_len = 30
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    dataset_train = CreateDataset(train['TITLE'], y_train, tokenizer, max_len)
    dataset_valid = CreateDataset(valid['TITLE'], y_valid, tokenizer, max_len)
    dataset_test = CreateDataset(test['TITLE'], y_test, tokenizer, max_len)

    # パラメータの設定
    DROP_RATE = 0.4
    OUTPUT_SIZE = 4
    BATCH_SIZE = 16
    NUM_EPOCHS = 4
    LEARNING_RATE = 2e-5
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = BERTClass(DROP_RATE, OUTPUT_SIZE)

    criterion = torch.nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(params=model.parameters(), lr=LEARNING_RATE)

    train_model(dataset_train, dataset_valid, BATCH_SIZE, model, criterion, optimizer, NUM_EPOCHS, DEVICE)

    dataloader_train = DataLoader(dataset_train, batch_size=len(dataset_train), shuffle=False)
    dataloader_test = DataLoader(dataset_test, batch_size=len(dataset_test), shuffle=False)

    _, acc_train = calculate_loss_accuracy(model, dataloader_train, device=DEVICE)
    _, acc_test = calculate_loss_accuracy(model, dataloader_test, device=DEVICE)

    print(f'{acc_train}')
    print(f'{acc_test}')

if __name__ == '__main__':
    main()
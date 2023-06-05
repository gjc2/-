import datasets
import torch
import torch.nn as nn
from transformers import BertTokenizer
import torch.utils.data as data
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
# 1 data processor

class Dataset(data.Dataset):
    def __init__(self, split):
        self.dataset = datasets.load_dataset(path='CAIL2019-相似案例匹配', split=split)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        text = self.dataset[i]['A'] + "#" + self.dataset[i]['B'] + "#" + self.dataset[i]['C']
        label = [0, 1]
        if self.dataset[i]['label'] == 'B':
            label = [1, 0]
        else:
            label = [0, 1]
        return text, label


train_data = Dataset('train')
valid_data = Dataset('validation')
test_data = Dataset('test')

tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")


def collate_fn(data):
    texts = [i[0] for i in data]
    labels = [i[1] for i in data]
    texts = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=2500,
        return_tensors="pt"
    )
    input_ids = texts['input_ids']
    attention_mask = texts['attention_mask']
    token_type_ids = texts['token_type_ids']
    labels = torch.tensor(labels, dtype=torch.float)
    return input_ids.to(device), labels.to(device)


device = torch.device("cuda")


train_dataloader = data.DataLoader(train_data, 64, collate_fn=collate_fn, shuffle=True, drop_last=True)
valid_dataloader = data.DataLoader(valid_data, 64, collate_fn=collate_fn, shuffle=True, drop_last=True)
test_dataLoader = data.DataLoader(test_data, 64, collate_fn=collate_fn, shuffle=True, drop_last=True)

# 2 define network


class Linear(nn.Module):
    def __init__(self, vocab_size, embedding_dim, seq_length, output_dim, dropout):
        super(Linear, self).__init__()
        self.embedded = nn.Embedding(vocab_size, embedding_dim)
        self.pooling = nn.AvgPool2d(kernel_size=(1, embedding_dim), stride=(1, embedding_dim))
        self.linear = nn.Linear(seq_length, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.embedded(x)
        x = self.pooling(x).squeeze(2)
        x = self.dropout(self.linear(x))
        # x = nn.functional.softmax(x, dim=1)
        return x


# 3 define parameters

VOCAB_SIZE = tokenizer.vocab_size
EMBEDDING_DIM = 100
SEQ_LENGTH = 2500
OUTPUT_DIM = 2
DROPOUT = 0.3


# 4 create model


model = Linear(VOCAB_SIZE, EMBEDDING_DIM, SEQ_LENGTH, OUTPUT_DIM, DROPOUT).to(device)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(model.parameters())


# 5 train model

def accuracy(pred, label):
    _, pred = torch.max(pred, 1)
    _, label = torch.max(label, 1)
    correct = pred.detach().cpu().numpy() == label.detach().cpu().numpy()
    acc = np.sum(correct) / 64
    return acc


def train(model, loader, optimizer, criterion):
    model.train()
    epoch_loss = 0
    epoch_acc = 0
    for batch_data, batch_label in tqdm(loader):
        pred = model(batch_data).to(device)
        loss = criterion(pred, batch_label)
        acc = accuracy(pred, batch_label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_acc += acc.item()
    return epoch_loss / len(loader), epoch_acc / len(loader)


def eval(model, loader, criterion):
    model.eval()
    epoch_loss = 0
    epoch_acc = 0
    with torch.no_grad():
        for batch_data, batch_label in tqdm(loader):
            pred = model(batch_data).to(device)
            loss = criterion(pred, batch_label)
            acc = accuracy(pred, batch_label)
            epoch_loss += loss.item()
            epoch_acc += acc.item()
    return epoch_loss / len(loader), epoch_acc / len(loader)


N_EPOCHS = 10


def train_model():
    best_valid_loss = float('inf')
    train_l, train_a, test_l, test_a = [], [], [], []
    for epoch in range(N_EPOCHS):
        train_loss, train_acc = train(model, train_dataloader, optimizer, criterion)
        valid_loss, valid_acc = eval(model, valid_dataloader, criterion)
        train_l.append(train_loss)
        train_a.append(train_acc)
        test_l.append(valid_loss)
        test_a.append(valid_acc)
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            #torch.save(model.state_dict(), f'checkpoints/linear/linear_model_{epoch}.pt')
        print(
            f'Epoch : {epoch + 1}, train_loss :{train_loss} ,train_acc = {train_acc}, valid_loss:{valid_loss}, valid_acc={valid_acc}')
    plt.figure
    plt.title('linear network')
    plt.xlabel('epoch')
    plt.ylabel('loss/acc')
    plt.plot([x for x in range(len(train_l))], train_l, 'r-', label="train_loss")
    plt.plot([x for x in range(len(train_a))], train_a, 'b-.', label="train_acc")
    plt.plot([x for x in range(len(test_l))], test_l, 'g-',  label="valid_loss")
    plt.plot([x for x in range(len(test_a))], test_a, 'y-.', label="valid_acc")
    plt.legend()
    plt.savefig('linear.png', dpi=1000, bbox_inches='tight')
    plt.show()
    print("finish !")
    test_loss, test_acc = eval(model, test_dataLoader, criterion)
    print(
        f'test_loss:{valid_loss}, test_acc={valid_acc}')
# model.load_state_dict(torch.load('/'))


if __name__ == '__main__':
    train_model()


























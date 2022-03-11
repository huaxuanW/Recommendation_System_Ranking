from dataset import MyDataset
from torch.utils.data import DataLoader
from model import LSTM
from tqdm import tqdm
import torch
import torch.nn as nn 

train_dataset = MyDataset('lstm_dataset_train.csv')
train_dataloder = DataLoader(train_dataset,  batch_size=256, shuffle= True)

test_dataset = MyDataset('lstm_dataset_test.csv')
test_dataloder = DataLoader(test_dataset,  batch_size=128, shuffle= True)

mod = LSTM()

optimizer = torch.optim.Adam(mod.parameters(), lr = 1e-4)
criterion = nn.CrossEntropyLoss()



for epoch in range(5):
    train_bar = tqdm(train_dataloder)
    for x, y in train_bar:
        train_bar.set_description(f'Epoch {epoch} train')
        x = x.view(-1, 90, 1)
        optimizer.zero_grad()
        pred_y = mod(x)
        loss = criterion(pred_y, y)
        loss.backward()
        optimizer.step()
        pred_y = torch.argmax(pred_y, dim=1)
        acc = torch.sum(pred_y == y).item() / len(y)
        train_bar.set_postfix(loss=round(loss.item(),2), acc=acc)

    test_bar = tqdm(test_dataloder)
    for x, y in test_bar:
        train_bar.set_description(f'Epoch {epoch} test')
        with torch.no_grad():
            mod.eval()
            pred_y = mod(x)
            loss = criterion(pred_y, y)
            pred_y = torch.argmax(pred_y, dim=1)
            acc = torch.sum(pred_y == y).item() / len(y)
            test_bar.set_postfix(loss=round(loss.item(),2), acc=acc)
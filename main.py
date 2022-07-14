import torch
import numpy as np
from torch.optim import Adam
from torch.nn import CrossEntropyLoss

from data import get_list, preprocess, get_data_loader
from model import MultiModalModel

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
epochs = 10

train_data_list, test_data_list = get_list()
train_data_list, test_data_list, vocab = preprocess(train_data_list, test_data_list)
train_data_loader, valid_data_loader, test_data_loader = get_data_loader(train_data_list, test_data_list)
model = MultiModalModel(num_embeddings=len(vocab) + 1)
model.to(device)

optimizer = Adam(lr=3e-4, params=model.parameters())
criterion = CrossEntropyLoss()
best_rate = 0

def model_train():
    print("Train Start!")
    for epoch in range(epochs):
        correct = 0
        total = 0
        model.train()
        print('[EPOCH{:03d}]'.format(epoch + 1), end='')
        for guid, tag, jpg, txt in train_data_loader:
            tag = tag.to(device)
            jpg = jpg.to(device)
            txt = txt.to(device)
            out = model(jpg, txt)
            pred = torch.max(out, 1)[1]
            total +=  len(guid)
            correct += (pred == tag).sum()
        rate = correct / total * 100
        print('ACC:{:.2f}%'.format(rate), end='')
        print()

        total_loss = 0
        correct = 0
        total = 0
        model.eval()
        print()
        for guid, tag, jpg, txt in valid_data_loader:
            tag = tag.to(device)
            jpg = jpg.to(device)
            txt = txt.to(device)
            out = model(jpg, txt)
            pred = torch.max(out, 1)[1]
            total +=  len(guid)
            correct += (pred == tag).sum()
        rate = correct / total * 100
        print('ACC:{:.2f}%'.format(rate), end='')
        print("Train End!")

def model_test():
    total_loss = 0
    correct = 0
    total = 0
    guid_list = []
    pred_list = []
    model.eval()
    print('Test Start!')
    for guid, tag, jpg, txt in test_data_loader:
        jpg = jpg.to(device)
        txt = txt.to(device)
        out = model(jpg, txt)
        pred = torch.max(out, 1)[1]
        guid_list.extend(guid)
        pred_list.extend(pred.cpu().tolist())

    sentiments = {
        'negative': 0,
        'neutral': 1,
        'positive': 2,
    }
    with open('test_with_label.txt', 'w', encoding='utf-8') as f:
        f.write('guid,tag\n')
        for guid, pred in zip(guid_list, pred_list):
            f.write(f'{guid},{sentiments[pred]}\n')
        f.close()
    print('Test End!')

if __name__ == "__main__":
    model_train()
    model_test()
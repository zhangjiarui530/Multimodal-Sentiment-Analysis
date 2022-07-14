import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import numpy as np
import pandas as pd
import os
import cv2
from collections import defaultdict
from nltk import sent_tokenize, word_tokenize
from sklearn.metrics import precision_score, recall_score, f1_score

torch.manual_seed(555)
np.random.seed(555)

def get_jpg(jpg_path):
    jpg = Image.open(jpg_path)
    return jpg


def get_txt(txt_path):
    with open(txt_path, 'rb') as f:
        txt = f.readline().strip()
    return txt


def get_file(file_path, guid):
    jpg_path = os.path.join(file_path, f'{int(guid)}.jpg')
    txt_path = os.path.join(file_path, f'{int(guid)}.txt')
    return get_jpg(jpg_path), get_txt(txt_path)


def get_list(file_path = 'data', tag_path = 'label') -> (list, list):
    # 读取训练和测试数据，返回训练集和测试集
    train_list = []
    test_list = []
    train_tag_path = os.path.join(tag_path, 'train.txt')
    test_tag_path = os.path.join(tag_path, 'test_without_label.txt')

    train_tag = pd.read_csv(train_tag_path)
    test_tag = pd.read_csv(test_tag_path)

    sentiments = {
        'negative': 0,
        'neutral': 1,
        'positive' : 2,
    }

    for guid, tag in train_tag.values:
        dict = {}
        dict['guid'] = int(guid)
        dict['tag'] = sentiments[tag]
        dict['jpg'], dict['txt'] = get_file(file_path, guid)
        train_list.append(dict)

    for guid, tag in test_tag.values:
        dict = {}
        dict['guid'] = int(guid)
        dict['tag'] = None
        dict['jpg'], dict['txt'] = get_image_and_text(data_folder_path, guid)
        test_list.append(dict)

    return train_list, test_list


def decode(text: bytes):
    return str(text.decode(encoding='utf-8'))


def preprocess(train_list, test_list):
    # 数据预处理
    for train in train_list:
        train['txt'] = decode(train['txt'])

    for data in test_list:
        test['txt'] = test(data['txt'])

    return train_list, test_list


def collate_fn(data_list):
    guid = [data['guid'] for data in data_list]
    tag = [data['tag'] for data in data_list]
    # jpg = [data['jpg'].cpu().numpy() for data in data_list]
    # jpg = np.array(jpg)
    jpg = np.array([data['jpg'].cpu().numpy() for data in data_list])
    txt = [data['txt'] for data in data_list]

    return guid, torch.LongTensor(tag), torch.Tensor(jpg), torch.LongTensor(txt)

def get_data_loader(train_list, test_list) -> (DataLoader, DataLoader, DataLoader):
    # 生成数据负载器
    train_length = int(len(train_list) * 0.8)
    valid_length = len(train_list) - train_length
    train_loader, valid_loader = random_split(dataset=train_list, lengths = [train_length, valid_length])
    test_loader = test_list

    train_data_loader = DataLoader(
        dataset=train_loader,
        collate_fn=collate_fn,
        batch_size=16,
        shuffle=True,
        drop_last=False,
    )

    valid_data_loader = DataLoader(
        dataset=valid_loader,
        collate_fn=collate_fn,
        batch_size=16,
        shuffle=True,
        drop_last=False,
    )

    test_data_loader = DataLoader(
        dataset=test_loader,
        collate_fn=collate_fn,
        batch_size=16,
        shuffle=False,
        drop_last=False,
    )

    return train_data_loader, valid_data_loader, test_data_loader

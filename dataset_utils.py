import torch
from torch.utils.data import Dataset
import pandas as pd
from settings import max_position_embeddings,bert_path
from regex_patterns import *
from pytorch_pretrained_bert import BertTokenizer

tokenizer=BertTokenizer.from_pretrained(bert_path)

def get_articals(data_path):
    data=pd.read_pickle(data_path)
    return data

def sentence_segment(artical):
    # 分句
    return end_pattern.split(artical)

def artical_tokenize(artical):
    sentences = sentence_segment(artical)
    total_tokens = []
    for sentence in sentences:
        if not total_tokens:
            total_tokens.append("[CLS]")
        total_tokens.extend(tokenizer.tokenize(sentence))
        total_tokens.append("[SEP]")
    if len(total_tokens)>max_position_embeddings-1:
        total_tokens=total_tokens[:max_position_embeddings-1]
        total_tokens.append("[SEP]")
    ids = torch.tensor(tokenizer.convert_tokens_to_ids(total_tokens))
    return ids

class Artical_Dataset(Dataset):
    def __init__(self,data_path):
        self.data = get_articals(data_path)
        self.length = len(self.data)

    def __getitem__(self, idx):
        idx = idx % self.length
        data=self.data.iloc[idx]
        x = data['para'].values
        x = artical_tokenize(x)
        y = list(data['ppt_vec'].values)
        y = torch.tensor(y)
        return x,y

    def __len__(self):
        return self.length

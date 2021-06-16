# -*- coding: utf-8 -*-

from torch.utils.data import Dataset
from transformers import BertTokenizer
from transformers import BertTokenizer,BertModel
import json
import torch
from torch import nn as nn
import numpy as np

class Dataset(Dataset):
    def __init__(self, data_path, max_seq_len):
        self.path = data_path
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased') # Bert的分词器
        self.max_seq_len = max_seq_len
        self.c_input_ids = []
        self.c_attention_mask = []
        self.labels = []
        
        with open(self.path, encoding='utf8') as fp:
            data = json.load(fp)
            for item in data: 
                
                text = ""

                for sentence in item[0]:
                    text += sentence
                    text += ", "
                text = text.replace('"', "'")

                for d in item[1]:

                    newtext = "[CLS] " + text + " [SEP] "
                    if d['x'] == "Speaker 1":                       #原数据集中x为subject，y为object
                        newtext = text.replace("Speaker 1", "X")       #[S1]:X    [S2]:Y
                        newtext += " X"
                    elif d['x'] == "Speaker 2":
                        newtext = text.replace("Speaker 2", "Y")
                        newtext += " Y"
                    else:
                        newtext = newtext + " " + d['x']
                    newtext += " [SEP]"
                    if d['y'] == "Speaker 1":
                        text = text.replace("Speaker 1", "X")
                        newtext += " X"
                    elif d['y'] == "Speaker 2":
                        text = text.replace("Speaker 2", "Y")
                        newtext += " Y"
                    else:
                        newtext = newtext + " " + d['y']
                    newtext += " [SEP]"
                                  
                    text_dict = self.tokenizer.encode_plus(newtext, return_attention_mask=True)
                    l = len(text_dict['input_ids'])
                    if(l > self.max_seq_len):
                        self.c_input_ids.append(torch.tensor(text_dict['input_ids'])[-self.max_seq_len-1:-1])
                        self.c_attention_mask.append(torch.tensor(text_dict['attention_mask'])[-self.max_seq_len-1:-1])     

                    else:
                        pad = nn.ZeroPad2d(padding=(0, self.max_seq_len - l, 0, 0))
                        self.c_input_ids.append(pad(torch.tensor(text_dict['input_ids']).unsqueeze(0)).squeeze(0))
                        self.c_attention_mask.append(pad(torch.tensor(text_dict['attention_mask']).unsqueeze(0)).squeeze(0))
                    
                    label = np.zeros(37)
                    label = label.astype(np.float32)
                    if(type(d['rid']) == list):
                        for i in range(len(d['rid'])):
                            label[d['rid'][i] - 1] = 1
                    if(type(d['rid']) == int):
                        label[d['rid'] - 1] = 1
                    self.labels.append(torch.from_numpy(label))
                        
    def __getitem__(self, index):
        return {'input_ids':self.c_input_ids[index], 'attention_mask':self.c_attention_mask[index], 'labels':self.labels[index]}

    def __len__(self):
        return len(self.labels)
        

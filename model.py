# -*- coding: utf-8 -*-
"""
Created on 13 15:44:48 2021
"""

from transformers import BertTokenizer
from transformers import BertTokenizer,BertModel,BertConfig
import torch
from torch import nn as nn

class DREmodel(nn.Module):
    '''
    baseline model
    '''
    def __init__(self):
        super(DREmodel, self).__init__()
        model_config = BertConfig.from_pretrained('bert-base-uncased')
        self.bert = BertModel.from_pretrained('bert-base-uncased', config = model_config)
        for param in self.bert.parameters():
            param.requires_grad = True
        
        # layer
        self.cls = nn.Linear(768, 37)
        self.sigmoid = nn.Sigmoid()


    def forward(self, input_ids, mask):
        # Bert
        H = self.bert(input_ids, attention_mask = mask)['last_hidden_state']
        pooler_output = self.bert(input_ids, attention_mask = mask)['pooler_output']
        #layer
        output = self.sigmoid(self.cls(pooler_output))
        
        return output
    
class new_DREmodel(nn.Module):
    '''
    The model with asymmetrical classifiers
    '''
    def __init__(self):
        super(new_DREmodel, self).__init__()
        model_config = BertConfig.from_pretrained('bert-base-uncased')
        self.bert = BertModel.from_pretrained('bert-base-uncased', config = model_config)
        for param in self.bert.parameters():
            param.requires_grad = True
        # layer
        self.cls_1 = nn.Sequential(nn.Linear(768 * 3, 37), nn.Dropout(p=0.3), nn.Sigmoid())
        self.cls_2 = nn.Sequential(nn.Linear(768 * 3, 37), nn.Dropout(p=0.3), nn.Sigmoid())


    def forward(self, input_ids, mask):
        # Bert
        batch_size = input_ids.shape[0]
        indices = []
        for i in range(batch_size):
            indices.append(torch.nonzero(input_ids[i] == 102))
        
        H = self.bert(input_ids, attention_mask = mask)['last_hidden_state']
        pooler_output = self.bert(input_ids, attention_mask = mask)['pooler_output']
        
        for i in range(batch_size):
            if(i == 0):
                #max pool
                a_1 = H[i, indices[i][0]:indices[i][1], :].unsqueeze(0).max(dim=1)[0]
                a_2 = H[i, indices[i][1]:, :].unsqueeze(0).max(dim=1)[0]
                c = H[i, :indices[i][0],:].unsqueeze(0).max(dim=1)[0]
            else:
                #max pool
                a_1 = torch.cat((a_1, H[i, indices[i][0]:indices[i][1], :].unsqueeze(0).max(dim=1)[0]), dim=0)
                a_2 = torch.cat((a_2, H[i, indices[i][1]:, :].unsqueeze(0).max(dim=1)[0]), dim=0)
                c = torch.cat((c, H[i, :indices[i][0],:].unsqueeze(0).max(dim=1)[0]), dim=0)
        
        x_1 = torch.cat((a_1, c, a_2), dim=1)
        x_2 = torch.cat((a_2, c, a_1), dim=1)
        
        y_1 = self.cls_1(x_1)
        y_2 = self.cls_2(x_2)
        
        output = 0.5 * (y_1 + y_2)
        
        return output


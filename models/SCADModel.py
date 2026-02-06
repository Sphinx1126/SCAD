# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 19:25:26 2024

@author: 28257
"""
import torch
import torch.nn as nn
import pandas as pd
import random
from collections import defaultdict
from torch.nn import functional as nnf
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer,AutoModel
from models.AdaptedBERT import AdaptedBERT
from typing import List, Optional, Tuple, Union
from enum import Enum
from tqdm import tqdm
import os
import pickle
import sys
import argparse
import json
from typing import Tuple, Optional, Union
from os.path import join

class SCADModel(nn.Module):
    def __init__(self, bert, 
                 alltxts,
                 adversarial_temperature,
                 negative_sample_size,
                 bottleneck_size = 64, bert_max_len = 512):
        super(SCADModel, self).__init__()
        self.allusions = alltxts
        self.bottleneck_size = bottleneck_size
        self.hidden_dim = bert.config.hidden_size
        self.negative_sample_size = negative_sample_size
        self.bert_max_len = bert_max_len
        self.bert = AdaptedBERT(bert,self.bottleneck_size,self.hidden_dim)
        self.dense = nn.Linear(self.hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)
        self.T = adversarial_temperature
    
    def predict(self, input_ids,
                token_type_ids,
                attention_masks):
        emb = self.bert(input_ids=input_ids,
                      token_type_ids=token_type_ids,
                      attention_mask=attention_masks)
        logits = self.dense(emb)
        prob = self.sigmoid(logits)
        return prob,logits
    
    def forward(self, input_ids,
                token_type_ids,
                attention_masks):
        bs = input_ids.size(0)
        input_ids = input_ids.reshape(bs*(self.negative_sample_size+1),self.bert_max_len)
        token_type_ids = token_type_ids.reshape(bs*(self.negative_sample_size+1),self.bert_max_len)
        attention_masks = attention_masks.reshape(bs*(self.negative_sample_size+1),self.bert_max_len)
        
        prob,logits=self.predict(input_ids,
                                 token_type_ids,
                                 attention_masks)
        prob=prob.reshape(bs,self.negative_sample_size+1)
        logits=logits.reshape(bs,self.negative_sample_size+1)
        
        positive_prob=prob[:,0]
        negative_prob=prob[:,1:]
        positive_loss=-torch.mean(torch.log(positive_prob+1e-15))
        #negative_loss=-torch.sum(torch.log(1-negative_prob+1e-15))
        
        negative_weight=self.softmax(logits[:,1:]/self.T)
        negative_loss=-torch.sum(torch.log(1-negative_prob+1e-15)*negative_weight)
        
        loss=(positive_loss+negative_loss)/2
        return (prob,loss)

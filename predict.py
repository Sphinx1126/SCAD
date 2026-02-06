# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 02:07:24 2024

@author: 28257
"""

import pandas as pd
import numpy as np
import math
import torch
import random
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import os
import argparse
from PoemDataset import PoemDataset
from transformers import AutoTokenizer,AutoModel
from models.AdaptedBERT import AdaptedBERT
from models.SCADModel import SCADModel
import time
from loguru import logger
from torch.utils.tensorboard import SummaryWriter
from os.path import join
import transformers
import torch.nn.functional as F
import pickle

def set_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--cuda', action='store_true', help='use GPU', default=True)

    parser.add_argument('--output_path', default='output/')
    parser.add_argument('--data_dir', default='data/')
    parser.add_argument('--bert_path', default='bert/')
    
    parser.add_argument('--negative_sample_size', default=16, type=int)
    parser.add_argument('--bottleneck_size', default=64, type=int)
    parser.add_argument('--bert_max_len', default=256, type=int)
    parser.add_argument('--adversarial_temperature', default=10, type=int)
    
    parser.add_argument('--bs_eval', default=32)

    parser.add_argument('-randomSeed', default=0, type=int)

    args = parser.parse_args([])
    return args

if __name__ == '__main__':
    args = set_args()
    args.device = torch.device("cuda:0" if args.cuda and torch.cuda.is_available() else "cpu")
    
    random.seed(args.randomSeed)
    np.random.seed(args.randomSeed)
    torch.manual_seed(args.randomSeed)
    if args.cuda:
        torch.cuda.manual_seed(args.randomSeed)
    
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    
    cur_time = time.strftime("%Y%m%d%H%M%S", time.localtime())
    logger.add(join(args.output_path, 'predict.log'))
    logger.info(args)
    writer = SummaryWriter(args.output_path)
    
    with open(args.data_dir+'global_info.pkl', 'rb') as fp:
        alltxts,poem2all=pickle.load(fp)

    pretrained_bert=AutoModel.from_pretrained(args.bert_path)
    tokenizer = AutoTokenizer.from_pretrained(args.bert_path)
    
    model=SCADModel(pretrained_bert, alltxts,
                    args.adversarial_temperature,
                    args.negative_sample_size,
                    args.bottleneck_size, args.bert_max_len)

    model.load_state_dict(torch.load(args.output_path+'Model.pt'))
    model.negative_sample_size=0
    model = model.to(args.device)
    model.eval()
    
    df=pd.read_csv(args.data_dir+'test.csv')
    truths,preds,ranks=[],[],[]
    step=0
    for _,row in tqdm(df.iterrows()):
        poem=row['poem']
        allusion=row['text']
        candtxts=[allusion]+[alltxt for alltxt in alltxts if alltxt not in poem2all[poem]]
        truth_index=candtxts.index(allusion)
        probs=torch.tensor([]).to(args.device)
        for i in range(math.ceil(len(candtxts)/args.bs_eval)):
            begin=i*args.bs_eval
            end=min(len(candtxts),(i+1)*args.bs_eval)
            inputs=tokenizer(text = [poem]*(end-begin),
                             text_pair = candtxts[begin:end],
                             truncation = True,
                             padding = 'max_length',
                             max_length = args.bert_max_len,
                             return_tensors = "pt",
                             return_token_type_ids = True,
                             return_attention_mask = True,
                             return_special_tokens_mask = True,
                             return_length = True).to(args.device)
            input_ids=inputs.input_ids
            token_type_ids=inputs.token_type_ids
            attention_masks=inputs.attention_mask
            with torch.no_grad():
                outputs,_ = model.predict(input_ids=input_ids,
                                          token_type_ids=token_type_ids,
                                          attention_masks=attention_masks)
            probs=torch.concat([probs,outputs.squeeze(1)])
        probs=probs.tolist()
        pred=candtxts[probs.index(max(probs))]
        prob=probs[truth_index]
        probs.sort(reverse=True)
        rank=probs.index(prob)
        truths.append(allusion)
        preds.append(pred)
        ranks.append(rank)
        if step%100==0:
            mrr=sum([1/(r+1) for r in ranks])/len(ranks)
            hit1=sum([r<1 for r in ranks])/len(ranks)
            hit3=sum([r<3 for r in ranks])/len(ranks)
            hit10=sum([r<10 for r in ranks])/len(ranks)
            logger.info('MRR at step {} is {}, Hits@1 is {}, Hits@3 is {}, Hits@10 is {}'.format(step, mrr,hit1,hit3,hit10))
        step+=1
    mrr=sum([1/(r+1) for r in ranks])/len(ranks)
    hit1=sum([r<1 for r in ranks])/len(ranks)
    hit3=sum([r<3 for r in ranks])/len(ranks)
    hit10=sum([r<10 for r in ranks])/len(ranks)
    logger.info('Test MRR is {}, Hits@1 is {}, Hits@3 is {}, Hits@10 is {}'.format(mrr,hit1,hit3,hit10))
    
    result=pd.DataFrame()
    result['poem']=df['poem'].tolist()
    result['truth']=truths
    result['pred']=preds
    result['rank']=ranks
    result.to_csv('output/predict.csv')
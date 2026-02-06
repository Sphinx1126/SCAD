# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 14:15:48 2024

@author: 28257
"""

import pandas as pd
import numpy as np
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
    
    parser.add_argument('--bs_train', default=4)
    parser.add_argument('--epochs', default=15)
    parser.add_argument('--warmup_steps', default=10000)
    parser.add_argument('--eval_step', default=10000)
    parser.add_argument('--bs_eval', default=32)
    parser.add_argument('--save_step', default=100000)

    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('-randomSeed', default=0, type=int)

    args = parser.parse_args([])
    return args

def train(model, train_loader, dev_dataloader, optimizer, scheduler, args):
    model.train()
    logger.info("start training")
    device = args.device
    for epoch in range(args.epochs):
        logger.info('start {}-th epoch training'.format(epoch + 1))
        for batch_idx, data in enumerate(tqdm(train_loader)):
            step = epoch * len(train_loader) + batch_idx + 1
            
            input_ids,token_type_ids,attention_masks=data
            input_ids=input_ids.to(device)
            token_type_ids = token_type_ids.to(device)
            attention_masks=attention_masks.to(device)
            
            model.train()
            opts = model(input_ids=input_ids,token_type_ids=token_type_ids,attention_masks=attention_masks)
            loss=opts[1]

            if step % args.eval_step == 0:
                dev_acc,dev_loss,_,_ = evaluate(args, model, dev_dataloader)
                writer.add_scalar('loss', dev_loss, step)
                logger.info('accuracy at step {} is {}, loss is {}'.format(step, dev_acc, dev_loss.item()))
                model.train()

            if step % args.save_step == 0:
                logger.info('saving checkpoint at step {}'.format(step))
                save_path = join(args.output_path, 'checkpoint-{}.pt'.format(step))
                torch.save(model.state_dict(), save_path)
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
    logger.info('saving model')
    save_path = join(args.output_path, 'Model.pt')
    torch.save(model.state_dict(), save_path)
    
def evaluate(args, model, dataloader):
    model.eval()
    device = args.device
    logger.info("Running evaluation")
    eval_loss = 0.0
    probs=[]
    pred=[]
    with torch.no_grad():
        for data in tqdm(dataloader):
            input_ids,token_type_ids,attention_masks=data
            input_ids=input_ids.to(device)
            token_type_ids = token_type_ids.to(device)
            attention_masks=attention_masks.to(device)
            
            opts = model(input_ids=input_ids,token_type_ids=token_type_ids,attention_masks=attention_masks)
            loss=opts[1]
            prob=opts[0]
            probs+=prob[:,0].tolist()
            pred+=torch.argmax(prob,dim=1).tolist()
            eval_loss += loss
    acc=sum([p==0 for p in pred])/len(pred)
    return acc,eval_loss/len(dataloader),pred,probs

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
    logger.add(join(args.output_path, 'train.log'))
    logger.info(args)
    writer = SummaryWriter(args.output_path)
    
    with open(args.data_dir+'global_info.pkl', 'rb') as fp:
        alltxts,poem2all=pickle.load(fp)

        
    pretrained_bert=AutoModel.from_pretrained(args.bert_path)
    
    model=SCADModel(pretrained_bert, alltxts,
                    args.adversarial_temperature,
                    args.negative_sample_size,
                    args.bottleneck_size, args.bert_max_len)
    if args.cuda:
        model = model.to(args.device)
    
    train_dataset = PoemDataset(args.data_dir+'train.csv',
                                alltxts,poem2all,
                                args.negative_sample_size,
                                args.bert_path,args.bert_max_len)
    train_dataloader = DataLoader(train_dataset, batch_size=args.bs_train, shuffle=True)
    del train_dataset
    logger.info('Traning Set Loaded.')
    
    val_dataset = PoemDataset(args.data_dir+'valid.csv',
                              alltxts,poem2all,
                              args.negative_sample_size,
                              args.bert_path,args.bert_max_len)
    val_dataloader = DataLoader(val_dataset, batch_size=args.bs_eval, shuffle=False)
    del val_dataset
    logger.info('Validation Set Loaded.')
    
    t_total = len(train_dataloader) * args.epochs
    optimizer = transformers.AdamW(model.parameters(), lr=args.lr)
    scheduler = transformers.get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )
    train(model, train_dataloader, val_dataloader, optimizer, scheduler, args)
    del train_dataloader
    del val_dataloader
    
    test_dataset = PoemDataset(args.data_dir+'test.csv',
                               alltxts,poem2all,
                               args.negative_sample_size,
                               args.bert_path,args.bert_max_len)
    test_dataloader = DataLoader(test_dataset, batch_size=args.bs_eval, shuffle=False)
    del test_dataset
    logger.info('Test Set Loaded.')
    test_acc,test_loss,pred,probs=evaluate(args, model, test_dataloader)
    del test_dataloader
    logger.info('accuracy in Test Dataset is {}, loss is {}'.format(test_acc, test_loss.item()))
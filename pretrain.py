#encoding=utf-8

import os
import json
import argparse
import numpy as np


import torch
import torch.nn as nn

from typing import NamedTuple
from transformers import AdamW, get_linear_schedule_with_warmup

from dataset import MyCLVCGTokenizer
from models import Config as ModelConfig
from models import MyCLVCG_POINTER
from dataset import Dataset
from train import train, inference, print_results



parser = argparse.ArgumentParser(description='pretrain.py')
parser.add_argument('-input_path', type=str, default='/input/folder/path', help="input folder path")
parser.add_argument('-workspace_path', type=str, default='/output/and/config/folders/path', help="output and config folders path")
parser.add_argument('-cfg_path', type=str, default='/config/file/path', help="config file path")
parser.add_argument('-model_cfg_file', type=str, default='model.json', help="model config file")
parser.add_argument('-pretrain_cfg_file', type=str, default='pretrain.json', help="pretrain config file")
parser.add_argument('-img_file', type=str, default=os.path.join('LB','image','res18.pkl'), help="image file")
parser.add_argument('-corpus_file', type=str, default=os.path.join('LB','corpus','LB_train_context.json'), help="train corpus json file")
parser.add_argument('-eval_corpus_file', type=str, default=os.path.join('LB','corpus','LB_dev_context.json'), help="evaluation corpus json file")
parser.add_argument('-vocab_file', type=str, default='vocabulary.json', help="vocabulary json file")
parser.add_argument('-preprocess_dir', type=str, default=os.path.join('LB','preprocessed_data'), help="path of preprocessed files")
parser.add_argument('-model_file', type=str, default=None, help="Restoring model file for train")
parser.add_argument('-eval_model_file', type=str, default='best-model.pt', help="Restoring model file for eval")
parser.add_argument('-save_dir', type=str, default='ckpt', help="checkpoint folder")
parser.add_argument('-eval', default=False, action='store_true', help="evaluate mod")
parser.add_argument('-parallel', default=False, action='store_true', help="DataParallel")   
       


class PretrainConfig(NamedTuple):
    """ Hyperparameters for training """
    seed: int = 3431 # random seed
    batch_size: int = 4
    predict_batch_size: int = 4
    lr: int = 1e-4 # learning rate
    n_epochs: int = 12 # the number of epoch
    save_steps: int = 10000 # interval for saving model
    print_steps: int = 100 # interval for print time
    eval_steps: int = 10000# interval for evaluation
    mask_prob: int = 0.15
    next_sentence_prob : float = 0.5
    p_geom : float = 0.5
    adam_epsilon: float = 1e-6
    gradient_accumulation_steps: int = 10
    max_grad_norm : float = 1.0
    weight_decay : float = 1e-3

    @classmethod
    def load_from_json(cls, file): # load config from json file
        return cls(**json.load(open(file, "r")))


def pretrain():
    
    opt = parser.parse_args()

    print(torch.cuda.is_available())
    
    pretrain_cfg = PretrainConfig.load_from_json(os.path.join(opt.cfg_path, opt.pretrain_cfg_file))
    model_cfg = ModelConfig.load_from_json(os.path.join(opt.cfg_path, opt.model_cfg_file))

    img_file = os.path.join(opt.input_path, opt.img_file)
    corpus_file = os.path.join(opt.input_path, opt.corpus_file)
    eval_corpus_file = os.path.join(opt.input_path, opt.eval_corpus_file)
    vocab_file = os.path.join(opt.input_path, opt.vocab_file)
    preprocess_dir = os.path.join(opt.input_path, opt.preprocess_dir)
    if not os.path.exists(preprocess_dir):
        os.mkdir(preprocess_dir)
    save_dir = os.path.join(opt.workspace_path, opt.save_dir)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    if opt.model_file is not None:
        model_file = os.path.join(save_dir, opt.model_file)
    else:
        model_file = None

    
    tokenizer = MyCLVCGTokenizer(vocab_file)

    dev_data = Dataset(eval_corpus_file, img_file, preprocess_dir, model_cfg, pretrain_cfg, imgs=None, is_training=False, type ='pretrain')
    dev_data.load_dataset_LB(tokenizer, model_type = 'POINTER')
    dev_data.load_dataloader()

    if opt.eval is False:
        train_data = Dataset(corpus_file, img_file, preprocess_dir, model_cfg, pretrain_cfg, imgs=dev_data.imgs, is_training=True, type ='pretrain')
        train_data.load_dataset_LB(tokenizer, model_type = 'POINTER')
        train_data.load_dataloader()
    
    device = torch.device("cuda", 0)
    
    print("model..")
    model = MyCLVCG_POINTER(model_cfg, type="pretrain") 
    
    if opt.eval is False:
        #Train
        if model_file is not None:
            model.load_state_dict(torch.load(model_file))
        if opt.parallel:
            model = nn.DataParallel(model,device_ids=[0,1])

        print("Data num:",len(train_data))
        print("Total steps:",int(len(train_data)*pretrain_cfg.n_epochs/pretrain_cfg.batch_size))

        optimizer = AdamW(filter(lambda p: p.requires_grad,model.parameters()), lr=pretrain_cfg.lr, eps=pretrain_cfg.adam_epsilon, weight_decay = pretrain_cfg.weight_decay)
        scheduler =  get_linear_schedule_with_warmup(optimizer,
                                        num_warmup_steps= int(len(train_data)/pretrain_cfg.batch_size),
                                        num_training_steps= int(1.1*len(train_data)*pretrain_cfg.n_epochs/pretrain_cfg.batch_size))
        print("training...")
        train(pretrain_cfg, save_dir, model, train_data, dev_data, optimizer, scheduler, device, opt.parallel, model_type = 'POINTER', type = 'pretrain')
    
    else:
        #Evaluation
        checkpoint = os.path.join(save_dir, opt.eval_model_file)
        model.load_state_dict(torch.load(checkpoint))
        if opt.parallel:
            model = nn.DataParallel(model,device_ids=[0,1])
            
        if torch.cuda.is_available():
            model.to(device)
        model.eval()

        with(torch.no_grad()):
            total_loss, total_cls_loss, total_ns_acc, predictions, ns_predictions, input_ids, masked_lm_labels = inference(pretrain_cfg, model, dev_data, device, opt.parallel, type, model_type = 'POINTER')

        print_results(save_dir, 0, total_loss, total_cls_loss, total_ns_acc, predictions, ns_predictions, input_ids, masked_lm_labels)
        

if __name__ == '__main__':
    pretrain()
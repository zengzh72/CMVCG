#encoding=utf-8

import os
import json
import argparse
import numpy as np


import torch
from torch import nn

from typing import NamedTuple
from transformers import AdamW, get_linear_schedule_with_warmup

from dataset import MyCLVCGTokenizer
from models import Config as ModelConfig
from models import MyCLVCG_CMLM
from dataset import Dataset
from train import train, inference, print_results



parser = argparse.ArgumentParser(description='pretrain.py')
parser.add_argument('-input_path', type=str, default='/input/folder/path', help="input folder path")
parser.add_argument('-workspace_path', type=str, default='/output/and/config/folders/path', help="output and config folders path")
parser.add_argument('-cfg_path', type=str, default='/config/file/path', help="config file path")
parser.add_argument('-model_cfg_file', type=str, default='model.json', help="model config file")
parser.add_argument('-fine_tuning_cfg_file', type=str, default='fine_tuning.json', help="fine tuning config file")
parser.add_argument('-img_path', type=str, default=os.path.join('Bili','image'), help="image path")
parser.add_argument('-corpus_file', type=str, default=os.path.join('Bili','CMLM','corpus','Bili_train_context.json'), help="train corpus json file")
parser.add_argument('-eval_corpus_file', type=str, default=os.path.join('Bili','CMLM','corpus','Bili_dev_context.json'), help="evaluation corpus json file")
parser.add_argument('-vocab_file', type=str, default='vocabulary.json', help="vocabulary json file")
parser.add_argument('-pretrain_path', type=str, default='pretrain/model/path', help="pretrain model path")
parser.add_argument('-pretrain_file', type=str, default='best-model.pt', help="pretrain model file")
parser.add_argument('-preprocess_dir', type=str, default=os.path.join('Bili','CMLM','preprocessed_data'), help="path of preprocessed files")
parser.add_argument('-model_file', type=str, default=None, help="Restoring model file")
parser.add_argument('-eval_model_file', type=str, default='best-model.pt', help="Restoring model file for eval")
parser.add_argument('-save_dir', type=str, default='ckpt', help="checkpoint folder")
parser.add_argument('-eval', default=False, action='store_true', help="evaluate mod")
parser.add_argument('-parallel', default=False, action='store_true', help="DataParallel")  
parser.add_argument('-LB_dataset', default=False, action='store_true', help="use LB dataset")  
       


class PretrainConfig(NamedTuple):
    """ Hyperparameters for training """
    seed: int = 3431 # random seed
    batch_size: int = 4
    predict_batch_size: int = 4
    lr: int = 1e-4 # learning rate
    n_epochs: int = 10 # the number of epoch
    save_steps: int = 10000 # interval for saving model
    print_steps: int = 100 # interval for print time
    eval_steps: int = 10000# interval for evaluation
    adam_epsilon: float = 1e-6
    gradient_accumulation_steps: int = 20
    max_grad_norm : float = 1.0
    max_output_length: int = 20
    weight_decay : float = 1e-2
    non_mask_tokens : int = 3
    without_visual: bool = False
    without_video_time: bool = False
    without_context: bool = False
    without_color: bool = True

    @classmethod
    def load_from_json(cls, file): # load config from json file
        return cls(**json.load(open(file, "r")))


def fine_tuning():
    
    opt = parser.parse_args()
    
    
    fine_tuning_cfg = PretrainConfig.load_from_json(os.path.join(opt.cfg_path, opt.fine_tuning_cfg_file))
    model_cfg = ModelConfig.load_from_json(os.path.join(opt.cfg_path, opt.model_cfg_file))

    if not opt.LB_dataset:
        train_img_file = os.path.join(opt.input_path, opt.img_path,'Bili_train.pkl')
        dev_img_file = os.path.join(opt.input_path, opt.img_path,'Bili_dev.pkl')
    else:
        train_img_file = os.path.join(opt.input_path, opt.img_path,'res18.pkl')
        dev_img_file = os.path.join(opt.input_path, opt.img_path,'res18.pkl')
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

    pretrain_file = os.path.join(opt.pretrain_path, 'ckpt', opt.pretrain_file)
    
    tokenizer = MyCLVCGTokenizer(vocab_file)

    dev_data = Dataset(eval_corpus_file, dev_img_file, preprocess_dir, model_cfg, fine_tuning_cfg, imgs=None, is_training=False, type ='fine_tuning')
    dev_data.load_dataset_Bili_CMLM(tokenizer, 'CMLM')
    dev_data.load_dataloader()
    if opt.LB_dataset:
        dev_data.dataset.load_imgs=dev_data.dataset.load_imgs_LB

    if opt.eval is False:
        train_data = Dataset(corpus_file, train_img_file, preprocess_dir, model_cfg, fine_tuning_cfg, imgs=None, is_training=True, type ='fine_tuning')
        #train_data = Dataset(eval_corpus_file, img_file, preprocess_dir, model_cfg, pretrain_cfg, imgs=dev_data.imgs, is_training=False, type ='pretrain')
        train_data.load_dataset_Bili_CMLM(tokenizer, 'CMLM')
        train_data.load_dataloader()
        if opt.LB_dataset:
            train_data.dataset.load_imgs=train_data.dataset.load_imgs_LB
    
    device = torch.device("cuda", 0)
    
    print("model..")
    model = MyCLVCG_CMLM(model_cfg, type="fine_tuning") 
    
    model_dict =  model.state_dict()
    print("Loading pretrain file...")
    pretrained_dict = torch.load(pretrain_file)
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    
    if opt.eval is False:
        #Train
        if model_file is not None:
            print("Restoring model...")
            model.load_state_dict(torch.load(model_file))
        if opt.parallel:
            model = nn.DataParallel(model,device_ids=[0,1])
            
        print("Data num:",len(train_data))
        print("Total steps:",int(len(train_data)*fine_tuning_cfg.n_epochs/fine_tuning_cfg.batch_size))

        optimizer = AdamW(filter(lambda p: p.requires_grad,model.parameters()), lr=fine_tuning_cfg.lr, eps=fine_tuning_cfg.adam_epsilon, weight_decay = fine_tuning_cfg.weight_decay)
        scheduler =  get_linear_schedule_with_warmup(optimizer,
                                        num_warmup_steps= int(len(train_data)/fine_tuning_cfg.batch_size),
                                        num_training_steps= int(1.1*len(train_data)*fine_tuning_cfg.n_epochs/fine_tuning_cfg.batch_size))
        print("training...")
        train(fine_tuning_cfg, save_dir, model, train_data, dev_data, optimizer, scheduler, device, opt.parallel, model_cfg, model_type = 'CMLM', type = 'pretrain')
    
    else:
        #Evaluation
        checkpoint = os.path.join(save_dir, opt.eval_model_file)
        model.load_state_dict(torch.load(checkpoint))
        if opt.parallel:
            model = nn.DataParallel(model,device_ids=[0,1])
        if torch.cuda.is_available():
            model.to(torch.device("cuda"))
        model.eval()

        with(torch.no_grad()):
            total_loss, total_cls_loss, total_ns_acc, predictions, ns_predictions, pos_predictions, input_ids, masked_lm_labels = inference(fine_tuning_cfg, model, dev_data, device, opt.parallel, model_cfg, 'CMLM', type)
        print_results(save_dir, 0, total_loss, total_cls_loss, total_ns_acc, predictions, ns_predictions, pos_predictions, input_ids, masked_lm_labels)
                 

if __name__ == '__main__':
    fine_tuning()
#encoding=utf-8

import torch
import torch.nn as nn
import os
import json
import argparse
import logging
import numpy as np

from typing import NamedTuple

from transformers import AdamW, get_linear_schedule_with_warmup


from dataset import MyCLVCGTokenizer
from models import Config as ModelConfig
from models import MyCLVCG_CMLM
from dataset import Dataset
from train import test_generation

parser = argparse.ArgumentParser(description='generate.py')


parser.add_argument('-input_path', type=str, default='/input/folder/path', help="input folder path")
parser.add_argument('-workspace_path', type=str, default='/output/and/config/folders/path', help="output and config folders path")
parser.add_argument('-cfg_path', type=str, default='/config/file/path', help="config file path")
parser.add_argument('-model_cfg_file', type=str, default='model.json', help="model config file")
parser.add_argument('-generate_cfg_file', type=str, default='generate.json', help="generate config file")
parser.add_argument('-img_path', type=str, default=os.path.join('Bili','image'), help="image path")
parser.add_argument('-test_corpus_file', type=str, default=os.path.join('Bili','CMLM','corpus','Bili_test_context.json'), help="train corpus json file")
parser.add_argument('-vocab_file', type=str, default='vocabulary.json', help="vocabulary json file")
parser.add_argument('-fine_tuning_path', type=str, default='/pretrain/model/path', help="fine tuning model path")
parser.add_argument('-fine_tuning_file', type=str, default='best-model.pt', help="fine tuning model file")
parser.add_argument('-preprocess_dir', type=str, default=os.path.join('Bili','CMLM','preprocessed_data'), help="path of preprocessed files")
parser.add_argument('-save_dir', type=str, default='ckpt', help="checkpoint folder")
parser.add_argument('-generate_dir', type=str, default='generate', help="generate folder")
parser.add_argument('-LB_dataset', default=False, action='store_true', help="use LB dataset")  
       

class GenerateConfig(NamedTuple):
    """ Hyperparameters for training """
    seed: int = 3431 # random seed
    predict_batch_size: int = 1
    print_steps: int = 1
    num_beams : int = 5
    vocab_size : int = 40443
    candidate_num : int = 5
    repetition_penalty : int = 2
    do_sample : bool = False
    temperature : float = 1.0
    print_steps : int = 100
    noi_decay : float = 1.0
    max_pair_num : int = 15 
    max_turn : int = 5
    non_mask_tokens : int = 3
    pos_do_sample: bool = True
    token_do_sample: bool = False
    temperature : float = 1.0
    pos_top_k : int = 5
    token_top_k : int = 10
    without_visual: bool = False
    without_video_time: bool = False
    without_context: bool = False
    without_color: bool = True
    
    @classmethod
    def load_from_json(cls, file): # load config from json file
        return cls(**json.load(open(file, "r")))



def generate():
    
  
    opt = parser.parse_args()
    
    
    generate_cfg = GenerateConfig.load_from_json(os.path.join(opt.cfg_path, opt.generate_cfg_file))
    model_cfg = ModelConfig.load_from_json(os.path.join(opt.cfg_path, opt.model_cfg_file))
    if not opt.LB_dataset:
        test_img_file = os.path.join(opt.input_path, opt.img_path,'Bili_test.pkl')
    else:
        test_img_file = os.path.join(opt.input_path, opt.img_path,'res18.pkl')

    
    
    test_corpus_file = os.path.join(opt.input_path, opt.test_corpus_file)
    vocab_file = os.path.join(opt.input_path, opt.vocab_file)
    preprocess_dir = os.path.join(opt.input_path, opt.preprocess_dir)
    if not os.path.exists(preprocess_dir):
        os.mkdir(preprocess_dir)
    generate_dir = os.path.join(opt.workspace_path, opt.generate_dir)
    if not os.path.exists(generate_dir):
        os.mkdir(generate_dir)



    model_file = os.path.join(opt.fine_tuning_path, opt.save_dir, opt.fine_tuning_file)
    
    tokenizer = MyCLVCGTokenizer(vocab_file)

    test_data = Dataset(test_corpus_file, test_img_file, preprocess_dir, model_cfg, generate_cfg, imgs=None, is_training=False, type ='generate')
    test_data.load_dataset_Bili_CMLM(tokenizer, 'CMLM')
    test_data.load_dataloader()
    if opt.LB_dataset:
        test_data.dataset.load_imgs=test_data.dataset.load_imgs_LB    

    
    model = MyCLVCG_CMLM(model_cfg, type="generate") 
    model.load_state_dict(torch.load(model_file))

    if torch.cuda.is_available():
        model.to(torch.device("cuda"))
    model.eval()

    with(torch.no_grad()):
        generateds, ground_truths, contexts, keywords = test_generation(model, generate_cfg, model_cfg, test_data, model_type='CMLM')

    if opt.LB_dataset:
        res_f = open(os.path.join(generate_dir, 'generated_LB.txt'),"w", encoding='utf8')
    else:
        out_file_name = 'generated_Bili'
        if generate_cfg.without_visual:
            out_file_name += '_novisual'
        if generate_cfg.without_video_time:
            out_file_name += '_notime'
        if generate_cfg.without_context:
            out_file_name += '_nocontext'
        if generate_cfg.without_color:
            out_file_name += '_nocolor'
        res_f = open(os.path.join(generate_dir, out_file_name+'.txt'),"w", encoding='utf8')

    
    for gen,gt,ct,kw in zip(generateds, ground_truths, contexts, keywords):
        
        ct_decode = '\t'.join(ct)
        end = ct_decode.find("<PAD>")
        if end != -1:
            ct_decode = ct_decode[:end]
        res_f.write("%s\n\nground_truth:\n"%(ct_decode))
        
        g_decode = '\t'.join(gt)
        end = g_decode.find("<EOS>")
        if end != -1:
            g_decode = g_decode[:end+5]
        res_f.write("\t%s\n"%(g_decode))
        res_f.write("\ngiven keywords:\n")
        
        kw_decode = '\t'.join(kw)
        res_f.write("\t%s\n"%(kw_decode))
        res_f.write("\ngenerated:\n")
        
        s = '\t'.join(gen)
        end = s.find("<EOS>")
        if end != -1:
            s = s[:end+5]
        res_f.write("\t%s\n"%(s))
        res_f.write("\n=============================\n\n")
    
    


if __name__ == '__main__':
    generate()
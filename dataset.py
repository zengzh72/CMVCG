# encoding=utf-8
# modified from https://github.com/shmsw25/bart-closed-book-qa/blob/master/data.py

import os
import json
import re
import string
import numpy as np
import copy
import random
from scipy.stats import geom

import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader, RandomSampler, SequentialSampler

from transformers import PreTrainedTokenizer
from transformers.tokenization_utils import AddedToken
from scipy._lib._ccallback_c import idx


class MyCLVCGTokenizer():
    # merges and vocab same as Roberta
    def __init__(self,vocab_file, unk_token='<UNK>', pad_token='<PAD>', exp_min_id=None):
        vocab = self.load_vocab(vocab_file)
        self.vocab_to_id = vocab['word2id']
        self.id_to_vocab = vocab['id2word']
        self.vocab_size = len(self.vocab_to_id)
        self.unk_token = unk_token
        self.pad_token = pad_token
        self.exp_min_id = exp_min_id
        
    def convert_tokens_to_ids(self, tokens):
        ids = []
        for token in tokens:
            ids.append(self._convert_token_to_id(token))
        return ids

    def _convert_token_to_id(self, token):
        return self.vocab_to_id.get(token, self.vocab_to_id.get(self.unk_token))

    def convert_ids_to_tokens(self,token_ids):
        if token_ids is None:
            return []
        tokens = []
        for id in token_ids:
            tokens.append(self._convert_id_to_token(id))
        return tokens

    def _convert_id_to_token(self, id):
        if id == -1:
            return self.pad_token
        if self.exp_min_id is not None and id > self.exp_min_id:
            return ''
        t = self.id_to_vocab.get(str(id), self.unk_token)
        return t


    def load_vocab(self,vocab_file):
        vocab = json.load(open(vocab_file, 'r', encoding='utf8'))
        return vocab
    
    def batch_convert_tokens_to_ids(self,texts):
        ids = []
        for text in texts:
            ids.append(self.convert_tokens_to_ids(text))
        return ids

    def decode(self,token_ids):
        tokens = self.convert_ids_to_tokens(token_ids)
        return tokens

    def decode_batch(self, tokens):
        return [self.decode(_tokens) for _tokens in tokens]
    
    def get_IELM_exp_maps(self):
        exps = [t for i,t in enumerate(self.vocab_to_id.keys()) if '-' in t and i>=self.exp_min_id]
        self.exp_maps = {}

        for exp in exps:
            tokens = exp[1:-1].split('-')
            #print(self.vocab_to_id[exp],exp,tokens)
            self.exp_maps[self.vocab_to_id[exp]] = [self.vocab_to_id['['+t+']'] for t in tokens ]#if t !='HEAD']
        
        self.exp_maps[self.vocab_to_id['[HEAD]']] = [self.vocab_to_id['[HEAD]']]
        
        #print(self.exp_maps)

       
class Dataset(object):

    def __init__(self, corpus_file, img_file, preprocess_path, model_cfg, train_cfg, imgs=None, is_training=True, type = 'pretrain'):

        self.corpus_file = corpus_file


        data_type = "train" if is_training else "dev"
        
        if type == "generate" :
            data_type = "test"
        '''
        if type == "test":
            data_type = "test"
        '''
        print(data_type)
        self.preprocessed_file = os.path.join(preprocess_path,data_type)

        #print("corpus length:",len(self.corpus))
        
        print("Loading img...")
        if imgs is None:
            self.imgs = torch.load(open(img_file, 'rb'))
        else:
            self.imgs = imgs

        img_len = 0
        for e in self.imgs.keys():
            for v in self.imgs[e].keys():
                img_len += len(self.imgs[e][v])
        print("%s image_len:%d"%(data_type,img_len))
        
        self.model_cfg = model_cfg
        self.train_cfg = train_cfg
        self.is_training = is_training
        self.type = type



    def __len__(self):
        return len(self.dataset)

    def load_from_json(self,corpus_file):
        corpus = json.load(corpus_file)
        return corpus



    def load_dataset_LB(self,tokenizer, model_type='CLVCG'):

        self.tokenizer = tokenizer
        if os.path.exists(self.preprocessed_file):
            print("Loading pre-tokenized data from {}".format(self.preprocessed_file))
            with open(self.preprocessed_file, "r") as f:
                contexts, context_position_ids, context_segment_ids, comments, video_ids, episode_ids, video_times, colors = json.load(f)
        else:
            corpus = self.load_from_json(open(self.corpus_file, 'r', encoding='utf8'))
            print ("Start tokenizing...")
            video_ids = []
            episode_ids = []
            video_times = []
            contexts = []
            context_position_ids = []
            context_segment_ids = []
            comments = []
            colors = []
            for data in corpus:#[:500]:
                if len(video_ids) % 1000 == 0:
                    print(len(video_ids))
                video_ids.append(data['anime'])
                episode_ids.append(data['episode'])
                video_times.append(int(data['video_time']))
                
                context = ["<&&&>"] + data['context']
                context_id = tokenizer.convert_tokens_to_ids(context)
                position_ids, segment_ids = self.get_position_segment_ids(context_id)
                context_position_ids.append(position_ids)
                context_segment_ids.append(segment_ids)
                contexts.append(context_id)
                
                comment = ["<CLS>"] + ['<BOS>'] + data['comment'] + ['<EOS>']
                comments_id = tokenizer.convert_tokens_to_ids(comment)
                comments.append(comments_id)
                
                colors.append(data['color'])
            
            print(contexts[0])
            print(context_position_ids[0])
            print(context_segment_ids[0])
            print(comments[0])
                        
            #context_input = tokenizer.batch_encode_plus(contexts, pad_to_max_length=True, max_length=self.model_cfg.max_context_len, truncation=True, add_special_tokens=False)
            #comment_input = tokenizer.batch_encode_plus(comments, pad_to_max_length=True, max_length=self.model_cfg.max_comment_len, truncation=True, add_special_tokens=False)
            #input_ids, attention_mask = context_input["input_ids"], context_input["attention_mask"]
            #decoder_input_ids, decoder_attention_mask = comment_input["input_ids"], comment_input["attention_mask"]
            
            preprocessed_data = [contexts, context_position_ids, context_segment_ids, comments, video_ids, episode_ids, video_times, colors]
            with open(self.preprocessed_file, "w") as f:
                json.dump(preprocessed_data, f)

        #self.contexts = contexts
        #self.comments = comments


        if model_type == 'POINTER':
            self.dataset = MyLBDataset_POINTER(self.model_cfg, self.train_cfg, contexts, context_position_ids, context_segment_ids, comments, video_ids, episode_ids, video_times, colors,
                                       self.imgs, tokenizer.vocab_size, self.is_training, type=self.type)
        else:
            self.dataset = MyLBDataset(self.model_cfg, self.train_cfg, contexts, context_position_ids, context_segment_ids, comments, video_ids, episode_ids, video_times, colors,
                                       self.imgs, tokenizer.vocab_size, self.is_training, type=self.type)
        
        return self.dataset



    def load_dataset_Bili_POINTER(self, tokenizer, model_type='POINTER',noi_token='<NOI>'):
        self.tokenizer = tokenizer  

        if os.path.exists(self.preprocessed_file):
            print("Loading pre-tokenized data from {}".format(self.preprocessed_file))
            with open(self.preprocessed_file, "r") as f:
                contexts, context_position_ids, context_segment_ids, input_comments, output_comments, video_ids, episode_ids, video_times, colors = json.load(f)
        else:
            corpus = self.load_from_json(open(self.corpus_file, 'r', encoding='utf8'))
            print ("Start tokenizing...")
            video_ids = []
            episode_ids = []
            video_times = []
            contexts = []
            context_position_ids = []
            context_segment_ids = []
            input_comments = []
            output_comments = []
            colors = []
            for data in corpus:#[:500]:
                if len(video_ids) % 1000 == 0:
                    print(len(video_ids))

                
                context = ["<&&&>"] + data['context']
                context_id = tokenizer.convert_tokens_to_ids(context)
                position_ids, segment_ids = self.get_position_segment_ids(context_id)
               
                for sample in data['samples']:
                    input_comment_id = tokenizer.convert_tokens_to_ids(sample[0])
                    output_comment_id = tokenizer.convert_tokens_to_ids(sample[1])
                
                    video_ids.append(data['anime'])
                    episode_ids.append(data['episode'])
                    video_times.append(int(data['video_time']))
                    
                    context_position_ids.append(position_ids)
                    context_segment_ids.append(segment_ids)
                    contexts.append(context_id)
                
                    output_comments.append(output_comment_id)
                    input_comments.append(input_comment_id )
                
                    colors.append(data['color'])
            
            print(contexts[0])
            print(context_position_ids[0])
            print(context_segment_ids[0])
            print(output_comments[0])
            print(input_comments[0])
                        
            #context_input = tokenizer.batch_encode_plus(contexts, pad_to_max_length=True, max_length=self.model_cfg.max_context_len, truncation=True, add_special_tokens=False)
            #comment_input = tokenizer.batch_encode_plus(comments, pad_to_max_length=True, max_length=self.model_cfg.max_comment_len, truncation=True, add_special_tokens=False)
            #input_ids, attention_mask = context_input["input_ids"], context_input["attention_mask"]
            #decoder_input_ids, decoder_attention_mask = comment_input["input_ids"], comment_input["attention_mask"]
            
            preprocessed_data = [contexts, context_position_ids, context_segment_ids, input_comments, output_comments, video_ids, episode_ids, video_times, colors]
            with open(self.preprocessed_file, "w") as f:
                json.dump(preprocessed_data, f)

        self.dataset = MyBiliDataset_POINTER(self.model_cfg, self.train_cfg, contexts, context_position_ids, context_segment_ids, input_comments, output_comments, video_ids, episode_ids, video_times, colors,
                               self.imgs, tokenizer.vocab_size, self.is_training, type=self.type)

        
        return self.dataset



    def load_dataset_Bili_IELM(self, tokenizer, model_type='IELM'):
        self.tokenizer = tokenizer  

        if os.path.exists(self.preprocessed_file):
            print("Loading pre-tokenized data from {}".format(self.preprocessed_file))
            with open(self.preprocessed_file, "r") as f:
                contexts, context_position_ids, context_segment_ids, input_comments, output_tokens, output_exps, video_ids, episode_ids, video_times, colors = json.load(f)
        else:
            corpus = self.load_from_json(open(self.corpus_file, 'r', encoding='utf8'))
            print ("Start tokenizing...")
            video_ids = []
            episode_ids = []
            video_times = []
            contexts = []
            context_position_ids = []
            context_segment_ids = []
            input_comments = []
            output_tokens = []
            output_exps = []
            colors = []
            for data in corpus:#[:500]:
                if len(video_ids) % 1000 == 0:
                    print(len(video_ids))

                
                context = ["<&&&>"] + data['context']
                context_id = tokenizer.convert_tokens_to_ids(context)
                position_ids, segment_ids = self.get_position_segment_ids(context_id)
               
                for sample in data['samples']:
                    input_comment_id = tokenizer.convert_tokens_to_ids(sample[0])
                    output_token_id = tokenizer.convert_tokens_to_ids(sample[1])
                    output_exp_id =  tokenizer.convert_tokens_to_ids(sample[2])
                
                    video_ids.append(data['anime'])
                    episode_ids.append(data['episode'])
                    video_times.append(int(data['video_time']))
                    
                    context_position_ids.append(position_ids)
                    context_segment_ids.append(segment_ids)
                    contexts.append(context_id)
                
                    output_tokens.append(output_token_id)
                    output_exps.append(output_exp_id)
                    input_comments.append(input_comment_id )
                
                    colors.append(data['color'])
            
            print(contexts[0])
            print(context_position_ids[0])
            print(context_segment_ids[0])
            print(output_tokens[0])
            print(output_exp_id[0])
            print(input_comments[0])
                        
            #context_input = tokenizer.batch_encode_plus(contexts, pad_to_max_length=True, max_length=self.model_cfg.max_context_len, truncation=True, add_special_tokens=False)
            #comment_input = tokenizer.batch_encode_plus(comments, pad_to_max_length=True, max_length=self.model_cfg.max_comment_len, truncation=True, add_special_tokens=False)
            #input_ids, attention_mask = context_input["input_ids"], context_input["attention_mask"]
            #decoder_input_ids, decoder_attention_mask = comment_input["input_ids"], comment_input["attention_mask"]
            
            preprocessed_data = [contexts, context_position_ids, context_segment_ids, input_comments, output_tokens, output_exps, video_ids, episode_ids, video_times, colors]
            with open(self.preprocessed_file, "w") as f:
                json.dump(preprocessed_data, f)

        self.dataset = MyBiliDataset_IELM(self.model_cfg, self.train_cfg, contexts, context_position_ids, context_segment_ids, input_comments, output_tokens, output_exps, video_ids, episode_ids, video_times, colors,
                               self.imgs, tokenizer.vocab_size, self.is_training, type=self.type)

        
        return self.dataset
    
    def load_dataset_Bili_CMLM(self, tokenizer, model_type='CMLM'):
        self.tokenizer = tokenizer  

        if os.path.exists(self.preprocessed_file):
            print("Loading pre-tokenized data from {}".format(self.preprocessed_file))
            with open(self.preprocessed_file, "r") as f:
                contexts, context_position_ids, context_segment_ids, comments , non_mask_pos, video_ids, episode_ids, video_times, colors = json.load(f)
        else:
            corpus = self.load_from_json(open(self.corpus_file, 'r', encoding='utf8'))
            print ("Start tokenizing...")
            video_ids = []
            episode_ids = []
            video_times = []
            contexts = []
            context_position_ids = []
            context_segment_ids = []
            comments = []
            non_mask_pos = []
            colors = []
            
            pos_count = [{},{},{},{},{},{}]
            
            for data in corpus:
                if len(video_ids) % 1000 == 0:
                    print(len(video_ids))

                
                context = ["<&&&>"] + data['context']
                context_id = tokenizer.convert_tokens_to_ids(context)
                position_ids, segment_ids = self.get_position_segment_ids(context_id)
               

                comment_id = tokenizer.convert_tokens_to_ids(data['comment'])
            
                video_ids.append(data['anime'])
                episode_ids.append(data['episode'])
                video_times.append(int(data['video_time']))
                
                context_position_ids.append(position_ids)
                context_segment_ids.append(segment_ids)
                contexts.append(context_id)
            
                comments.append(comment_id)
                non_mask_pos.append(data['non_mask_pos'])
                
                tmp = data['non_mask_pos'][:self.train_cfg.non_mask_tokens+2]
                tmp = sorted([max(min(i,49),0) for i in tmp])
                for i,t in enumerate(tmp):
                    if t  not in pos_count[i].keys():
                        pos_count[i][t] =0
                    else:
                        pos_count[i][t] += 1
                    
                
            
                colors.append(data['color'])
            
            print(pos_count)
            for j in range(5):
                for i in range(49):
                    print(j,i,pos_count[j].get(i))
            print()
            
            print(contexts[0])
            print(context_position_ids[0])
            print(context_segment_ids[0])
            print(comments[0])
            print(non_mask_pos[0])

            preprocessed_data = [contexts, context_position_ids, context_segment_ids, comments, non_mask_pos, video_ids, episode_ids, video_times, colors]
            with open(self.preprocessed_file, "w") as f:
                json.dump(preprocessed_data, f)

        if self.type == 'generate':
            self.dataset = MyBiliDataset_CMLM_Generate(self.model_cfg, self.train_cfg, contexts, context_position_ids, context_segment_ids, comments, non_mask_pos, video_ids, episode_ids, video_times, colors,
                   self.imgs, tokenizer.vocab_size, self.is_training, type=self.type)
        else:
            self.dataset = MyBiliDataset_CMLM(self.model_cfg, self.train_cfg, contexts, context_position_ids, context_segment_ids, comments, non_mask_pos, video_ids, episode_ids, video_times, colors,
                               self.imgs, tokenizer.vocab_size, self.is_training, type=self.type)


        
        return self.dataset

    def load_gengrate_dataset_Bili_CMLM_for_POINTER(self, tokenizer, model_type='POINTER'):
        self.tokenizer = tokenizer  

        if os.path.exists(self.preprocessed_file):
            print("Loading pre-tokenized data from {}".format(self.preprocessed_file))
            with open(self.preprocessed_file, "r") as f:
                contexts, context_position_ids, context_segment_ids, input_comments, ground_truth_comments, video_ids, episode_ids, video_times, colors = json.load(f)
        else:
            corpus = self.load_from_json(open(self.corpus_file, 'r', encoding='utf8'))
            print ("Start tokenizing...")
            video_ids = []
            episode_ids = []
            video_times = []
            contexts = []
            context_position_ids = []
            context_segment_ids = []
            ground_truth_comments = []
            input_comments = []
            non_mask_pos = []
            colors = []
            

            
            for data in corpus:
                if len(video_ids) % 1000 == 0:
                    print(len(video_ids))

                
                context = ["<&&&>"] + data['context']
                context_id = tokenizer.convert_tokens_to_ids(context)
                position_ids, segment_ids = self.get_position_segment_ids(context_id)
               

                comment_id = tokenizer.convert_tokens_to_ids(data['comment'])
            
                video_ids.append(data['anime'])
                episode_ids.append(data['episode'])
                video_times.append(int(data['video_time']))
                
                context_position_ids.append(position_ids)
                context_segment_ids.append(segment_ids)
                contexts.append(context_id)
            
                ground_truth_comments.append(comment_id)
                
                non_mask_pos = sorted(data['non_mask_pos'][2:self.train_cfg.non_mask_tokens_for_CMLM+2])
                non_mask_pos = sorted([max(min(i,self.model_cfg.max_comment_len-1),0) for i in non_mask_pos])
                input_comment_id = [comment_id[i] for i in non_mask_pos]
                print(input_comment_id)
                input_comments.append(input_comment_id)
                
            
                colors.append(data['color'])
            

            preprocessed_data = [contexts, context_position_ids, context_segment_ids, input_comments, ground_truth_comments, video_ids, episode_ids, video_times, colors]
            with open(self.preprocessed_file, "w") as f:
                json.dump(preprocessed_data, f)

        self.dataset = MyBiliDataset_POINTER_Generate(self.model_cfg, self.train_cfg, contexts, context_position_ids, context_segment_ids, input_comments, ground_truth_comments, video_ids, episode_ids, video_times, colors,
                           self.imgs, tokenizer.vocab_size, self.is_training, type=self.type)
        
        
        return self.dataset


    def load_gengrate_dataset_Bili_POINTER(self, tokenizer, model_type='POINTER'):
        self.tokenizer = tokenizer  

        if os.path.exists(self.preprocessed_file):
            print("Loading pre-tokenized data from {}".format(self.preprocessed_file))
            with open(self.preprocessed_file, "r") as f:
                contexts, context_position_ids, context_segment_ids, input_comments, ground_truth_comments, video_ids, episode_ids, video_times, colors = json.load(f)
        else:
            corpus = self.load_from_json(open(self.corpus_file, 'r', encoding='utf8'))
            print ("Start tokenizing...")
            video_ids = []
            episode_ids = []
            video_times = []
            contexts = []
            context_position_ids = []
            context_segment_ids = []
            input_comments = []
            ground_truth_comments = []
            colors = []
            for data in corpus:#[:500]:
                if len(video_ids) % 1000 == 0:
                    print(len(video_ids))
                
                video_ids.append(data['anime'])
                episode_ids.append(data['episode'])
                video_times.append(int(data['video_time']))        
                
                        
                context = ["<&&&>"] + data['context']
                context_id = tokenizer.convert_tokens_to_ids(context)
                position_ids, segment_ids = self.get_position_segment_ids(context_id)
               
                input_comment_id = tokenizer.convert_tokens_to_ids(data['samples'][-1][0])
                ground_truth_comment_id = tokenizer.convert_tokens_to_ids(data['comment'])

                
                context_position_ids.append(position_ids)
                context_segment_ids.append(segment_ids)
                contexts.append(context_id)
            
                ground_truth_comments.append(ground_truth_comment_id)
                input_comments.append(input_comment_id )
            
                colors.append(data['color'])
        
            print(contexts[0])
            print(context_position_ids[0])
            print(context_segment_ids[0])
            print(ground_truth_comments[0])
            print(input_comments[0])
                        
            #context_input = tokenizer.batch_encode_plus(contexts, pad_to_max_length=True, max_length=self.model_cfg.max_context_len, truncation=True, add_special_tokens=False)
            #comment_input = tokenizer.batch_encode_plus(comments, pad_to_max_length=True, max_length=self.model_cfg.max_comment_len, truncation=True, add_special_tokens=False)
            #input_ids, attention_mask = context_input["input_ids"], context_input["attention_mask"]
            #decoder_input_ids, decoder_attention_mask = comment_input["input_ids"], comment_input["attention_mask"]
            
            preprocessed_data = [contexts, context_position_ids, context_segment_ids, input_comments, ground_truth_comments, video_ids, episode_ids, video_times, colors]
            with open(self.preprocessed_file, "w") as f:
                json.dump(preprocessed_data, f)

        self.dataset = MyBiliDataset_POINTER_Generate(self.model_cfg, self.train_cfg, contexts, context_position_ids, context_segment_ids, input_comments, ground_truth_comments, video_ids, episode_ids, video_times, colors,
                           self.imgs, tokenizer.vocab_size, self.is_training, type=self.type)
        

    def load_gengrate_dataset_Bili_IELM(self, tokenizer, model_type='IELM'):
        self.tokenizer = tokenizer  

        if os.path.exists(self.preprocessed_file):
            print("Loading pre-tokenized data from {}".format(self.preprocessed_file))
            with open(self.preprocessed_file, "r") as f:
                contexts, context_position_ids, context_segment_ids, input_comments, output_tokens, output_exps, video_ids, episode_ids, video_times, colors, ground_truth_comment_ids = json.load(f)
        else:
            corpus = self.load_from_json(open(self.corpus_file, 'r', encoding='utf8'))
            print ("Start tokenizing...")
            video_ids = []
            episode_ids = []
            video_times = []
            contexts = []
            context_position_ids = []
            context_segment_ids = []
            input_comments = []
            output_tokens = []
            output_exps = []
            colors = []
            ground_truth_comment_ids = []
            for data in corpus:#[:500]:
                if len(video_ids) % 1000 == 0:
                    print(len(video_ids))

                
                context = ["<&&&>"] + data['context']
                context_id = tokenizer.convert_tokens_to_ids(context)
                position_ids, segment_ids = self.get_position_segment_ids(context_id)
               
               
                if len(data['samples'])<=self.train_cfg.given_step:
                    continue
                
                sample = data['samples'][self.train_cfg.given_step]
                input_comment_id = tokenizer.convert_tokens_to_ids(sample[0])
                output_token_id = tokenizer.convert_tokens_to_ids(sample[1])
                output_exp_id =  tokenizer.convert_tokens_to_ids(sample[2])
            
                video_ids.append(data['anime'])
                episode_ids.append(data['episode'])
                video_times.append(int(data['video_time']))
                
                context_position_ids.append(position_ids)
                context_segment_ids.append(segment_ids)
                contexts.append(context_id)
            
                output_tokens.append(output_token_id)
                output_exps.append(output_exp_id)
                input_comments.append(input_comment_id )
            
                colors.append(data['color'])
                ground_truth_comment_ids.append(tokenizer.convert_tokens_to_ids(data['comment']))
            
            print(contexts[0])
            print(context_position_ids[0])
            print(context_segment_ids[0])
            print(output_tokens[0])
            print(output_exp_id[0])
            print(input_comments[0])
                        
            #context_input = tokenizer.batch_encode_plus(contexts, pad_to_max_length=True, max_length=self.model_cfg.max_context_len, truncation=True, add_special_tokens=False)
            #comment_input = tokenizer.batch_encode_plus(comments, pad_to_max_length=True, max_length=self.model_cfg.max_comment_len, truncation=True, add_special_tokens=False)
            #input_ids, attention_mask = context_input["input_ids"], context_input["attention_mask"]
            #decoder_input_ids, decoder_attention_mask = comment_input["input_ids"], comment_input["attention_mask"]
            
            preprocessed_data = [contexts, context_position_ids, context_segment_ids, input_comments, output_tokens, output_exps, video_ids, episode_ids, video_times, colors, ground_truth_comment_ids]
            with open(self.preprocessed_file, "w") as f:
                json.dump(preprocessed_data, f)

        self.dataset = MyBiliDataset_IELM(self.model_cfg, self.train_cfg, contexts, context_position_ids, context_segment_ids, input_comments, output_tokens, output_exps, video_ids, episode_ids, video_times, colors, 
                               self.imgs, tokenizer.vocab_size, ground_truth_comment_ids, self.is_training, type=self.type)

        
        return self.dataset


    def get_position_segment_ids(self, text, sep_id=4):
        position_ids = [0]*len(text)
        segment_ids =  [0]*len(text)
        
        i = 1
        pos = 1
        segment = 2

        while i < len(text):
            if text[i] != sep_id:
                position_ids[i] = pos
                segment_ids[i] = segment
                pos += 1
            else:
                pos = 1
                segment += 1
            i += 1
        
        return position_ids,segment_ids
    
        

    def load_dataloader(self):
        self.dataloader = MyDataLoader(self.train_cfg, self.dataset, self.is_training)
        



class MyDataset():

    def __init__(self, model_cfg, train_cfg, contexts_ids, context_position_ids, context_segment_ids, video_ids, episode_ids, video_times, colors, imgs, vocab_size, is_training= True, type="pretrain" ):
        self.model_cfg = model_cfg
        self.train_cfg = train_cfg
        self.contexts_ids = contexts_ids
        self.context_position_ids = context_position_ids
        self.context_segment_ids = context_segment_ids

        self.is_training = is_training
        self.type = type
        self.video_ids = video_ids
        self.episode_ids = episode_ids
        self.colors = colors
        self.video_times = video_times
        self.imgs = imgs
        self.vocab_size = vocab_size
        self.device = torch.device("cuda")
        


    def get_position_segment_ids(self, text, seg_min=1):
        position_ids = [0]*len(text)
        segment_ids =  [0]*len(text)
        
        i = 1
        pos = 1
        segment = seg_min

        while i < len(text) :
            position_ids[i] = pos
            segment_ids[i] = segment
            pos += 1
            i += 1
        return position_ids,segment_ids  


    def padding(self,ids,length,direction='left', pad_id=0, ending_id=0):
        if len(ids) > length:
            ids = ids[:length]
            if ending_id !=0 and ids[-1] != ending_id and ids[-1] != pad_id:
                ids[-1] = ending_id
        if direction == 'left':
            ids = [pad_id] * (length-len(ids)) + ids
        else:
            ids = ids + [pad_id] * (length-len(ids))
        return ids



    def get_mask(self,train_input_id):
        return [0 if id==0 else 1 for id in train_input_id]
    

    def mask_token(self, token_id, mask_token_id = 5):
        rand = np.random.random()
        if rand < 0.8:
            return mask_token_id
        elif rand < 0.9:
            # sample random token according to input distribution
            return random.randint(8, self.vocab_size-1)
        else:
            return token_id

    def get_pairs(self, masked_lm_labels):
        pairs = []
        pair_targets = []
        for i in range(len(masked_lm_labels)):
            if masked_lm_labels[i] != -1:
                left,right = i-1, i+1
                while left>0 and masked_lm_labels[left] != -1 :
                    left -= 1
                while right<self.model_cfg.max_len and masked_lm_labels[right] != -1:
                    right += 1
                if right < len(masked_lm_labels):
                    pairs.append([left,i,right])
                    pair_targets.append(masked_lm_labels[i])
        pairs += [[0,0,0]] * (self.train_cfg.max_pair_num - len(pairs))
        pair_targets += [-1] * (self.train_cfg.max_pair_num - len(pair_targets))     
        
        pairs = pairs[:self.train_cfg.max_pair_num]
        pair_targets = pair_targets[:self.train_cfg.max_pair_num] 
        return pairs, pair_targets
    
    def load_imgs_Bili(self,video_id,episode_id,video_time):
        
        previous = [i for i in range(-self.model_cfg.max_n_clips + 1, 1)]

        V_t = torch.zeros([self.model_cfg.max_n_clips,self.model_cfg.dim], device = self.device)
        i = self.model_cfg.max_n_clips-1
        for t in previous[::-1]:
            if video_time + t >= 0 and video_time + t < len(self.imgs[video_id][episode_id]):
                V_t[i] = torch.cat((self.imgs[video_id][episode_id][video_time + t],self.imgs[video_id][episode_id][video_time + t]), dim=-1)
                i -= 1
        return V_t.to(self.device)

    def load_imgs_LB(self,video_id,episode_id,video_time):
        episode_id = int(episode_id)
        previous = [i for i in range(-self.model_cfg.max_n_clips + 1, 1)]

        V_t = torch.zeros([self.model_cfg.max_n_clips,self.model_cfg.dim], device = self.device)
        i = self.model_cfg.max_n_clips-1
        for t in previous[::-1]:
            if video_time + t >= 0 and video_time + t < len(self.imgs[episode_id]):
                V_t[i] = torch.cat((self.imgs[episode_id][video_time + t],self.imgs[episode_id][video_time + t]), dim=-1)
                i -= 1
        return V_t.to(self.device)

class MyLBDataset(MyDataset):
    def __init__(self, model_cfg, train_cfg, contexts_ids, context_position_ids, context_segment_ids, comments_ids, video_ids, episode_ids, video_times, colors, imgs, vocab_size, is_training= True, type="pretrain" ):
        self.comments_ids = comments_ids#torch.LongTensor(decoder_input_ids)
        super(MyLBDataset, self).__init__(model_cfg, train_cfg, contexts_ids, context_position_ids, context_segment_ids, video_ids, episode_ids, video_times, colors, imgs, vocab_size, is_training, type )
        
    def cotext_mask(self, context_id_input, mask_token_id = 5, special_ids = [0,1,2,3,4,5,6,7]):
        context_id = copy.deepcopy(context_id_input)
        masked_tokens = 0

        i = 1
        while i<len(context_id):
            prob = random.random()
            if prob < self.train_cfg.mask_prob:

                if context_id[i] in special_ids:
                    continue
                context_id[i] = self.mask_token(context_id[i], mask_token_id)
                masked_tokens += 1
                i += 2
            i += 1
    
        return(context_id)

    def __len__(self):
        return len(self.contexts_ids)

    def __getitem__(self, idx):
        video_time = self.video_times[idx]
        visual = self.load_imgs_LB(self.video_ids[idx], self.episode_ids[idx], video_time)
        video_time = float(video_time)
        color = float(self.colors[idx])
        
        context_id, context_gt_id, masked_lm_labels_context = self.infilling(self.contexts_ids[idx],'context')
        context_id = self.padding(context_id, self.model_cfg.max_context_len, direction='left')
        context_gt_id = self.padding(context_gt_id, self.model_cfg.max_context_len, direction='left')
        masked_lm_labels_context = self.padding(masked_lm_labels_context, self.model_cfg.max_context_len, direction='left', pad_id=-1)
        context_position_id = self.padding(self.context_position_ids[idx] , self.model_cfg.max_context_len, direction='left')
        context_segment_id =  self.padding(self.context_segment_ids[idx], self.model_cfg.max_context_len, direction='left')

        comment_id, comment_gt_id, masked_lm_labels_comment = self.infilling(self.comments_ids[idx],'comment')
        comment_position_id, comment_segment_id = self.get_position_segment_ids(comment_id)
        comment_id = self.padding(comment_id, self.model_cfg.max_comment_len, direction='right', ending_id=2)
        comment_gt_id = self.padding(comment_gt_id, self.model_cfg.max_comment_len, direction='right', ending_id=2)
        masked_lm_labels_comment = self.padding(masked_lm_labels_comment, self.model_cfg.max_comment_len, direction='right', pad_id=-1, ending_id=-1)
        comment_position_id = self.padding(comment_position_id, self.model_cfg.max_comment_len, direction='right')
        comment_segment_id = self.padding(comment_segment_id, self.model_cfg.max_comment_len, direction='right')
        
        
        
        train_input_id = context_id + comment_id
        input_mask = self.get_mask(train_input_id)
        input_mask = [1] * (self.model_cfg.max_n_clips + 2) + input_mask
        position_ids = context_position_id + comment_position_id
        segment_ids = context_segment_id + comment_segment_id
        
        masked_lm_labels = [-1] * (self.model_cfg.max_n_clips + 2) + masked_lm_labels_context + masked_lm_labels_comment
        
        pairs, pair_targets = self.get_pairs(masked_lm_labels)
        
        return torch.LongTensor(train_input_id), torch.LongTensor(input_mask), \
               torch.LongTensor(position_ids), torch.LongTensor(segment_ids), \
               torch.LongTensor(masked_lm_labels), visual, \
               torch.LongTensor([color]*self.model_cfg.dim).float(), \
               torch.LongTensor([video_time]*self.model_cfg.dim).float(),\
               torch.LongTensor(pairs), torch.LongTensor(pair_targets)
        


    def infilling(self, input_id, type='context', mask_token_id=5, noi_token_id=7):
        length = len(input_id)
        train_input_id = copy.deepcopy(input_id)
        #print(train_input_id)
        
        masked_lm_labels = [-1]*length
        
        if type == 'context':
            max_masked = self.train_cfg.max_masked_context
        else:
            max_masked = self.train_cfg.max_masked_comment
            
        rand_geom = geom.rvs(size=max_masked*2,p=self.train_cfg.p_geom)
        mask_len = [min(self.train_cfg.max_mask_len,int(l)) for l in rand_geom]
        
        begin_pos = 2
        end_pos = length-2
        

        
        
        for i in range(max_masked):
            if begin_pos>=end_pos:
                #print("begin,end",begin_pos,end_pos)
                break
            #print(i)
            mask_pos = random.randint(begin_pos,end_pos)
            #print(mask_pos)
            
            n = 0
            #print("left",mask_len[2*i])
            while mask_len[2*i] > n and (mask_pos-1-n)>=begin_pos and input_id[mask_pos-1-n] not in [0,1,2,3,4,5,6,7]:
                #print("rewrite",n,mask_pos-1-n)
                train_input_id[mask_pos-1-n] = self.mask_token(train_input_id[mask_pos-1-n], mask_token_id)
                masked_lm_labels[mask_pos-1-n] = input_id[mask_pos-1-n]
                n += 1
            '''
            if n == 0 or mask_len[2*i]>n:
                #print("insert",n,mask_pos-1-n+1)
                train_input_id.insert(mask_pos-1-n+1, mask_token_index)
                input_id.insert(mask_pos-1-n+1, noi_token_index)
                lm_pos.insert(mask_pos-1-n+1, 1)
                mask_pos += 1
                end_pos += 1
            '''
            #print(train_input_id)
            #print(input_id)
            n = 0
            #print("right",mask_len[2*i+1])
            while mask_len[2*i+1] > n and (mask_pos+1+n)<end_pos and input_id[mask_pos+1+n] not in [0,1,2,3,4,5,6,7]:
                #print("rewrite",n,mask_pos+1+n)
                train_input_id[mask_pos+1+n] = self.mask_token(train_input_id[mask_pos+1+n], mask_token_id)
                masked_lm_labels[mask_pos+1+n] = input_id[mask_pos+1+n]
                n += 1
            '''
            if n == 0 or mask_len[2*i]>n:
                #print("insert",n,mask_pos+1+n)
                train_input_id.insert(mask_pos+1+n, mask_token_index)
                input_id.insert(mask_pos+1+n, noi_token_index)
                lm_pos.insert(mask_pos+1+n, 1)
                end_pos += 1
            '''
            #print(train_input_id)
            #print(input_id)
            begin_pos = mask_pos + n + 3
            
        
        return train_input_id,input_id,masked_lm_labels



    
                  
class MyBiliDataset(MyDataset):
    def __init__(self, model_cfg, train_cfg, contexts_ids, context_position_ids, context_segment_ids, input_comments_ids=None, output_comments_ids=None, video_ids=None, episode_ids=None, video_times=None, colors=None, imgs=None, vocab_size=None, is_training= True,  type="pretrain" ):
        self.input_comments_ids = input_comments_ids#torch.LongTensor(decoder_input_ids)
        self.output_comments_ids = output_comments_ids
        super(MyBiliDataset, self).__init__(model_cfg, train_cfg, contexts_ids, context_position_ids, context_segment_ids, video_ids, episode_ids, video_times, colors, imgs, vocab_size, is_training, type )
        
        self.device = torch.device("cuda")



    def __len__(self):
        return len(self.contexts_ids)


    def get_output_weight(self):
        vocab_dic =  dict().fromkeys([i for i in range(self.vocab_size)], 1)
        words = self.vocab_size
        for ids in self.output_comments_ids:
            for id in ids:
                vocab_dic[id] += 1
            words += len(ids)
        
        words = words / self.vocab_size
        for i in range(self.vocab_size):
            vocab_dic[i] = words/vocab_dic[i]
        
        print(vocab_dic.values())
        return list(vocab_dic.values())
        

    def insert_masks(self,comment_id, comment_gt_id, pad_token_id=0, mask_token_id=5):
        masked_comment_id = []
        masked_lm_labels = []
        
        for tc,tg in zip(comment_id[:-1],comment_gt_id[:-1]):
            masked_comment_id.append(tc)
            masked_lm_labels.append(-1)
            
            masked_comment_id.append(mask_token_id)
            masked_lm_labels.append(tg)
        
        masked_comment_id.append(comment_id[-1])
        masked_lm_labels.append(-1)
        
        return masked_comment_id,masked_lm_labels


class MyLBDataset_POINTER(MyLBDataset):
    def __init__(self, model_cfg, train_cfg, contexts_ids, context_position_ids, context_segment_ids, comments_ids, video_ids, episode_ids, video_times, colors, imgs, vocab_size, is_training= True, next_sentence_pred=True, type="pretrain" ):
        
        self.comments_ids = comments_ids#torch.LongTensor(decoder_input_ids)
        super(MyLBDataset_POINTER, self).__init__(model_cfg, train_cfg, contexts_ids, context_position_ids, context_segment_ids, comments_ids, video_ids, episode_ids, video_times, colors, imgs, vocab_size, is_training= is_training, type=type)
    
        self.next_sentence_pred = next_sentence_pred
        self.load_imgs=self.load_imgs_LB
    
    def __getitem__(self, idx):
        video_time = self.video_times[idx]
        visual = self.load_imgs(self.video_ids[idx], self.episode_ids[idx], video_time)
        video_time = float(video_time)
        color = float(self.colors[idx])
        next_sentence_label = 1
        
        context_id, context_gt_id, masked_lm_labels_context = self.infilling(self.contexts_ids[idx],'context')
        context_id = self.padding(context_id, self.model_cfg.max_context_len, direction='left')
        context_gt_id = self.padding(context_gt_id, self.model_cfg.max_context_len, direction='left')
        masked_lm_labels_context = self.padding(masked_lm_labels_context, self.model_cfg.max_context_len, direction='left', pad_id=-1)
        context_position_id = self.padding(self.context_position_ids[idx] , self.model_cfg.max_context_len, direction='left')
        context_segment_id =  self.padding(self.context_segment_ids[idx], self.model_cfg.max_context_len, direction='left')


        
        id = idx
        
        if self.next_sentence_pred:
            if random.random()<self.train_cfg.next_sentence_prob:
                rand_id = random.randint(0,len(self.contexts_ids)-1)
                while self.episode_ids[rand_id] == self.episode_ids[idx]:
                    rand_id = random.randint(0,len(self.contexts_ids)-1)
                id = rand_id
                next_sentence_label = 0


        comment_id, comment_gt_id, masked_lm_labels_comment = self.infilling(self.comments_ids[id],'comment')
        comment_position_id, comment_segment_id = self.get_position_segment_ids(comment_id)
        comment_id = self.padding(comment_id, self.model_cfg.max_comment_len, direction='right', ending_id=2)
        comment_gt_id = self.padding(comment_gt_id, self.model_cfg.max_comment_len, direction='right', ending_id=2)
        masked_lm_labels_comment = self.padding(masked_lm_labels_comment, self.model_cfg.max_comment_len, direction='right', pad_id=-1, ending_id=-1)
        comment_position_id = self.padding(comment_position_id, self.model_cfg.max_comment_len, direction='right')
        comment_segment_id = self.padding(comment_segment_id, self.model_cfg.max_comment_len, direction='right')
        
        
        
        train_input_id = context_id + comment_id
        input_mask = self.get_mask(train_input_id)
        train_input_id = [0] * (self.model_cfg.max_n_clips + 2) + train_input_id
        input_mask = [1] * (self.model_cfg.max_n_clips + 2) + input_mask
        position_ids =  [0] * (self.model_cfg.max_n_clips + 2) +  context_position_id + comment_position_id
        segment_ids = [0] * (self.model_cfg.max_n_clips + 2) + context_segment_id + comment_segment_id
        
        masked_lm_labels = [-1] * (self.model_cfg.max_n_clips + 2) + masked_lm_labels_context + masked_lm_labels_comment
        
        return torch.LongTensor(train_input_id), torch.LongTensor(input_mask), \
               torch.LongTensor(position_ids), torch.LongTensor(segment_ids), \
               torch.LongTensor(masked_lm_labels), visual, \
               torch.LongTensor([color]*self.model_cfg.dim).float(), \
               torch.LongTensor([video_time]*self.model_cfg.dim).float(),\
               torch.LongTensor([next_sentence_label])
        
    def infilling(self, input_id, type='context', mask_token_id=5, special_ids=[0,1,2,3,4,5,6,7]):
        length = len(input_id)
        train_input_id = copy.deepcopy(input_id)
        masked_lm_labels = [-1]*length
        
        for i in range(length):
            if train_input_id[i] in special_ids:
                continue
            if random.random() < self.train_cfg.mask_prob:
                train_input_id[i] = self.mask_token(train_input_id[i], mask_token_id)
                masked_lm_labels[i] = input_id[i]
        return train_input_id,input_id,masked_lm_labels

        

class MyBiliDataset_POINTER(MyBiliDataset):
    def __init__(self, model_cfg, train_cfg, contexts_ids, context_position_ids, context_segment_ids, input_comments_ids, output_comments_ids, video_ids, episode_ids, video_times, colors, imgs, vocab_size, is_training= True, type="pretrain" ):
        super(MyBiliDataset_POINTER, self).__init__( model_cfg, train_cfg, contexts_ids, context_position_ids, context_segment_ids, input_comments_ids, output_comments_ids, video_ids, episode_ids, video_times, colors, imgs, vocab_size, is_training, type)
        
        self.device = torch.device("cuda")
        self.load_imgs=self.load_imgs_Bili

    def __getitem__(self, idx):
        video_time = self.video_times[idx]
        visual = self.load_imgs(self.video_ids[idx], self.episode_ids[idx], video_time)
        video_time = float(video_time)
        color = float(self.colors[idx])

        
        context_id = self.padding(self.contexts_ids[idx], self.model_cfg.max_context_len, direction='left')
        context_position_id = self.padding(self.context_position_ids[idx] , self.model_cfg.max_context_len, direction='left')
        context_segment_id =  self.padding(self.context_segment_ids[idx], self.model_cfg.max_context_len, direction='left')

        #if self.is_training and self.type=="pretrain":
        comment_id, comment_gt_id = self.input_comments_ids[idx], self.output_comments_ids[idx]
        comment_position_id, comment_segment_id = self.get_position_segment_ids(comment_id)
        
        comment_id = self.padding(comment_id, self.model_cfg.max_comment_len, direction='right')
        comment_position_id = self.padding(comment_position_id, self.model_cfg.max_comment_len, direction='right')
        comment_segment_id = self.padding(comment_segment_id, self.model_cfg.max_comment_len, direction='right')
        
        masked_lm_labels = self.padding(comment_gt_id, self.model_cfg.max_comment_len, direction='right', pad_id=-1)
        
        train_input_id = context_id + comment_id
        
        input_mask = self.get_mask(train_input_id)
        train_input_id = [0] * (self.model_cfg.max_n_clips + 2) + train_input_id
        input_mask = [1] * (self.model_cfg.max_n_clips + 2) + input_mask
        position_ids =  [0] * (self.model_cfg.max_n_clips + 2) +  context_position_id + comment_position_id
        segment_ids = [0] * (self.model_cfg.max_n_clips + 2) + context_segment_id + comment_segment_id
        

        masked_lm_labels = [-1] * (self.model_cfg.max_n_clips + 2 + len(context_id)) + masked_lm_labels

           
        return torch.LongTensor(train_input_id), torch.LongTensor(input_mask), \
               torch.LongTensor(position_ids), torch.LongTensor(segment_ids), \
               torch.LongTensor(masked_lm_labels), visual, \
               torch.LongTensor([color]*self.model_cfg.dim).float(), \
               torch.LongTensor([video_time]*self.model_cfg.dim).float(),\
               torch.LongTensor([[0,0,0]]*self.train_cfg.max_pair_num),torch.LongTensor([-1]*self.train_cfg.max_pair_num)

class MyBiliDataset_CMLM(MyBiliDataset):
    def __init__(self, model_cfg, train_cfg, contexts_ids, context_position_ids, context_segment_ids, comments_ids, non_mask_pos, video_ids, episode_ids, video_times, colors, imgs, vocab_size, is_training= True, type="pretrain" ):
        super(MyBiliDataset_CMLM, self).__init__( model_cfg, train_cfg, contexts_ids, context_position_ids, context_segment_ids, comments_ids, None, video_ids, episode_ids, video_times, colors, imgs, vocab_size, is_training, type)
        self.non_mask_pos = non_mask_pos
        self.device = torch.device("cuda")
        self.load_imgs=self.load_imgs_Bili
        self.train_cfg = train_cfg

    def __getitem__(self, idx):
        orig_idx = idx
        idx = idx//2
        #print(orig_idx,orig_idx%2,idx)
        
            
        video_time = self.video_times[idx]
        visual = self.load_imgs(self.video_ids[idx], self.episode_ids[idx], video_time)
        video_time = float(video_time)
        color = float(self.colors[idx])

        if self.train_cfg.without_visual:
            visual = torch.zeros_like(self.load_imgs(self.video_ids[idx], self.episode_ids[idx], video_time))
        if self.train_cfg.without_video_time:
            video_time = 0.0 
        if self.train_cfg.without_color:
            color = 0.0
        
        context_id = self.padding(self.contexts_ids[idx], self.model_cfg.max_context_len, direction='left')
        if self.train_cfg.without_context:
            context_id = self.padding([], self.model_cfg.max_context_len, direction='left')
        context_position_id = self.padding(self.context_position_ids[idx] , self.model_cfg.max_context_len, direction='left')
        context_segment_id =  self.padding(self.context_segment_ids[idx], self.model_cfg.max_context_len, direction='left')

        non_mask_pos = self.non_mask_pos[idx][:self.train_cfg.non_mask_tokens+2]
        non_mask_pos = sorted([max(min(i,self.model_cfg.max_comment_len_CMLM-1),0) for i in non_mask_pos])
        
        if orig_idx%2 == 1:
            comment_id, masked_lm_labels_comment = self.infilling(self.input_comments_ids[idx],non_mask_pos)
            non_mask_pos = []
        else:
            comment_id, masked_lm_labels_comment = [],[]
            
        comment_position_id, comment_segment_id = self.get_position_segment_ids(comment_id)
        comment_id = self.padding(comment_id, self.model_cfg.max_comment_len_CMLM, direction='right',ending_id=2)
        comment_position_id = self.padding(comment_position_id, self.model_cfg.max_comment_len_CMLM, direction='right')
        comment_segment_id = self.padding(comment_segment_id, self.model_cfg.max_comment_len_CMLM, direction='right')
        masked_lm_labels = self.padding(masked_lm_labels_comment, self.model_cfg.max_comment_len_CMLM, direction='right', pad_id=-1)
        
        
        pos_id_gt = self.padding(non_mask_pos, self.model_cfg.max_pos_len_CMLM, direction='right', pad_id=-1)
        pos_labels = [-1] * (self.model_cfg.max_n_clips + 2) + [-1] *(self.model_cfg.max_context_len) + pos_id_gt + [-1]*self.model_cfg.max_comment_len_CMLM
        pos_position_id = [0]*self.model_cfg.max_pos_len_CMLM 
        pos_segment_id = [min(max(context_segment_id)+1,99)]*self.model_cfg.max_pos_len_CMLM  
        pos_id_input = [self.input_comments_ids[idx][i] for i in non_mask_pos]
        pos_id_input = self.padding(pos_id_input, self.model_cfg.max_pos_len_CMLM, direction='right')
        
        train_input_id = context_id + pos_id_input + comment_id
        
        input_mask = self.get_mask(train_input_id)
        train_input_id = [0] * (self.model_cfg.max_n_clips + 2) + train_input_id
        input_mask = [1] * (self.model_cfg.max_n_clips + 2) + input_mask
        position_ids = [0] * (self.model_cfg.max_n_clips + 2) + context_position_id + pos_position_id + comment_position_id
        segment_ids = [0] * (self.model_cfg.max_n_clips + 2) + context_segment_id + pos_segment_id + comment_segment_id
          
        masked_lm_labels = [-1] * (self.model_cfg.max_n_clips + 2) + [-1] *(self.model_cfg.max_context_len+self.model_cfg.max_pos_len_CMLM) + masked_lm_labels

        #print(train_input_id)
        #print(position_ids)
        #print(segment_ids)

        return torch.LongTensor(train_input_id), torch.LongTensor(input_mask), \
               torch.LongTensor(position_ids), torch.LongTensor(segment_ids), \
               torch.LongTensor(masked_lm_labels), \
               visual,  torch.LongTensor([color]*self.model_cfg.dim).float(), \
               torch.LongTensor([video_time]*self.model_cfg.dim).float(),\
               torch.LongTensor(pos_labels)

    def infilling(self, input_id, non_mask_pos, mask_token_id=5, special_ids=[0,1,2,3,4,5,6,7]):
        length = len(input_id)
        train_input_id = copy.deepcopy(input_id)
        masked_lm_labels = [-1]*length
        
        sample_size = random.randint(min(3,length-2), length-2)
        ind = np.random.RandomState(114514).choice(length , size=sample_size, replace=False)
        
        for p in ind:
            if p not in non_mask_pos and train_input_id[p] not in special_ids:
                train_input_id[p] = self.mask_token(train_input_id[p], mask_token_id)
                masked_lm_labels[p] = input_id[p]
        
        
        if sum(masked_lm_labels) == -1 * length:
            p = random.randint(1, length-2)
            if train_input_id[p] not in special_ids:
                train_input_id[p] = self.mask_token(train_input_id[p], mask_token_id)
                masked_lm_labels[p] = input_id[p]

        return train_input_id, masked_lm_labels
    
    def mask_token(self, token_id, mask_token_id = 5):
        return mask_token_id
    
    def __len__(self):
        return len(self.contexts_ids*2)
        
        
class MyBiliDataset_CMLM_Generate(MyBiliDataset_CMLM):
    def __init__(self, model_cfg, train_cfg, contexts_ids, context_position_ids, context_segment_ids, comments_ids, non_mask_pos, video_ids, episode_ids, video_times, colors, imgs, vocab_size, is_training= True, type="pretrain" ):
        super(MyBiliDataset_CMLM_Generate, self).__init__( model_cfg, train_cfg, contexts_ids, context_position_ids, context_segment_ids, comments_ids, None, video_ids, episode_ids, video_times, colors, imgs, vocab_size, is_training, type)
        self.non_mask_pos = non_mask_pos
        self.ground_truth_comment_ids = comments_ids
        self.device = torch.device("cuda")
        self.load_imgs=self.load_imgs_Bili
        print(len(self.video_times),len(contexts_ids))

    def __getitem__(self, idx):
        video_time = self.video_times[idx]
        visual = self.load_imgs(self.video_ids[idx], self.episode_ids[idx], video_time)
        video_time = float(video_time)
        color = float(self.colors[idx])


        if self.train_cfg.without_visual:
            visual = torch.zeros_like(self.load_imgs(self.video_ids[idx], self.episode_ids[idx], video_time))
        if self.train_cfg.without_video_time:
            video_time = 0.0 
        if self.train_cfg.without_color:
            color = 0.0

        context_id = self.padding(self.contexts_ids[idx], self.model_cfg.max_context_len, direction='left')
        if self.train_cfg.without_context:
            context_id = self.padding([], self.model_cfg.max_context_len, direction='left')
        context_position_id = self.padding(self.context_position_ids[idx] , self.model_cfg.max_context_len, direction='left')
        context_segment_id =  self.padding(self.context_segment_ids[idx], self.model_cfg.max_context_len, direction='left')

        non_mask_pos = self.non_mask_pos[idx][:self.train_cfg.non_mask_tokens+2]
        non_mask_pos = sorted([max(min(i,self.model_cfg.max_comment_len_CMLM-1),0) for i in non_mask_pos])
        
                
        gt_comment_id = self.padding(self.input_comments_ids[idx], self.model_cfg.max_comment_len_CMLM, direction='right',ending_id=2)
        gt_comment_id = [0] * (self.model_cfg.max_n_clips + 2) + [0] *(self.model_cfg.max_context_len)  + [0]*(self.model_cfg.max_pos_len_CMLM) + gt_comment_id
        
        comment_id =  []
        masked_lm_labels_comment = []

        comment_position_id, comment_segment_id = self.get_position_segment_ids(comment_id)
        comment_id = self.padding(comment_id, self.model_cfg.max_comment_len_CMLM, direction='right',ending_id=2)
        comment_position_id = self.padding(comment_position_id, self.model_cfg.max_comment_len_CMLM, direction='right')
        comment_segment_id = self.padding(comment_segment_id, self.model_cfg.max_comment_len_CMLM, direction='right')
        masked_lm_labels = self.padding(masked_lm_labels_comment, self.model_cfg.max_comment_len_CMLM, direction='right', pad_id=-1)
        
        
        pos_id_gt = self.padding(non_mask_pos, self.model_cfg.max_pos_len_CMLM, direction='right', pad_id=-1)
        pos_labels = [-1] * (self.model_cfg.max_n_clips + 2) + [-1] *(self.model_cfg.max_context_len) + pos_id_gt + [-1]*self.model_cfg.max_comment_len_CMLM
        pos_position_id = [0]*self.model_cfg.max_pos_len_CMLM 
        pos_segment_id = [min(max(context_segment_id)+1,99)]*self.model_cfg.max_pos_len_CMLM  
        pos_id_input = [self.input_comments_ids[idx][i] for i in non_mask_pos]
        pos_id_input = self.padding(pos_id_input, self.model_cfg.max_pos_len_CMLM, direction='right')
        
        train_input_id = context_id + pos_id_input + comment_id
        
        input_mask = self.get_mask(train_input_id)
        train_input_id = [0] * (self.model_cfg.max_n_clips + 2) + train_input_id
        input_mask = [1] * (self.model_cfg.max_n_clips + 2) + input_mask
        position_ids = [0] * (self.model_cfg.max_n_clips + 2) + context_position_id + pos_position_id + comment_position_id
        segment_ids = [0] * (self.model_cfg.max_n_clips + 2) + context_segment_id + pos_segment_id + comment_segment_id
        

        masked_lm_labels = [-1] * (self.model_cfg.max_n_clips + 2) + [-1] *(self.model_cfg.max_context_len+self.model_cfg.max_pos_len_CMLM) + masked_lm_labels

        #print(train_input_id)
        #print(position_ids)
        #print(segment_ids)
        #print("pos_id_input",pos_id_input)
        
        return torch.LongTensor(train_input_id), torch.LongTensor(input_mask), \
               torch.LongTensor(position_ids), torch.LongTensor(segment_ids), \
               torch.LongTensor(masked_lm_labels), \
               visual,  torch.LongTensor([color]*self.model_cfg.dim).float(), \
               torch.LongTensor([video_time]*self.model_cfg.dim).float(),\
               torch.LongTensor(pos_labels),torch.LongTensor(gt_comment_id),\
               torch.LongTensor(pos_id_input),torch.LongTensor([idx])
    
    def __len__(self):
        return len(self.contexts_ids)-1


class MyBiliDataset_CLVCG(MyBiliDataset):
    def __init__(self, model_cfg, train_cfg, contexts_ids, context_position_ids, context_segment_ids, input_comments_ids, output_comments_ids, video_ids, episode_ids, video_times, colors, imgs, vocab_size, is_training= True, type="pretrain" ):
        super(MyBiliDataset_CLVCG, self).__init__( model_cfg, train_cfg, contexts_ids, context_position_ids, context_segment_ids, input_comments_ids, output_comments_ids, video_ids, episode_ids, video_times, colors, imgs, vocab_size, is_training, type)
        self.load_imgs=self.load_imgs_Bili
        print(len(self.video_times),len(contexts_ids))

    def __getitem__(self, idx):
        video_time = self.video_times[idx]
        visual = self.load_imgs(self.video_ids[idx], self.episode_ids[idx], video_time)
        video_time = float(video_time)
        color = float(self.colors[idx])
        
        context_id = self.padding(self.contexts_ids[idx], self.model_cfg.max_context_len, direction='left')
        context_position_id = self.padding(self.context_position_ids[idx] , self.model_cfg.max_context_len, direction='left')
        context_segment_id =  self.padding(self.context_segment_ids[idx], self.model_cfg.max_context_len, direction='left')

        #if self.is_training and self.type=="pretrain":
        masked_comment_id,masked_lm_labels = self.insert_masks(self.input_comments_ids[idx], self.output_comments_ids[idx])

        comment_position_id, comment_segment_id = self.get_position_segment_ids(masked_comment_id)
        
        masked_comment_id = self.padding(masked_comment_id, self.model_cfg.max_comment_len, direction='right', ending_id=2)
        comment_position_id = self.padding(comment_position_id, self.model_cfg.max_comment_len, direction='right')
        comment_segment_id = self.padding(comment_segment_id, self.model_cfg.max_comment_len, direction='right')
        
        masked_lm_labels = self.padding(masked_lm_labels, self.model_cfg.max_comment_len, direction='right', pad_id=-1)
        
        train_input_id = context_id + masked_comment_id
        
        input_mask = self.get_mask(train_input_id)
        train_input_id = [0] * (self.model_cfg.max_n_clips + 2) + train_input_id
        input_mask = [1] * (self.model_cfg.max_n_clips + 2) + input_mask
        position_ids = [0] * (self.model_cfg.max_n_clips + 2) +context_position_id + comment_position_id
        segment_ids = [0] * (self.model_cfg.max_n_clips + 2) +context_segment_id + comment_segment_id
        

        masked_lm_labels = [-1] * (self.model_cfg.max_n_clips + 2 + len(context_id)) + masked_lm_labels


        pairs, pair_targets = self.get_pairs(masked_lm_labels)

        
        return torch.LongTensor(train_input_id), torch.LongTensor(input_mask), \
               torch.LongTensor(position_ids), torch.LongTensor(segment_ids), \
               torch.LongTensor(masked_lm_labels), visual, \
               torch.LongTensor([color]*self.model_cfg.dim).float(), \
               torch.LongTensor([video_time]*self.model_cfg.dim).float(),\
               torch.LongTensor(pairs), torch.LongTensor(pair_targets)

    

class MyBiliDataset_POINTER_Generate(MyBiliDataset):
    def __init__(self, model_cfg, train_cfg, contexts_ids, context_position_ids, context_segment_ids, input_comments_ids, ground_truth_comment_ids, video_ids, episode_ids, video_times, colors, imgs, vocab_size, is_training= True, type="pretrain" ):
        super(MyBiliDataset_POINTER_Generate, self).__init__( model_cfg, train_cfg, contexts_ids, context_position_ids, context_segment_ids, input_comments_ids, ground_truth_comment_ids, video_ids, episode_ids, video_times, colors, imgs, vocab_size, is_training, type)
        self.ground_truth_comment_ids = ground_truth_comment_ids
        self.device = torch.device("cuda")
        self.load_imgs=self.load_imgs_Bili

    def __getitem__(self, idx):
        video_time = self.video_times[idx]
        visual = self.load_imgs(self.video_ids[idx], self.episode_ids[idx], video_time)
        video_time = float(video_time)
        color = float(self.colors[idx])

        
        context_id = self.padding(self.contexts_ids[idx], self.model_cfg.max_context_len, direction='left')
        context_position_id = self.padding(self.context_position_ids[idx] , self.model_cfg.max_context_len, direction='left')
        context_segment_id =  self.padding(self.context_segment_ids[idx], self.model_cfg.max_context_len, direction='left')

        #if self.is_training and self.type=="pretrain":
        masked_comment_id,masked_lm_labels = self.input_comments_ids[idx], self.ground_truth_comment_ids[idx]
        comment_position_id, comment_segment_id = self.get_position_segment_ids(masked_comment_id)
        

        masked_comment_id = self.padding(masked_comment_id, self.model_cfg.max_comment_len, direction='right', ending_id=2)
        comment_position_id = self.padding(comment_position_id, self.model_cfg.max_comment_len, direction='right')
        comment_segment_id = self.padding(comment_segment_id, self.model_cfg.max_comment_len, direction='right')
        
        masked_lm_labels = self.padding(masked_lm_labels, self.model_cfg.max_comment_len, direction='right', pad_id=-1)
        
        input_id = context_id + masked_comment_id
        
        input_mask = self.get_mask(input_id)
        input_id = [0] * (self.model_cfg.max_n_clips + 2) + input_id
        input_mask = [1] * (self.model_cfg.max_n_clips + 2) + input_mask
        position_ids = [0] * (self.model_cfg.max_n_clips + 2) +context_position_id + comment_position_id
        segment_ids = [0] * (self.model_cfg.max_n_clips + 2) +context_segment_id + comment_segment_id
        
        masked_lm_labels = [-1] * (self.model_cfg.max_n_clips + 2 + len(context_id)) + masked_lm_labels

           
        return torch.LongTensor(input_id), torch.LongTensor(input_mask), \
               torch.LongTensor(position_ids), torch.LongTensor(segment_ids), \
               torch.LongTensor(masked_lm_labels), visual, \
               torch.LongTensor([color]*self.model_cfg.dim).float(), \
               torch.LongTensor([video_time]*self.model_cfg.dim).float(),\
               torch.LongTensor([idx])

    def __len__(self):
        return len(self.contexts_ids)-1  
    
    

class MyBiliDataset_IELM(MyBiliDataset):
    def __init__(self, model_cfg, train_cfg, contexts_ids, context_position_ids, context_segment_ids, input_comments_ids, output_tokens_ids, output_exps_ids, video_ids, episode_ids, video_times, colors, imgs, vocab_size, ground_truth_comment_ids=None, is_training= True, type="pretrain" ):
        super(MyBiliDataset_IELM, self).__init__( model_cfg, train_cfg, contexts_ids, context_position_ids, context_segment_ids, None, None, video_ids, episode_ids, video_times, colors, imgs, vocab_size, is_training, type)
        self.input_comments_ids = input_comments_ids 
        self.output_tokens_ids = output_tokens_ids
        self.output_exps_ids = output_exps_ids
        self.device = torch.device("cuda")
        self.load_imgs=self.load_imgs_Bili
        self.ground_truth_comment_ids=ground_truth_comment_ids

    def __getitem__(self, idx):
        video_time = self.video_times[idx]
        visual = self.load_imgs(self.video_ids[idx], self.episode_ids[idx], video_time)
        video_time = float(video_time)
        color = float(self.colors[idx])

        
        context_id = self.padding(self.contexts_ids[idx], self.model_cfg.max_context_len, direction='right')
        context_position_id = self.padding(self.context_position_ids[idx] , self.model_cfg.max_context_len, direction='right')
        context_segment_id =  self.padding(self.context_segment_ids[idx], self.model_cfg.max_context_len, direction='right')

        #if self.is_training and self.type=="pretrain":
        comment_id = self.input_comments_ids[idx]
        #print(comment_id)
        comment_position_id, comment_segment_id = self.get_position_segment_ids(comment_id)
        
        decoder_input_id = self.padding(comment_id, self.model_cfg.max_comment_len, direction='right')
        decoder_input_mask = self.get_mask(decoder_input_id)
        decoder_position_id = self.padding(comment_position_id, self.model_cfg.max_comment_len, direction='right')
        decoder_segment_id = self.padding(comment_segment_id, self.model_cfg.max_comment_len, direction='right')
        
        masked_lm_labels_tokens = self.padding(self.output_tokens_ids[idx], self.model_cfg.max_comment_len, direction='right', pad_id=0)
        masked_lm_labels_exps = self.padding(self.output_exps_ids[idx], self.model_cfg.max_comment_len, direction='right', pad_id=0)
        
        encoder_input_id = context_id 
        
        encoder_input_mask = self.get_mask(encoder_input_id)
        encoder_input_id = [0] * (self.model_cfg.max_n_clips + 2) + encoder_input_id
        encoder_input_mask = [1] * (self.model_cfg.max_n_clips + 2) + encoder_input_mask
        encoder_position_ids =  [0] * (self.model_cfg.max_n_clips + 2) +  context_position_id 
        encoder_segment_ids = [0] * (self.model_cfg.max_n_clips + 2) + context_segment_id 
        



           
        return torch.LongTensor(encoder_input_id), torch.LongTensor(encoder_input_mask), \
               torch.LongTensor(encoder_position_ids), torch.LongTensor(encoder_segment_ids), \
               torch.LongTensor(decoder_input_id), torch.LongTensor(decoder_input_mask), \
               torch.LongTensor(decoder_position_id), torch.LongTensor(decoder_segment_id), \
               torch.LongTensor(masked_lm_labels_tokens), torch.LongTensor(masked_lm_labels_exps), \
               visual, \
               torch.LongTensor([color]*self.model_cfg.dim).float(), \
               torch.LongTensor([video_time]*self.model_cfg.dim).float(),\
               torch.LongTensor([idx])

    def __len__(self):
        return len(self.contexts_ids)-1  
'''
class MyBiliDataset_CLVCG_Generate(MyBiliDataset_CLVCG):
    def __init__(self, model_cfg, train_cfg, contexts_ids, context_position_ids, context_segment_ids, input_comments_ids, ground_truth_comment_ids, video_ids, episode_ids, video_times, colors, imgs, vocab_size, is_training= True, type="generate" ):
        super(MyBiliDataset_CLVCG_Generate, self).__init__( model_cfg, train_cfg, contexts_ids, context_position_ids, context_segment_ids, input_comments_ids, ground_truth_comment_ids, video_ids, episode_ids, video_times, colors, imgs, vocab_size, is_training, type)
        self.ground_truth_comment_ids = ground_truth_comment_ids
        self.load_imgs=self.load_imgs_Bili
        
    def __getitem__(self, idx):
        video_time = self.video_times[idx]
        visual = self.load_imgs(self.video_ids[idx], self.episode_ids[idx], video_time)
        video_time = float(video_time)
        color = float(self.colors[idx])
        
        context_id = self.padding(self.contexts_ids[idx], self.model_cfg.max_context_len, direction='left')
        context_position_id = self.padding(self.context_position_ids[idx] , self.model_cfg.max_context_len, direction='left')
        context_segment_id =  self.padding(self.context_segment_ids[idx], self.model_cfg.max_context_len, direction='left')

        #if self.is_training and self.type=="pretrain":
        masked_comment_id,masked_lm_labels = self.insert_masks(self.input_comments_ids[idx], self.ground_truth_comment_ids[idx])
        #print()
        #print(masked_comment_id)
        #print(masked_lm_labels)
        comment_position_id, comment_segment_id = self.get_position_segment_ids(masked_comment_id)
        
        masked_comment_id = self.padding(masked_comment_id, self.model_cfg.max_comment_len, direction='right', ending_id=2)
        comment_position_id = self.padding(comment_position_id, self.model_cfg.max_comment_len, direction='right')
        comment_segment_id = self.padding(comment_segment_id, self.model_cfg.max_comment_len, direction='right')
        
        masked_lm_labels = self.padding(masked_lm_labels, self.model_cfg.max_comment_len, direction='right', pad_id=-1)
        
        input_id = context_id + masked_comment_id
        
        input_mask = self.get_mask(input_id)
        input_id = [0] * (self.model_cfg.max_n_clips + 2) + input_id
        input_mask = [1] * (self.model_cfg.max_n_clips + 2) + input_mask
        position_ids = [0] * (self.model_cfg.max_n_clips + 2) +context_position_id + comment_position_id
        segment_ids = [0] * (self.model_cfg.max_n_clips + 2) +context_segment_id + comment_segment_id
        
        masked_lm_labels = [-1] * (self.model_cfg.max_n_clips + 2 + len(context_id)) + masked_lm_labels
        

        pairs, pair_targets = self.get_pairs(masked_lm_labels)
        
        return torch.LongTensor(input_id), torch.LongTensor(input_mask), \
               torch.LongTensor(position_ids), torch.LongTensor(segment_ids), \
               torch.LongTensor(masked_lm_labels), visual, \
               torch.LongTensor([color]*self.model_cfg.dim).float(), \
               torch.LongTensor([video_time]*self.model_cfg.dim).float(),\
               torch.LongTensor(pairs), torch.LongTensor(pair_targets),\
               torch.LongTensor([idx])
    
    def __len__(self):
        return len(self.contexts_ids[:500])    
'''
    
class MyDataLoader(DataLoader):

    def __init__(self, train_cfg, dataset, is_training):
        if is_training:
            sampler=SequentialSampler(dataset)
            batch_size = train_cfg.batch_size
        else:
            sampler=SequentialSampler(dataset)
            batch_size = train_cfg.predict_batch_size
        super(MyDataLoader, self).__init__(dataset, sampler=sampler, batch_size=batch_size)


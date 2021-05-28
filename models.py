#encoding=utf-8
# modified from https://github.com/huggingface/transformers/blob/master/src/transformers/modeling_bart.py
import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import json
import random


from typing import NamedTuple, Optional, Iterable, Tuple

from transformers import BertModel, BertPreTrainedModel
from transformers.modeling_bert import BertLMPredictionHead, ACT2FN
from transformers.configuration_bert import BertConfig
from transformers.modeling_outputs import  BaseModelOutputWithPooling

BertLayerNorm = nn.LayerNorm

class Config(NamedTuple):
    "Configuration for BERT model"
    vocab_size: int = 40443 # Size of Vocabulary
    dim: int = 1024 # Dimension of Hidden Layer in Transformer Encoder
    layers: int = 12 # Numher of Encoder Layers
    n_heads: int = 8 # Numher of Heads in Multi-Headed Attention Layers
    dim_ff: int = 768*4 # Dimension of Intermediate Layers in Positionwise Feedforward Net
    p_drop_hidden: float = 0.3 # Probability of Dropout of various Hidden Layers
    p_drop_attn: float = 0.3 # Probability of Dropout of Attention Layers
    max_n_clips: int = 10        # Maximum video clips for each comment
    max_comment_len: int = 56  # Maximun words for each comment 
    max_comment_len_CMLM: int = 50
    max_pos_len_CMLM: int = 6
    max_context_len: int = 128  # Maximum words for context comments
    max_len : int = 196
    pair_loss_weight : float = 1.0
    next_sentence_loss_weight: float = 5
    pos_loss_weight: float = 1
    @classmethod
    def load_from_json(cls, file):
        return cls(**json.load(open(file, "r")))


   
class MyBertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config):
        super(MyBertEmbeddings, self).__init__()
        
        
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))

    def forward(self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]
        
        
        #print("input_ids",input_ids[0,:])
        #print("token_type_ids",token_type_ids[0,:])
        #print("position_ids",position_ids[0,:])

        
        input_ids = input_ids[:,self.visual.size()[1]+2:]
        
        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]

        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        
        
        self.video_time = self.video_time.unsqueeze( dim=1)
        self.color = self.color.unsqueeze( dim=1)
         
        inputs_embeds = torch.cat([self.visual,self.video_time,self.color,inputs_embeds], dim=1) 
        

        #visual_zeros = torch.zeros([self.visual.size()[0],self.visual.size()[1]+2], dtype=input_ids.dtype).to(torch.device("cuda"))
        
        #position_embeddings = self.position_embeddings(torch.cat([visual_zeros,position_ids], dim=1))
        #token_type_embeddings = self.token_type_embeddings(torch.cat([visual_zeros,token_type_ids], dim=1))

        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)


        embeddings = inputs_embeds + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        
        #print("embeddings",embeddings[0,:])
        
        return embeddings


class MLPWithLayerNorm(nn.Module):
    def __init__(self, config, input_size):
        super(MLPWithLayerNorm, self).__init__()
        self.config = config
        self.linear1 = nn.Linear(input_size, config.hidden_size)
        self.non_lin1 = ACT2FN[config.hidden_act] if isinstance(config.hidden_act, str) else config.hidden_act
        self.layer_norm1 = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.linear2 = nn.Linear(config.hidden_size, config.hidden_size)
        self.non_lin2 = ACT2FN[config.hidden_act] if isinstance(config.hidden_act, str) else config.hidden_act
        self.layer_norm2 = BertLayerNorm(config.hidden_size, eps=1e-12)

    def forward(self, hidden):
        return self.layer_norm2(self.non_lin2(self.linear2(self.layer_norm1(self.non_lin1(self.linear1(hidden))))))


class BertPairTargetPredictionHead(nn.Module):
    def __init__(self, config):
        super(BertPairTargetPredictionHead, self).__init__()
        self.mlp_layer_norm = MLPWithLayerNorm(config, config.hidden_size * 3)
        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

    def forward(self, hidden_states, pairs):
        bs, num_pairs, _ = pairs.size()
        bs, seq_len, dim = hidden_states.size()
        # pair indices: (bs, num_pairs)
        left, here, right = pairs[:,:, 0], pairs[:, :, 1], pairs[:, :, 2]
        # (bs, num_pairs, dim)
        
        left_hidden = torch.gather(hidden_states, 1, left.unsqueeze(2).repeat(1, 1, dim))
        # pair states: bs * num_pairs, max_targets, dim
        #left_hidden = left_hidden.contiguous().view(bs * num_pairs, dim).unsqueeze(1)#.repeat(1, self.max_targets, 1)
        here_hidden = torch.gather(hidden_states, 1, here.unsqueeze(2).repeat(1, 1, dim))
        # bs * num_pairs, max_targets, dim
        #here_hidden = here_hidden.contiguous().view(bs * num_pairs, dim).unsqueeze(1)#.repeat(1, self.max_targets, 1)
        right_hidden = torch.gather(hidden_states, 1, right.unsqueeze(2).repeat(1, 1, dim))
        # bs * num_pairs, max_targets, dim
        #right_hidden = right_hidden.contiguous().view(bs * num_pairs, dim).unsqueeze(1)#.repeat(1, self.max_targets, 1)

        #print(right_hidden)
        
        
        # (max_targets, dim)
        hidden_states = self.mlp_layer_norm(torch.cat((left_hidden, right_hidden, here_hidden), -1))
        # target scores : bs * num_pairs, max_targets, vocab_size
        target_scores = self.decoder(hidden_states) + self.bias
        
        return target_scores



class BertPreTrainingHeads(nn.Module):
    def __init__(self, config):
        super(BertPreTrainingHeads, self).__init__()
        self.predictions = BertLMPredictionHead(config)
        self.pair_target_predictions = BertPairTargetPredictionHead(config)

    def forward(self, sequence_output, pairs):
        prediction_scores = self.predictions(sequence_output)
        pair_target_scores = self.pair_target_predictions(sequence_output, pairs)
        return prediction_scores, pair_target_scores

class BertPreTrainingHeads_WithoutPair(BertPreTrainingHeads):
    def __init__(self, config):
        super(BertPreTrainingHeads_WithoutPair, self).__init__(config)
        self.pair_target_predictions = None
        
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, sequence_output, pooled_output):
        prediction_scores = self.predictions(sequence_output)
        seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score


class BertPreTrainingHeads_WithPos(BertPreTrainingHeads):
    def __init__(self, config,model_cfg):
        super(BertPreTrainingHeads_WithPos, self).__init__(config)
        self.pair_target_predictions = None
        
        self.pos_pred = nn.Linear(config.hidden_size, model_cfg.max_comment_len_CMLM)

    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        seq_pos_score = self.pos_pred(sequence_output)
        return prediction_scores, seq_pos_score

class MyBertModel(BertModel):

    def __init__(self, config,fix_mask=False,model_cfg=None):
        super(MyBertModel, self).__init__(config)
        self.embeddings = MyBertEmbeddings(config)
        self.fix_mask = fix_mask
        self.model_cfg = model_cfg
    
    '''
    def get_extended_attention_mask(self, attention_mask, input_shape: Tuple[int], device=None):
        begin_pos = self.model_cfg.max_n_clips + 2 + self.model_cfg.max_context_len
        if self.fix_mask:
            batch_size, seq_length = input_shape
            #seq_ids = torch.arange(seq_length, device=device)
            #causal_mask = seq_ids[None, None, :].repeat(batch_size, seq_length, 1) <= seq_ids[None, :, None]
            
            causal_mask = torch.ones((batch_size,seq_length,seq_length), device=attention_mask.device)
            causal_mask[:,begin_pos:begin_pos+self.model_cfg.max_pos_len_CMLM,begin_pos+self.model_cfg.max_pos_len_CMLM:] = 0
            causal_mask = causal_mask.to(attention_mask.dtype)
            extended_attention_mask = causal_mask[:, None, :, :] * attention_mask[:, None, None, :]
        else:
            extended_attention_mask = attention_mask[:, None, None, :]

        
        #print("extended_attention_mask",extended_attention_mask[0,0,begin_pos-1:begin_pos+self.model_cfg.max_pos_len_CMLM+1,begin_pos+self.model_cfg.max_pos_len_CMLM-1:])
        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=self.dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    '''

class MyCLVCG(BertPreTrainedModel):
    def __init__(self, model_cfg, type="pretrain",  pad_token_id=0):
        config = BertConfig(
                    vocab_size = model_cfg.vocab_size,
                    hidden_size = model_cfg.dim,
                    num_hidden_layers = model_cfg.layers,
                    num_attention_heads = model_cfg.n_heads,
                    intermediate_size = model_cfg.dim_ff,
                    hidden_dropout_prob = model_cfg.p_drop_hidden,
                    attention_probs_dropout_prob = model_cfg.p_drop_attn,
                    max_position_embeddings = model_cfg.max_len,
                    pad_token_id=pad_token_id,
                    type_vocab_size = 100
                )
        super(MyCLVCG, self).__init__(config)
        self.config = config
        self.type = type
        self.model_cfg = model_cfg 

        self.bert = MyBertModel(config)
        self.cls = BertPreTrainingHeads(config)
        self.pad_token_id = pad_token_id
        self.init_weights()
        self.tie_weights()
        
        self.vocab_weight = None



        self.apply(self.inplace_gelu)

    def inplace_gelu(self,m):
        classname = m.__class__.__name__
        if classname.find('GeLU') != -1:
            m.inplace=True

    def tie_weights(self):
        """ Make sure we are sharing the input and output embeddings.
            Export to TorchScript can't handle parameter sharing so we are cloning them instead.
        """
        self._tie_or_clone_weights(self.cls.predictions.decoder,
                                   self.bert.embeddings.word_embeddings)

    def forward(self,input_ids, attention_mask,  position_ids, segment_ids, masked_lm_labels, visual, color, video_time, pairs, pair_targets, head_mask=None, is_training=True):
        self.bert.embeddings.visual = visual
        self.bert.embeddings.color = color
        self.bert.embeddings.video_time = video_time
        
        
        outputs = self.bert(input_ids, position_ids=position_ids, token_type_ids=segment_ids,
                            attention_mask=attention_mask, head_mask=head_mask)
        
        
        sequence_output = outputs[0]
        prediction_scores, pair_target_scores = self.cls(sequence_output, pairs)



        if self.vocab_weight is None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-1,reduction='none')
        else:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-1,reduction='none', weight=self.vocab_weight)


        if masked_lm_labels is not None:
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))
        

        
        ntokens = torch.sum(torch.ne(masked_lm_labels,-1))
        masked_lm_loss = torch.sum(masked_lm_loss)/ntokens
        # SBO loss

        pair_loss = loss_fct(
            pair_target_scores.view(-1, self.config.vocab_size),
            pair_targets.view(-1)
        )
        pair_loss = torch.sum(pair_loss)/ntokens


        loss = masked_lm_loss + self.model_cfg.pair_loss_weight * pair_loss 
        
        
        return loss, prediction_scores, pair_target_scores #,  outputs[2:]
        

class MyCLVCG_POINTER(MyCLVCG):
    def __init__(self, model_cfg, type="pretrain",  pad_token_id=0):
        config = BertConfig(
                    vocab_size = model_cfg.vocab_size,
                    hidden_size = model_cfg.dim,
                    num_hidden_layers = model_cfg.layers,
                    num_attention_heads = model_cfg.n_heads,
                    intermediate_size = model_cfg.dim_ff,
                    hidden_dropout_prob = model_cfg.p_drop_hidden,
                    attention_probs_dropout_prob = model_cfg.p_drop_attn,
                    max_position_embeddings = model_cfg.max_len,
                    pad_token_id=pad_token_id,
                    type_vocab_size = 100
                )
        super(MyCLVCG_POINTER, self).__init__(model_cfg)
        self.cls = BertPreTrainingHeads_WithoutPair(config)

        
        self.tie_weights()

    def forward(self,input_ids, attention_mask,  position_ids, segment_ids, masked_lm_labels, visual, color, video_time, next_sentence_label = None, head_mask=None, is_training=True):
        self.bert.embeddings.visual = visual
        self.bert.embeddings.color = color
        self.bert.embeddings.video_time = video_time
        
        #print(input_ids[:,self.model_cfg.max_context_len])
        outputs = self.bert(input_ids, position_ids=position_ids, token_type_ids=segment_ids,
                            attention_mask=attention_mask, head_mask=head_mask)
        
        
        sequence_output = outputs[0]
        cls_output = sequence_output[:,self.model_cfg.max_context_len]
        prediction_scores, seq_relationship_score = self.cls(sequence_output, cls_output)


        loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
        masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))
        total_loss = masked_lm_loss
        
        next_sentence_loss = torch.LongTensor(0).to(total_loss.device)
        if next_sentence_label is not None:
            #print(seq_relationship_score.view(-1, 2), next_sentence_label.view(-1))
            next_sentence_loss = loss_fct(seq_relationship_score.view(-1, 2), next_sentence_label.view(-1))
            total_loss += self.model_cfg.next_sentence_loss_weight * next_sentence_loss
            #print(next_sentence_loss)
        

        return total_loss, self.model_cfg.next_sentence_loss_weight * next_sentence_loss, prediction_scores, seq_relationship_score
    
    
class MyCLVCG_CMLM(MyCLVCG):
    def __init__(self, model_cfg, type="pretrain",  pad_token_id=0):
        config = BertConfig(
                    vocab_size = model_cfg.vocab_size,
                    hidden_size = model_cfg.dim,
                    num_hidden_layers = model_cfg.layers,
                    num_attention_heads = model_cfg.n_heads,
                    intermediate_size = model_cfg.dim_ff,
                    hidden_dropout_prob = model_cfg.p_drop_hidden,
                    attention_probs_dropout_prob = model_cfg.p_drop_attn,
                    max_position_embeddings = model_cfg.max_len,
                    pad_token_id=pad_token_id,
                    type_vocab_size = 100,
                )
        super(MyCLVCG_CMLM, self).__init__(model_cfg)

        self.bert = MyBertModel(config,fix_mask=True,model_cfg=model_cfg)
        self.cls = BertPreTrainingHeads_WithPos(config,model_cfg)
        self.tie_weights()

    def forward(self,input_ids, attention_mask,  position_ids, segment_ids, masked_lm_labels, visual=None, color=None, video_time=None, pos_labels = None, head_mask=None, is_training=True):
        
        '''
        begin_pos = self.model_cfg.max_n_clips + 2 + self.model_cfg.max_context_len + self.model_cfg.max_pos_len_CMLM
        print("input_ids0",input_ids[0,begin_pos:])
        print("input_ids1",input_ids[1,begin_pos:])
        print("input_ids0",input_ids[0,self.model_cfg.max_n_clips + 2:])
        print("input_ids1",input_ids[1,self.model_cfg.max_n_clips + 2:])
        print("attention_mask",attention_mask[1,:])
        print("position_ids",position_ids[1,:])
        print("segment_ids",segment_ids[1,:])
        print("visual",visual[1,:])
        print("color",color[1,:])
        print("video_time",video_time[1,:])
        print("head_mask",head_mask)
        
        print("masked_lm_labels",masked_lm_labels[1,self.model_cfg.max_n_clips + 2 + self.model_cfg.max_context_len:])
        print("pos_labels",pos_labels[1,self.model_cfg.max_n_clips + 2 + self.model_cfg.max_context_len:])
        '''
        
        
        self.bert.embeddings.visual = visual
        self.bert.embeddings.color = color
        self.bert.embeddings.video_time = video_time
        
        #print(input_ids[:,self.model_cfg.max_context_len])
        outputs = self.bert(input_ids, position_ids=position_ids, token_type_ids=segment_ids,
                            attention_mask=attention_mask, head_mask=head_mask)
        
        
        
        sequence_output = outputs[0]
        
        prediction_scores, seq_pos_score = self.cls(sequence_output)


        loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
        
        #print(prediction_scores.view(-1, self.config.vocab_size))
        #print(masked_lm_labels.view(-1))
        masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))
        total_loss = masked_lm_loss



        pos_loss = loss_fct(seq_pos_score.view(-1, self.model_cfg.max_comment_len_CMLM), pos_labels.view(-1))
        total_loss += self.model_cfg.pos_loss_weight * pos_loss

        begin_pos = self.model_cfg.max_n_clips + 2 + self.model_cfg.max_context_len
        

        
        '''
        print("sequence_output",sequence_output[0,begin_pos:begin_pos+self.model_cfg.max_pos_len_CMLM])
        pos = seq_pos_score[:,begin_pos:begin_pos+self.model_cfg.max_pos_len_CMLM]
        print("pos_pred",pos.argmax(dim=2)[0])
        print("pos_labels0",pos_labels[0,begin_pos:])
        print("pos_labels1",pos_labels[1,begin_pos:])
        print("masked_lm_labels0",masked_lm_labels[0,begin_pos:])
        print("masked_lm_labels1",masked_lm_labels[1,begin_pos:])
        print("\n\n")
        os._exit(0)
        '''
        return total_loss, self.model_cfg.pos_loss_weight * pos_loss, prediction_scores, seq_pos_score[:,begin_pos:begin_pos+self.model_cfg.max_pos_len_CMLM]
    


import torch
import numpy as np
import os
import time
import sys

from torch.nn import functional as F
from transformers.generation_utils import BeamHypotheses, calc_banned_ngram_tokens, top_k_top_p_filtering

def train(cfg, save_dir, model, train_data, dev_data, optimizer, scheduler, device=None, parallel=False, model_cfg = None, model_type='CLVCG', type = 'pretrain'):
    if torch.cuda.is_available() and device is None:
        device = torch.device("cuda")
    model = model.to(device) 
    model.train()
    global_step = 0
    train_losses = []
    best_loss = sys.maxsize
    begin_time = time.time()
    total_loss = 0
    
    
    #model.vocab_weight = torch.FloatTensor(train_data.dataset.get_output_weight()).to(torch.device("cuda"))

    for epoch in range(int(cfg.n_epochs)):
        loss_print = 0
        cls_loss_print = 0
        for batch in train_data.dataloader:
            global_step += 1
            #try:
            if torch.cuda.is_available():
                batch = [b.to(device) for b in batch]

            if model_type == 'POINTER' and type=='pretrain':
                loss,cls_loss,_,_ = model(input_ids=batch[0], attention_mask=batch[1],
                               position_ids=batch[2], segment_ids=batch[3],
                             masked_lm_labels=batch[4],
                             visual=batch[5], color=batch[6],
                             video_time=batch[7], next_sentence_label=batch[8],
                             is_training=True)
            elif model_type == 'POINTER':
                loss,_,_,_ = model(input_ids=batch[0], attention_mask=batch[1],
                               position_ids=batch[2], segment_ids=batch[3],
                             masked_lm_labels=batch[4],
                             visual=batch[5], color=batch[6],
                             video_time=batch[7], is_training=True)
                cls_loss = loss-loss
            elif model_type == 'CMLM':
                loss,cls_loss,_,_ = model(input_ids=batch[0], attention_mask=batch[1],
                               position_ids=batch[2], segment_ids=batch[3],
                             masked_lm_labels=batch[4],
                             visual=batch[5], color=batch[6],
                             video_time=batch[7], pos_labels=batch[8],
                             is_training=True)
            elif model_type == 'IELM':
                loss,cls_loss,_,_ = model(encoder_input_id=batch[0], encoder_input_mask=batch[1],
                               encoder_position_ids=batch[2], encoder_segment_ids=batch[3],
                             decoder_input_id=batch[4],
                             decoder_input_mask=batch[5], decoder_position_id=batch[6],
                             decoder_segment_id=batch[7], masked_lm_labels_tokens=batch[8],
                             masked_lm_labels_exps=batch[9],visual = batch[10],
                             color=batch[11],video_time=batch[12], is_training=True)
            else:
                loss,_,_ = model(input_ids=batch[0], attention_mask=batch[1],
                               position_ids=batch[2], segment_ids=batch[3],
                             masked_lm_labels=batch[4],
                             visual=batch[5], color=batch[6],
                             video_time=batch[7], pairs=batch[8], 
                             pair_targets=batch[9],is_training=True)
                cls_loss = loss-loss
                
            if parallel:
                loss = loss.mean()  
                cls_loss = cls_loss.mean()
            loss_print += loss.item()
            cls_loss_print += cls_loss.item()
            
            if torch.isnan(loss).data:
                break
            train_losses.append(loss.detach().cpu())
            loss.backward()
                
            if global_step % cfg.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                optimizer.step()    # We have accumulated enought gradients
                scheduler.step()
                model.zero_grad()
                
            if global_step % cfg.print_steps == 0:
                print( "global_step:%d \t time:%d \t loss:%5.3f \t cls_loss:%5.3f"%(global_step,time.time()-begin_time,loss_print,cls_loss_print))
                loss_print = 0
                cls_loss_print = 0

            
            if global_step % cfg.eval_steps == 0:
                model.eval()
                total_loss = 0
                with(torch.no_grad()):
                    if model_type == 'IELM':
                        total_loss, total_exp_loss,  token_predictions, exp_predictions, ground_truth_token, ground_truth_exp, input_ids = inference_IELM(cfg, model, dev_data, device, parallel, type)
                        print_results_IELM(save_dir, global_step, total_loss, total_exp_loss,  token_predictions, exp_predictions, ground_truth_token, ground_truth_exp, input_ids)   
                    else:
                        total_loss, total_cls_loss, total_ns_acc, predictions, ns_predictions, pos_predictions, input_ids, masked_lm_labels = inference(cfg, model, dev_data, device, parallel, model_cfg, model_type, type)
                        print_results(save_dir, global_step, total_loss, total_cls_loss, total_ns_acc, predictions, ns_predictions, pos_predictions, input_ids, masked_lm_labels)
                model.train()

            if global_step % cfg.save_steps == 0:    
                save_model(model,parallel,os.path.join(save_dir, 'model_steps_%06d.pt'%(global_step)))
                
                if best_loss > total_loss:
                    save_model(model,parallel, os.path.join(save_dir, "best-model.pt"))
                    best_loss = total_loss
            #except:
            #    print("Got an Error! Continue..")
            #    continue

    save_model(model,parallel,os.path.join(save_dir, 'model_steps_%06d.pt'%(global_step)))

def save_model(model,parallel,file):
    if parallel:
        model_state_dict = {k[7:]:v.cpu() for (k, v) in model.state_dict().items()}
    else:
        model_state_dict = {k:v.cpu() for (k, v) in model.state_dict().items()}
    torch.save(model_state_dict, file)

def inference(cfg, model, dev_data, device=None, parallel=False, model_cfg = None, model_type = 'CLVCG', type='pretrain'):
    print("evaluating...")
    if torch.cuda.is_available() and device is None:
        device = torch.device("cuda")
        
    predictions = []
    ns_predictions = []
    pos_predictions = []
    input_ids = []
    ground_truth_ids = []
    total_loss = 0
    total_cls_loss = 0
    total_ns_acc = 0
    
    for i, batch in enumerate(dev_data.dataloader):
        if i % 100 == 0:
            print("\t eval:%d"%(i))
        if torch.cuda.is_available():
            batch = [b.to(device) for b in batch]

        ns_logits = None
        pair_logits = None
        pos_logits = None
        
        if model_type == 'POINTER'  and type=='pretrain':
            loss,cls_loss,lm_logits,ns_logits  = model(input_ids=batch[0], attention_mask=batch[1],
                               position_ids=batch[2], segment_ids=batch[3],
                             masked_lm_labels=batch[4],
                             visual=batch[5], color=batch[6],
                             video_time=batch[7], next_sentence_label=batch[8],
                             is_training=True)
        elif model_type == 'POINTER':
            loss,_,lm_logits,_ = model(input_ids=batch[0], attention_mask=batch[1],
                           position_ids=batch[2], segment_ids=batch[3],
                         masked_lm_labels=batch[4],
                         visual=batch[5], color=batch[6],
                         video_time=batch[7], is_training=True)
            cls_loss = loss-loss
        elif model_type == 'CMLM':
            loss,cls_loss,lm_logits,pos_logits = model(input_ids=batch[0], attention_mask=batch[1],
                           position_ids=batch[2], segment_ids=batch[3],
                         masked_lm_labels=batch[4],
                         visual=batch[5], color=batch[6],
                         video_time=batch[7], pos_labels=batch[8],
                         is_training=True)
        else:
            loss,lm_logits,pair_logits  = model(input_ids=batch[0], attention_mask=batch[1],
                               position_ids=batch[2], segment_ids=batch[3],
                             masked_lm_labels=batch[4],
                             visual=batch[5], color=batch[6],
                             video_time=batch[7], pairs=batch[8], 
                             pair_targets=batch[9], is_training=True)
            cls_loss = loss-loss
        if parallel:
            loss = loss.mean() 
            

        pred_lm = lm_logits.argmax(dim=2)
        
        pred_ns =  [None]*cfg.batch_size
        pred_pos = [None]*cfg.batch_size
        pred_pair = [None]*cfg.batch_size
        if ns_logits is not None:
            pred_ns = ns_logits.argmax(dim=1)
        if pos_logits is not None:
            pred_pos = pos_logits.argmax(dim=2)  
        if pair_logits is not None:
            pred_pair = pair_logits.argmax(dim=2)

            
        #pred_type = clf_logits.argmax(dim=1)
        
        for input_, pl, pn, pps, pp, lm_label, label in zip(batch[0], pred_lm, pred_ns, pred_pos, pred_pair, batch[4], batch[8]): 
            decoded_pl = dev_data.tokenizer.decode(pl.cpu().numpy().tolist()[10:])
            decoded_pp = None
            pps_l = None
            if pp is not None:
                decoded_pp = dev_data.tokenizer.decode(pp.cpu().numpy().tolist())
            if pps is not None:
                pps_l = pps.cpu().numpy().tolist()
            if pn == label:
                total_ns_acc += 1
            
            predictions.append((decoded_pl,decoded_pp))
            ns_predictions.append((pn,label))
            if pps_l is not None:
                begin_pos = model_cfg.max_n_clips + 2 + model_cfg.max_context_len
                label = label[begin_pos:begin_pos+cfg.non_mask_tokens +2]
                pos_predictions.append((pps_l[:cfg.non_mask_tokens +2],label.cpu()))
            else:
                pos_predictions.append((None,label.cpu()))
            
            decoded_i = dev_data.tokenizer.decode(input_.cpu().numpy().tolist())
            input_ids.append(decoded_i)
            
            decoded_gt = dev_data.tokenizer.decode(lm_label.cpu().numpy().tolist()[10:])
            ground_truth_ids.append(decoded_gt)


                
                
        total_loss += loss.sum().item()
        total_cls_loss += cls_loss.sum().item()
    total_ns_acc = total_ns_acc/len(dev_data)
    print("\t total_loss:%f total_cls_loss:%f"%(total_loss,total_cls_loss))
    print("\t next_acc:%f\n"%(total_ns_acc))
    return total_loss, total_cls_loss, total_ns_acc, predictions, ns_predictions, pos_predictions, input_ids, ground_truth_ids #, predictions_type, all_logits


def inference_IELM(cfg, model, dev_data, device=None, parallel=False, type='pretrain'):
    print("evaluating...")
    if torch.cuda.is_available() and device is None:
        device = torch.device("cuda")
        
    token_predictions = []
    exp_predictions = []
    input_ids = []
    ground_truth_token = []
    ground_truth_exp = []
    total_loss = 0
    total_exp_loss = 0
    
    for i, batch in enumerate(dev_data.dataloader):
        if i % 100 == 0:
            print("\t eval:%d"%(i))
        if torch.cuda.is_available():
            batch = [b.to(device) for b in batch]

            loss,exp_loss,token_logits,exp_logits = model(encoder_input_id=batch[0], encoder_input_mask=batch[1],
                           encoder_position_ids=batch[2], encoder_segment_ids=batch[3],
                         decoder_input_id=batch[4],
                         decoder_input_mask=batch[5], decoder_position_id=batch[6],
                         decoder_segment_id=batch[7], masked_lm_labels_tokens=batch[8],
                         masked_lm_labels_exps=batch[9],visual = batch[10],
                         color=batch[11],video_time=batch[12], is_training=True)
            pred_token = token_logits.argmax(dim=2)
            pred_exp = exp_logits.argmax(dim=2)  
            if parallel:
                loss = loss.mean() 
                exp_loss = exp_loss.mean()
                
            for input_, pt, pe, token_label, exp_label in zip(batch[0], pred_token, pred_exp, batch[8], batch[9]):
                decoded_pt = dev_data.tokenizer.decode(pt.cpu().numpy().tolist())
                decoded_pe = dev_data.tokenizer.decode(pe.cpu().numpy().tolist())
                token_predictions.append(decoded_pt)
                exp_predictions.append(decoded_pe)
                decoded_gt_token = dev_data.tokenizer.decode(token_label.cpu().numpy().tolist())
                decoded_exp = dev_data.tokenizer.decode(exp_label.cpu().numpy().tolist())
                ground_truth_token.append(decoded_gt_token)
                ground_truth_exp.append(decoded_exp)
                decoded_i = dev_data.tokenizer.decode(input_.cpu().numpy().tolist())
                input_ids.append(decoded_i)
        total_loss += loss.sum().item()
        total_exp_loss += exp_loss.sum().item()
    print("\t total_loss:%f total_cls_loss:%f"%(total_loss,total_exp_loss))
    return total_loss, total_exp_loss,  token_predictions, exp_predictions, ground_truth_token, ground_truth_exp, input_ids #, predictions_type, all_logits


 
def logit_pred(logits, thread = 0.5):
    softmax_logits = torch.softmax(logits,dim=1)
    pred = softmax_logits[:,1].gt(thread).long()
    return pred
    
            

def print_results(save_dir, global_step, total_loss, total_cls_loss, total_ns_acc=None, outputs=None, ns_predictions=None, pos_predictions=None, inputs=None, lm_labels=None):
    res_f = open(os.path.join(save_dir, 'res_%06d.txt')%(global_step),"w", encoding='utf8')
    res_f.write("\tloss:"+str(total_loss) + '\n')
    res_f.write("\tcls_loss:"+str(total_cls_loss) + '\n')
    res_f.write("\ttotal_ns_acc:"+str(total_ns_acc) + '\n')

    for o_ids, ns, pos, in_id,lm in zip(outputs, ns_predictions, pos_predictions, inputs, lm_labels):
        
        preds_lm = []
        gts = []

        ns_pred,ns_label = ns
        pos_pred,pos_label = pos
        
        for i in range(len(lm)):
            if lm[i] != '<PAD>':
                preds_lm.append(o_ids[0][i])
                gts.append(lm[i])

        if "<&&&>" in in_id:
            in_id = in_id[in_id.index("<&&&>"):]#gt.index("<EOS>")]
        
        preds_pair = ''
        if o_ids[1] is not None:
            preds_pair = ' '.join(o_ids[1][:len(preds_lm)])
        if ns_pred is not None:
            res_f.write("pred:%d gt:%d ||"%(ns_pred,ns_label))
        if pos_pred is not None:
            res_f.write("pred:%s gt:%s ||"%(' '.join([str(i) for i in pos_pred]),' '.join([str(i.item()) for i in pos_label])))
        res_f.write("%s\t||\t%s\t||\t%s\t||\t%s\n"%(' '.join(preds_lm), preds_pair, ' '.join(gts), ' '.join(in_id)))

    res_f.close()


def print_results_IELM(save_dir, global_step, total_loss, total_exp_loss,  token_predictions, exp_predictions, ground_truth_token, ground_truth_exp, input_ids):
    res_f = open(os.path.join(save_dir, 'res_%06d.txt')%(global_step),"w", encoding='utf8')
    res_f.write("\tloss:"+str(total_loss) + '\n')
    res_f.write("\texp_loss:"+str(total_exp_loss) + '\n')


    for tp,ep,gtt,gte,in_id in zip(token_predictions, exp_predictions, ground_truth_token, ground_truth_exp, input_ids):
        
        gtt = [s for s in gtt if s !='<PAD>']
        s_tp = ' '.join(tp[:len(gtt)])
        s_gtt = ' '.join(gtt)
        res_f.write("tp:%s gt:%s ||\t"%(s_tp,s_gtt))
        
        gte = [s for s in gte if s !='<PAD>']
        s_ep = ' '.join(ep[:len(gte)])
        s_gte = ' '.join(gte)
        res_f.write("ep:%s gt:%s ||\t"%(s_ep,s_gte))
        if "<&&&>" in in_id:
            in_id = in_id[in_id.index("<&&&>"):]
        res_f.write("%s\n"%(' '.join([s for s in in_id if s !='<PAD>'])))

    res_f.close()



def test_generation(model, generate_cfg, model_cfg, test_data, device=None, type='greedy', exp_maps=None, bos_token_id = 1, model_type = 'POINTER', eos_token_id = 2):
    if torch.cuda.is_available() and device is None:
        device = torch.device("cuda")
    predictions = []
    contexts = []
    ground_truths = []
    keywords = [] 
    
    begin_time = time.time()
    for i, batch in enumerate(test_data.dataloader):
        if i % generate_cfg.print_steps == 0:
            print("steps:%d time:%f"%(i,time.time()-begin_time))


        if torch.cuda.is_available():
            batch = [b.to(device) for b in batch]
        
        if type =='greedy' and model_type == 'POINTER':
            outputs = greedy_generate_POINTER(model, generate_cfg, model_cfg, batch)
        elif  type =='greedy' and model_type == 'CMLM':
            outputs = greedy_generate_CMLM(model, generate_cfg, model_cfg, batch)
        elif type =='greedy' and model_type == 'IELM':
            outputs = greedy_generate_IELM(model, generate_cfg, model_cfg, batch, exp_maps)
        #print("!!!!!")

        for ct, gen, idx  in zip(batch[0], outputs, batch[-1]):
            
            if model_type=='POINTER' or  model_type=='IELM':
                gt = [bos_token_id] + test_data.dataset.ground_truth_comment_ids[idx.item()] + [eos_token_id]
            else:
                gt = test_data.dataset.ground_truth_comment_ids[idx.item()] + [eos_token_id]
            #print(gt)

            decoded_ct = test_data.tokenizer.decode(ct.cpu().numpy().tolist()[model_cfg.max_n_clips+2:model_cfg.max_n_clips+2+model_cfg.max_context_len])
            #print(ct.cpu().numpy().tolist()[10:model_cfg.max_context_len])
            contexts.append(decoded_ct)
            decoded_gen = test_data.tokenizer.decode(gen)
            predictions.append(decoded_gen)
            decoded_gt = test_data.tokenizer.decode(gt)
            ground_truths.append(decoded_gt)
            
            
            begin_pos = model_cfg.max_n_clips+2+model_cfg.max_context_len
            if model_type=='POINTER' :
                #print(test_data.tokenizer.decode(ct.cpu().numpy().tolist()))
                #print(test_data.tokenizer.decode(ct.cpu().numpy().tolist()[begin_pos:]))
                #print()
                decode_keyword_withmask = test_data.tokenizer.decode(ct.cpu().numpy().tolist()[begin_pos:])
            elif model_type=='IELM':
                ips =  test_data.dataset.input_comments_ids[idx.item()] 
                #print(ips)
                decode_keyword_withmask = test_data.tokenizer.decode(ips)
            else:
                decode_keyword_withmask = test_data.tokenizer.decode(ct.cpu().numpy().tolist()[begin_pos:begin_pos+model_cfg.max_pos_len_CMLM])

            decode_keyword = []
            for w in decode_keyword_withmask:
                if w != "<MASK>" and w !="<PAD>":
                    decode_keyword.append(w)
            keywords.append(decode_keyword)

    #print(predictions)
    #print(contexts)
    #print(ground_truths)
    return predictions, ground_truths, contexts, keywords


def assign_single_value_byte(x, i, y):
    x.view(-1)[i.view(-1).nonzero()] = y

def assign_multi_value_long(x, i, y):
    b, l = x.size()
    i = i + torch.arange(0, b*l, l, device=i.device).unsqueeze(1)
    x.view(-1)[i.view(-1)] = y.view(-1)[i.view(-1)]

def assign_single_value_long(x, i, y):
    b, l = x.size()
    i = i + torch.arange(0, b*l, l, device=i.device).unsqueeze(1)
    x.view(-1)[i.view(-1)] = y

def select_worst(token_probs, num_mask):
    bsz, seq_len = token_probs.size()
    masks = [token_probs[batch, :].topk(max(1, num_mask[batch]), largest=False, sorted=False)[1] for batch in range(bsz)]
    masks = [torch.cat([mask, mask.new(seq_len - mask.size(0)).fill_(mask[0])], dim=0) for mask in masks]
    return torch.stack(masks, dim=0)


def greedy_generate_IELM(model, generate_cfg, model_cfg, batch, exp_maps, not_generate_ids = [0,1,2,3,4,5,6,7], pad_id=0):

    head_token_id = 41328

    decoder_input_id = batch[4]
    decoder_input_mask=batch[5]
    decoder_position_id=batch[6]
    decoder_segment_id=batch[7]
    
    with torch.no_grad():
        loss,exp_loss,token_logits,exp_logits = model(encoder_input_id=batch[0], encoder_input_mask=batch[1],
                       encoder_position_ids=batch[2], encoder_segment_ids=batch[3],
                     decoder_input_id=decoder_input_id,
                     decoder_input_mask=decoder_input_mask, decoder_position_id=decoder_position_id,
                     decoder_segment_id=decoder_segment_id, masked_lm_labels_tokens=batch[8],
                     masked_lm_labels_exps=batch[9],visual = batch[10],
                     color=batch[11],video_time=batch[12], is_training=True)
        
        encoder_output = model.encoder_output
        
        for turn in range(generate_cfg.max_turn):
            #print("turn:%d"%(turn))
            #print("decoder_input_id",decoder_input_id)
            token_logits[:,:,3] = -float("Inf")
            
            if generate_cfg.token_do_sample:
                print()
            else:
                pred_token = token_logits.argmax(dim=2)
            
            if generate_cfg.exp_do_sample:
                print()
            else:
                pred_exp = exp_logits.argmax(dim=2)
                
            #print("pred_token",pred_token)
            #print("pred_exp",pred_exp)
                
            decoder_input_id_new = torch.zeros_like(decoder_input_id, dtype=decoder_input_id.dtype, device=decoder_input_id.device)
            decoder_input_mask_new = torch.zeros_like(decoder_input_mask, dtype=decoder_input_mask.dtype, device=decoder_input_mask.device)
            decoder_position_id_new = torch.zeros_like(decoder_position_id, dtype=decoder_position_id.dtype, device=decoder_position_id.device)
            decoder_segment_id_new = torch.zeros_like(decoder_segment_id, dtype=decoder_segment_id.dtype, device=decoder_segment_id.device)
            
            exp_num = 0
            for b in range(generate_cfg.predict_batch_size):
                i = 0
                k = 0
                
                while max(i,k)<model_cfg.max_comment_len-1:
                    #print("input:",i,decoder_input_id)
                    #print("out_put:",k,decoder_input_id_new)
                    if decoder_input_id[b,i] == pad_id:
                        break
                    if decoder_input_id[b,i] != pad_id and decoder_input_id[b,i] < model_cfg.exp_min_id:
                        decoder_input_id_new[b,k] =  decoder_input_id[b,i]
                        i+=1
                        k+=1 
                        if k >= model_cfg.max_comment_len: break
                    elif decoder_input_id[b,i] >= model_cfg.exp_min_id:
                        exps = exp_maps[pred_exp[b,i].item()]
                        #print("exps:",exps)
                        for exp in exps:
                            if exp == head_token_id:
                                decoder_input_id_new[b,k] = pred_token[b,i]
                            else:
                                decoder_input_id_new[b,k] = exp
                                exp_num += 1
                            k+=1
                            if k >= model_cfg.max_comment_len: break
                        i+=1
                
                        
                        #decoder_input_id_new[b,k] =  pred_token[b,i]
                        #i+=1
                        #k+=1

                    
                    
                
                decoder_input_mask_new[b,:k] = torch.ones([k], dtype=decoder_input_mask_new.dtype, device=decoder_input_mask_new.device)
                decoder_position_id_new[b,:k] = torch.arange(0, k ,dtype=decoder_position_id_new.dtype, device=decoder_position_id_new.device)
                decoder_segment_id_new[b,:k] = torch.ones([k], dtype=decoder_segment_id_new.dtype, device=decoder_segment_id_new.device)
                
            decoder_input_id = decoder_input_id_new
            decoder_input_mask= decoder_input_mask_new
            decoder_position_id= decoder_position_id_new
            decoder_segment_id= decoder_segment_id_new

        
            if exp_num == 0:
                break     
            
            
            loss,exp_loss,token_logits,exp_logits = model(encoder_input_id=batch[0], encoder_input_mask=batch[1],
                   encoder_position_ids=batch[2], encoder_segment_ids=batch[3],
                 decoder_input_id=decoder_input_id,
                 decoder_input_mask=decoder_input_mask, decoder_position_id=decoder_position_id,
                 decoder_segment_id=decoder_segment_id, masked_lm_labels_tokens=batch[8],
                 masked_lm_labels_exps=batch[9],visual = batch[10],
                 color=batch[11],video_time=batch[12],encoder_output = encoder_output, is_training=True)
            
        #print('===========\n')
    out_ids = []    
    for b in range(generate_cfg.predict_batch_size):
        out_list = decoder_input_id[b].cpu().numpy().tolist()
        out_ids.append([0]+[t for t in out_list if t != 0]+[1])
    return out_ids  



def greedy_generate_CMLM(model, generate_cfg, model_cfg,  batch, noi_token_id = 7, mask_token_id = 5, not_generate_ids = [0,1,2,3,4,5,6,7], not_copy_ids = [0,4,5,6,7]):  
    input_ids = batch[0]
    attention_mask = batch[1]
    position_ids = batch[2]
    segment_ids = batch[3]
    masked_lm_labels = batch[4]
    pos_labels = batch[8]
    gt_comment_id = batch[9]
    pos_id_input = batch[10]
    
    

    with torch.no_grad():

        begin_pos = model_cfg.max_n_clips + 2 + model_cfg.max_context_len + model_cfg.max_pos_len_CMLM
        
        #print("input_ids",input_ids[:,model_cfg.max_n_clips + 2 + model_cfg.max_context_len:])
        #print("pos_labels",pos_labels[:,model_cfg.max_n_clips + 2 + model_cfg.max_context_len:])
        loss,cls_loss,lm_logits,pos_logits = model(input_ids=input_ids, attention_mask=attention_mask,
                       position_ids=position_ids, segment_ids=segment_ids,
                     masked_lm_labels=masked_lm_labels,
                     visual=batch[5], color=batch[6],
                     video_time=batch[7], pos_labels=pos_labels,
                     is_training=True)
        
        pos_logits[:,0,1:] = -float("Inf")
        
        for i in range(1,generate_cfg.non_mask_tokens+2):
            pos_logits[:,i,0:i] =  -float("Inf")
        pos_logits = F.softmax(pos_logits, dim=-1)    

        if generate_cfg.temperature != 1.0:
            pos_logits = pos_logits / generate_cfg.temperature     
        
        pred_pos_all = torch.zeros(pos_logits.size()[:-1],device=pos_logits.device,dtype=int)
        
        if generate_cfg.pos_do_sample:
            for i in range(1,generate_cfg.non_mask_tokens+2):
            # Top-p/top-k filtering
                _scores = pos_logits[0,i]
                #_scores[:pred_pos_all[0,i-1].item()+1] = -float("Inf")
                _scores = top_k_top_p_filtering(
                    _scores, top_k=generate_cfg.pos_top_k, min_tokens_to_keep=2
                )
                #print(pred_pos_all[0,i-1].item())
                
                _scores = _scores.contiguous().view(
                    1, pos_logits.size()[2]
                )  # (batch_size, num_beams * vocab_size)
    
                # Sample 2 next tokens for each beam (so we have some spare tokens and match output of greedy beam search)
                #print("probs",_scores)
                probs = F.softmax(_scores, dim=-1)
                #print("probs",probs)
                pred_pos = torch.multinomial(probs, 1)  # (batch_size, num_beams * 2)
                # Compute next scores
                pred_pos_scores = torch.gather(_scores, -1, pred_pos)  # (batch_size, num_beams * 2)
                # sort the sampled vector to make sure that the first num_beams samples are the best
                next_scores, next_pos_scores_indices = torch.sort(pred_pos_scores, descending=True, dim=1)
                pred_pos = torch.gather(pred_pos, -1, next_pos_scores_indices)  # (batch_size, num_beams * 2)
                pred_pos_all[0,i] = pred_pos[:,0]
        else:
            pred_pos_all = pos_logits.argmax(dim=2)
        

        #print("pred_pos:",pred_pos_all)
        
        for b in range(generate_cfg.predict_batch_size):
            for i in range(generate_cfg.non_mask_tokens+1):
                if pred_pos_all[b,i+1] <= pred_pos_all[b,i]:
                    pred_pos_all[b,i+1] = min(model_cfg.max_comment_len_CMLM-1 ,pred_pos_all[b,i]+1)
            
            pred_len = min(max(pred_pos_all[b]),model_cfg.max_comment_len_CMLM-generate_cfg.non_mask_tokens+i)

            input_ids[b,begin_pos:begin_pos+pred_len] = torch.LongTensor([mask_token_id] * pred_len)
            attention_mask[b,begin_pos:begin_pos+pred_len] = torch.LongTensor([1] * pred_len)
            position_ids[b,begin_pos:begin_pos+pred_len] = torch.arange(1, pred_len+1)
            segment_ids[b,begin_pos:begin_pos+pred_len] = torch.LongTensor([1] * pred_len)
            #print("pos_id_input",pos_id_input[b])
            for i in range(generate_cfg.non_mask_tokens+2): 
                if pos_id_input[b,i].item() != -1:
                    #print(pred_pos_all[b,i].item(),pos_id_input[b,i].item())
                    input_ids[b,begin_pos+pred_pos_all[b,i].item()] = pos_id_input[b,i]
                
        
        #print("input_ids",begin_pos,input_ids[:,begin_pos:])
        #print("gt_comment_id",gt_comment_id)
        pad_mask = input_ids.ne(mask_token_id)
        
        for ip in range(1,generate_cfg.max_turn):

            loss,cls_loss,lm_logits,pos_logits = model(input_ids=input_ids, attention_mask=attention_mask,
                           position_ids=position_ids, segment_ids=segment_ids,
                         masked_lm_labels=masked_lm_labels,
                         visual=batch[5], color=batch[6],
                         video_time=batch[7], pos_labels=pos_labels,
                         is_training=True)
            
            
            
            lm_logits[:,:,not_generate_ids] = -float("Inf")
            lm_logits[:,:,23] = -float("Inf")
            _scores = F.softmax(lm_logits, dim=-1)
            if generate_cfg.token_do_sample:
                _scores = _scores.contiguous().view(
                    lm_logits.size()[1], lm_logits.size()[2]
                )  # (batch_size, num_beams * vocab_size)
                _scores = top_k_top_p_filtering(
                    _scores, top_k=generate_cfg.token_top_k, min_tokens_to_keep=2
                )

                # Sample 2 next tokens for each beam (so we have some spare tokens and match output of greedy beam search)
                #print("probs",_scores)
                _scores = F.softmax(_scores, dim=-1)
                #print("probs",probs)
                gen_tokens = torch.multinomial(_scores, 1)  # (batch_size, num_beams * 2)
                # Compute next scores
                token_probs = torch.gather(_scores, -1, gen_tokens)  # (batch_size, num_beams * 2)
                gen_tokens = gen_tokens.view(lm_logits.size()[0],lm_logits.size()[1])
                token_probs = token_probs.view(lm_logits.size()[0],lm_logits.size()[1])
            else:
                token_probs, gen_tokens = _scores.max(dim=-1)
            
            tmp_ones = torch.ones([pad_mask.size()[0],1],device=pad_mask.device)
            generate_mask = pad_mask + torch.cat([tmp_ones,pad_mask[:,:-1]],dim=-1) +torch.cat([pad_mask[:,1:],tmp_ones],dim=-1)
            token_probs = token_probs * generate_mask
            
            assign_single_value_byte(gen_tokens, pad_mask, 0)
            assign_single_value_byte(token_probs, pad_mask, 1.0)
            num_mask = input_ids.eq(mask_token_id).sum(dim=1)
            num_mask_next = (num_mask-1).long()#(num_mask.float() * (1.0 - (ip / generate_cfg.max_turn))).long()

            #print("num_mask_next",num_mask_next)
            assign_single_value_byte(token_probs, pad_mask, 1.0)
            #print("token_probs",begin_pos,token_probs[:,begin_pos:])
            #print("gen_tokens",begin_pos,gen_tokens[:,begin_pos:])
            mask_ind = select_worst(token_probs, num_mask_next)
            #print("mask_ind",mask_ind)
            
            target_ids = (input_ids * pad_mask + gen_tokens)[:,begin_pos:]
            
            assign_single_value_long(gen_tokens, mask_ind, mask_token_id)
            assign_single_value_byte(gen_tokens, pad_mask, 0)

            #print("gen_tokens",begin_pos,gen_tokens[:,begin_pos:])
            
            
            
            pad_mask_new = gen_tokens.ne(mask_token_id)
            
            

            
            if pad_mask_new.equal(pad_mask):
                break
            else:
                
                #print("input_ids",input_ids)
                input_ids = input_ids * pad_mask + gen_tokens
                pad_mask = pad_mask_new
                #print("input_ids",input_ids)
                #print("target_ids",target_ids)
                #print("\n")
                
    out_ids = []    
    for b in range(generate_cfg.predict_batch_size):
        out_ids.append(target_ids[b].cpu().numpy().tolist())
    #print("out_ids",out_ids)
    
    #print(ip,"============================\n\n")        


    return out_ids  

def greedy_generate_POINTER(model, generate_cfg, model_cfg,  batch, noi_token_id = 7, mask_token_id = 5, not_generate_ids = [0,1,2,3,4,5,6,7], not_copy_ids = [0,4,5,6,7]):  
    input_ids = batch[0]
    attention_mask=batch[1]
    position_ids=batch[2]
    segment_ids=batch[3]
    masked_lm_labels=batch[4]

    begin_pos = model_cfg.max_n_clips + 2 + model_cfg.max_context_len 
    
    #print("============")
    #print(input_ids[:,begin_pos:])
    for ip in range(generate_cfg.max_turn):
        with torch.no_grad():  
            #print('-')
            #print(input_ids[:,begin_pos:])
            #print(masked_lm_labels[:,110:])
            
            _,_,lm_logits,_  = model(input_ids=input_ids, attention_mask=attention_mask,
                           position_ids=position_ids, segment_ids=segment_ids,
                         masked_lm_labels=masked_lm_labels,
                         visual=batch[5], color=batch[6],
                         video_time=batch[7], is_training=True)
            

            #print(lm_logits[:,begin_pos:,].argmax(dim=2))

            
            

            logits = lm_logits
 

            #noi_temp = min((float(ip)+1) / generate_cfg.noi_decay, 1.0) 
            #logits[:,:,noi_token_id] = logits[:,:,noi_token_id] * noi_temp 
            logits[:,:, 1] = -float("Inf")
            logits[:,:, 2] = -float("Inf")
            logits[:,:, 3] = -float("Inf")
            logits[:,:, 0] = -float("Inf")
            logits[:,:,23] =  -float("Inf")
            
            probs = F.softmax(logits, dim=-1)
            if generate_cfg.temperature != 1.0:
                probs = probs / generate_cfg.temperature    

            noi_temp = min((float(ip)+1) / generate_cfg.noi_decay, 1.0) 
            probs[:,:,noi_token_id] = probs[:,:,noi_token_id] * noi_temp * 7

            input_ids_new = torch.zeros_like(input_ids)
            position_ids_new = torch.zeros_like(position_ids) 
            segment_ids_new = torch.zeros_like(segment_ids) 
            logit_new = torch.zeros_like(input_ids,dtype=torch.float)
            top_predicts = torch.zeros([input_ids.shape[0], input_ids.shape[1], 3], dtype=torch.long)
            
            
            input_ids_new[:,:begin_pos] = input_ids[:,:begin_pos] 
            position_ids_new[:,:begin_pos] = position_ids[:,:begin_pos] 
            segment_ids_new[:,:begin_pos] = segment_ids[:,:begin_pos] 

            mask_predicts = get_pred_tokens(probs,generate_cfg,model_cfg,begin_pos) #probs.argmax(2)
            #print(mask_predicts[:,begin_pos:])
            for t in range(model_cfg.max_comment_len-1):
                top_predicts[:,begin_pos+t] = torch.topk(probs[:,begin_pos+t,:], k=3)[1]
            top_predicts_new = torch.zeros_like(top_predicts)

            

            
            for b in range(generate_cfg.predict_batch_size):
                i = 0
                j = 0
                k = 0
                #print("input_ids",input_ids[b,begin_pos:])
                #print("mask_predicts",mask_predicts[b,begin_pos:])
                #print()
                while np.max([i,j,k]) < model_cfg.max_comment_len-1:
                    #print(i,j,k,"input_ids_new",input_ids_new[:,begin_pos:])
                    input_ids_new[b,begin_pos+k] = input_ids[b,begin_pos+i]
                    if input_ids[b,begin_pos+i] == 0: # padding, ignore prediction
                        break
                    i += 1
                    k += 1
                

                    if mask_predicts[b,begin_pos+j].cpu().numpy() != noi_token_id:
                        input_ids_new[b,begin_pos+k] = mask_predicts[0,begin_pos+j]
                        logit_new[0,begin_pos+k] = probs[b,begin_pos+j,mask_predicts[b,begin_pos+j]]
                        top_predicts_new[b,begin_pos+k,:] = top_predicts[b,begin_pos+j,:]    
                        k+=1
                        j+=1
                    else:
                        j+=1
                    

                mask_pos = input_ids_new > 1
                input_ids = input_ids_new
                attention_mask = mask_pos
                
                segment_ids_new[b,begin_pos:begin_pos+k] =  torch.LongTensor([1] * k)
                position_ids_new[b,begin_pos:begin_pos+k] =  torch.arange(1,k+1)
                
                logit_new = logit_new.detach().cpu().numpy()
                top_predicts_new = top_predicts_new.detach().cpu().numpy()
                
    out_ids = [b[begin_pos:].cpu().numpy().tolist() for b in input_ids]
    return out_ids         
                    
def get_pred_tokens(logits,generate_cfg,model_cfg,begin_pos):  
    if  not generate_cfg.do_sample:
        pred_all = logits.argmax(dim=2)
    else:
        pred_all = torch.zeros(logits.size()[:-1],device=logits.device,dtype=int)
        for i in range(begin_pos,begin_pos+model_cfg.max_comment_len-1):
        # Top-p/top-k filtering
            _scores = logits[0,i]
            #_scores[:pred_pos_all[0,i-1].item()+1] = -float("Inf")
            _scores = top_k_top_p_filtering(
                _scores, top_k=20, min_tokens_to_keep=2
            )
            #print(pred_pos_all[0,i-1].item())
            
            _scores = _scores.contiguous().view(
                1, logits.size()[2]
            )  # (batch_size, num_beams * vocab_size)

            # Sample 2 next tokens for each beam (so we have some spare tokens and match output of greedy beam search)
            #print("probs",_scores)
            probs = F.softmax(_scores, dim=-1)
            #print("probs",probs)
            pred = torch.multinomial(probs, 1)  # (batch_size, num_beams * 2)
            pred_all[0,i] = pred[:,0]


    return pred_all


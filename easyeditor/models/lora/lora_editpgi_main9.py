from copy import deepcopy
import gc
import random
from typing import Any, Dict, List, Tuple
from peft import get_peft_model, AdaLoraConfig, TaskType, LoraConfig
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor
from torch.nn import CrossEntropyLoss
from .lora_hparams import LoRAHyperParams
from .lora_multimodal_hparams import LoRAMultimodalHyperParams

from .data_engine9_template import KnowledgeGraphWrapper, EditPGIDataEngine

def apply_lora_to_model(
        model: AutoModelForCausalLM,
        tok: AutoTokenizer,
        requests: List[Dict],
        hparams: LoRAHyperParams,
        copy=False,
        return_orig_weights=False,
        keep_original_weight=False,
        **kwargs: Any,
) -> Tuple[AutoModelForCausalLM, Dict[str, Any]]:
    
    weights_copy = {}
    if copy: model = deepcopy(model)
    edited_model = execute_lora(model, tok, requests, hparams, keep_original_weight, **kwargs)
    return edited_model, weights_copy


def execute_lora(
        model: AutoModelForCausalLM,
        tok: AutoTokenizer,
        requests: List[Dict],
        hparams: LoRAHyperParams,
        keep_original_weight=False,
        **kwargs: Any,
) -> Dict[str, Tuple[torch.Tensor]]:
    
    model.config.use_cache = False
    model.supports_gradient_checkpointing = True  
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    
    if hparams.lora_type == "lora":
        Config = LoraConfig 
        use_dora = getattr(hparams, 'use_dora', True)      
        # use_dora = getattr(hparams, 'use_dora', False)      
        use_rslora = getattr(hparams, 'use_rslora', False) 
    else:
        Config = AdaLoraConfig
        
    if not keep_original_weight and hasattr(model,'peft_config'):
        peft_model = model
    else:
        peft_config = Config(
            task_type=TaskType.CAUSAL_LM, inference_mode=False,
            r=hparams.rank, lora_alpha=hparams.lora_alpha, lora_dropout=hparams.lora_dropout,
            layers_to_transform=hparams.layers if len(hparams.layers) > 0 else None,
            target_modules=hparams.target_modules,
            use_dora=use_dora,
            use_rslora=use_rslora,
        )
        peft_model = get_peft_model(model, peft_config)

    peft_model.is_parallelizable = True
    peft_model.model_parallel = True
    
    if isinstance(requests[0],list):
        requests = requests[0]
    requests = deepcopy(requests)
    tmp_requests =[]
    kg_datas = kwargs.get("graph_data", None)
    
    if len(requests) == 1 and isinstance(requests[0]['prompt'],list):
        for i in range(len(requests[0]['prompt'])):
            tmp_requests.append({
                'prompt': requests[0]['prompt'][i],
                'subject': requests[0]['subject'][i],
                'target_new': requests[0]['target_new'][i],
                'triples': None,
            })
        requests = tmp_requests
    elif isinstance(requests[0]['prompt'],list):
        for rnum,req in enumerate(requests):
            if isinstance(req['prompt'],list):
                tmp_req1 =[]
                for i in range(len(req['prompt'])):
                    tmp_req1.append({
                        'prompt': req['prompt'][i],
                        'subject': req['subject'][i],
                        'target_new': req['target_new'][i],
                        'triples': None,
                    })
                tmp_requests.append(tmp_req1)
        requests = tmp_requests
    
    expanded_requests =[]
    

    if kg_datas is not None and isinstance(kg_datas,dict):
        if 'triples' in kg_datas:
            kg_wrapper = KnowledgeGraphWrapper([kg_datas['triples']])
        else:
            kg_wrapper = KnowledgeGraphWrapper([])
        try:
            engine = EditPGIDataEngine(kg_wrapper)
            expanded_requests =[]
            if kg_datas is not None and 'triples' in kg_datas:
                for req in requests:
                    req['triples'] = kg_datas['triples']
                    req['summary'] = kg_datas['summary']
            expanded_requests.extend(engine.generate_EditPGI_requests(requests))    
            requests = expanded_requests
        except Exception as e:
            print(f"⚠️[Engine Error]: {e}")

    elif kg_datas is not None and isinstance(kg_datas,list):        
        for i,(kg_data,reqs) in enumerate(zip(kg_datas,requests)):
            if kg_data is not None and 'triples' in kg_data:
                kg_wrapper = KnowledgeGraphWrapper([kg_data['triples']])
            else:
                kg_wrapper = KnowledgeGraphWrapper([])
            try:
                engine = EditPGIDataEngine(kg_wrapper)
                expanded_request =[]
                if not isinstance(reqs, list):
                    reqs = [reqs]
                if kg_data is not None and 'triples' in kg_data:
                    for req in reqs:
                        req['triples'] = kg_data['triples']
                        req['summary'] = kg_data['summary']
                expanded_requests.extend(engine.generate_EditPGI_requests(reqs))
            except Exception as e:
                print(f"⚠️[Engine Error]: {e}")
        requests = expanded_requests
 

    for request in requests:
        if '{}' in request['prompt']:
            request['prompt'] = request['prompt'].format(request['subject'])
        if request["target_new"] != " " and not request["target_new"].startswith(" "):
            request["target_new"] = " " + request["target_new"]


    stage1_tasks =["Context_Memorization","Subgraph_Summarization" ] # 
    stage2_tasks =[ "Standard_Edit","Reasoning_QA"] #,"Bridge_Anchoring"] 
    stage1_reqs =[r for r in requests if r.get("task_type") in stage1_tasks]
    
    stage2_reqs =[r for r in requests if r.get("task_type") in stage2_tasks]

    training_stages =[]
    if stage1_reqs: training_stages.append(("Stage 1: Context Memorization & Summarization", stage1_reqs))
    if stage2_reqs: training_stages.append(("Stage 2: Context QA & CoT Reasoning", stage2_reqs))
    if not training_stages: training_stages.append(("Standard LoRA Edit", requests))

    device = torch.device(f'cuda:{hparams.device}' if hasattr(hparams, 'device') else 'cuda')
    lr_stage1 = getattr(hparams, 'lr_stage1', hparams.lr * 1)
    lr_stage2 = getattr(hparams, 'lr_stage2', hparams.lr * 0.6) 
    lr_standard = hparams.lr
    step_stage1 = hparams.num_steps
    step_stage2 = hparams.num_steps
    # 依次执行两阶段串行训练
    for stage_name, stage_requests in training_stages:
        if "Stage 1" in stage_name:
            current_lr = lr_stage1
            current_num_steps = step_stage1
        elif "Stage 2" in stage_name:
            current_lr = lr_stage2
            current_num_steps = step_stage2
        else:
            current_lr = lr_standard

        opt = torch.optim.Adam(peft_model.parameters(), lr=current_lr, weight_decay=hparams.weight_decay)
        print(f"\n" + "="*80)
        print(f" {stage_name} (Tasks: {len(stage_requests)})")
        print("="*80)
        # -----print requests------
        # for req in stage_requests:
        #     trunc_tgt = req['target_new'][:100] + "..." if len(req['target_new']) > 100 else req['target_new']
        #     print(f"[{req.get('task_type', 'Standard')}] \n   P: {req['prompt']}  \n   T: {trunc_tgt}\n")


        seen_pairs = set()
        unique_requests = []
        for r in stage_requests:
            pair = (r["prompt"], r["target_new"], r["subject"])
            if pair not in seen_pairs:
                seen_pairs.add(pair)
                unique_requests.append(r)
        
        texts = [r["prompt"] for r in unique_requests]
        targets = [r["target_new"] for r in unique_requests]
        task_types = [r.get("task_type", "Standard") for r in unique_requests]
        
        if len(unique_requests) < len(stage_requests):
            print(f"[INFO] Removed {len(stage_requests) - len(unique_requests)} duplicate (prompt, target_new) pairs")
            print(f" {stage_name} (Tasks: {len(unique_requests)})")


        # count_req = len(texts)
        loss_meter = AverageMeter()

        for it in range(current_num_steps):
        # for it in range(current_num_steps):
            print(f"--- Epoch: {it} ---")
            loss_meter.reset()
            counts = 0
            all_loss =[]
                
            for txt, tgt in zip(chunks(texts, hparams.batch_size), chunks(targets, hparams.batch_size)):
                opt.zero_grad()
                
                if 't5' in hparams.model_name.lower():
                    inputs = tok(txt, return_tensors="pt", padding=True).to(device)
                    target_ids = tok(tgt, return_tensors="pt", padding=True)["input_ids"].to(device)
                    inputs['decoder_input_ids'] = target_ids
                    logits = peft_model(**inputs).logits
                    unmasked_log_probs = logits.log_softmax(-1).gather(-1, inputs['decoder_input_ids'].unsqueeze(-1)).squeeze(-1)
                    mask = inputs['decoder_input_ids'] != -100
                    n_tokens = mask.float().sum()
                    loss = -(unmasked_log_probs * mask.float()).sum() / n_tokens
                    
                elif 'chatglm' in hparams.model_name.lower():
                    input_ids = tok(txt, return_tensors="pt", padding=True)["input_ids"].tolist()
                    labels = tok(tgt, return_tensors="pt", padding=True)["input_ids"].tolist()
                    len_batches =[len(input_ids[i]) + len(labels[i]) + 1 for i in range(len(input_ids))]
                    len_max_batch = max(len_batches)
                    batch_input_ids, batch_labels = [],[]
                    for x, y in zip(input_ids, labels):
                        len_padding = len_max_batch - len(x) - len(y)
                        if tok.padding_side and tok.padding_side == "left":
                            batch_label = [-100] * len_padding +[-100] * len(x) + y
                            batch_input_id = [0] * (len_padding) + x + y
                        else:
                            batch_label =[-100] * len(x) + y + [-100] * len_padding
                            batch_input_id = x + y + [0] * (len_padding)
                        batch_input_ids.append(torch.tensor(batch_input_id, dtype=torch.long).to(device))
                        batch_labels.append(torch.tensor(batch_label, dtype=torch.long).to(device))
                    
                    batch_input_ids = torch.stack(batch_input_ids).to(device)
                    batch_labels = torch.stack(batch_labels).to(device)
                    lm_logits = peft_model(input_ids=batch_input_ids)['logits'].to(torch.float32)
                    shift_logits = lm_logits[..., :-1, :].contiguous()
                    shift_labels = batch_labels[..., 1:].contiguous()
                    loss_fct = CrossEntropyLoss(ignore_index=-100)
                    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)).to(lm_logits.dtype)
                    
                else:
                    inputs_targets =[txt_ + tgt_ for txt_, tgt_ in zip(txt, tgt)]
                    inputs_targets = tok(inputs_targets, return_tensors="pt", padding=True).to(device)
                    inputs = tok(txt, return_tensors="pt", padding=True).to(device)
                    
                    num_prompt_toks =[int((i != tok.pad_token_id).sum()) for i in inputs['input_ids'].cpu()]
                    num_pad_toks =[int((i == tok.pad_token_id).sum()) for i in inputs_targets['input_ids'].cpu()]
                    prompt_len =[x + y for x, y in zip(num_pad_toks, num_prompt_toks)]
                    prompt_target_len = inputs_targets['input_ids'].size(1)
                    
                    label_mask = torch.tensor([
                        [False] * length +[True] * (prompt_target_len - length) for length in prompt_len
                    ]).to(device)
                    
                    logits = peft_model(**inputs_targets).logits
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = inputs_targets['input_ids'][..., 1:].contiguous()
                    

                    batch_task_types = task_types[counts:counts + len(txt)]
                    
                    
                    
                    # loss_fct = CrossEntropyLoss(reduction='none')
                    # loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                    # loss = loss.view(inputs_targets['input_ids'].shape[0], -1)
                    # loss = (loss * label_mask[:, 1:]).sum(1) / label_mask[:, 1:].sum(1)


                    # for i, task_type in enumerate(batch_task_types):
                    #     if task_type == "Reasoning_QA":
                    #         loss[i] = loss[i] * 3
                    
                    # loss = loss.mean()

                    ### new kl loss
                    ###
                    per_sample_loss = []

                    subgraph_indices = [i for i, task_type in enumerate(batch_task_types) if task_type == "Subgraph_Summarization"]
                    other_indices = [i for i, task_type in enumerate(batch_task_types) if task_type != "Subgraph_Summarization"]
                    
                    per_sample_loss = torch.zeros(len(txt)).to(device)
                    
                    if subgraph_indices:
                        from torch.nn import KLDivLoss
                        kl_loss = KLDivLoss(reduction='none')
                        
                        subgraph_mask = torch.tensor([i in subgraph_indices for i in range(len(txt))]).to(device)
                        
                        log_probs = shift_logits.log_softmax(dim=-1)
                        target_probs = torch.zeros_like(log_probs).to(device)
                        
                        batch_valid_positions = label_mask[:, 1:].unsqueeze(-1)  # 形状: (batch_size, seq_length, 1)
                        subgraph_indicator = torch.tensor([i in subgraph_indices for i in range(len(txt))]).unsqueeze(-1).unsqueeze(-1).to(device)  # 形状: (batch_size, 1, 1)
                        
                        one_hot_labels = torch.nn.functional.one_hot(shift_labels, num_classes=target_probs.size(-1)).float()  # 形状: (batch_size, seq_length, vocab_size)
                        
                        target_probs = one_hot_labels * batch_valid_positions * subgraph_indicator
                        
                        kl_div = kl_loss(log_probs, target_probs).sum(dim=-1)
                        kl_div = (kl_div * label_mask[:, 1:]).sum(dim=-1) / label_mask[:, 1:].sum(dim=-1)
                        per_sample_loss[subgraph_mask] = kl_div[subgraph_mask]
                    
                    if other_indices:
                        loss_fct = CrossEntropyLoss(reduction='none')
                        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                        loss = loss.view(inputs_targets['input_ids'].shape[0], -1)
                        loss = (loss * label_mask[:, 1:]).sum(1) / label_mask[:, 1:].sum(1)
                        
                        other_mask = torch.tensor([i in other_indices for i in range(len(txt))]).to(device)
                        per_sample_loss[other_mask] = loss[other_mask]
                    

                    for i, task_type in enumerate(batch_task_types):
                        if task_type == "Reasoning_QA":
                            per_sample_loss[i] = per_sample_loss[i] * 3
                        # elif task_type == "Context_Memorization":
                        #     per_sample_loss[i] = per_sample_loss[i] * 0.5
                    
                    loss = per_sample_loss.sum()

                counts += len(txt)
                bs = inputs["input_ids"].shape[0] if 't5' in hparams.model_name.lower() else inputs_targets["input_ids"].shape[0]
                print(f"  Batch loss: {loss.item():.4f}")
                loss_meter.update(loss.item(), n=bs) 
                all_loss.append(loss.item())
                # if stage_name.startswith("Stage 1"):
                #     if loss.item() >= 1e-1:
                #         loss.backward()
                #         opt.step()
                # else:
                if loss.item() >= 1e-1:
                    loss.backward()
                    opt.step()
                # torch.cuda.empty_cache()   
                # gc.collect()               
            print(f"  Stage Total loss: {loss_meter.avg:.4f}")
            if all(loss < 1e-1 for loss in all_loss):
                print(f" {stage_name} converged early at epoch {it}.")
                break
            # if stage_name.startswith("Stage 1"):
            #     # if all(loss < 1e-1 for loss in all_loss):
            #     if loss_meter.avg < 5e-2:
            #         print(f" {stage_name} converged early at epoch {it}.")
            #         break
            # else:
            #     if all(loss < 1e-1 for loss in all_loss):
            #         print(f" {stage_name} converged early at epoch {it}.")
            #         break
    print("Training finished.")
    return peft_model


def apply_lora_to_multimodal_model(
        model: AutoModelForCausalLM,
        tok: AutoProcessor,
        requests: List[Dict],
        hparams: LoRAMultimodalHyperParams,
        copy=False,
        return_orig_weights=False,
        keep_original_weight=False,
        **kwargs: Any,
) -> Tuple[AutoModelForCausalLM, Dict[str, Any]]:
    weights_copy = {}
    device = f'cuda:{hparams.device}' if torch.cuda.is_available() else 'cpu'
    if copy:
        model = deepcopy(model)
        model.to(device)

    edited_model = execute_multimodal_lora(model, tok, requests, hparams, keep_original_weight)
    return edited_model, weights_copy


def execute_multimodal_lora(
        model: AutoModelForCausalLM,
        processor: AutoProcessor,
        requests: List[Dict],
        hparams: LoRAMultimodalHyperParams,
        keep_original_weight=False,
        **kwargs: Any,
) -> AutoModelForCausalLM:
    model.config.use_cache = False
    model.supports_gradient_checkpointing = True
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    
    if hparams.lora_type == "lora":
        Config = LoraConfig
    elif hparams.lora_type == "adalora":
        Config = AdaLoraConfig
    else:
        raise NotImplementedError
        
    if not keep_original_weight and hasattr(model, 'peft_config'):
        peft_model = model
    else:
        peft_config = Config(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=hparams.rank,
            lora_alpha=hparams.lora_alpha, 
            lora_dropout=hparams.lora_dropout,
            target_modules=hparams.target_modules
        )
        peft_model = get_peft_model(model, peft_config)

    peft_model.to(dtype=torch.float32)
    peft_model.is_parallelizable = True
    peft_model.model_parallel = True
    
    from torch.optim.lr_scheduler import ExponentialLR
    opt = torch.optim.SGD(
        peft_model.parameters(),
        lr=hparams.lr,
        weight_decay=hparams.weight_decay,
    )
    scheduler = ExponentialLR(opt, gamma=hparams.sh_lr) if hasattr(hparams, 'sh_lr') else None
    
    if hasattr(peft_model, 'print_trainable_parameters'):
        peft_model.print_trainable_parameters()
    
    requests = deepcopy(requests)
    for request in requests:
        print(f"执行多模态编辑: [{request['prompt']}] -> [{request['target']}]")
    
    device = torch.device(f'cuda:{hparams.device}' if torch.cuda.is_available() else 'cpu')
    prompts = [r["prompt"] for r in requests]
    labels = [r["target"] for r in requests]
    file_type = requests[0].get('file_type', 'image') if requests else 'image'
    input_images = [r.get('image', None) for r in requests] if requests else []
    loss_meter = AverageMeter()
    
    for it in range(hparams.num_steps):
        print(20 * "=")
        print(f"Epoch: {it}")
        loss_meter.reset()

        for txt, tgt in zip(chunks(prompts, hparams.batch_size), chunks(labels, hparams.batch_size)):
            mask_token = -100
            opt.zero_grad()
            
            multimodal_inputs = None
            if hasattr(hparams, 'use_chat_template') and hparams.use_chat_template and processor:
                if file_type == "video" and input_images:
                    temp_prompt = [processor.apply_chat_template([
                        {"role": "user", "content": [{"type": "video"}, {"type": "text", "text": p}]},
                    ], add_generation_prompt=True, tokenize=False) + l
                    for p, l in zip(prompts, labels)]
                    
                elif file_type in ["image", "single-image", "multi-image"]:
                    num_images = len(input_images[0]) if file_type == "multi-image" and input_images else 1
                    temp_prompt = [processor.apply_chat_template([
                        {"role": "user", "content": [{"type": "image"}] * num_images + [{"type": "text", "text": p}]},
                    ], add_generation_prompt=True, tokenize=False) + l
                    for p, l in zip(prompts, labels)]              
                else:
                    temp_prompt = [f"{p} {l}" for p, l in zip(prompts, labels)]
                
                full_prompt = temp_prompt
                if processor and input_images:
                    if file_type in ["image", "single-image", "multi-image"]:
                        multimodal_inputs = processor(images=input_images, text=full_prompt, 
                                                      return_tensors="pt", padding=True).to(device, dtype=torch.float32)
                    elif file_type == "video":
                        multimodal_inputs = processor(videos=input_images[0] if input_images else None, 
                                                      text=full_prompt, return_tensors="pt", 
                                                      padding=True).to(device, dtype=torch.float32)
            
            if multimodal_inputs is None:
                full_prompt = [f"{p} {l}" for p, l in zip(txt, tgt)]
                tokens = processor.tokenizer(full_prompt, return_tensors="pt", padding=True, truncation=True) if processor else None
                if tokens is None:
                    continue
                tokens = tokens.to(device)
                tokens["labels"] = tokens["input_ids"].clone()
            else:
                tokens = multimodal_inputs
                if processor and 'target' in requests[0]:
                    targets = processor.tokenizer(labels[0], add_special_tokens=False,
                                               return_tensors="pt", padding=True, 
                                               max_length=tokens["input_ids"].size(1))["input_ids"] if processor else None
                    if targets is not None:
                        labels_ids = torch.full_like(tokens["input_ids"], -100)
                        labels_ids[:, -targets.size(1):] = targets
                        tokens["labels"] = labels_ids
            
            pred = peft_model(**tokens)
            loss = pred.loss
            print(f"Batch loss: {loss.item()}")
            loss.backward()
            opt.step()
            if scheduler:
                scheduler.step()

        print(f"Total loss: {loss_meter.avg}")
    
    return peft_model


class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def chunks(arr, n):
    """Yield successive n-sized chunks from arr."""
    chunk = []
    for a in arr:
        chunk.append(a)
        if len(chunk) == n:
            yield chunk
            chunk = []
    if len(chunk) > 0:
        yield chunk
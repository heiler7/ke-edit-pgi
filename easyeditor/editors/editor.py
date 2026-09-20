import gc
import sys
from typing import Optional, Union, List, Tuple, Dict
from time import time
from tqdm import tqdm
import json
from os import path
import torch
import numpy as np
import random
import copy
import time
import types
from typing import List, Dict, Optional
from peft import LoraConfig, get_peft_model, PeftModel
from peft.utils import set_peft_model_state_dict
from ..models.melo.melo import LORA
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel, BitsAndBytesConfig
from transformers import LlamaTokenizer,PreTrainedTokenizerFast, LlamaTokenizerFast
from transformers import T5ForConditionalGeneration, T5Tokenizer
from transformers import GPT2TokenizerFast, GPT2Tokenizer
from ..util.globals import *
from .utils import _chunks, _prepare_requests, summary_metrics
from .batch_editor import BatchEditor
from ..evaluate import compute_edit_quality, compute_icl_edit_quality, compute_sent_metric
from ..util import nethook
from ..util.hparams import HyperParams
from ..util.alg_dict import *
from ..evaluate.evaluate_utils import test_generation_quality

from ..modified_models import _LlamaForCausalLM, _GPTJForCausalLM

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)

LOG = logging.getLogger(__name__)
def make_logs():

    f_h, s_h = get_handler('logs', log_name='run.log')
    LOG.addHandler(f_h)
    LOG.addHandler(s_h)

def seed_everything(seed):
    if seed >= 10000:
        raise ValueError("seed number should be less than 10000")
    if torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
    else:
        rank = 0
    seed = (rank * 100000) + seed

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

seed_everything(42)





def eval(datas):
    #data_rome_counterfact['post'].keys()  dict_keys(['rewrite_acc', 'locality', 'portability'])
    Edit_Succ_list=[data_rome_counterfact['post']['rewrite_acc'][0] for data_rome_counterfact in datas]
    Edit_Succ=sum(Edit_Succ_list)/len(Edit_Succ_list)*100
    print('Edit_Succ:',Edit_Succ)
    # rephrase_Succ_list=[data_rome_counterfact['post']['rephrase_acc'][0] for data_rome_counterfact in datas] if 'rephrase_acc' in data_rome_counterfact['post'].keys() else []
    # rephrase_Succ=sum(rephrase_Succ_list)/len(rephrase_Succ_list)*100 if len(rephrase_Succ_list) != 0 else 0
    # print('rephrase_Succ:',rephrase_Succ)
    Portability_list=[]
    Portability_dict={}
    for data_rome_counterfact in datas:
        case_list=[]
        for key in data_rome_counterfact['post']['portability'].keys():
            if key == "single_hop_acc":
                values = data_rome_counterfact['post']['portability'][key]
                if all(v == 1 for v in values):
                    accuracy = 100
                else:
                    accuracy = min(values) * 100
            else:
                accuracy = sum(data_rome_counterfact['post']['portability'][key])/len(data_rome_counterfact['post']['portability'][key])*100
            if key not in Portability_dict.keys():
                Portability_dict[key]=[accuracy]
            else:
                Portability_dict[key].append(accuracy)
            case_list.append(accuracy)
        if len(case_list) != 0:
            Portability_list.append(np.mean(case_list))
    Overall_portability = np.mean(Portability_list)
    print('Overall_portability:',Overall_portability)
    for key in Portability_dict.keys():
        print("Portability_",key,':',np.mean(Portability_dict[key]))
    Locality_list=[]
    Locality_dict={}
    for data_rome_counterfact in datas:
        case_list=[]
        for key in data_rome_counterfact['post']['locality'].keys():
            if key not in Locality_dict.keys():
                Locality_dict[key]=[sum(data_rome_counterfact['post']['locality'][key])/len(data_rome_counterfact['post']['locality'][key])*100]
            else:
                Locality_dict[key].append(sum(data_rome_counterfact['post']['locality'][key])/len(data_rome_counterfact['post']['locality'][key])*100)
            case_list.append(sum(data_rome_counterfact['post']['locality'][key])/len(data_rome_counterfact['post']['locality'][key])*100)
        if len(case_list) != 0:
            Locality_list.append(np.mean(case_list))
    Overall_locality = np.mean(Locality_list)
    print('Overall_locality:',Overall_locality)
    for key in Locality_dict.keys():
        print("Locality_",key,':',np.mean(Locality_dict[key]))

    
    Fluency_list=[x['post']['fluency']['ngram_entropy'] for x in datas]
    Fluency=sum(Fluency_list)/len(Fluency_list)*100
    print('Fluency:',Fluency)


def safe_del_dispatched_model(model, *, verbose=True, max_referrers_print=10):
    if model is None:
        if verbose:
            print("[safe_del_dispatched_model] model is already None")
        return

    try:
        refcount = sys.getrefcount(model)
    except Exception:
        refcount = None

    if verbose:
        print(f"[safe_del_dispatched_model] refcount before cleaning: {refcount}")

    referrers = gc.get_referrers(model)
    if verbose:
        print(f"[safe_del_dispatched_model] total referrers found: {len(referrers)} (show up to {max_referrers_print})")

    cleaned = 0
    for i, ref in enumerate(referrers[:max_referrers_print]):
        if isinstance(ref, dict):
            for k, v in list(ref.items()):
                if v is model:
                    try:
                        ref[k] = None
                        cleaned += 1
                        if verbose:
                            print(f"  cleared dict value for key={repr(k)[:120]}")
                    except Exception:
                        pass
                if k is model:
                    try:
                        del ref[k]
                        cleaned += 1
                        if verbose:
                            print(f"  removed dict key={repr(k)[:120]}")
                    except Exception:
                        pass


        elif isinstance(ref, list):
            for idx, v in enumerate(list(ref)):
                if v is model:
                    try:
                        ref[idx] = None
                        cleaned += 1
                        if verbose:
                            print(f"  cleared list item at index {idx}")
                    except Exception:
                        pass


        elif isinstance(ref, set):
            if model in ref:
                try:
                    ref.discard(model)
                    cleaned += 1
                    if verbose:
                        print("  discarded model from set")
                except Exception:
                    pass


        elif isinstance(ref, types.ModuleType):
            for name, val in list(vars(ref).items()):
                if val is model:
                    try:
                        setattr(ref, name, None)
                        cleaned += 1
                        if verbose:
                            print(f"  cleared module attribute {ref.__name__}.{name}")
                    except Exception:
                        pass

        else:
            if verbose:
                tname = type(ref).__name__
                print(f"  other ref type: {tname} (repr sliced) -> {repr(ref)[:200]}")

    if verbose:
        print(f"[safe_del_dispatched_model] cleaned {cleaned} container references (within first {max_referrers_print} referrers)")


    try:
        del model
    except Exception:
        pass

    gc.collect()
    torch.cuda.empty_cache()

    try:
        allocated_mb = torch.cuda.memory_allocated() / 1024**2
        reserved_mb = torch.cuda.memory_reserved() / 1024**2
    except Exception:
        allocated_mb = reserved_mb = -1

    if verbose:
        print(f"[safe_del_dispatched_model] GPU memory -> allocated: {allocated_mb:.2f} MB, reserved: {reserved_mb:.2f} MB")
        print("[safe_del_dispatched_model] Done. 请在调用处再执行 `your_var = None`（比如 self.model = None），并再次运行 gc.collect() + torch.cuda.empty_cache() 检查是否完全释放。")

    return {"cleaned_ref_entries": cleaned, "gpu_allocated_mb": allocated_mb, "gpu_reserved_mb": reserved_mb}
def safe_delete_model(model, verbose=True, max_referrers=5):
    if model is None:
        if verbose:
            print("[safe_delete_model] model is already None")
        return

    try:
        refcount = sys.getrefcount(model)
        if verbose:
            print(f"[safe_delete_model] refcount before delete: {refcount}")
        
        referrers = gc.get_referrers(model)
        if verbose:
            print(f"[safe_delete_model] found {len(referrers)} referrers")
            for i, obj in enumerate(referrers[:max_referrers]):
                print(f"  -> {i+1}: type={type(obj)}, repr={repr(obj)[:200]}")
            if len(referrers) > max_referrers:
                print(f"  ... ({len(referrers)-max_referrers} more)")

    except Exception as e:
        print(f"[safe_delete_model] error inspecting refcount: {e}")


    try:
        del model
    except Exception as e:
        if verbose:
            print(f"[safe_delete_model] error deleting model: {e}")


    gc.collect()

    torch.cuda.empty_cache()
    if verbose:
        allocated = torch.cuda.memory_allocated() / 1024**2
        reserved = torch.cuda.memory_reserved() / 1024**2
        print(f"[safe_delete_model] GPU memory -> allocated: {allocated:.2f} MB, reserved: {reserved:.2f} MB")

def find_gpu_tensors():
    tensors = []
    for obj in gc.get_objects(): 
        try:
            if torch.is_tensor(obj) and obj.device.type == 'cuda':  # 只检查 GPU 张量
                size_mb = obj.element_size() * obj.numel() / (1024 ** 2)  # 估算大小 (MB)
                ref_count = sys.getrefcount(obj)  # 引用计数（>1 表示被其他变量引用）
                tensors.append({
                    'tensor': obj,
                    'shape': obj.shape,
                    'size_mb': size_mb,
                    'ref_count': ref_count,
                    'repr': repr(obj)[:100] 
                })
        except Exception:
            pass 

    tensors.sort(key=lambda x: x['size_mb'], reverse=True)
    for t in tensors:
        print(f"张量: {t['repr']}, 形状: {t['shape']}, 大小: {t['size_mb']:.2f} MB, 引用计数: {t['ref_count']}")

    if not tensors:
        print("未找到 GPU 张量。占用可能来自其他进程或 PyTorch 缓存。")
        
class BaseEditor:
    """Base editor for all methods"""

    @classmethod
    def from_hparams(cls, hparams: HyperParams):
        return cls(hparams)

    def __init__(self, hparams: HyperParams):
        assert hparams is not None, 'Error: hparams is None.'
        self.model_name = hparams.model_name
        self.apply_algo = ALG_DICT[hparams.alg_name]
        self.alg_name = hparams.alg_name
        make_logs()
        LOG.info("Instantiating model")
        print(self.model_name)
        if type(self.model_name) is str:
            device_map = 'auto' if hparams.model_parallel else None
            torch_dtype = torch.float16 if hasattr(hparams, 'fp16') and hparams.fp16 else torch.float32

            # QLoRA configuration
            if hparams.alg_name == 'QLoRA':
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=(hparams.quantization_bit == 4),
                    bnb_4bit_use_double_quant=hparams.double_quant,
                    bnb_4bit_quant_type=hparams.quant_type,
                    bnb_4bit_compute_dtype=torch.bfloat16
                )
                model_kwargs = {
                    "quantization_config": bnb_config,
                    "torch_dtype": torch_dtype,
                    "device_map": {'': hparams.device}
                }
            else:
                model_kwargs = {
                    "torch_dtype": torch_dtype,
                    "device_map": device_map
                }

            if 't5' in os.path.basename(self.model_name).lower():
                self.model = T5ForConditionalGeneration.from_pretrained(self.model_name, **model_kwargs)
                self.tok = T5Tokenizer.from_pretrained(self.model_name)
            elif 'chatglm-api' in os.path.basename(self.model_name).lower():
                self.model, self.tok = None, None
                self.hparams = hparams
                return
            elif 'gpt-3.5' in os.path.basename(os.path.basename(self.model_name)).lower():
                self.model, self.tok = None, None
            elif 'gpt' in os.path.basename(self.model_name).lower():
                self.model = AutoModelForCausalLM.from_pretrained(self.model_name, **model_kwargs)
                self.tok = GPT2Tokenizer.from_pretrained(self.model_name)
                self.tok.pad_token_id = self.tok.eos_token_id
            elif 'llama' in os.path.basename(self.model_name).lower():
                self.model = AutoModelForCausalLM.from_pretrained(self.model_name, **model_kwargs)
                self.tok = AutoTokenizer.from_pretrained(self.model_name)
                self.tok.pad_token_id = self.tok.eos_token_id
            elif 'vicuna' in os.path.basename(self.model_name).lower():
                self.model = AutoModelForCausalLM.from_pretrained(self.model_name, **model_kwargs)
                self.tok = AutoTokenizer.from_pretrained(self.model_name)
                self.tok.pad_token_id = self.tok.eos_token_id
                self.tok.padding_side = 'left'
            elif 'baichuan' in os.path.basename(self.model_name).lower():
                self.model = AutoModelForCausalLM.from_pretrained(self.model_name, **model_kwargs, trust_remote_code=True)
                self.tok = AutoTokenizer.from_pretrained(self.model_name,trust_remote_code=True)
                self.tok.pad_token_id = self.tok.eos_token_id
            elif 'chatglm' in os.path.basename(self.model_name).lower():
                self.model = AutoModel.from_pretrained(self.model_name,trust_remote_code=True, **model_kwargs)
                self.tok = AutoTokenizer.from_pretrained(self.model_name,trust_remote_code=True)
                if 'chatglm2'in os.path.basename(self.model_name).lower():
                    self.tok.unk_token_id = 64787
                else:
                    self.tok.pad_token_id = self.tok.eos_token_id
            elif 'internlm' in os.path.basename(self.model_name).lower():
                self.model = AutoModel.from_pretrained(self.model_name,trust_remote_code=True, **model_kwargs)
                self.tok = AutoTokenizer.from_pretrained(self.model_name,trust_remote_code=True)
                self.tok.pad_token_id = self.tok.eos_token_id
            elif 'qwen2' in os.path.basename(self.model_name).lower():
                self.model = AutoModelForCausalLM.from_pretrained(self.model_name,trust_remote_code=True, torch_dtype=torch_dtype if hparams.alg_name not in ['MEND'] else torch.bfloat16, device_map=device_map)
                self.tok = AutoTokenizer.from_pretrained(self.model_name, eos_token='<|endoftext|>', pad_token='<|endoftext|>',unk_token='<|endoftext|>', trust_remote_code=True)
                self.tok.pad_token_id = self.tok.eos_token_id
            elif 'qwen' in os.path.basename(self.model_name).lower():
                # self.model = AutoModelForCausalLM.from_pretrained(self.model_name,fp32=False,trust_remote_code=True, **model_kwargs)
                self.model = AutoModelForCausalLM.from_pretrained(self.model_name,trust_remote_code=True, **model_kwargs)
                self.tok = AutoTokenizer.from_pretrained(self.model_name, eos_token='<|endoftext|>', pad_token='<|endoftext|>',unk_token='<|endoftext|>', trust_remote_code=True)
                self.tok.pad_token_id = self.tok.eos_token_id
            elif 'mistral' in os.path.basename(self.model_name).lower():
                self.model = AutoModelForCausalLM.from_pretrained(self.model_name, **model_kwargs)
                self.tok = AutoTokenizer.from_pretrained(self.model_name)
                self.tok.pad_token_id = self.tok.eos_token_id
            else:
                raise NotImplementedError

            if self.tok is not None and (isinstance(self.tok, GPT2Tokenizer) or isinstance(self.tok, GPT2TokenizerFast) or isinstance(self.tok, LlamaTokenizer) or isinstance(self.tok, LlamaTokenizerFast) or isinstance(self.tok, PreTrainedTokenizerFast)) and (hparams.alg_name not in ['ROME', 'MEMIT', 'EMMET', 'R-ROME','AlphaEdit','CORE']):
                LOG.info('AutoRegressive Model detected, set the padding side of Tokenizer to left...')
                self.tok.padding_side = 'left'
            if self.tok is not None and ('mistral' in os.path.basename(self.model_name).lower() or 'llama' in os.path.basename(self.model_name).lower() or 'qwen' in os.path.basename(self.model_name).lower()) and (hparams.alg_name in ['ROME', 'MEMIT', 'EMMET', 'R-ROME','AlphaEdit', 'CORE']):
                LOG.info('AutoRegressive Model detected, set the padding side of Tokenizer to right...')
                self.tok.padding_side = 'right'
        else:
            self.model, self.tok = self.model_name
        


        if hparams.model_parallel:
            hparams.device = str(self.model.device).split(":")[1]
        if not hparams.model_parallel and hasattr(hparams, 'device') and hparams.alg_name != 'QLoRA':
            self.model.to(f'cuda:{hparams.device}')

        self.hparams = hparams

    def edit(self,
             prompts: Union[str, List[str]],
             target_new: Union[str, List[str]],
             ground_truth: Optional[Union[str, List[str]]] = None,
             target_neg: Optional[Union[str, List[str]]] = None,
             rephrase_prompts: Optional[Union[str, List[str]]] = None,
             locality_inputs:  Optional[Dict] = None,
             portability_inputs: Optional[Dict] = None,
             sequential_edit=False,
             keep_original_weight=False,
             verbose=True,
             summary_metrics = False, 
             **kwargs
             ):
        """
        `prompts`: list or str
            the prompts to edit
        `ground_truth`: str
            the ground truth / expected output
        `locality_inputs`: dict
            for locality
        """
        test_generation = kwargs.pop('test_generation', False)
        save_gen_sentence = kwargs['save_gen_sentence'] if 'save_gen_sentence' in kwargs.keys() else False
        if isinstance(prompts, List):
            assert len(prompts) == len(target_new)
        else:
            prompts, target_new = [prompts,], [target_new,]

        if hasattr(self.hparams, 'batch_size') and not BatchEditor.is_batchable_method(self.alg_name):  # For Singleton Editing, bs=1
            assert self.hparams.batch_size == 1, 'Single Editing: batch_size should be set to 1'

        if ground_truth is not None:
            ground_truth = [ground_truth,] if isinstance(ground_truth, str) else ground_truth
        else:# Default ground truth is <|endoftext|>
            ground_truth = ['<|endoftext|>'] * (len(prompts))

        if "requests" in kwargs.keys():
            requests = kwargs["requests"]
        else:
            requests = _prepare_requests(prompts, target_new, ground_truth, target_neg, rephrase_prompts, locality_inputs, portability_inputs, **kwargs)

        return self.edit_requests(requests, sequential_edit, verbose, test_generation=test_generation, **kwargs)

    def batch_edit(self,
                   prompts: List[str],
                   target_new: List[str],
                   ground_truth: Optional[List[str]] = None,
                   target_neg: Optional[List[str]] = None,
                   rephrase_prompts: Optional[List[str]] = None,
                   locality_inputs: Optional[Dict] = None,
                   portability_inputs: Optional[Dict] = None,
                   sequential_edit=False,
                   verbose=True,
                   **kwargs
                   ):
        """
        `prompts`: list or str
            the prompts to edit
        `ground_truth`: str
            the ground truth / expected output
        """
        assert len(prompts) == len(target_new)
        test_generation = kwargs['test_generation'] if 'test_generation' in kwargs.keys() else False
        if ground_truth is not None:
            if isinstance(ground_truth, str):
                ground_truth = [ground_truth,]
            else:
                assert len(ground_truth) == len(prompts)
        else: # Default ground truth is <|endoftext|>
            ground_truth = ['<|endoftext|>' for _ in range(len(prompts))]


        assert BatchEditor.is_batchable_method(self.alg_name), f'The Method {self.alg_name} can not batch edit examples.'

        requests = _prepare_requests(prompts, target_new, ground_truth, target_neg, rephrase_prompts, locality_inputs, portability_inputs, **kwargs)

        assert hasattr(self.hparams, 'batch_size'), f'Method {self.alg_name} found, pls specify the batch_size....'
        all_metrics = []
        for record_chunks in _chunks(requests, self.hparams.batch_size):
            start = time()

            edited_model, weights_copy = self.apply_algo(
                self.model,
                self.tok,
                record_chunks,
                self.hparams,
                copy=False,
                return_orig_weights=True
            )
            exec_time = time() - start
            LOG.info(f"Execution editing took {exec_time}")

            start = time()
            chunk_metrics = []
            for i, request in enumerate(record_chunks):

                metrics = {
                    'case_id': i,
                    "requested_rewrite": request,
                    "time": exec_time,
                    "post": compute_edit_quality(edited_model, self.model_name, self.hparams, self.tok, request, self.hparams.device, test_generation=test_generation),
                }

                chunk_metrics.append(metrics)

            if sequential_edit:
                self.model = edited_model
            else:
                if self.alg_name == 'KN' or self.alg_name == 'GRACE' or self.alg_name == 'WISE':
                    with torch.no_grad():
                        weights_copy()
                elif self.alg_name == 'LoRA' or self.alg_name == 'QLoRA' or self.alg_name == 'DPO':
                    edited_model.unload()
                    del self.model.peft_config
                elif self.alg_name == 'MELO':
                    self.model = edited_model
                else:
                    with torch.no_grad():
                        for k, v in weights_copy.items():
                            nethook.get_parameter(self.model, k)[...] = v.to(f"cuda:{self.hparams.device}")

            for i, request in enumerate(record_chunks):
                chunk_metrics[i]["pre"] = compute_edit_quality(self.model, self.model_name, self.hparams, self.tok, request, self.hparams.device, test_generation=test_generation)

                if verbose:
                    LOG.info(
                        f"{i} editing: {request['prompt']} -> {request['target_new']}  \n {chunk_metrics[i]}"
                    )

            LOG.info(f"Evaluation took {time() - start}")
            all_metrics.extend(chunk_metrics)
        return all_metrics, edited_model, weights_copy

    def edit_requests(self,
             requests,
             sequential_edit=False,
             verbose=True,
             test_generation=False,
             **kwargs
             ):
        """
        `prompts`: list or str
            the prompts to edit
        `ground_truth`: str
            the ground truth / expected output
        `locality_inputs`: dict
            for locality
        """
        eval_metric= kwargs['eval_metric'] if 'eval_metric' in kwargs.keys() else 'exact match'
        # if hasattr(self.hparams, 'batch_size'):  # For Singleton Editing, bs=1
        #     assert self.hparams.batch_size == 1, 'Single Editing: batch_size should be set to 1'
        all_metrics = []
        if 'pre_edit' in kwargs and kwargs['pre_edit'] is not None:
            metrics = kwargs['pre_edit']
            all_metrics = metrics
        else:
            for i, request in enumerate(tqdm(requests)):
                if self.alg_name == 'IKE':
                    assert 'train_ds' in kwargs.keys(), print('IKE need train_ds(For getting In-Context prompt)')
                    metrics = {"pre": compute_icl_edit_quality(self.model, self.model_name, self.hparams, self.tok, [''], request, self.hparams.device, pre_edit=True)}
                else:
                    metrics = {"pre": compute_edit_quality(self.model, self.model_name, self.hparams, self.tok, request, self.hparams.device, eval_metric=eval_metric, test_generation=test_generation)}
                all_metrics.append(metrics)
            if 'pre_file' in kwargs and kwargs['pre_file'] is not None:
                json.dump(all_metrics, open(kwargs['pre_file'], 'w'), indent=4)

        def edit_func(request,case_id):
            
            if self.alg_name == 'IKE' or self.alg_name == 'ICE':
                edited_model, weights_copy, icl_examples = self.model, {}, self.apply_algo(
                    self.model,
                    self.tok,
                    [request],
                    self.hparams,
                    copy=False,
                    return_orig_weights=True,
                    keep_original_weight=False,
                    train_ds=kwargs['train_ds'] if self.alg_name == 'IKE' else None,
                    graph_data = kwargs["graph_data"][case_id-1]
                )
            else:
                if isinstance(case_id,list):
                    edited_model, weights_copy = self.apply_algo(
                        self.model,
                        self.tok,
                        [request],
                        self.hparams,
                        copy=False,
                        return_orig_weights=True,
                        keep_original_weight=False,
                        train_ds=kwargs['train_ds'] if self.alg_name == 'IKE' else None,
                        graph_data = kwargs["graph_data"][case_id[0]:case_id[1]]
                    )
                else:
                    edited_model, weights_copy = self.apply_algo(
                        self.model,
                        self.tok,
                        [request],
                        self.hparams,
                        copy=False,
                        return_orig_weights=True,
                        keep_original_weight=False,
                        train_ds=kwargs['train_ds'] if self.alg_name == 'IKE' else None,
                        graph_data = kwargs["graph_data"][case_id-1]
                    )
                icl_examples = None
            return edited_model, weights_copy, icl_examples

        def edit_evaluation(all_metrics, request, edited_model, idx, test_generation, icl_examples, **kwargs):
            eval_metric= kwargs['eval_metric'] if 'eval_metric' in kwargs.keys() else 'exact match'
            if self.alg_name == 'IKE':
                all_metrics[idx].update({
                    'case_id': idx,
                    "requested_rewrite": request,
                    "post": compute_icl_edit_quality(self.model, self.model_name, self.hparams, self.tok, icl_examples, request, self.hparams.device ,test_generation=test_generation),
                })
                if "metric_kwargs" in kwargs:
                    all_metrics[idx].update(compute_sent_metric(self.model, edited_model, self.model_name, self.hparams, self.tok,metric_kwargs=kwargs["metric_kwargs"][idx], device=self.hparams.device))

            else:
                all_metrics[idx].update({
                    'case_id': idx,
                    "requested_rewrite": request,
                    "post": compute_edit_quality(edited_model, self.model_name, self.hparams, self.tok, request, self.hparams.device, eval_metric=eval_metric, test_generation=test_generation),
                })
                if "metric_kwargs" in kwargs:
                    all_metrics[idx].update(compute_sent_metric(self.model, edited_model, self.model_name, self.hparams, self.tok,metric_kwargs=kwargs["metric_kwargs"][idx], device=self.hparams.device))
                if 'locality' in all_metrics[idx]['post'].keys() and not hasattr(self.hparams, 'evaluation_type'):
                    for locality_key in request['locality'].keys():
                        locality_result = []
                        if hasattr(self.hparams, 'evaluation_type'):
                            locality_result.append(float(all_metrics[idx]['post']['locality'][f'{locality_key}_output']==all_metrics[idx]['pre']['locality'][f'{locality_key}_output']))
                        else:
                            for ans, label in zip(all_metrics[idx]['post']['locality'][f'{locality_key}_output'], all_metrics[idx]['pre']['locality'][f'{locality_key}_output']):
                                locality_result.append(np.mean(np.equal(ans, label)))
                        all_metrics[idx]['post']['locality'][f'{locality_key}_acc'] = locality_result
                        all_metrics[idx]['post']['locality'].pop(f'{locality_key}_output')
                    all_metrics[idx]['pre'].pop('locality')

            if verbose:
                LOG.info(f"{idx} editing: {request['prompt']} -> {request['target_new']}  \n\n {all_metrics[idx]}")

        device_map = 'auto' if self.hparams.model_parallel else None
        torch_dtype = torch.float16 if hasattr(self.hparams, 'fp16') and self.hparams.fp16 else torch.float32
        if self.hparams.alg_name == 'QLoRA':
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=(self.hparams.quantization_bit == 4),
                bnb_4bit_use_double_quant=self.hparams.double_quant,
                bnb_4bit_quant_type=self.hparams.quant_type,
                bnb_4bit_compute_dtype=torch.bfloat16
            )
            model_kwargs = {
                "quantization_config": bnb_config,
                "torch_dtype": torch_dtype,
                "device_map": {'': self.hparams.device}
            }
        else:
            model_kwargs = {
                "torch_dtype": torch_dtype,
                "device_map": device_map
            }





         


        if sequential_edit:
            # set epochs
            epoch = 1
            EMA_LAMBDA = 0.997

            
            for j in range(epoch):
               
                all_time = 0.0
                all_metrics_copy_1 =copy.deepcopy(all_metrics)
                # teacher = copy.deepcopy(self.model)
                # ==========================================
                chunk_size = 100  # 设定的 n 值，即每个 batch 包含的连续编辑数据量
                # chunk_size = len(requests)
                # chunk_size = None


                # ==========================================
                # request_tmp = []
                # request_step = []
                pre = 0
                if chunk_size:
                    chunks = [requests[i : i + chunk_size] for i in range(0, len(requests), chunk_size)]
                    for i, chunk_requests in enumerate(tqdm(chunks, desc=f"Sequential Batch Editing (n={chunk_size})")):    
                        # if current_lora_sd is not None:
                        #     self.model.load_state_dict(current_lora_sd,strict=False)
                        # else:
                        #     self.model = get_peft_model(self.model, lora_config)
                        # request_tmp.append(request)
                        # request_step.append(request)
                        startt = time.time()
                        if isinstance(chunk_requests[0],list):
                            chunk_requests = chunk_requests[0]
                        edited_model, weights_copy, icl_examples = edit_func(chunk_requests,[i*chunk_size+pre,(i+1)*chunk_size+pre])
                        endt = time.time()
                        # print("times per edit:",endt-startt)
                        all_time += endt-startt

                        for ii, request in enumerate(chunk_requests):
                            edit_evaluation(all_metrics_copy_1, request, edited_model, ii+i*chunk_size+pre, test_generation, icl_examples, **kwargs)
                        eval(all_metrics_copy_1[i*chunk_size+pre:(i+1)*chunk_size+pre])
                        if self.alg_name == 'KN' or self.alg_name == 'GRACE' or self.alg_name == 'WISE':
                            with torch.no_grad():
                                weights_copy()
                        elif self.alg_name == 'LoRA' or self.alg_name == 'QLoRA' or self.alg_name == 'DPO':
                            edited_model.unload()
                            del self.model.peft_config
                        elif self.alg_name == 'MELO':
                            self.model = edited_model
                        else:
                            with torch.no_grad():
                                for k, v in weights_copy.items():
                                    nethook.get_parameter(self.model, k)[...] = v.to(f"cuda:{self.hparams.device}") 
                    eval(all_metrics_copy_1)
                    continue    
                else:
                    for i, request in enumerate(tqdm(requests, total=len(requests))):
                        startt = time.time()
                        edited_model, weights_copy, icl_examples = edit_func(request,i+1)
                        endt = time.time()
                        # print("times per edit:",endt-startt)
                        all_time += endt-startt    
                    for i, request in enumerate(requests):
                        edit_evaluation(all_metrics_copy_1, request, edited_model, i, test_generation, icl_examples, **kwargs)
                    eval(all_metrics_copy_1)
                print("all_time:",all_time/len(requests))
                
            all_metrics = all_metrics_copy_1
            
        else:
            # if hasattr(self.hparams, 'batch_size'):  # For Singleton Editing, bs=1
            #     assert self.hparams.batch_size == 1
            # teacher = copy.deepcopy(self.model)
            sta = 0
            for i, request in enumerate(tqdm(requests[sta:], total=len(requests[sta:]))):
                edited_model, weights_copy, icl_examples = edit_func(request,i+1+sta)
                edit_evaluation(all_metrics, request, edited_model, i, test_generation, icl_examples, **kwargs)
                # eval(all_metrics[:2471])
                if self.alg_name == 'KN' or self.alg_name == 'GRACE' or self.alg_name == 'WISE':
                    with torch.no_grad():
                        weights_copy()
                elif self.alg_name == 'LoRA' or self.alg_name == 'QLoRA' or self.alg_name == 'DPO':
                    edited_model.unload()
                    del self.model.peft_config
                elif self.alg_name == 'MELO':
                    self.model = edited_model
                else:
                    with torch.no_grad():
                        for k, v in weights_copy.items():
                            nethook.get_parameter(self.model, k)[...] = v.to(f"cuda:{self.hparams.device}")
            eval(all_metrics)

        if isinstance(edited_model, LORA):
            edited_model = edited_model.model
        if not hasattr(self.hparams, 'evaluation_type') or self.hparams.evaluation_type != "generate-text":
            summary_metrics(all_metrics)
        
        return all_metrics, edited_model, weights_copy

    def normal_edit(
        self,
        prompts: List[str],
        target_new: List[str],
        sequential_edit=False,
    ):
        """
        `prompts`: list or str
            the prompts to edit
        `ground_truth`: str
            the ground truth / expected output
        """
        assert len(prompts) == len(target_new)
        ground_truth = ['<|endoftext|>' for _ in range(len(prompts))]


        assert BatchEditor.is_batchable_method(self.alg_name), f'The Method {self.alg_name} can not batch edit examples.'

        requests = _prepare_requests(prompts, target_new, ground_truth)

        assert hasattr(self.hparams, 'batch_size'), f'Method {self.alg_name} found, pls specify the batch_size....'

        # print(f"[editor.py][batch_edit] `batch_size`={self.hparams.batch_size}")
        # for epc in range(epoch):
        #     print(f"[editor.py][batch_edit] `Epoch` = {epc+1}")
        #     for record_chunks in self._chunks(requests, self.hparams.batch_size):
        start = time()

        edited_model, weights_copy = self.apply_algo(
            self.model,
            self.tok,
            requests,  # record_chunks -> requests
            self.hparams,
            copy=False,
            return_orig_weights=True,
            keep_original_weight=False,
        )
        exec_time = time() - start
        LOG.info(f"Execution editing took {exec_time}")

        with torch.no_grad():
            for k, v in weights_copy.items():
                nethook.get_parameter(self.model, k)[...] = v.to(f"cuda:{self.hparams.device}")

        return None, edited_model, weights_copy

    def generate_edit(
        self,
        prompts: Union[str, List[str]],
        target_new: Union[str, List[str]],
        ground_truth: Optional[Union[str, List[str]]] = None,
        target_neg: Optional[Union[str, List[str]]] = None,
        rephrase_prompts: Optional[Union[str, List[str]]] = None,
        locality_inputs:  Optional[Dict] = None,
        portability_inputs: Optional[Dict] = None,
        sequential_edit=False,
        verbose=True,
        **kwargs
    ):
        eval_metric= kwargs['eval_metric'] if 'eval_metric' in kwargs.keys() else 'exact match'
        test_generation = kwargs.pop('test_generation', False)

        assert len(prompts) == len(target_new)

        if hasattr(self.hparams, 'batch_size'):
            assert self.hparams.batch_size == 1, 'Single Editing: batch_size should be set to 1'

        if "requests" in kwargs.keys():
            requests = kwargs["requests"]
        else:
            requests = _prepare_requests(prompts, target_new, ground_truth, target_neg, rephrase_prompts, locality_inputs, portability_inputs, **kwargs)

        def text_generate(
            model,
            model_name,
            hparams: HyperParams,
            tok: AutoTokenizer,
            query,
            device,
            eval_metric: str = 'token_em',
            test_generation = False
        ):
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": query}
            ]
            text = self.tok.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            model_inputs = tok.encode(text, return_tensors="pt").to(f"cuda:{device}")
            template_length = len(model_inputs[0])
            generated_ids = model.generate(
                input_ids=model_inputs,
                max_new_tokens=512
            )
            trimmed_generated_ids = generated_ids[0][template_length:]
            response = tok.decode(trimmed_generated_ids, skip_special_tokens=True)
            return response

        all_results = []
        if 'pre_edit' in kwargs and kwargs['pre_edit'] is not None:
            results = kwargs['pre_edit']
            all_results = results
        else:
            for i, request in enumerate(tqdm(requests)):
                results = {}
                results['pre'] = {}
                results['pre']['rewrite_ans'] = text_generate(self.model, self.model_name, self.hparams, self.tok, request['prompt'], self.hparams.device, eval_metric=eval_metric, test_generation=test_generation)
                results['pre']['rephrase_ans'] = text_generate(self.model, self.model_name, self.hparams, self.tok, request['rephrase_prompt'], self.hparams.device, eval_metric=eval_metric, test_generation=test_generation)
                por_results = []
                for pr in request['portability']['por_hop']['prompt']:
                    por_results.append(text_generate(self.model, self.model_name, self.hparams, self.tok, pr, self.hparams.device, eval_metric=eval_metric, test_generation=test_generation))
                if 'locality' in request.keys() and 'loc_hop' in request['locality'].keys():
                    loc_results = []
                    for pr in request['locality']['loc_hop']['prompt']:
                        loc_results.append(text_generate(self.model, self.model_name, self.hparams, self.tok, pr, self.hparams.device, eval_metric=eval_metric, test_generation=test_generation))
                    results['pre']['locality_ans'] = loc_results
                results['pre']['portability_ans'] = por_results
                all_results.append(results)
            if 'pre_file' in kwargs and kwargs['pre_file'] is not None:
                json.dump(all_results, open(kwargs['pre_file'], 'w'), indent=4)

        def edit_func(request):
            if self.alg_name == 'IKE':
                edited_model, weights_copy, icl_examples = self.model, {}, self.apply_algo(
                    self.model,
                    self.tok,
                    [request],
                    self.hparams,
                    copy=False,
                    return_orig_weights=True,
                    keep_original_weight=False,
                    train_ds=kwargs['train_ds'] if self.alg_name == 'IKE' else None
                )
            else:
                edited_model, weights_copy = self.apply_algo(
                    self.model,
                    self.tok,
                    [request],
                    self.hparams,
                    copy=False,
                    return_orig_weights=True,
                    keep_original_weight=False,
                    train_ds=kwargs['train_ds'] if self.alg_name == 'IKE' else None
                )
                icl_examples = None
            return edited_model, weights_copy, icl_examples

        def post_edit_results(all_results, request, edited_model, idx, eval_metric, test_generation, icl_examples, **kwargs):
            if self.alg_name == 'IKE':
                all_results[idx].update({
                    'case_id': idx,
                    "requested_rewrite": request,
                    "post": compute_icl_edit_quality(self.model, self.model_name, self.hparams, self.tok, icl_examples, request, self.hparams.device),
                })
            else:
                results_post = {}
                results_post['rewrite_ans'] = text_generate(edited_model, self.model_name, self.hparams, self.tok, request['prompt'], self.hparams.device, eval_metric=eval_metric, test_generation=test_generation)
                results_post['rephrase_ans'] = text_generate(edited_model, self.model_name, self.hparams, self.tok, request['rephrase_prompt'], self.hparams.device, eval_metric=eval_metric, test_generation=test_generation)
                por_results = []
                for pr in request['portability']['por_hop']['prompt']:
                    por_results.append(text_generate(edited_model, self.model_name, self.hparams, self.tok, pr, self.hparams.device, eval_metric=eval_metric, test_generation=test_generation))
                if 'locality' in request.keys() and 'loc_hop' in request['locality'].keys():
                    loc_results = []
                    for pr in request['locality']['loc_hop']['prompt']:
                        loc_results.append(text_generate(edited_model, self.model_name, self.hparams, self.tok, pr, self.hparams.device, eval_metric=eval_metric, test_generation=test_generation))
                    results_post['locality_ans'] = loc_results
                results_post['portability_ans'] = por_results
                if test_generation:
                    if self.hparams.alg_name == 'GRACE':
                        results_post['fluency'] = test_generation_quality(model=edited_model,tok=self.tok,prefixes=request['prompt'] if isinstance(request['prompt'],list) else [request['prompt'],], max_out_len=100, vanilla_generation=True)
                    else:
                        results_post['fluency'] = test_generation_quality(model=edited_model,tok=self.tok,prefixes=request['prompt'] if isinstance(request['prompt'],list) else [request['prompt'],], max_out_len=100, vanilla_generation=False)
                all_results[idx].update({
                    'case_id': idx,
                    "requested_rewrite": request,
                    "post": results_post
                })
            if verbose:
                LOG.info(f"{idx} editing: {request['prompt']} -> {request['target_new']}")

        if sequential_edit:
            for i, request in enumerate(tqdm(requests, total=len(requests))):
                edited_model, weights_copy, icl_examples = edit_func(request)
            for i, request in enumerate(requests):
                post_edit_results(all_results, request, edited_model, i, eval_metric, test_generation, icl_examples, **kwargs)
        else:
            for i, request in enumerate(tqdm(requests, total=len(requests))):
                edited_model, weights_copy, icl_examples = edit_func(request)
                post_edit_results(all_results, request, edited_model, i, eval_metric, test_generation, icl_examples, **kwargs)
                if self.alg_name == 'KN' or self.alg_name == 'GRACE' or self.alg_name == 'WISE':
                    with torch.no_grad():
                        weights_copy()
                elif self.alg_name == 'LoRA' or self.alg_name == 'QLoRA' or self.alg_name == 'DPO':
                    edited_model.unload()
                    del self.model.peft_config
                elif self.alg_name == 'MELO':
                    self.model = edited_model
                elif self.alg_name == 'LoRA' or self.alg_name == 'QLoRA' or self.alg_name == 'DPO':
                    self.model = edited_model
                else:
                    with torch.no_grad():
                        for k, v in weights_copy.items():
                            nethook.get_parameter(self.model, k)[...] = v.to(f"cuda:{self.hparams.device}")

        if isinstance(edited_model, LORA):
            edited_model = edited_model.model
        if len(all_results) != 0:
            summary_metrics(all_results)

        return all_results, edited_model, weights_copy

    def deep_edit(
        self,
        datasets
    ):
        metrics = self.apply_algo(datasets, self.hparams)
        return metrics



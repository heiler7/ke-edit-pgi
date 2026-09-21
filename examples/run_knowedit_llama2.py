import os





import os.path as path
import sys
import json
import random
sys.path.append('..')
sys.path.append('.')
from easyeditor import (
    AlphaEditHyperParams,
    FTHyperParams, 
    IKEHyperParams, 
    KNHyperParams, 
    MEMITHyperParams, 
    ROMEHyperParams, 
    LoRAHyperParams,
    MENDHyperParams,
    SERACHparams
    )
from easyeditor import BaseEditor
from easyeditor.models.ike import encode_ike_facts
from sentence_transformers import SentenceTransformer
from easyeditor import KnowEditDataset
import torch

import argparse
import numpy as np

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

class DebugLogger:
    def __init__(self, filename):
        self.file = open(filename, 'w')
        self.stdout = sys.stdout
    
    def write(self, msg):
        self.stdout.write(msg)
        self.file.write(msg)
    
    def flush(self):
        self.stdout.flush()
        self.file.flush()

if __debug__:  
    sys.stdout = DebugLogger('debug1.log')


def eval(result_path):
    if path.exists(result_path):
        
        with open(result_path,'r') as file:
            datas=json.load(file)
        #data_rome_counterfact['post'].keys()  dict_keys(['rewrite_acc', 'locality', 'portability'])
        Edit_Succ_list=[data_rome_counterfact['post']['rewrite_acc'][0] for data_rome_counterfact in datas]
        Edit_Succ=sum(Edit_Succ_list)/len(Edit_Succ_list)*100
        print('Edit_Succ:',Edit_Succ)
         # rephrase_Succ_list=[data_rome_counterfact['post']['rephrase_acc'][0] for data_rome_counterfact in datas if 'rephrase_acc' in data_rome_counterfact['post'].keys() else 0] 
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
            print("Portability_",key,':',np.mean(Locality_dict[key]))

        Fluency_list=[x['post']['fluency']['ngram_entropy'] for x in datas]
        Fluency=sum(Fluency_list)/len(Fluency_list)*100
        print('Fluency:',Fluency)

dataset_graph_dict = {
    'zsre.json': 'zsre_with_relations_fixed_merged_graph.json',
    'mquake_easyedit_3k.json': 'MQuAKE-CF-3k_merged_graph_expect_summarys_cleaned.json',
    'mquake_easyedit_3k_1_add_alias.json': 'MQuAKE-3k_merged_graph_expect_withouttest_api_summarys_filtered_5.json',
    'mquake_easyedit_T_1_add_alias.json': 'MQuAKE-T_merged_graph_expect_withouttest_api_summarys_filtered_5.json'
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--editing_method', default='LoRA', type=str)
    # parser.add_argument('--hparams_dir', default='../hparams/LoRA/llama-7b', type=str)
    parser.add_argument('--hparams_dir', default='../hparams/LoRA/vicuna-7b', type=str)
    parser.add_argument('--data_dir', default='./data/mquake_easyedit_3k_1_add_alias.json', type=str)
    parser.add_argument('--ds_size', default=None, type=int) 
    parser.add_argument('--metrics_save_dir', default='./output', type=str)
    # parser.add_argument('--datatype', default='counterfact',type=str)
    parser.add_argument('--datatype', default='mquake',type=str)
    parser.add_argument('--train_data_path', type=str)
    parser.add_argument('--pre_file', default='./seq_pre.json', type=str)
    parser.add_argument('--testname', default='', type=str)
    
    args = parser.parse_args()

    if args.editing_method == 'FT':
        editing_hparams = FTHyperParams
    elif args.editing_method == 'IKE':
        editing_hparams = IKEHyperParams
    elif args.editing_method == 'ICE':
        editing_hparams = IKEHyperParams
    elif args.editing_method == 'KN':
        editing_hparams = KNHyperParams
    elif args.editing_method == 'MEMIT':
        editing_hparams = MEMITHyperParams
    elif args.editing_method == 'ROME':
        editing_hparams = ROMEHyperParams
    elif args.editing_method == 'LoRA':
        editing_hparams = LoRAHyperParams
    elif args.editing_method == 'SERAC':
        editing_hparams = SERACHparams
    elif args.editing_method == 'MEND':
        editing_hparams = MENDHyperParams
    elif args.editing_method == 'AlphaEdit':
        editing_hparams = AlphaEditHyperParams
    else:
        raise NotImplementedError
    
    graph_data_path = os.path.join(os.path.dirname(args.data_dir),dataset_graph_dict[os.path.basename(args.data_dir)])
    with open(graph_data_path, "r") as f:
        graph_data = json.load(f)
    
    datas = KnowEditDataset(args.data_dir,size=args.ds_size)
    if args.datatype == 'counterfact' or args.datatype == 'recent' or args.datatype == 'zsre':
        prompts=[data['prompt'] for data in datas]
        subjects=[data['subject'] for data in datas]
        target_new = [data['target_new'] for data in datas]
        context = [data['context'] for data in datas]
        rewrite = [data['rewrite'] for data in datas]
        rephrase_example = [data['rephrase_example'] for data in datas]
        
        portability_r =[data['portability_r'] for data in datas]
        portability_s =[data['portability_s'] for data in datas]
        portability_l =[data['portability_l'] for data in datas]
        portability_m =[data['portability_m'] for data in datas]
        
        portability_reasoning_prompts=[]
        portability_reasoning_ans=[]
        portability_Logical_Generalization_prompts=[]
        portability_Logical_Generalization_ans=[]
        portability_Subject_Aliasing_prompts=[]
        portability_Subject_Aliasing_ans=[]
        
        portability_data = [portability_r,portability_s,portability_l]
        portability_prompts = [portability_reasoning_prompts,portability_Subject_Aliasing_prompts,portability_Logical_Generalization_prompts]
        portability_answers = [portability_reasoning_ans,portability_Subject_Aliasing_ans,portability_Logical_Generalization_ans]
        for data, portable_prompts, portable_answers in zip(portability_data,portability_prompts,portability_answers):
            for item in data:
                if item is None:
                    portable_prompts.append(None)
                    portable_answers.append(None)
                else:
                    temp_prompts = []
                    temp_answers = []
                    for pr in item:
                        prompt=pr["prompt"]
                        an=pr["ground_truth"]
                        while isinstance(an,list):
                            an = an[0]
                        if an.strip() =="":
                            continue
                        temp_prompts.append(prompt)
                        temp_answers.append(an)
                    portable_prompts.append(temp_prompts)
                    portable_answers.append(temp_answers)
        assert len(prompts) == len(portability_reasoning_prompts) == len(portability_Logical_Generalization_prompts) == len(portability_Subject_Aliasing_prompts)
        
        locality_rs = [data['locality_rs'] for data in datas]
        locality_f = [data['locality_f'] for data in datas]
        locality_Relation_Specificity_prompts=[]
        locality_Relation_Specificity_ans=[]
        locality_Forgetfulness_prompts=[]        
        locality_Forgetfulness_ans=[]
        
        locality_data = [locality_rs, locality_f]
        locality_prompts = [locality_Relation_Specificity_prompts,locality_Forgetfulness_prompts]
        locality_answers = [locality_Relation_Specificity_ans,locality_Forgetfulness_ans]
        for data, local_prompts, local_answers in zip(locality_data,locality_prompts,locality_answers):
            for item in data:
                if item is None:
                    local_prompts.append(None)
                    local_answers.append(None)
                else:
                    temp_prompts = []
                    temp_answers = []
                    for pr in item:
                        prompt=pr["prompt"]
                        an=pr["ground_truth"]
                        while isinstance(an,list):
                            an = an[0]
                        if an.strip() =="":
                            continue
                        temp_prompts.append(prompt)
                        temp_answers.append(an)
                    local_prompts.append(temp_prompts)
                    local_answers.append(temp_answers)
        assert len(prompts) == len(locality_Relation_Specificity_prompts) == len(locality_Forgetfulness_prompts)
        locality_inputs = {}
        portability_inputs = {}
        
        locality_inputs = {
            'Relation_Specificity':{
                'prompt': locality_Relation_Specificity_prompts,
                'ground_truth': locality_Relation_Specificity_ans
            },
            'Forgetfulness':{
                'prompt':locality_Forgetfulness_prompts,
                'ground_truth':locality_Forgetfulness_ans
            }
        }
        portability_inputs = {
            'Subject_Aliasing':{
                'prompt': portability_Subject_Aliasing_prompts,
                'ground_truth': portability_Subject_Aliasing_ans
            },
            'reasoning':{
                'prompt': portability_reasoning_prompts,
                'ground_truth': portability_reasoning_ans           
            },
            'Logical_Generalization':{
                'prompt': portability_Logical_Generalization_prompts,
                'ground_truth': portability_Logical_Generalization_ans           
            }
        }
    if args.datatype == 'mquake':
        prompts=[data['prompt'] for data in datas]
        subjects=[data['subject'] for data in datas]
        target_new = [data['target_new'] for data in datas]
        context = [data['context'] for data in datas]
        rewrite = [data['rewrite'] for data in datas]
        rephrase_example = [data['rephrase_example'] for data in datas]
        
        portability_r =[data['portability_r'] for data in datas]
        portability_s =[data['portability_s'] for data in datas]
        portability_l =[data['portability_l'] for data in datas]
        portability_m =[data['portability_m'] for data in datas]
        portability_h =[data['portability_h'] for data in datas]
        
        portability_reasoning_prompts=[]
        portability_reasoning_ans=[]
        portability_Logical_Generalization_prompts=[]
        portability_Logical_Generalization_ans=[]
        portability_Subject_Aliasing_prompts=[]
        portability_Subject_Aliasing_ans=[]
        portability_multihop_prompts=[]
        portability_multihop_ans=[]
        portability_singlehop_prompts=[]
        portability_singlehop_ans=[]
        portability_data = [portability_r,portability_s,portability_l,portability_m,portability_h]
        portability_prompts = [portability_reasoning_prompts,portability_Subject_Aliasing_prompts,portability_Logical_Generalization_prompts,portability_multihop_prompts,portability_singlehop_prompts]
        portability_answers = [portability_reasoning_ans,portability_Subject_Aliasing_ans,portability_Logical_Generalization_ans,portability_multihop_ans,portability_singlehop_ans]
        for data, portable_prompts, portable_answers, is_singlehop in zip(portability_data,portability_prompts,portability_answers,[False,False,False,False,True]):
            for item in data:
                if item is None:
                    portable_prompts.append(None)
                    portable_answers.append(None)
                else:
                    temp_prompts = []
                    temp_answers = []
                    for pr in item:
                        prompt=pr["prompt"]
                        an=pr["ground_truth"]
                        if not is_singlehop:
                            while isinstance(an,list):
                                an = an[0]
                            if an.strip() =="":
                                continue
                        # else:
                        #     if isinstance(an, list):
                        #         an = an[0]
                        #     if an.strip() =="":
                        #         continue
                        temp_prompts.append(prompt)
                        temp_answers.append(an)
                    portable_prompts.append(temp_prompts)
                    portable_answers.append(temp_answers)
        assert len(prompts) == len(portability_reasoning_prompts) == len(portability_Logical_Generalization_prompts) == len(portability_Subject_Aliasing_prompts)
        
        locality_rs = [data['locality_rs'] for data in datas]
        locality_f = [data['locality_f'] for data in datas]
        locality_Relation_Specificity_prompts=[]
        locality_Relation_Specificity_ans=[]
        locality_Forgetfulness_prompts=[]        
        locality_Forgetfulness_ans=[]
        
        locality_data = [locality_rs, locality_f]
        locality_prompts = [locality_Relation_Specificity_prompts,locality_Forgetfulness_prompts]
        locality_answers = [locality_Relation_Specificity_ans,locality_Forgetfulness_ans]
        for data, local_prompts, local_answers in zip(locality_data,locality_prompts,locality_answers):
            for item in data:
                if item is None:
                    local_prompts.append(None)
                    local_answers.append(None)
                else:
                    temp_prompts = []
                    temp_answers = []
                    for pr in item:
                        prompt=pr["prompt"]
                        an=pr["ground_truth"]
                        while isinstance(an,list):
                            an = an[0]
                        if an.strip() =="":
                            continue
                        temp_prompts.append(prompt)
                        temp_answers.append(an)
                    local_prompts.append(temp_prompts)
                    local_answers.append(temp_answers)
        assert len(prompts) == len(locality_Relation_Specificity_prompts) == len(locality_Forgetfulness_prompts)
        locality_inputs = {}
        portability_inputs = {}
        
        locality_inputs = {
            'Relation_Specificity':{
                'prompt': locality_Relation_Specificity_prompts,
                'ground_truth': locality_Relation_Specificity_ans
            },
            'Forgetfulness':{
                'prompt':locality_Forgetfulness_prompts,
                'ground_truth':locality_Forgetfulness_ans
            }
        }
        portability_inputs = {
            'Subject_Aliasing':{
                'prompt': portability_Subject_Aliasing_prompts,
                'ground_truth': portability_Subject_Aliasing_ans
            },
            'reasoning':{
                'prompt': portability_reasoning_prompts,
                'ground_truth': portability_reasoning_ans           
            },
            'Logical_Generalization':{
                'prompt': portability_Logical_Generalization_prompts,
                'ground_truth': portability_Logical_Generalization_ans           
            },
            'multihop':{
                'prompt':portability_multihop_prompts,
                'ground_truth':portability_multihop_ans
            },
            'single_hop':{
                'prompt': portability_singlehop_prompts,
                'ground_truth':portability_singlehop_ans
            }
        }
    if args.datatype == 'wikibio':
        prompts=[data['prompt'] for data in datas]
        subjects=[data['subject'] for data in datas]
        target_new = [data['target_new'] for data in datas]
        context = [data['context'] for data in datas]
        rephrase_example = [data['rephrase_example'] for data in datas]
        locality_rs = [data['locality_rs'] for data in datas]
        locality_f = [data['locality_f'] for data in datas]
        locality_Relation_Specificity_prompts=[]
        locality_Relation_Specificity_ans=[]
        
        locality_data = [locality_rs]
        locality_prompts = [locality_Relation_Specificity_prompts]
        locality_answers = [locality_Relation_Specificity_ans]
        for data, local_prompts, local_answers in zip(locality_data,locality_prompts,locality_answers):
            for item in data:
                if item is None:
                    local_prompts.append(None)
                    local_answers.append(None)
                else:
                    temp_prompts = []
                    temp_answers = []
                    for pr in item:
                        prompt=pr["prompt"]
                        an=pr["ground_truth"]
                        while isinstance(an,list):
                            an = an[0]
                        if an.strip() =="":
                            continue
                        temp_prompts.append(prompt)
                        temp_answers.append(an)
                    local_prompts.append(temp_prompts)
                    local_answers.append(temp_answers)
        assert len(prompts) == len(locality_Relation_Specificity_prompts)
        portability_inputs = None
        locality_inputs = {}
        locality_inputs = {
            'Relation_Specificity':{
                'prompt': locality_Relation_Specificity_prompts,
                'ground_truth': locality_Relation_Specificity_ans
            }
        }
    # print(locality_inputs)
    hparams = editing_hparams.from_hparams(args.hparams_dir)
    print("hparams: ",hparams)
    datadir = args.data_dir.split('/')[-1].split('.')[0]
    args.pre_file = f"./{hparams.model_name.split('/')[-1]}_{datadir}_{args.datatype}_pre_edit.json"
    print("args.pre_file",args.pre_file)
    if args.pre_file is not None and os.path.exists(args.pre_file):
        pre_edit = json.load(open(args.pre_file,'r'))
        if args.ds_size is not None:
            pre_edit = pre_edit[:args.ds_size]
        assert len(pre_edit) == len(prompts)
    else:
        pre_edit = None
    if args.editing_method == 'IKE':
        train_ds = KnowEditDataset(args.train_data_path)
        sentence_model = SentenceTransformer(hparams.sentence_model_name).to(f'cuda:{hparams.device}')
        encode_ike_facts(sentence_model, train_ds, hparams)
    elif args.editing_method == 'ICE':
        hparams.use_icl_examples = False
        train_ds = None
    else:
        train_ds = None

    _sequential_edit = True
    # _sequential_edit = False
    editor = BaseEditor.from_hparams(hparams)
    metrics, edited_model, _ = editor.edit(
        prompts=prompts,
        target_new=target_new,
        subject=subjects,
        rephrase_prompts=rewrite,
        locality_inputs=locality_inputs,
        portability_inputs=portability_inputs,
        train_ds=train_ds,
        keep_original_weight=True,
        pre_file=args.pre_file,
        pre_edit = pre_edit,
        test_generation=True,
        sequential_edit=_sequential_edit,   
        # sequential_edit=False,
        save_gen_sentence = True,
        context=context,
        rephrase_example=rephrase_example,
        graph_data=graph_data
    )
    if not os.path.exists(args.metrics_save_dir):
        os.makedirs(args.metrics_save_dir)
    result_path = os.path.join(args.metrics_save_dir, f'{args.editing_method}_{args.datatype}_{"sequential" if _sequential_edit else "non_sequential"}_{args.ds_size if args.ds_size is not None else "full"}_{args.data_dir.split("/")[-1].split(".")[0]}_{hparams.model_name.split("/")[-1]}_results_add_alias_'+args.testname+'.json')
    json.dump(metrics, open(result_path, 'w'), indent=4)
    eval(result_path)


    print(args.editing_method)
    print(hparams.model_name)
   
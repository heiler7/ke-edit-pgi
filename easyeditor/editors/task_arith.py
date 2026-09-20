import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
import os
import torch
from typing import List, Optional

def merge_with_task_arithmetic_v2(
    base_state_dict: dict,
    state_dicts_list: List[dict],
    model_weights: Optional[List[float]] = None,
    density_list: Optional[List[float]] = None,
    lambda_merge: float = 1.0,
    layers: Optional[List[int]] = None,
    rewrite_module_tmp: str = "layers.{}.mlp",
    normalize_by_weights: bool = True,
    keep_min_per_row: int = 1,
    per_row_topk_for_matrix: bool = True,
    compute_dtype=torch.float32
):
    """
    改进点：
    - 支持每个模型不同 weight（model_weights）和不同 density（density_list）
    - 在 compute_dtype 上使用 float32 计算，最后转换回原 dtype
    - 对矩阵参数支持按行 top-k（保留每行一定比例），对小张量保证至少保留元素
    - 支持是否用 weights 归一化（normalize_by_weights）
    """
    if layers is None:
        layers = list(range(32))

    num_models = len(state_dicts_list)
    if num_models < 1:
        raise ValueError("至少需要一个微调模型的 state_dict。")

    if model_weights is None:
        model_weights = [0.5] * num_models
    if density_list is None:
        density_list = [1.0] * num_models

    merged_state_dict = {}
    # 计算权重和用于归一化
    total_weight = sum(model_weights) if normalize_by_weights else 1.0

    for key, base_val in base_state_dict.items():
        # 是否匹配需要合并的模块
        if any(rewrite_module_tmp.format(layer) in key for layer in layers):
            base_tensor = base_val.to(compute_dtype).clone()
            trimmed_weighted_taus = []
            for i, state_dict in enumerate(state_dicts_list):
                if key not in state_dict:
                    raise ValueError(f"键 {key} 在第 {i} 个 state_dict 中缺失。")
                other = state_dict[key].to(compute_dtype)
                tau = other - base_tensor
                density = density_list[i]
                weight = model_weights[i]

                abs_tau = torch.abs(tau)
                numel = abs_tau.numel()

                # 若是矩阵并要求按行topk，则对每行计算 topk
                if per_row_topk_for_matrix and tau.ndim >= 2:
                    # 以最后一维为行内维度（通常 linear weights: [out, in]）
                    last_dim = tau.shape[-1]
                    keep_per_row = max(keep_min_per_row, int(round(density * last_dim)))
                    if keep_per_row >= last_dim:
                        trimmed_tau = tau  # 全部保留
                    else:
                        # reshape to (..., rows, last_dim) — 这里按最后一维做 topk 保留
                        reshaped = tau.view(-1, last_dim)  # (R, last_dim)
                        abs_reshaped = torch.abs(reshaped)
                        # 每行 topk
                        k = keep_per_row
                        # 如果 k == 0 handled by keep_min_per_row
                        topk_vals, topk_idx = torch.topk(abs_reshaped, k=k, dim=1, largest=True, sorted=False)
                        mask = torch.zeros_like(reshaped, dtype=torch.bool)
                        rows = reshaped.shape[0]
                        # 构建 mask
                        idx = topk_idx.view(-1)
                        # create flat mask
                        flat_mask = mask.view(-1)
                        # compute offsets for each row
                        offsets = (torch.arange(rows, device=reshaped.device) * last_dim).repeat_interleave(k)
                        flat_mask[idx + offsets] = True
                        mask = flat_mask.view_as(reshaped)
                        mask = mask.view_as(reshaped)
                        trimmed = reshaped * mask.to(dtype=reshaped.dtype)
                        trimmed_tau = trimmed.view_as(tau)
                else:
                    # 全展平 topk
                    keep_num = max(keep_min_per_row, int(round(density * numel)))
                    if keep_num >= numel:
                        trimmed_tau = tau
                    else:
                        flat_abs = abs_tau.view(-1)
                        _, topk_idx = torch.topk(flat_abs, keep_num, sorted=False)
                        mask = torch.zeros_like(flat_abs, dtype=torch.bool)
                        mask[topk_idx] = True
                        mask = mask.view_as(tau)
                        trimmed_tau = tau * mask.to(dtype=tau.dtype)

                # 乘以该模型的权重（合并前加权）
                trimmed_weighted_taus.append(trimmed_tau * weight)

            # 合并：按权重求和并可选归一化
            summed = sum(trimmed_weighted_taus)
            if normalize_by_weights and total_weight != 0:
                summed = summed / total_weight

            # 最终缩放因子 lambda_merge
            summed = summed * lambda_merge

            merged = base_tensor + summed
            # 转回原 dtype（比如 base_val.dtype 可能是 float16）
            merged = merged.to(base_val.dtype)
            merged_state_dict[key] = merged
        else:
            merged_state_dict[key] = base_val
    return merged_state_dict


def merge_with_task_arithmetic(base_state_dict, state_dicts_list, lambda_merge=0.5, density=0.1, layers=None, rewrite_module_tmp="layers.{}.mlp"):
    """内存中Task Arithmetic融合，支持多个模型，只对指定模块的参数应用Task Arithmetic，其余保持base_state_dict不变（CPU上逐键处理）。
    density控制保留top-k比例的变化参数（类似于topk_ratio）。"""
    if layers is None:
        layers = list(range(32))  # 假设Llama2-7B有32层，根据实际模型调整
    
    if len(state_dicts_list) < 1:
        raise ValueError("至少需要一个微调模型的state_dict进行融合。")
    
    num_models = len(state_dicts_list)
    
    merged_state_dict = {}
    for key in base_state_dict.keys():
        if any(rewrite_module_tmp.format(layer) in key for layer in layers):
            base_tensor = base_state_dict[key]
            taus = []
            for state_dict in state_dicts_list:
                if key in state_dict:
                    tau = state_dict[key] - base_tensor
                    taus.append(tau)
                else:
                    raise ValueError(f"键 {key} 在某个state_dict中缺失。")
            
            if not taus:
                merged_state_dict[key] = base_tensor.clone()
                continue
            
            # Trim每个任务向量：保留density比例的最大幅度参数（top-k基于绝对值）
            trimmed_taus = []
            for tau in taus:
                abs_tau = torch.abs(tau)
                num_elements = abs_tau.numel()
                keep_num = int(density * num_elements)
                if keep_num == 0:
                    trimmed_taus.append(torch.zeros_like(tau))
                    continue
                # 获取top-k索引（变化最大的）
                flat_abs = abs_tau.view(-1)
                _, topk_indices = torch.topk(flat_abs, keep_num, sorted=False)
                # 创建mask
                mask = torch.zeros_like(flat_abs, dtype=torch.bool)
                mask[topk_indices] = True
                mask = mask.view_as(tau)
                trimmed_tau = tau * mask.float()
                trimmed_taus.append(trimmed_tau)
            
            # Task Arithmetic：线性求和（等权）
            summed_tau = sum(trimmed_taus) / num_models  # 等权平均
            
            # 应用缩放
            summed_tau *= lambda_merge
            
            # 最终参数 = base + 求和后的任务向量
            merged_tensor = base_tensor + summed_tau
            
            merged_state_dict[key] = merged_tensor
            del base_tensor, taus, trimmed_taus, summed_tau  # 释放内存
        else:
            merged_state_dict[key] = base_state_dict[key]
    return merged_state_dict

def fine_tune_in_memory(base_model_path, dataset):
    """内存中微调，返回state_dict（不保存到磁盘）"""
    model = AutoModelForCausalLM.from_pretrained(base_model_path, torch_dtype=torch.float16, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    
    training_args = TrainingArguments(
        output_dir="./temp",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        save_strategy="no",
        # ... 其他参数，根据需要调整
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
    )
    trainer.train()
    
    state_dict = model.state_dict()
    model.to("cpu")
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return state_dict, tokenizer

# 主流程：内存中微调多个模型，然后直接融合
# base_model_path = "meta-llama/Llama-2-7b-hf"

# # 加载base state_dict（在CPU上，以节省GPU内存）
# base_model = AutoModelForCausalLM.from_pretrained(base_model_path, torch_dtype=torch.float16, device_map="cpu")
# base_state_dict = base_model.state_dict()
# del base_model  # 释放临时模型
# torch.cuda.empty_cache() if torch.cuda.is_available() else None

# 假设多个数据集（替换为实际）
# state_dicts_list = []
# for dataset in datasets:  # datasets是你的多个数据集列表
#     state_dict, _ = fine_tune_in_memory(base_model_path, dataset)
#     state_dicts_list.append(state_dict)
# tokenizer = AutoTokenizer.from_pretrained(base_model_path)  # 只需加载一次

# 示例：融合多个state_dict，并指定layers和rewrite_module_tmp（根据hparams调整）
# merged_state_dict = merge_with_task_arithmetic(base_state_dict, state_dicts_list, lambda_merge=0.5, density=0.1, layers=list(range(32)), rewrite_module_tmp="layers.{}.mlp")

# 加载融合后的模型
# merged_model = AutoModelForCausalLM.from_pretrained(base_model_path, state_dict=merged_state_dict, torch_dtype=torch.float16, device_map="cpu")

# 可选：保存最终模型
# output_dir = "/path/to/merged_model"
# os.makedirs(output_dir, exist_ok=True)
# merged_model.save_pretrained(output_dir)
# tokenizer.save_pretrained(output_dir)

# print("融合过程已集成到代码中，使用Task Arithmetic方法，支持多个模型融合，并保持top-k保留程度（通过density控制）。")
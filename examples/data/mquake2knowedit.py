# import json

# def convert_mquake_to_easyedit(input_file, output_file):
#     with open(input_file, 'r', encoding='utf-8') as f:
#         mquake_data = json.load(f)

#     easyedit_data = []
    
#     for item in mquake_data:
#         rewrite = item['requested_rewrite'][0] # 获取编辑项
        
#         # 构造 EasyEdit 格式
#         entry = {
#             "prompt": rewrite['prompt'].format(rewrite['subject']),
#             "subject": rewrite['subject'],
#             "target_new": rewrite['target_new'],
#             "target_true": rewrite['target_true'],
#             # 将 MQUAKE 的多步推理题目放入 portability 模块
#             "portability": {
#                 "multihop": {
#                     "prompt": item['questions'],  # MQUAKE 包含多个问题列表
#                     "ground_truth": [item['new_answer']] * len(item['questions'])
#                 }
#             },
#             # 如果需要，可以加入 locality 测试（MQUAKE-CF 也有相关字段）
#             "locality": {} 
#         }
#         easyedit_data.append(entry)

#     with open(output_file, 'w', encoding='utf-8') as f:
#         json.dump(easyedit_data, f, indent=4, ensure_ascii=False)
#     print(f"转换完成，保存至: {output_file}")

# # 使用示例
# convert_mquake_to_easyedit('/EasyEdit/examples/data/MQuAKE-CF-3k.json', '/EasyEdit/examples/data/mquake_easyedit.json')


import json
import os

def convert_mquake_simple(input_file, output_file):
    if not os.path.exists(input_file):
        print(f"找不到输入文件: {input_file}")
        return

    with open(input_file, 'r', encoding='utf-8') as f:
        mquake_data = json.load(f)

    converted_data = []

    for item in mquake_data:
        # 获取核心编辑信息
        subjects = []
        prompts = []
        target_new_strs = []
        target_true_strs = []

        for rewrite in item['requested_rewrite']:
            subject = rewrite.get('subject', "")
            subjects.append(subject)
            
            # 确保 target_new 和 target_true 是纯字符串
            target_new_strs.append(rewrite['target_new']['str'])
            target_true_strs.append(rewrite['target_true']['str'])
            
            # prompts.append(rewrite['prompt'].format(subject) if "{}" in rewrite['prompt'] else rewrite['prompt'])
            prompts.append(rewrite["question"])

        # 构造 multihop 推理列表
        multihop_list = []
        single_hop_list = []
        # MQUAKE 的推理答案
        final_answer = item.get('new_answer', "")
        final_answer_alias = item.get("new_answer_alias","")
        sin_hops = item["new_single_hops"]
        for i,hop in enumerate(sin_hops):
            single_hop_list.append({
                "prompt": hop["question"],
                "ground_truth": [[hop["answer"]]+hop["answer_alias"]] # 评测器匹配需要嵌套列表 +hop["answer_alias"]
            })
        for q in item.get('questions', []):
            multihop_list.append({
                "prompt": q,
                "ground_truth": [[final_answer]+final_answer_alias] # 评测器匹配需要嵌套列表
            })

        # 构造简化后的 EasyEdit 格式
        entry = {
            "case_id": item['case_id'],
            "subject": subjects,
            "prompt": prompts,
            "target_new": target_new_strs,
            "ground_truth": target_true_strs,
            "portability": {
                "multihop": multihop_list,
                "single_hop": single_hop_list
            },
            "locality" : {},
            "rephrase": "",
            "context": []
            
        }
        converted_data.append(entry)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(converted_data, f, indent=4, ensure_ascii=True)
    
    print(f"成功转换 {len(converted_data)} 条数据至: {output_file}")

# 执行转换
# convert_mquake_simple('MQUAKE-CF.json', 'mquake_easyedit_simple.json')
# convert_mquake_simple('/EasyEdit/examples/data/MQuAKE-CF-3k.json', '/EasyEdit/examples/data/mquake_easyedit_3k_1_add_alias.json')
convert_mquake_simple('/EasyEdit/examples/data/MQuAKE-T.json', '/EasyEdit/examples/data/mquake_easyedit_T_1_add_alias.json')

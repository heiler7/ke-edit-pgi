import json

# with open('/EasyEdit/examples/data/MQuAKE-CF-3k.json', 'r', encoding='utf-8') as f:
#     mquake_t = json.load(f)

# with open('/EasyEdit/examples/data/MQuAKE-CF-3k_merged_graph_expect_withouttest_api_summarys.json', 'r', encoding='utf-8') as f:
#     merged_graph = json.load(f)

with open('/EasyEdit/examples/data/MQuAKE-T.json', 'r', encoding='utf-8') as f:
    mquake_t = json.load(f)

with open('/EasyEdit/examples/data/MQuAKE-T_merged_graph_expect_withouttest_api_summarys.json', 'r', encoding='utf-8') as f:
    merged_graph = json.load(f)

# with open('/EasyEdit/examples/data/MQuAKE-T.json', 'r', encoding='utf-8') as f:
#     mquake_t = json.load(f)

# with open('/EasyEdit/examples/data/MQuAKE-T_merged_graph_expect_withouttest_api_summarys.json', 'r', encoding='utf-8') as f:
#     merged_graph = json.load(f)

all_new_triples_labeled = []
triple_to_cases = {}
all_required_edit_len = []
for item in mquake_t:
    case_id = item['case_id']
    orig = item.get('orig', {})
    triples = orig.get('new_triples', [])
    new_triples_labeled = orig.get('new_triples_labeled', [])
    edit_triples = orig.get('edit_triples', [])
    required_edit_len = len(edit_triples)
    all_required_edit_len.append(required_edit_len)
    
    edit_indices = set()
    for edit_triple in edit_triples:
        if edit_triple in triples:
            edit_idx = triples.index(edit_triple)
            edit_indices.add(edit_idx)
    
    for idx, triple in enumerate(new_triples_labeled):
        if idx in edit_indices:
            continue
        if len(triple) >= 3:
            key = (triple[0], triple[1], triple[2])
            all_new_triples_labeled.append(key)
            if key not in triple_to_cases:
                triple_to_cases[key] = []
            triple_to_cases[key].append(case_id)

triples_removed_detail = {}

filtered_case_ids = []

for i,item in enumerate(merged_graph):
    case_id = item['case_id']
    triples = item.get('triples', [])

    triples_to_remove_by_subject_relation = []
    for idx, triple in enumerate(triples):
        s, r, o = triple.get('subject', ''), triple.get('relation', ''), triple.get('target', '')
        for all_s, all_r, all_o in all_new_triples_labeled:
            if all_s == s and all_r == r :
                triples_to_remove_by_subject_relation.append(idx)
                key = (s, r, o)
                if key not in triples_removed_detail:
                    conflict_cases = [c for c in triple_to_cases.get((all_s, all_r, all_o), [])]
                    triples_removed_detail[key] = conflict_cases
                break

    all_indices_to_remove = set(triples_to_remove_by_subject_relation)

    actually_removed = []
    for idx in sorted(all_indices_to_remove, reverse=True):
        if idx < len(triples) and idx >= all_required_edit_len[i]:
            removed = triples.pop(idx)
            actually_removed.append(idx)
            print(f"Case {case_id}: 移除三元组 - {removed['subject']} | {removed['relation']} | {removed['target']}")

    if actually_removed:
        filtered_case_ids.append(case_id)

# print("\n被移除三元组所在的new_triples_labeled case列表：")
# for (s, r, o), cases in triples_removed_detail.items():
#     print(f"  {s} | {r} | {o} -> 出现在 Case {cases}")

# 保存修改后的文件
output_path = '/EasyEdit/examples/data/MQuAKE-T_merged_graph_expect_withouttest_api_filtered_5.json'
# output_path = '/EasyEdit/examples/data/MQuAKE-3k_merged_graph_expect_withouttest_api_filtered_5.json'
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(merged_graph, f, ensure_ascii=False, indent=2)

print(f"\n处理完成，已保存到: {output_path}")

print(f"\n被filter的case_id列表: {filtered_case_ids}")
print(f"总共 {len(filtered_case_ids)} 个case被filter")
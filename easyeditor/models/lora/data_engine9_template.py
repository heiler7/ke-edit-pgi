import networkx as nx
import random
import logging
from typing import List, Dict, Tuple, Any, Optional
from transformers import PreTrainedTokenizer
import numpy as np


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_GEN_MAX_LENGTH = 1000
DEFAULT_MAX_PATHS = 5    
DEFAULT_MAX_DEPTH = 4    
DEFAULT_MAX_DEGREE = 15  
MAX_RELEVANT_PATHS = 3   
MAX_CONTEXT_QA_PER_TRIPLE = 1    
MAX_REASONING_QA_PER_PATH = 1    
MAX_TOTAL_AUGMENTED_SAMPLES = 25

class KnowledgeGraphWrapper:
    def __init__(self, triples_data: Optional[List[List[Dict[str, str]]]] = None, random_seed: int = 42):
        self.graph = nx.DiGraph()
        self.random_seed = random_seed
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)

        if triples_data:
            self.add_triples(triples_data)

    def add_triples(self, triples_data: List[List[Dict[str, str]]]):
        if not isinstance(triples_data, list): return
        total_added = 0
        for sublist in triples_data:
            if not isinstance(sublist, list): continue
            for triple in sublist:
                h, r, t = triple.get("subject"), triple.get("relation"), triple.get("target")
                if all([h, r, t]):
                    self.graph.add_edge(h.strip(), t.strip(), relation=r.strip().lower())
                    total_added += 1

    def get_diverse_paths(self, start_node: str, num_paths: int = DEFAULT_MAX_PATHS, max_depth: int = DEFAULT_MAX_DEPTH, max_degree: int = DEFAULT_MAX_DEGREE) -> List[List[Tuple[str, str, str]]]:
        start_node = start_node.strip()
        if start_node not in self.graph: return[]

        paths =[]
        for _ in range(num_paths * 5):
            back_depth = random.randint(0, max_depth)
            fwd_depth = max_depth - back_depth
            
            back_path =[]
            curr_back = start_node
            for _ in range(back_depth):
                preds = list(self.graph.predecessors(curr_back))
                if not preds: break  
                if len(preds) > max_degree:
                    preds = random.sample(preds, max_degree)
                
                prev_node = random.choice(preds)
                rel = self.graph.get_edge_data(prev_node, curr_back).get('relation', 'related to')
                back_path.insert(0, (prev_node, rel, curr_back))
                curr_back = prev_node

            fwd_path =[]
            curr_fwd = start_node
            for _ in range(fwd_depth):
                succs = list(self.graph.successors(curr_fwd))
                if not succs: break  
                if len(succs) > max_degree:
                    succs = random.sample(succs, max_degree)
                
                next_node = random.choice(succs)
                rel = self.graph.get_edge_data(curr_fwd, next_node).get('relation', 'related to')
                fwd_path.append((curr_fwd, rel, next_node))
                curr_fwd = next_node

            full_path = back_path + fwd_path
            if full_path and full_path not in paths:
                paths.append(full_path)
            if len(paths) >= num_paths: 
                break
                
        paths.sort(key=len, reverse=True)
        return paths
    
class EditPGIDataEngine:
    def __init__(self, kg_wrapper: KnowledgeGraphWrapper, tokenizer: Optional[PreTrainedTokenizer] = None, gen_max_length: int = DEFAULT_GEN_MAX_LENGTH):
        self.kg = kg_wrapper
        self.tokenizer = tokenizer
        self.gen_max_length = gen_max_length
        # 加载关系模板
        import json
        import os
        template_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'relation_templates1.json')
        try:
            with open(template_path, 'r', encoding='utf-8') as f:
                self.relation_templates = json.load(f)
        except FileNotFoundError:
            print(f"⚠️ Relation templates file not found at {template_path}, using default templates")
            self.relation_templates = {
                "default": {
                    "one_hop": [
                        "What is the {relation} of {subject}?",
                        "{subject} has what {relation}?",
                        "Which {relation} is associated with {subject}?",
                        "What {relation} does {subject} have?"
                    ],
                    "multi_hop": [
                        "the {relation} of {subject}"
                    ]
                }
            }

    def _truncate(self, text: str) -> str:
        if not self.tokenizer: return text[:self.gen_max_length]
        encoded = self.tokenizer(text, truncation=True, max_length=self.gen_max_length, padding=False, return_tensors="pt")
        return self.tokenizer.decode(encoded["input_ids"][0], skip_special_tokens=True)

    def _assemble_linear_paths(self, triples: List[Dict], core_triples_count: int) -> List[List[Tuple[str, str, str]]]:
        if not triples: return []
        mini_kg = nx.DiGraph()
        all_edges =[]
        for t in triples:
            s, r, tg = t.get("subject"), t.get("relation"), t.get("target")
            if s and tg and s != tg:
                mini_kg.add_edge(s.strip(), tg.strip(), relation=r.strip().lower())
                all_edges.append((s.strip(), tg.strip()))
        
        core_edges = set(all_edges[:core_triples_count])
        target_edges = set(all_edges)
        candidate_paths =[]
        
        consecutive_path = []
        if core_triples_count > 1:
            core_triples = triples[:core_triples_count]

            current_path = []
            for i, triple in enumerate(core_triples):
                s = triple.get("subject").strip()
                r = triple.get("relation").strip().lower()
                t = triple.get("target").strip()
                
                if i == 0:
                    current_path.append((s, r, t))
                else:

                    if current_path[-1][2] == s:
                        current_path.append((s, r, t))
                    else:
                        break
            
            if len(current_path) == core_triples_count:
                consecutive_path = current_path
        

        for u, v in core_edges:
            if not mini_kg.has_edge(u, v): continue
            in_paths = [[u]]
            for node in mini_kg.nodes():
                if node != u:
                    for p in nx.all_simple_paths(mini_kg, source=node, target=u, cutoff=2):
                        in_paths.append(p)
            out_paths = [[v]]
            for node in mini_kg.nodes():
                if node != v:
                    for p in nx.all_simple_paths(mini_kg, source=v, target=node, cutoff=2):
                        out_paths.append(p)
            
            for in_p in in_paths:
                for out_p in out_paths:
                    full_node_path = in_p + out_p
                    edge_path =[]
                    is_valid = True
                    for i in range(len(full_node_path)-1):
                        n1, n2 = full_node_path[i], full_node_path[i+1]
                        if mini_kg.has_edge(n1, n2):
                            rel = mini_kg[n1][n2]['relation']
                            edge_path.append((n1, rel, n2))
                        else:
                            is_valid = False
                            break
                    if is_valid and edge_path and edge_path not in candidate_paths:
                        candidate_paths.append(edge_path)
                        
        candidate_paths.sort(key=len, reverse=True)
        final_paths =[]
        covered_edges = set()
        

        if consecutive_path:
            final_paths.append(consecutive_path)
            covered_edges.update((u, v) for u, r, v in consecutive_path)
        
        for path in candidate_paths:
            if consecutive_path and path == consecutive_path:
                continue
                
            path_edges = set((u, v) for u, r, v in path)
            if not path_edges.issubset(covered_edges):
                final_paths.append(path)
                covered_edges.update(path_edges)
            if target_edges.issubset(covered_edges):
                break
                
        return final_paths

    def generate_EditPGI_requests(self, requests: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if isinstance(requests, dict):
            requests = [requests]
            
        enhanced_requests = []
        subjects =[]
        
        for req in requests:
            req["task_type"] = "Standard_Edit"
            enhanced_requests.append(req)
            subjects.append(req["subject"])
        
        if not requests:
            return enhanced_requests
            
        first_req = requests[0]
        subject = first_req.get("subject", "").strip()
        target_new = first_req.get("target_new", "").strip()
        prompt = first_req.get("prompt", "").strip()
        req_triples = first_req.get("triples",[])
        summary = first_req.get("summary","")
        # summary = ""
        if not target_new or not prompt:
            return enhanced_requests
        
        augmented_count = 0
            
        

        core_triples_count = len(requests)
        paths = self._assemble_linear_paths(req_triples, core_triples_count)
        
        injected_edges = set() 
        for path in paths:
            # if path[0][0] == test_sub and path[-1][-1] == test_ans and len(path) == len(test_triples):
            #     if len(path) > 2:
            #         sub_path_no_first = path[1:]
            #         sub_path_no_last = path[:-1]
            #         paths.append(sub_path_no_first)
            #         paths.append(sub_path_no_last)
            #     continue
            # print(path)    
            if augmented_count >= MAX_TOTAL_AUGMENTED_SAMPLES:
                break
                
            for u, rel, v in path:
                injected_edges.add((u, rel, v))

            if len(path) >= 2:
                bridge_entity = path[0][2] 
                bridge_prompt = f"To determine the final answer for '{prompt}', what is the essential intermediate bridge entity we must identify first?"
                enhanced_requests.append({
                    "prompt": self._truncate(bridge_prompt),
                    "target_new": self._truncate(bridge_entity),
                    "subject": subject,
                    "task_type": "Bridge_Anchoring"
                })
                augmented_count += 1
                

                final_target = path[-1][2]
                final_rel = path[-1][1]
                
                if len(path) == 2:
                    rel1 = path[0][1]
                    entity1 = path[0][0]
                    
                    if rel1 in self.relation_templates:
                        templates = self.relation_templates[rel1]["multi_hop"]
                        selected_template = templates[0]
                        if "{subject}" in selected_template:
                            prefix = selected_template.format(subject=entity1)
                        else:
                            prefix = entity1
                            if rel1.endswith("of"):
                                prefix = f"the {rel1} {prefix}"
                            else:
                                prefix = f"the {rel1} of {prefix}"
                    else:


                        templates = self.relation_templates["default"]["multi_hop"]
                        selected_template = templates[0]
                        prefix = selected_template.format(relation=rel1, subject=entity1)
                    

                    if final_rel in self.relation_templates:
                        templates = self.relation_templates[final_rel]["one_hop"]
                        selected_template = templates[0]
                        reasoning_prompt = selected_template.format(subject=prefix)
                    else:

                        templates = self.relation_templates["default"]["one_hop"]
                        selected_template = templates[0]
                        reasoning_prompt = selected_template.format(relation=final_rel, subject=prefix)
                        
                elif len(path) == 3:

                    entity1 = path[0][0]
                    rel1 = path[0][1]
                    rel2 = path[1][1]
                    

                    if rel1 in self.relation_templates:
                        templates = self.relation_templates[rel1]["multi_hop"]
                        selected_template = templates[0]
                        if "{subject}" in selected_template:
                            first_prefix = selected_template.format(subject=entity1)
                        else:
                            first_prefix = entity1
                            if rel1.endswith("of"):
                                first_prefix = f"the {rel1} {first_prefix}"
                            else:
                                first_prefix = f"the {rel1} of {first_prefix}"
                    else:
                        print(rel1)
                        templates = self.relation_templates["default"]["multi_hop"]
                        selected_template = templates[0]
                        first_prefix = selected_template.format(relation=rel1, subject=entity1)
                    

                    if rel2 in self.relation_templates:
                        templates = self.relation_templates[rel2]["multi_hop"]
                        selected_template = templates[0]
                        if "{subject}" in selected_template:
                            prefix = selected_template.format(subject=first_prefix)
                        else:
                            prefix = first_prefix
                            if rel2.endswith("of"):
                                prefix = f"the {rel2} {prefix}"
                            else:
                                prefix = f"the {rel2} of {prefix}"
                    else:
                        print(rel2)
                        templates = self.relation_templates["default"]["multi_hop"]
                        selected_template = templates[0]
                        prefix = selected_template.format(relation=rel2, subject=first_prefix)
                    

                    if final_rel in self.relation_templates:
                        templates = self.relation_templates[final_rel]["one_hop"]
                        selected_template = templates[0]
                        reasoning_prompt = selected_template.format(subject=prefix)
                    else:
                        print(final_rel)
                        templates = self.relation_templates["default"]["one_hop"]
                        selected_template = templates[0]
                        reasoning_prompt = selected_template.format(relation=final_rel, subject=prefix)
                        
                else:

                    prefix = path[0][0]
                    for i in range(len(path) - 1): 
                        p_rel = path[i][1]
                        if p_rel in self.relation_templates:
                            templates = self.relation_templates[p_rel]["multi_hop"]
                            selected_template = templates[0]
                            if "{subject}" in selected_template:
                                prefix = selected_template.format(subject=prefix)
                            else:
                                if p_rel.endswith("of"):
                                    prefix = f"the {p_rel} {prefix}"
                                else:
                                    prefix = f"the {p_rel} of {prefix}"
                        else:
                            print(p_rel)
                            templates = self.relation_templates["default"]["multi_hop"]
                            selected_template = templates[0]
                            prefix = selected_template.format(relation=p_rel, subject=prefix)
                    

                    if final_rel in self.relation_templates:
                        templates = self.relation_templates[final_rel]["one_hop"]
                        selected_template = templates[0]
                        reasoning_prompt = selected_template.format(subject=prefix)
                    else:
                        print(final_rel)
                        templates = self.relation_templates["default"]["one_hop"]
                        selected_template = templates[0]
                        reasoning_prompt = selected_template.format(relation=final_rel, subject=prefix)
                
                enhanced_requests.append({
                    "prompt": self._truncate(reasoning_prompt.strip()),
                    "target_new": self._truncate(final_target),
                    "subject": subject,
                    "task_type": "Reasoning_QA"
                })
                augmented_count += 1
                
                if augmented_count >= MAX_TOTAL_AUGMENTED_SAMPLES:
                    break

       
        if injected_edges and augmented_count < MAX_TOTAL_AUGMENTED_SAMPLES:
            if summary == "":
                summary_sentences =[]
                for i, (u, rel, v) in enumerate(injected_edges):
                    if i == 0:
                        summary_sentences.append(f"To begin with, it is established that the {rel} of {u} is {v}.")
                    else:
                        connectors =["Furthermore,", "Additionally,", "By the way,", "In relation to this,", "Moreover,", "We also note that", "Finally,"]
                        summary_sentences.append(f"{random.choice(connectors)} {u} connects to {v} through the '{rel}' relationship.")
                
                summary = f"Summary: {' '.join(summary_sentences)}"
                
            enhanced_requests.append({
                "prompt": self._truncate(f"Summarize and recite the most relevant knowledge subgraph for {subject}: {injected_edges}"),
                "target_new": self._truncate(summary),
                "subject": subject,
                "task_type": "Subgraph_Summarization"
            })
            augmented_count += 1
        # Context_Memorization
        unique_triples = set()
        triples_list =[]
        for t in req_triples:
            s, r, tg = t.get("subject"), t.get("relation"), t.get("target")
            if s and tg and (s, r, tg) not in unique_triples:
                unique_triples.add((s, r, tg))
                triples_list.append((s, r, tg))
        
        for s, r, tg in triples_list:
            if augmented_count >= MAX_TOTAL_AUGMENTED_SAMPLES:
                break
            templates_c = self.relation_templates[r]["one_hop"]
            selected_template_c = templates_c[0]
            context_prompt = selected_template_c.format(subject=s)
            enhanced_requests.append({
                # "prompt": self._truncate(context_prompt.strip()),
                # "prompt": self._truncate(f"Based on your internal memory, what is the {r} of {s}?"),
                "prompt": self._truncate(f"Based on your internal memory, the node {s} has the relation {r} with the node"),
                "target_new": self._truncate(tg),
                "subject": subject,
                "task_type": "Context_Memorization"
            })

            # enhanced_requests.append({
            #     # "prompt": self._truncate(context_prompt.strip()),
            #     # "prompt": self._truncate(f"Based on your internal memory, what is the {r} of {s}?"),
            #     "prompt": self._truncate(f"Based on your internal memory, the node {s} and the node {tg} have the relation"),
            #     "target_new": self._truncate(r),
            #     "subject": subject,
            #     "task_type": "Context_Memorization"
            # })
            augmented_count += 1

        

        return enhanced_requests
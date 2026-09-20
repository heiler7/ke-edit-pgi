import json
import os
import re
import time
import socket
import urllib.request
import urllib.parse
from urllib.error import HTTPError, URLError
from SPARQLWrapper import SPARQLWrapper, JSON
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


os.environ["http_proxy"] = "http://127.0.0.1:7890"
os.environ["https_proxy"] = "http://127.0.0.1:7890"

wikidata_url = "https://query.wikidata.org/sparql"
# User-Agent
user_agent = "MyGraphExtractionBot/1.0 (mailto:xxx@gmail.com) Python/urllib"
sparql = SPARQLWrapper(wikidata_url, agent=user_agent)
sparql.setTimeout(120)
global_datastr_name = ""

http_session = requests.Session()
http_session.proxies = {"http": "http://127.0.0.1:7890", "https": "http://127.0.0.1:7890"}
http_session.headers.update({'User-Agent': user_agent})
retry_strategy = Retry(total=5, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
adapter = HTTPAdapter(max_retries=retry_strategy, pool_connections=20, pool_maxsize=20)
http_session.mount("https://", adapter)
http_session.mount("http://", adapter)


BLACKLIST_PROPERTIES = "wdt:P31, wdt:P214, wdt:P225, wdt:P244, wdt:P213, wdt:P269, wdt:P345, wdt:P646, wdt:P1566, wdt:P910, wdt:P1423, wdt:P373, wdt:P1343, wdt:P530, wdt:P161, wdt:P460, wdt:P421, wdt:P47, wdt:P150, wdt:P1906, wdt:P1313, wdt:P1151, wdt:P1204, wdt:P2633, wdt:P194, wdt:P5008, wdt:P2184, wdt:P279, wdt:P361, wdt:P1269, wdt:P171, wdt:P212, wdt:P580, wdt:P582, wdt:P18, wdt:P734, wdt:P1344, wdt:P854, wdt:P248, wdt:P813, wdt:P143, wdt:P2581, wdt:P1332, wdt:P1465, wdt:P8744, wdt:P1304, wdt:P562,wdt:P9241, wdt:P1464, wdt:P1465, wdt:P2853, wdt:P208, wdt:P2596, wdt:P2852, wdt:P1365, wdt:P237, wdt:P1423, wdt:P1424, wdt:P163, wdt:P1792, wdt:P102, wdt:P1791, wdt:P1740, wdt:P5125, wdt:P7867, wdt:P10280, wdt:P8402, wdt:P1889"
# WHITELIST_PROPERTIES = "wdt:P6, wdt:P17, wdt:P19, wdt:P20, wdt:P26, wdt:P27, wdt:P30, wdt:P35, wdt:P36, wdt:P37, wdt:P39, wdt:P40, wdt:P50, wdt:P69, wdt:P101, wdt:P103, wdt:P106, wdt:P108, wdt:P112, wdt:P127, wdt:P131, wdt:P136, wdt:P138, wdt:P140, wdt:P159, wdt:P169, wdt:P170, wdt:P175, wdt:P176, wdt:P178, wdt:P190, wdt:P264, wdt:P276, wdt:P286, wdt:P364, wdt:P407, wdt:P413, wdt:P449, wdt:P463, wdt:P488, wdt:P495, wdt:P641, wdt:P740, wdt:P800, wdt:P937, wdt:P1037, wdt:P1303, wdt:P1308, wdt:P1412"
WHITELIST_PROPERTIES = "wdt:P6, wdt:P19, wdt:P20, wdt:P26, wdt:P27, wdt:P30, wdt:P35, wdt:P36, wdt:P37, wdt:P40, wdt:P50, wdt:P69, wdt:P106, wdt:P108, wdt:P112, wdt:P136, wdt:P140, wdt:P159, wdt:P169, wdt:P170, wdt:P175, wdt:P176, wdt:P178, wdt:P286, wdt:P364, wdt:P407, wdt:P413, wdt:P449, wdt:P488, wdt:P495, wdt:P641, wdt:P740, wdt:P800, wdt:P937, wdt:P1037, wdt:P1308, wdt:P1412"


def safe_sparql_request(sparql, max_retries=5):
    retries = 0
    while retries < max_retries:
        try:
            results = sparql.query().convert()
            time.sleep(0.2) 
            return results
        except HTTPError as e:
            wait_time = 30 if e.code == 429 else 5
            print(f"[网络] HTTP {e.code}，暂停 {wait_time} 秒后重试... ({retries+1}/{max_retries})")
            time.sleep(wait_time)
        except Exception as e:
            print(f"[网络] 连接不稳定 ({e})，等待 5 秒后重试... ({retries+1}/{max_retries})")
            time.sleep(5)
        retries += 1
    return {"results": {"bindings": []}}

def get_wikidata_id_by_label(label: str, max_retries=3) -> str:
    wd_url = f"https://www.wikidata.org/w/api.php?action=wbsearchentities&search={urllib.parse.quote(label)}&language=en&format=json"
    for attempt in range(max_retries):
        try:
            response = http_session.get(wd_url, timeout=10)
            data = response.json()
            if data.get('search'): return data['search'][0]['id']
            break
        except Exception: time.sleep(2)
    return None

def fetch_label_by_id(qid: str) -> str:
    if not qid or not qid.startswith("Q") or not qid[1:].isdigit():
        return qid
    url = f"https://www.wikidata.org/w/api.php?action=wbgetentities&ids={qid}&props=labels|aliases&format=json"
    try:
        response = http_session.get(url, timeout=10)
        data = response.json()
        entities = data.get("entities", {})
        if qid in entities:
            labels = entities[qid].get("labels", {})
            aliases = entities[qid].get("aliases", {})
            if "en" in labels: return labels["en"]["value"]
            for lang in ["en-gb", "en-ca", "en-us"]:
                if lang in labels: return labels[lang]["value"]
            if "en" in aliases and len(aliases["en"]) > 0:
                return aliases["en"][0]["value"]
    except Exception as e:
        pass
    return qid

def get_property_label(property_id: str) -> str:
    queryString = f"""
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    SELECT ?propertyLabel WHERE {{
      wd:{property_id} rdfs:label ?propertyLabel.
      FILTER(LANG(?propertyLabel) = "en")
    }} LIMIT 1
    """
    sparql.setQuery(queryString)
    sparql.setReturnFormat(JSON)
    results = safe_sparql_request(sparql)
    if results and "results" in results and results['results']['bindings']:
        return results['results']['bindings'][0]['propertyLabel']['value']
    return property_id

def get_incoming_triples(item_id: str, fetch_limit: int = 500, sample_pool: int = 20000) -> dict:
    whitelist_items = [item.strip() for item in WHITELIST_PROPERTIES.split(",") if item.strip()]
    values_tuples =[]
    for item in whitelist_items:
        if item.startswith("wdt:"):
            prop_id = item.split(":")[1]  # 提取纯数字部分，如 P17
            # 构建四个维度的确切 URI，避免查询引擎在运行时去动态 Join
            values_tuples.append(f"({item} p:{prop_id} ps:{prop_id} wd:{prop_id})")
            
    values_matrix = " ".join(values_tuples)


    queryString = """
    SELECT ?s ?sLabel ?p ?pLabel WHERE {
      {
        SELECT ?s ?p ?wdProp ?sitelinks WHERE {
          {
            SELECT ?s ?p ?wdProp WHERE {
              
              VALUES (?p ?p_prop ?ps ?wdProp) { %s }
              
              
              ?stmt ?ps wd:%s .
              ?s ?p_prop ?stmt .
              
              
              FILTER(isIRI(?s)) .
              
              
              OPTIONAL { ?stmt pq:P580 ?startTime }
              OPTIONAL { ?stmt pq:P582 ?endTime }
              FILTER (
                (!BOUND(?startTime) || ?startTime <= "2019-12-31T23:59:59Z"^^xsd:dateTime) &&
                (!BOUND(?endTime) || ?endTime >= "2019-12-31T23:59:59Z"^^xsd:dateTime)
              )
            }
            LIMIT %d
          }
          
          OPTIONAL { ?s wikibase:sitelinks ?sitelinks }
        }
        ORDER BY DESC(COALESCE(?sitelinks, 0))
        LIMIT %d
      }
      
      SERVICE wikibase:label {
        bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en".
        ?s rdfs:label ?sLabel .
        ?wdProp rdfs:label ?pLabel .
      }
    }
    """ % (values_matrix, item_id, sample_pool, fetch_limit)
    
    sparql.setQuery(queryString)
    sparql.setReturnFormat(JSON)
    return safe_sparql_request(sparql)

def get_incoming_triples_q30(item_id: str, fetch_limit: int = 5000, sample_pool: int = 200000) -> dict:
    whitelist_items =[item.strip() for item in WHITELIST_PROPERTIES.split(",") if item.strip()]
    values_matrix = " ".join(whitelist_items)
    
    current_pool = sample_pool
    

    while current_pool >= 100:
        queryString = """
        SELECT ?s ?sLabel ?p ?pLabel WHERE {
          {
            SELECT ?s ?p ?sitelinks WHERE {
              {
                SELECT ?s ?p WHERE {
                  VALUES ?p { %s }
                  ?s ?p wd:%s .
                }
                LIMIT %d
              }
            
              OPTIONAL { ?s wikibase:sitelinks ?sitelinks }
            }
            ORDER BY DESC(?sitelinks)
            LIMIT %d
          }

          BIND(IRI(REPLACE(STR(?p), "prop/direct", "entity")) AS ?wdProp)
          
          SERVICE wikibase:label {
            bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en".
            ?s rdfs:label ?sLabel .
            ?wdProp rdfs:label ?pLabel .
          }
        }
        """ % (values_matrix, item_id, current_pool, fetch_limit)
        
        sparql.setQuery(queryString)
        sparql.setReturnFormat(JSON)
        
        try:

            result = sparql.query().convert()
            
            if current_pool < sample_pool:
                print(f"sample_pool={current_pool} ")
            return result
            
        except Exception as e:

            error_msg = str(e).lower()
            if "timeout" in error_msg or "timed out" in error_msg or "503" in error_msg or "500" in error_msg:
                current_pool = current_pool // 2  
                time.sleep(2) 
            else:
                print(f"Error {e}")
                break


    return {}

def get_neighbor_triples(item_id: str, fetch_limit: int = 500, sample_pool: int = 5000) -> dict:
    
    
    values_whitelist = WHITELIST_PROPERTIES.replace(",", " ")
    
    queryString = """
    SELECT ?p ?pLabel ?o ?oLabel WHERE {
      {
        SELECT ?p ?wdProp ?o WHERE {
          {
            SELECT ?p ?wdProp ?o WHERE {
              
              VALUES ?p { %s }
              
              
              ?wdProp wikibase:directClaim ?p ;
                      wikibase:claim ?p_prop ;
                      wikibase:statementProperty ?ps .
                      
              
              wd:%s ?p_prop ?stmt .
              ?stmt ?ps ?o .
              
              
              FILTER(STRSTARTS(STR(?o), STR(wd:))) .
              
             
              OPTIONAL { ?stmt pq:P580 ?startTime }
              OPTIONAL { ?stmt pq:P582 ?endTime }
              FILTER (
                (!BOUND(?startTime) || ?startTime <= "2019-12-31T23:59:59Z"^^xsd:dateTime) &&
                (!BOUND(?endTime) || ?endTime >= "2019-12-31T23:59:59Z"^^xsd:dateTime)
              )
            }
            LIMIT %d
          }
          OPTIONAL { ?o wikibase:sitelinks ?sitelinks }
        }
        ORDER BY DESC(COALESCE(?sitelinks, 0))
        LIMIT %d
      }
      SERVICE wikibase:label {
        bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en".
        ?wdProp rdfs:label ?pLabel .
        ?o rdfs:label ?oLabel .
      }
    }
    """ % (values_whitelist, item_id, sample_pool, fetch_limit)
    
    sparql.setQuery(queryString)
    sparql.setReturnFormat(JSON)
    return safe_sparql_request(sparql)


def init_triples_dict(subject, relation, obj) -> dict:
    return {"subject": subject, "relation": relation, "target": obj}



def get_relation_between(s_id: str, s_label: str, o_id: str, o_label: str) -> list:
    queryString = f"""
    SELECT ?p ?pLabel WHERE {{
      wd:{s_id} ?p_prop ?stmt .
      ?stmt ?ps wd:{o_id} .
      ?wdProp wikibase:claim ?p_prop .
      ?wdProp wikibase:statementProperty ?ps .
      ?wdProp wikibase:directClaim ?p .
      
      FILTER(?p IN ({WHITELIST_PROPERTIES}))
      
      OPTIONAL {{ ?stmt pq:P580 ?startTime }}
      OPTIONAL {{ ?stmt pq:P582 ?endTime }}
      FILTER (
        (!BOUND(?startTime) || ?startTime <= "2019-12-31T23:59:59Z"^^xsd:dateTime) &&
        (!BOUND(?endTime) || ?endTime >= "2019-12-31T23:59:59Z"^^xsd:dateTime)
      )
      
      SERVICE wikibase:label {{
        bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en".
        ?wdProp rdfs:label ?pLabel .
      }}
    }} LIMIT 3
    """
    sparql.setQuery(queryString)
    sparql.setReturnFormat(JSON)
    results = safe_sparql_request(sparql)
    
    triples = []
    if results and "results" in results and results['results']['bindings']:
        for result in results['results']['bindings']:
            p_uri = result.get('p', {}).get('value', '')
            if not p_uri: continue
            p_id = p_uri.split('/')[-1]
            p_label = result.get('pLabel', {}).get('value', p_id)
            triples.append(init_triples_dict(s_label, p_label, o_label))
    return triples

def get_intermediate_entity(s_id: str, s_label: str, o_id: str, o_label: str) -> list:
    
    queryString = f"""
    SELECT ?e ?eLabel ?p1 ?p1Label ?p2 ?p2Label WHERE {{
      # First hop: S -> E
      wd:{s_id} ?p1_prop ?stmt1 .
      ?stmt1 ?ps1 ?e .
      ?wdProp1 wikibase:claim ?p1_prop .
      ?wdProp1 wikibase:statementProperty ?ps1 .
      ?wdProp1 wikibase:directClaim ?p1 .
      
      # Second hop: E -> O
      ?e ?p2_prop ?stmt2 .
      ?stmt2 ?ps2 wd:{o_id} .
      ?wdProp2 wikibase:claim ?p2_prop .
      ?wdProp2 wikibase:statementProperty ?ps2 .
      ?wdProp2 wikibase:directClaim ?p2 .
      
      FILTER(?p1 IN ({WHITELIST_PROPERTIES}))
      FILTER(?p2 IN ({WHITELIST_PROPERTIES}))
      
      # Time filter for first hop
      OPTIONAL {{ ?stmt1 pq:P580 ?startTime1 }}
      OPTIONAL {{ ?stmt1 pq:P582 ?endTime1 }}
      FILTER (
        (!BOUND(?startTime1) || ?startTime1 <= "2019-12-31T23:59:59Z"^^xsd:dateTime) &&
        (!BOUND(?endTime1) || ?endTime1 >= "2019-12-31T23:59:59Z"^^xsd:dateTime)
      )
      
      # Time filter for second hop
      OPTIONAL {{ ?stmt2 pq:P580 ?startTime2 }}
      OPTIONAL {{ ?stmt2 pq:P582 ?endTime2 }}
      FILTER (
        (!BOUND(?startTime2) || ?startTime2 <= "2019-12-31T23:59:59Z"^^xsd:dateTime) &&
        (!BOUND(?endTime2) || ?endTime2 >= "2019-12-31T23:59:59Z"^^xsd:dateTime)
      )

      SERVICE wikibase:label {{
        bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en".
        ?e rdfs:label ?eLabel .
        ?wdProp1 rdfs:label ?p1Label .
        ?wdProp2 rdfs:label ?p2Label .
      }}
    }} LIMIT 3
    """
    sparql.setQuery(queryString)
    sparql.setReturnFormat(JSON)
    results = safe_sparql_request(sparql)
    
    triples = []
    if results and "results" in results and results['results']['bindings']:
        for result in results['results']['bindings']:
            e_uri = result.get('e', {}).get('value', '')
            if not e_uri: continue
            e_id = e_uri.split('/')[-1]
            e_label = result.get('eLabel', {}).get('value', e_id)
            
            p1_uri = result.get('p1', {}).get('value', '')
            p1_id = p1_uri.split('/')[-1]
            p1_label = result.get('p1Label', {}).get('value', p1_id)
            
            p2_uri = result.get('p2', {}).get('value', '')
            p2_id = p2_uri.split('/')[-1]
            p2_label = result.get('p2Label', {}).get('value', p2_id)
            
            triples.append(init_triples_dict(s_label, p1_label, e_label))
            triples.append(init_triples_dict(e_label, p2_label, o_label))
    return triples

def get_incoming_entity_triplets(target_id: str, target_label: str, max_same_rel: int = 2) -> list:
    if target_id == "Q30":
        results = {}
        # result = get_incoming_triples_q30(target_id, fetch_limit=60)
    else:
        results = get_incoming_triples(target_id, fetch_limit=60)
    if not results or "results" not in results or "bindings" not in results["results"]:
        return []

    triples = []
    relation_counts = {}
    
    for result in results["results"]["bindings"]:
        s_uri, p_uri = result.get('s', {}).get('value', ''), result.get('p', {}).get('value', '')
        if not s_uri or not p_uri: continue
            
        s_id, p_id = s_uri.split('/')[-1], p_uri.split('/')[-1]
        s_label = result.get('sLabel', {}).get('value', s_id)
        p_label = result.get('pLabel', {}).get('value', p_id)
        
        if relation_counts.get(p_label, 0) >= max_same_rel:
            continue
            
        if s_label == s_id or (s_label.startswith("Q") and s_label[1:].isdigit()):
            s_label = fetch_label_by_id(s_id)
            
        triple = init_triples_dict(subject=s_label, relation=p_label, obj=target_label)
        triples.append(triple)
        relation_counts[p_label] = relation_counts.get(p_label, 0) + 1

    return triples

def get_entity_triplets(entity_id: str, entity_label: str, max_same_rel: int = 2) -> list:
    results = get_neighbor_triples(entity_id, fetch_limit=60)
    if not results or "results" not in results or "bindings" not in results["results"]:
        return []

    triples = []
    relation_counts = {}
    
    for result in results["results"]["bindings"]:
        p_uri, o_uri = result.get('p', {}).get('value', ''), result.get('o', {}).get('value', '')
        if not p_uri or not o_uri: continue
        
        p_id, o_id = p_uri.split('/')[-1], o_uri.split('/')[-1]
        p_label = result.get('pLabel', {}).get('value', p_id)
        o_label = result.get('oLabel', {}).get('value', o_id)
        
        
        if relation_counts.get(p_label, 0) >= max_same_rel:
            continue
            
        triple = init_triples_dict(subject=entity_label, relation=p_label, obj=o_label)

        if not triple["target"] or (triple["target"].startswith("Q") and triple["target"][1:].isdigit()):
            triple["target"] = fetch_label_by_id(o_id)
            if not triple["target"] or (triple["target"].startswith("Q") and triple["target"][1:].isdigit()):
                continue
                
        triples.append(triple)
        relation_counts[p_label] = relation_counts.get(p_label, 0) + 1

    return triples



def validate_and_add_triple(existing_triples, new_triple, core_subject, core_target, is_outgoing_layer=False):
    """验证并应用全局拓扑规则"""
    s, r, o = new_triple["subject"], new_triple["relation"], new_triple["target"]
    

    if s == o: return False
        

    existing_entities = set()
    for t in existing_triples:
        existing_entities.add(t["subject"])
        existing_entities.add(t["target"])
        
    existing_entities.discard(core_subject)
    existing_entities.discard(core_target)
    
    if s in existing_entities or o in existing_entities:
        return False 


    country_keywords = ["country", "republic", "kingdom", "states", "nation", "empire", "land"]
    is_subject_country = any(kw in s.lower() for kw in country_keywords) in ["country", "country of citizenship"]
    if is_subject_country and is_outgoing_layer:
        return False


    existing_triples.append(new_triple)
    return True


def get_triplets_from_dataset(dataset_path: str):
    with open(dataset_path, 'r', encoding='utf-8') as file:
        dataset = json.load(file)

    counter = 0
    relation_id_dict = {}
    new_json = []
    
    for idx, case in enumerate(dataset):
        case_id = case.get("case_id", idx)
        
        new_data = {"case_id": case_id, "triples": []}
        
        
        
        if "requested_rewrite" in case and isinstance(case["requested_rewrite"], list):
            is_multihop_dataset = True
            rewrites = case["requested_rewrite"]
            # new_triples_len = len(case["orig"]["new_triples_labeled"])
            MAX_TRIPLES_PER_BRANCH = 10+len(rewrites) 
            rewrite_infos = []
            
            
            for i in range(len(rewrites)):
                rewrite = rewrites[i]
                subject_label = rewrite["subject"]
                
                subject_id = case["orig"]["edit_triples"][i][0]
                target_id = rewrite["target_new"]["id"]
                target_label = rewrite["target_new"]["str"]
                relation_id = rewrite["relation_id"]
                
                if relation_id in relation_id_dict:
                    relation_label = relation_id_dict[relation_id]
                else:
                    relation_label = get_property_label(relation_id)
                    relation_id_dict[relation_id] = relation_label
                print(f"\nProcessing Case {case_id}: {subject_label}({subject_id}) -> {target_label}({target_id})")
                
                origin_triplet = init_triples_dict(subject_label, relation_label, target_label)
                new_data["triples"].append(origin_triplet)
                
                
                rewrite_infos.append({
                    "s_id": subject_id, "s_label": subject_label,
                    "t_id": target_id, "t_label": target_label,
                    "r_id": relation_id, "r_label": relation_label
                })

            if len(rewrites) == 1:
                info = rewrite_infos[0]
                subject_id, subject_label = info["s_id"], info["s_label"]
                target_id, target_label = info["t_id"], info["t_label"]
                

                if subject_id:
                    raw_in_candidates = get_incoming_entity_triplets(subject_id, subject_label, max_same_rel=1)
                    added_count = 0
                    for t in raw_in_candidates:
                        if validate_and_add_triple(new_data["triples"], t, subject_label, target_label, is_outgoing_layer=False):
                            added_count += 1
                            if len(new_data["triples"]) >= MAX_TRIPLES_PER_BRANCH: break
                    print(f"   [+] 头实体做尾实体 '{subject_label}' 获取候选入边 {len(raw_in_candidates)} 条，过滤后成功收录 {added_count} 条")


                if target_id:
                    raw_out_candidates = get_entity_triplets(target_id, target_label, max_same_rel=1)
                    added_count = 0
                    for t in raw_out_candidates:
                        if validate_and_add_triple(new_data["triples"], t, subject_label, target_label, is_outgoing_layer=True):
                            added_count += 1
                            if len(new_data["triples"]) >= MAX_TRIPLES_PER_BRANCH: break
                    print(f"   [+] 尾实体 '{target_label}' 获取候选出边 {len(raw_out_candidates)} 条，过滤后成功收录 {added_count} 条")

            # Case 2: len(rewrites) == 2
            elif len(rewrites) == 2:
                r1 = rewrite_infos[0]
                r2 = rewrite_infos[1]
                

                is_continuous = (r1["t_id"] == r2["s_id"])
                is_continuous1 = (r1["s_id"] == r2["t_id"])
                if is_continuous:

                    core_s_label = r1["s_label"]
                    core_t_label = r2["t_label"]
                    
                    if r1["s_id"]:
                        raw_in = get_incoming_entity_triplets(r1["s_id"], r1["s_label"], max_same_rel=1)
                        added = 0
                        for t in raw_in:
                            if validate_and_add_triple(new_data["triples"], t, core_s_label, core_t_label, is_outgoing_layer=False):
                                added += 1
                                if len(new_data["triples"]) >= MAX_TRIPLES_PER_BRANCH: break
                        print(f"   [+] 头实体做尾实体 '{core_s_label}' 获取候选入边 {len(raw_in)} 条，过滤后成功收录 {added} 条")    
                    
                    if r2["t_id"]:
                        raw_out = get_entity_triplets(r2["t_id"], r2["t_label"], max_same_rel=1)
                        added = 0
                        for t in raw_out:
                            if validate_and_add_triple(new_data["triples"], t, core_s_label, core_t_label, is_outgoing_layer=True):
                                added += 1
                                if len(new_data["triples"]) >= MAX_TRIPLES_PER_BRANCH: break
                        print(f"   [+] 尾实体 '{core_t_label}' 获取候选出边 {len(raw_out)} 条，过滤后成功收录 {added} 条")        
                
                else:



                    new_rels = get_intermediate_entity(r1["t_id"], r1["t_label"], r2["s_id"], r2["s_label"])
                    for t in new_rels:
                        new_data["triples"].append(t)
                    print(f"   [+] 补全关系 '{r1['t_label']} ->mid-> {r2['s_label']} ' 获取候选关系 {len(new_rels)} 条")
                    if len(new_rels) == 0:

                        new_rels = get_relation_between(r1["t_id"], r1["t_label"], r2["s_id"], r2["s_label"])
                        for t in new_rels:
                            new_data["triples"].append(t)
                        print(f"   [+] 补全关系 '{r1['t_label']} -> {r2['s_label']}' 获取候选关系 {len(new_rels)} 条") 
                        
                        core_s_label = r1["s_label"]
                        core_t_label = r2["t_label"]
                        
                        if r1["s_id"]:
                            raw_in = get_incoming_entity_triplets(r1["s_id"], r1["s_label"], max_same_rel=1)
                            added = 0
                            for t in raw_in:
                                if validate_and_add_triple(new_data["triples"], t, core_s_label, core_t_label, is_outgoing_layer=False):
                                    added += 1
                                    if len(new_data["triples"]) >= MAX_TRIPLES_PER_BRANCH: break
                            print(f"   [+] 头实体做尾实体 '{core_s_label}' 获取候选入边 {len(raw_in)} 条，过滤后成功收录 {added} 条")    
                        
                        if r2["t_id"]:
                            raw_out = get_entity_triplets(r2["t_id"], r2["t_label"], max_same_rel=1)
                            added = 0
                            for t in raw_out:
                                if validate_and_add_triple(new_data["triples"], t, core_s_label, core_t_label, is_outgoing_layer=True):
                                    added += 1
                                    if len(new_data["triples"]) >= MAX_TRIPLES_PER_BRANCH: break
                            print(f"   [+] 尾实体 '{core_t_label}' 获取候选出边 {len(raw_out)} 条，过滤后成功收录 {added} 条")        
                                

            # Case 3: len(rewrites) == 3
            elif len(rewrites) == 3:
                r1, r2, r3 = rewrite_infos[0], rewrite_infos[1], rewrite_infos[2]


                link1_ok = (r1["t_id"] == r2["s_id"])
                link2_ok = (r2["t_id"] == r3["s_id"])
                
                if link1_ok and link2_ok:

                    core_s_label = r1["s_label"]
                    core_t_label = r3["t_label"]
                    
                    if r1["s_id"]:
                        raw_in = get_incoming_entity_triplets(r1["s_id"], r1["s_label"], max_same_rel=1)
                        added = 0
                        for t in raw_in:
                            if validate_and_add_triple(new_data["triples"], t, core_s_label, core_t_label, is_outgoing_layer=False):
                                added += 1
                                if len(new_data["triples"]) >= MAX_TRIPLES_PER_BRANCH: break
                        print(f"   [+] 头实体做尾实体 '{core_s_label}' 获取候选入边 {len(raw_in)} 条，过滤后成功收录 {added} 条")    
                    
                    if r3["t_id"]:
                        raw_out = get_entity_triplets(r3["t_id"], r3["t_label"], max_same_rel=1)
                        added = 0
                        for t in raw_out:
                            if validate_and_add_triple(new_data["triples"], t, core_s_label, core_t_label, is_outgoing_layer=True):
                                added += 1
                                if len(new_data["triples"]) >= MAX_TRIPLES_PER_BRANCH: break
                        print(f"   [+] 尾实体 '{core_t_label}' 获取候选出边 {len(raw_out)} 条，过滤后成功收录 {added} 条")        
                    
                elif not link1_ok:
                    # Gap between r1 and r2 (B->C missing)
                    new_rels = get_relation_between(r1["t_id"], r1["t_label"], r2["s_id"], r2["s_label"])
                    for t in new_rels:
                        new_data["triples"].append(t)
                    print(f"   [+] 补全关系 '{r1['t_label']} -> {r2['s_label']}' 获取候选关系 {len(new_rels)} 条")
                        
                elif not link2_ok:
                    # Gap between r2 and r3 (C->D missing)
                    new_rels = get_relation_between(r2["t_id"], r2["t_label"], r3["s_id"], r3["s_label"])
                    for t in new_rels:
                        new_data["triples"].append(t)
                    print(f"   [+] 补全关系 '{r2['t_label']} -> {r3['s_label']}' 获取候选关系 {len(new_rels)} 条")
                            
        else:
            continue
        
        new_json.append(new_data)
        
        counter += 1    
        if counter % 100 == 0:
            file_name = f'{global_datastr_name}_graph_{counter - 99}_to_{counter}.json'
            with open(file_name, 'w', encoding='utf-8') as output_file:
                json.dump(new_json, output_file, indent=4)
            new_json = []


    if len(new_json) > 0:
        file_name = f'{global_datastr_name}_graph_{counter - len(new_json) + 1}_to_{counter}.json'
        with open(file_name, 'w', encoding='utf-8') as output_file:
            json.dump(new_json, output_file, indent=4)

if __name__ == "__main__":
    dataset_path = "./data/MQuAKE-CF-3k.json"
    # dataset_path = "./data/MQuAKE-T.json"
    data_str_name = os.path.basename(dataset_path).split(".")[0]
    print(f"Starting extraction for dataset: {data_str_name}")
    global_datastr_name = data_str_name
    get_triplets_from_dataset(dataset_path)
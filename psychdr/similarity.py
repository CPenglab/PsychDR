import numpy as np
import torch
import xml.etree.ElementTree as ET
import re
from collections import defaultdict, deque, Counter
from psychdr.utils import normalize_matrix_minmax

# ================= 1. 基于蛋白的疾病相似度 =================
def compute_protein_similarity(disease_names, protein_set, edges, df_gene_score, alpha=1.0, beta=0.7, gamma=0.5, max_steps=3):
    seed_proteins = set(df_gene_score["gene_symbol"].tolist()) & protein_set
    disease_to_protein = defaultdict(set)
    drug_target_proteins = set()
    ppi_graph = defaultdict(list)

    for _, row in edges.iterrows():
        p, d = row.get("from"), row.get("to")
        if row["edge_type"] == "Protein-Disease" and p in protein_set and d in disease_names:
            disease_to_protein[d].add(p)
        elif row["edge_type"] == "Protein-Drug" and p in protein_set:
            drug_target_proteins.add(p)
        elif row["edge_type"] == "Protein-Protein" and p in protein_set and d in protein_set:
            ppi_graph[p].append(d)
            ppi_graph[d].append(p)

    def bfs_min_dist(start, targets, graph, max_steps):
        if start in targets: return 0
        visited = set([start])
        q = deque([(start, 0)])
        while q:
            node, dist = q.popleft()
            if dist >= max_steps: continue
            for nxt in graph[node]:
                if nxt in visited: continue
                if nxt in targets: return dist + 1
                visited.add(nxt)
                q.append((nxt, dist+1))
        return None

    drug_target_dist = {}
    for p in drug_target_proteins:
        drug_target_dist[p] = bfs_min_dist(p, seed_proteins, ppi_graph, max_steps)

    protein_weights = defaultdict(float)
    for d in disease_names:
        for p in disease_to_protein[d]:
            if p in drug_target_proteins:
                protein_weights[p] = max(protein_weights[p], alpha)
            elif any(n in drug_target_proteins for n in ppi_graph.get(p, [])):
                protein_weights[p] = max(protein_weights[p], beta)

    for p in drug_target_proteins:
        dist = drug_target_dist[p]
        if dist is not None:
            protein_weights[p] = max(protein_weights[p], gamma / (dist + 1))

    n = len(disease_names)
    sim_matrix_np = np.zeros((n, n), dtype=np.float32)
    for i, d1 in enumerate(disease_names):
        p1 = disease_to_protein[d1]
        for j, d2 in enumerate(disease_names):
            if i > j: continue
            p2 = disease_to_protein[d2]
            w_inter = sum(protein_weights[p] for p in (p1 & p2))
            w_union = sum(protein_weights[p] for p in (p1 | p2)) + 1e-12
            sim = w_inter / w_union
            sim_matrix_np[i, j] = sim_matrix_np[j, i] = sim
    return sim_matrix_np

# ================= 2. 基于 MeSH 的疾病相似度 =================
def parse_mesh_with_synonyms(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    mesh_info = {}
    name_to_ui = {}
    for desc in root.findall("DescriptorRecord"):
        ui = desc.findtext("DescriptorUI")
        dname_el = desc.find("DescriptorName/String")
        dname = dname_el.text if dname_el is not None else ""
        tns = [tn.text for tn in desc.findall("TreeNumberList/TreeNumber")]
        syns = []
        for concept in desc.findall("ConceptList/Concept"):
            for term in concept.findall("TermList/Term"):
                t = term.find("String")
                if t is not None and t.text: syns.append(t.text)
        mesh_info[ui] = {"name": dname, "tree_numbers": tns, "synonyms": syns}
        if dname: name_to_ui[dname.lower()] = ui
        for s in syns:
            if s: name_to_ui[s.lower()] = ui
    return mesh_info, name_to_ui

def normalize_name(s):
    if s is None: return ""
    s = s.lower()
    s = re.sub(r'\([^)]*\)', '', s)
    s = s.replace('/', ' ').replace('-', ' ')
    s = re.sub(r'[^a-z0-9\s]', ' ', s)
    return re.sub(r'\s+', ' ', s).strip()

def tokens(s):
    return [t for t in normalize_name(s).split() if len(t) > 0]

def mesh_level_similarity(tree1, tree2):
    if not tree1 or not tree2: return 0.0
    def depth(x): return len(x.split('.'))
    def expand(tns):
        anc = set()
        for tn in tns:
            parts = tn.split('.')
            # 🔥 修复：这里原来的大写 I 改成了小写 i
            for i in range(1, len(parts)+1): 
                anc.add('.'.join(parts[:i]))
        return anc
    
    best = 0
    for t1 in tree1:
        p1 = t1.split('.'); d1 = len(p1)
        for t2 in tree2:
            p2 = t2.split('.'); d2 = len(p2)
            lcp = 0
            for a, b in zip(p1, p2):
                if a == b: lcp += 1
                else: break
            if lcp == 0: continue
            best = max(best, 2 * lcp / (d1 + d2))
            
    anc1, anc2 = expand(tree1), expand(tree2)
    overlap = len(anc1 & anc2) / len(anc1 | anc2) if len(anc1 | anc2) > 0 else 0
    return 0.6 * best + 0.4 * overlap

def compute_mesh_similarity(disease_names, mesh_xml_path):
    mesh_info, name_to_ui = parse_mesh_with_synonyms(mesh_xml_path)
    token_to_uis = {}
    for ui, info in mesh_info.items():
        seen = set()
        for nm in [info.get("name","")] + info.get("synonyms",[]):
            for t in normalize_name(nm).split():
                if t not in seen:
                    token_to_uis.setdefault(t, set()).add(ui)
                    seen.add(t)
    
    disease_to_mesh = {}
    for dn in disease_names:
        dn_norm = normalize_name(dn)
        if dn_norm in name_to_ui:
            disease_to_mesh[dn] = mesh_info[name_to_ui[dn_norm]]["tree_numbers"]
            continue
        found_ui = next((ui for k, ui in name_to_ui.items() if dn_norm in k or k in dn_norm), None)
        if found_ui:
            disease_to_mesh[dn] = mesh_info[found_ui]["tree_numbers"]
            continue
        tks = tokens(dn)
        cand_cnt = Counter()
        for t in tks:
            for ui in token_to_uis.get(t, []): cand_cnt[ui] += 1
        if cand_cnt:
            best_ui, cnt = cand_cnt.most_common(1)[0]
            if cnt / len(tks) >= 0.5:
                disease_to_mesh[dn] = mesh_info[best_ui]["tree_numbers"]
            else:
                disease_to_mesh[dn] = []
        else:
            disease_to_mesh[dn] = []

    n = len(disease_names)
    sim_matrix = np.zeros((n, n), dtype=float)
    for i, d1 in enumerate(disease_names):
        for j, d2 in enumerate(disease_names):
            sim_matrix[i,j] = mesh_level_similarity(disease_to_mesh[d1], disease_to_mesh[d2])
    return sim_matrix, disease_to_mesh

# ================= 3. Top-K 与门控 =================
def compute_topk_and_gate(multi_sim_np, top_k=2):
    sim = torch.tensor(multi_sim_np, dtype=torch.float32)
    if sim.shape[0] == sim.shape[1]:
        idx = torch.arange(sim.shape[0])
        sim[idx, idx] = -1.0
    top_k = min(top_k, sim.shape[1])
    topk_vals, topk_idx = torch.topk(sim, top_k, dim=1)
    topk_vals = torch.clamp(topk_vals, min=0.0)
    sum_vals = topk_vals.sum(dim=1, keepdim=True) + 1e-12
    norm_wts = topk_vals / sum_vals
    avg_sim = topk_vals.mean(dim=1)
    
    # 🔥 修复：这里必须加上 1.0 - avg_sim，否则门控逻辑完全反了
    gate_c = torch.clamp(1.0 - avg_sim, min=0.1, max=0.9).unsqueeze(1)
    return topk_idx.long(), topk_vals.float(), norm_wts.float(), gate_c.float()

# ================= 4. Fuzzy MeSH 匹配 =================
def fuzzy_map_disease_to_mesh_tns(
        disease_list, mesh_name_index, mesh_info, token_to_uis,
        min_token_overlap_ratio=0.5):
    mapping = {}
    not_found = []
    mesh_key_tokens = {
        ui: tokens(nm) for nm, ui in mesh_name_index.items()
    }
    for dn in disease_list:
        dn_norm = normalize_name(dn)
        # ① 精确或同义词匹配
        if dn_norm in mesh_name_index:
            ui = mesh_name_index[dn_norm]
            mapping[dn] = mesh_info[ui]["tree_numbers"]
            continue
        # ② substring
        found_ui = None
        for kname, ui in mesh_name_index.items():
            if dn_norm in kname or kname in dn_norm:
                found_ui = ui
                break
        if found_ui:
            mapping[dn] = mesh_info[found_ui]["tree_numbers"]
            continue
        # ③ token overlap
        tks = tokens(dn)
        if not tks:
            mapping[dn] = []
            not_found.append(dn)
            continue
        cand_cnt = Counter()
        for t in tks:
            for ui in token_to_uis.get(t, []):
                cand_cnt[ui] += 1
        if not cand_cnt:
            mapping[dn] = []
            not_found.append(dn)
            continue
        best_ui, cnt = cand_cnt.most_common(1)[0]
        ratio = cnt / len(tks)
        if ratio >= min_token_overlap_ratio:
            mapping[dn] = mesh_info[best_ui]["tree_numbers"]
        else:
            mapping[dn] = []
            not_found.append(dn)
    
    print(f"🔎 fuzzy MeSH 匹配成功 {len(disease_list)-len(not_found)} / {len(disease_list)}")
    return mapping
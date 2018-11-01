import os
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import HeteroData
from collections import defaultdict

def build_hetero_data(data_dir, drug_feat_file, protein_feat_file, disease_feat_file, nodes_file, edges_file):
    """
    根据传入的 CSV 文件构建异构图 HeteroData
    """
    print("Loading data from:", data_dir)
    
    # 1. 读取特征
    drug_feat = pd.read_csv(os.path.join(data_dir, drug_feat_file), index_col=0)
    protein_feat = pd.read_csv(os.path.join(data_dir, protein_feat_file), index_col=0)
    if "SYMBOL" in protein_feat.columns:
        protein_feat = protein_feat.set_index("SYMBOL")
    disease_feat = pd.read_csv(os.path.join(data_dir, disease_feat_file), index_col=0)
    
    # 2. 读取节点和边
    nodes = pd.read_csv(os.path.join(data_dir, nodes_file))
    edges = pd.read_csv(os.path.join(data_dir, edges_file))

    data = HeteroData()
    id_map, valid_ids = {}, {}

    # 3. 构建节点特征
    for node_type, feat_df in zip(['Drug', 'Protein', 'Disease'], [drug_feat, protein_feat, disease_feat]):
        node_df = nodes[nodes['type'] == node_type]
        ids = list(node_df['name'])
        feat_dim = feat_df.shape[1]
        feat = np.zeros((len(ids), feat_dim), dtype=np.float32)
        for i, nid in enumerate(ids):
            if nid in feat_df.index:
                feat[i] = feat_df.loc[nid].values
        data[node_type.lower()].x = torch.tensor(feat, dtype=torch.float)
        id_map[node_type] = {nid: i for i, nid in enumerate(ids)}
        valid_ids[node_type] = set(ids)

    # 4. 构建边
    edge_index_dict = defaultdict(lambda: [[], []])
    for _, row in edges.iterrows():
        src_id, dst_id, edge_type = row['from'], row['to'], row['edge_type']
        if '-' not in edge_type:
            continue
        src_type, dst_type = edge_type.split('-')
        src, dst = src_type.lower(), dst_type.lower()
        if src_id in id_map[src_type] and dst_id in id_map[dst_type]:
            edge_index_dict[(src, edge_type, dst)][0].append(id_map[src_type][src_id])
            edge_index_dict[(src, edge_type, dst)][1].append(id_map[dst_type][dst_id])

    for (src, rel, dst), (s, d) in edge_index_dict.items():
        data[(src, rel, dst)].edge_index = torch.tensor([s, d], dtype=torch.long)

    return data, id_map, nodes, edges
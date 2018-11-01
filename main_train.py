import os
import torch
import pandas as pd
import numpy as np
import random
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# 导入你自己包里的模块
from psychdr.models import FeatureReducer, HGTWithDiseaseSim, LinkPredictor
from psychdr.data_loader import build_hetero_data
from psychdr.similarity import (
    compute_protein_similarity, 
    compute_mesh_similarity, 
    compute_topk_and_gate
)

# ============================================================
# 1. 配置路径与参数 (请替换为你本地的实际路径)
# ============================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", DEVICE)

# 请修改为你当前的数据存放根目录
DATA_ROOT = "/public/home/liyu/Disease_drug/TXGNN/indication/No repeat mental illness" 
MESH_XML_PATH = "/public/home/liyu/Disease_drug/TXGNN/desc2025.xml"
GENE_SCORE_PATH = "/public/home/liyu/Disease_drug/TXGNN/NND_gene_score.xlsx"
TCM_LIST_PATH = os.path.join(DATA_ROOT, "Traditional_Chinese_Medicine_Database/20260522/TCM_ingredient_smiles_all_1104.csv")

# ============================================================
# 2. 辅助函数 (负采样、训练单轮、评估)
# ============================================================
def sample_neg_pairs_no_leak(pos_dst, all_positive_set, num_drugs, neg_ratio=1):
    neg_src, neg_dst = [], []
    for d in pos_dst.unique().tolist():
        num_pos = (pos_dst == d).sum().item()
        num_neg = int(num_pos * neg_ratio)
        candidates = [drug for drug in range(num_drugs) if (drug, d) not in all_positive_set]
        if not candidates: continue
        num_neg = min(num_neg, len(candidates))
        sampled_drugs = random.sample(candidates, num_neg)
        neg_src.extend(sampled_drugs)
        neg_dst.extend([d] * num_neg)
    return (torch.tensor(neg_src, dtype=torch.long, device=DEVICE),
            torch.tensor(neg_dst, dtype=torch.long, device=DEVICE))

def train_one_epoch(ddi_train, edge_index_dicts_train, feature_reducer, model, predictor, optimizer, all_positive_set, disease_sim_idx, disease_sim_wts, gate_c, lambda_rank=0.5):
    model.train(); predictor.train(); feature_reducer.train()
    optimizer.zero_grad()
    
    x_dict_raw = {k: data[k].x.to(DEVICE) for k in data.node_types}
    x_dict = feature_reducer(x_dict_raw)
    
    # 将边移到设备上
    edge_index_dicts_train_dev = [{k: v.to(DEVICE) for k, v in ed.items()} for ed in edge_index_dicts_train]
    h_dict = model(x_dict, edge_index_dicts_train_dev, disease_sim_idx, disease_sim_wts, gate_c)
    
    pos_edge = ddi_train.to(DEVICE)
    pos_src, pos_dst = pos_edge[0].view(-1), pos_edge[1].view(-1)
    
    neg_src, neg_dst = sample_neg_pairs_no_leak(pos_dst, all_positive_set, num_drugs=data['drug'].num_nodes, neg_ratio=1)
    
    pos_score = predictor(h_dict['drug'][pos_src], h_dict['disease'][pos_dst])
    
    bce_loss_fn = torch.nn.BCEWithLogitsLoss()
    
    if neg_src.numel() > 0:
        neg_score = predictor(h_dict['drug'][neg_src], h_dict['disease'][neg_dst])
        
        labels = torch.cat([torch.ones_like(pos_score), torch.zeros_like(neg_score)])
        scores = torch.cat([pos_score, neg_score])
        bce_loss = bce_loss_fn(scores, labels)
        
        # BPR Rank Loss
        min_len = min(len(pos_score), len(neg_score))
        rank_loss = -torch.log(torch.sigmoid(pos_score[:min_len] - neg_score[:min_len]) + 1e-8).mean()
        
        loss = bce_loss + lambda_rank * rank_loss
    else:
        loss = bce_loss_fn(pos_score, torch.ones_like(pos_score))
        
    loss.backward()
    optimizer.step()
    return loss.item()

@torch.no_grad()
def evaluate_on_set(ddi_edge_eval, edge_index_dicts_eval, feature_reducer, model, predictor, fixed_neg_src, fixed_neg_dst, disease_sim_idx, disease_sim_wts, gate_c):
    model.eval(); predictor.eval(); feature_reducer.eval()
    
    x_dict_raw = {k: data[k].x.to(DEVICE) for k in data.node_types}
    x_dict = feature_reducer(x_dict_raw)
    edge_index_dicts_eval_dev = [{k: v.to(DEVICE) for k, v in ed.items()} for ed in edge_index_dicts_eval]
    
    h_dict = model(x_dict, edge_index_dicts_eval_dev, disease_sim_idx, disease_sim_wts, gate_c)
    
    pos_edge = ddi_edge_eval.to(DEVICE)
    pos_src, pos_dst = pos_edge[0].view(-1), pos_edge[1].view(-1)
    
    pos_scores = predictor(h_dict['drug'][pos_src], h_dict['disease'][pos_dst])
    neg_scores = predictor(h_dict['drug'][fixed_neg_src], h_dict['disease'][fixed_neg_dst])
    
    labels = torch.cat([torch.ones_like(pos_scores), torch.zeros_like(neg_scores)]).cpu().numpy()
    preds = torch.cat([pos_scores, neg_scores]).cpu().numpy()
    
    auc = roc_auc_score(labels, preds)
    return auc

# ============================================================
# 3. 主训练流程
# ============================================================
def main():
    global data # 为了辅助函数能访问到 data
    
    print("1. 正在加载图谱数据...")
    data, id_map, nodes, edges = build_hetero_data(
        data_dir=os.path.join(DATA_ROOT, "new_kg_20261109"),
        drug_feat_file="drug_features_ecfp2048_20261109_548.csv",
        protein_feat_file="/public/home/liyu/Disease_drug/TXGNN/indication/No repeat mental illness/new_kg/protein_go_features.csv", 
        disease_feat_file="/public/home/liyu/Disease_drug/TXGNN/indication/No repeat mental illness/new_kg_20261103/disease_nodes_biobert_cls_features_20261104.csv",
        nodes_file="node_20261109.csv",
        edges_file="edge_20261109_final.csv"
    )
    
    disease_names = list(nodes[nodes['type'] == "Disease"]["name"])
    protein_names = list(nodes[nodes['type'] == "Protein"]["name"])
    protein_set = set(protein_names)
    df_gene_score = pd.read_excel(GENE_SCORE_PATH)
    
    print("2. 正在计算疾病相似度...")
    # 蛋白质相似度
    protein_sim = compute_protein_similarity(disease_names, protein_set, edges, df_gene_score)
    # MeSH 相似度
    mesh_sim, disease_to_mesh_fuzzy = compute_mesh_similarity(disease_names, MESH_XML_PATH)
    
    print("3. 正在划分训练/测试集并计算药物相似度...")
    ddi_edge_all = data[('drug', 'Drug-Disease', 'disease')].edge_index
    train_diseases, test_diseases = train_test_split(disease_names, test_size=0.20, random_state=SEED)
    
    # 划分边
    test_ids = [id_map['Disease'][d] for d in test_diseases]
    mask_test = torch.isin(ddi_edge_all[1], torch.tensor(test_ids))
    ddi_train = ddi_edge_all[:, ~mask_test]
    ddi_test = ddi_edge_all[:, mask_test]
    
    # 计算药物相似度矩阵
    def compute_drug_pairwise_from_ddi(ddi_edges_tensor, n_disease, n_drug):
        ddi_numpy = ddi_edges_tensor.cpu().numpy()
        ddi_mat = np.zeros((n_disease, n_drug), dtype=np.uint8)
        for s, d in zip(ddi_numpy[0].astype(int), ddi_numpy[1].astype(int)):
            if 0 <= d < n_disease and 0 <= s < n_drug: ddi_mat[d, s] = 1
        inter = ddi_mat @ ddi_mat.T
        sum_d = ddi_mat.sum(axis=1, keepdims=True)
        union = (sum_d + sum_d.T - inter).astype(np.float32) + 1e-9
        return inter.astype(np.float32) / union
    
    drug_sim = compute_drug_pairwise_from_ddi(ddi_train, len(disease_names), data['drug'].num_nodes)
    
    print("4. 多视图融合与 Top-K 聚合...")
    # 归一化
    def normalize_matrix_minmax(mat):
        _min, _max = mat.min(), mat.max()
        if _max - _min < 1e-9: return np.zeros_like(mat)
        return (mat - _min) / (_max - _min)

    protein_sim_norm = normalize_matrix_minmax(protein_sim)
    mesh_sim_norm = normalize_matrix_minmax(mesh_sim)
    
    # 自动权重
    vars = np.array([np.var(protein_sim_norm), np.var(mesh_sim_norm)], dtype=np.float32)
    view_weights = vars / vars.sum()
    w_protein, w_mesh = view_weights
    
    multi_sim_auto = w_protein * protein_sim_norm + w_mesh * mesh_sim_norm
    disease_sim_idx, _, disease_sim_wts, gate_c = compute_topk_and_gate(multi_sim_auto, top_k=2)
    
    # 转移到 GPU
    disease_sim_idx_dev = disease_sim_idx.to(DEVICE)
    disease_sim_wts_dev = disease_sim_wts.to(DEVICE)
    gate_c_dev = gate_c.to(DEVICE)

    print("5. 准备 TCM 数据增强与模型图结构...")
    # 读取中药列表并增强
    df_tcm = pd.read_csv(TCM_LIST_PATH, index_col=0)
    chinese_medicines = df_tcm.index.tolist()
    chinese_medicine_set = set(chinese_medicines)
    
    chinese_drug_ids = [id_map['Drug'][d] for d in chinese_medicines if d in id_map['Drug']]
    
    def augment_ddi_train_with_chinese(ddi_train, chinese_drug_ids, repeat_times=5):
        ddi_train_np = ddi_train.cpu().numpy()
        extra_edges = []
        for d_id in chinese_drug_ids:
            mask = ddi_train_np[0] == d_id
            if mask.sum() > 0:
                # 提取当前中药关联的边（形状为 2 x N）
                edges_for_this_drug = ddi_train_np[:, mask]
                for _ in range(repeat_times):
                    extra_edges.append(edges_for_this_drug)
                    
        if extra_edges:
            # 关键修复：使用 np.concatenate 在列方向（axis=1）拼接所有的边
            extra_edges_np = np.concatenate(extra_edges, axis=1)
            extra_edges_tensor = torch.tensor(extra_edges_np, dtype=torch.long)
            return torch.cat([ddi_train, extra_edges_tensor], dim=1)
        return ddi_train
    
    ddi_train = augment_ddi_train_with_chinese(ddi_train, chinese_drug_ids, repeat_times=5)
    
    # 构建 Edge Dicts
    def build_edge_index_dicts_zero_shot(ddi_train_edges):
        edge_index_layer1 = {
            ('protein', 'Protein-Disease', 'disease'): data[('protein', 'Protein-Disease', 'disease')].edge_index,
            ('protein', 'Protein-Drug', 'drug'): data[('protein', 'Protein-Drug', 'drug')].edge_index,
            ('protein', 'Protein-Protein', 'protein'): data[('protein', 'Protein-Protein', 'protein')].edge_index,
            ('drug', 'Drug-Disease', 'disease'): ddi_train_edges,
        }
        edge_index_layer3 = {}
        for (src, rel, dst) in data.edge_types:
            ei = data[(src, rel, dst)].edge_index
            if src == 'drug' and rel == 'Drug-Disease' and dst == 'disease':
                edge_index_layer3[(src, rel, dst)] = ddi_train_edges
            else:
                edge_index_layer3[(src, rel, dst)] = ei
        return [edge_index_layer1, edge_index_layer1, edge_index_layer3]

    edge_index_dicts_train = build_edge_index_dicts_zero_shot(ddi_train)

    print("6. 初始化模型与优化器...")
    node_in_dims = {ntype: data[ntype].x.size(1) for ntype in data.node_types}
    node_out_dims = {'drug':512, 'protein':256, 'disease':256} 
    
    feature_reducer = FeatureReducer(node_in_dims, node_out_dims).to(DEVICE)
    model = HGTWithDiseaseSim(
        hidden_channels=256, metadata=data.metadata(), 
        node_in_dims=node_out_dims, num_layers=3, heads=4, dropout=0.1
    ).to(DEVICE)
    predictor = LinkPredictor(256).to(DEVICE)
    
    optimizer = torch.optim.Adam(
        list(feature_reducer.parameters()) + list(model.parameters()) + list(predictor.parameters()),
        lr=0.001, weight_decay=1e-3
    )
    
    # 固定评估用负样本
    all_positive_set = set(zip(ddi_edge_all[0].tolist(), ddi_edge_all[1].tolist()))
    train_neg_src, train_neg_dst = sample_neg_pairs_no_leak(ddi_train[1], all_positive_set, data['drug'].num_nodes)
    test_neg_src, test_neg_dst = sample_neg_pairs_no_leak(ddi_test[1], all_positive_set, data['drug'].num_nodes)

    print("7. 开始训练...")
    EPOCHS = 200
    eval_interval = 10
    best_test_auc = -float("inf")
    best_epoch = None
    best_model_states = {}

    for epoch in range(1, EPOCHS + 1):
        loss = train_one_epoch(ddi_train, edge_index_dicts_train, feature_reducer, model, predictor, optimizer, all_positive_set, disease_sim_idx_dev, disease_sim_wts_dev, gate_c_dev)
        
        if epoch % eval_interval == 0 or epoch == EPOCHS:
            train_auc = evaluate_on_set(ddi_train, edge_index_dicts_train, feature_reducer, model, predictor, train_neg_src, train_neg_dst, disease_sim_idx_dev, disease_sim_wts_dev, gate_c_dev)
            test_auc = evaluate_on_set(ddi_test, edge_index_dicts_train, feature_reducer, model, predictor, test_neg_src, test_neg_dst, disease_sim_idx_dev, disease_sim_wts_dev, gate_c_dev)
            
            print(f"Epoch {epoch:03d} | Loss {loss:.4f} | Train AUC {train_auc:.4f} | Test AUC {test_auc:.4f}")
            
            if test_auc > best_test_auc:
                best_test_auc = test_auc
                best_epoch = epoch
                best_model_states = {
                    'feature_reducer': {k: v.cpu().clone() for k, v in feature_reducer.state_dict().items()},
                    'model': {k: v.cpu().clone() for k, v in model.state_dict().items()},
                    'predictor': {k: v.cpu().clone() for k, v in predictor.state_dict().items()}
                }
                print(f"*** New best model at epoch {epoch} with Test AUC {test_auc:.4f} ***")

    print("8. 保存最佳模型权重...")
    os.makedirs("saved_weights", exist_ok=True)
    torch.save(best_model_states, "saved_weights/psychdr_pretrained.pth")
    print(f"✅ 训练完成！最佳 Epoch: {best_epoch}, Test AUC: {best_test_auc:.4f}")
    print("✅ 权重已保存至 saved_weights/psychdr_pretrained.pth")

if __name__ == "__main__":
    main()
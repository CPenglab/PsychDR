import torch
import random
import numpy as np

def set_seed(seed=42):
    """设置全局随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def normalize_matrix_minmax(mat):
    """矩阵 Min-Max 归一化"""
    _min, _max = mat.min(), mat.max()
    if _max - _min < 1e-9:
        return np.zeros_like(mat)
    return (mat - _min) / (_max - _min)

def get_edge_index(data_obj, edge_key):
    """安全获取边的索引"""
    edge_data = data_obj[edge_key]
    if hasattr(edge_data, 'edge_index'):
        return edge_data.edge_index
    elif hasattr(edge_data, 'edge_label_index'):
        return edge_data.edge_label_index
    else:
        raise AttributeError(f"No edge_index found for {edge_key}")

# 把其他的通用函数也加进来... (比如 build_edge_index_dicts_zero_shot, move_edge_index_dicts_to_device)
def build_edge_index_dicts_zero_shot(data, ddi_train_edges):
    """构建用于零样本/少样本测试的边字典列表"""
    
    # 如果 utils 里没有 get_edge_index，记得把它也加上（或者 import）
    def get_edge_index(data_obj, edge_key):
        edge_data = data_obj[edge_key]
        if hasattr(edge_data, 'edge_index'):
            return edge_data.edge_index
        elif hasattr(edge_data, 'edge_label_index'):
            return edge_data.edge_label_index
        else:
            raise AttributeError(f"No edge_index found for {edge_key}")

    edge_index_layer1 = {
        ('protein', 'Protein-Disease', 'disease'): get_edge_index(data, ('protein', 'Protein-Disease', 'disease')),
        ('protein', 'Protein-Drug', 'drug'): get_edge_index(data, ('protein', 'Protein-Drug', 'drug')),
        ('protein', 'Protein-Protein', 'protein'): get_edge_index(data, ('protein', 'Protein-Protein', 'protein')),
        ('drug', 'Drug-Disease', 'disease'): ddi_train_edges,
    }
    
    edge_index_layer3 = {}
    for (src, rel, dst) in data.edge_types:
        ei = get_edge_index(data, (src, rel, dst))
        if src == 'drug' and rel == 'Drug-Disease' and dst == 'disease':
            edge_index_layer3[(src, rel, dst)] = ddi_train_edges
        else:
            edge_index_layer3[(src, rel, dst)] = ei
            
    return [edge_index_layer1, edge_index_layer1, edge_index_layer3]

def move_edge_index_dicts_to_device(edge_index_dicts, device):
    out = []
    for ed in edge_index_dicts:
        newd = {}
        for k,v in ed.items():
            if isinstance(v, torch.Tensor):
                newd[k] = v.to(device)
            else:
                newd[k] = torch.tensor(v,dtype=torch.long,device=device)
        out.append(newd)
    return out


def augment_ddi_train_with_chinese(ddi_train, chinese_drug_ids, repeat_times=5):
    """
    对中药对应的正样本重复采样 repeat_times 次
    """
    ddi_train_np = ddi_train.cpu().numpy()
    extra_edges = []
    for d_id in chinese_drug_ids:
        mask = ddi_train_np[0] == d_id
        if mask.sum() > 0:
            # 重复采样 repeat_times 次
            for _ in range(repeat_times):
                extra_edges.append(ddi_train_np[:, mask])
    if extra_edges:
        extra_edges = np.concatenate(extra_edges, axis=1)
        ddi_train_aug = np.concatenate([ddi_train_np, extra_edges], axis=1)
        return torch.tensor(ddi_train_aug, dtype=torch.long)
    else:
        return ddi_train


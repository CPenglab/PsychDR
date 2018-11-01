import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import softmax

class FeatureReducer(nn.Module):
    def __init__(self, node_in_dims, node_out_dims):
        super().__init__()
        self.reducers = nn.ModuleDict()
        for ntype, in_dim in node_in_dims.items():
            out_dim = node_out_dims[ntype]
            self.reducers[ntype] = nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.BatchNorm1d(out_dim),
                nn.ReLU(inplace=True)
            )
    def forward(self, x_dict):
        out = {}
        for ntype, x in x_dict.items():
            reducer = self.reducers[ntype]
            dev = next(reducer.parameters()).device
            x = x.to(dev)
            out[ntype] = reducer(x)
        return out

class HGTConvWithWeight(nn.Module):
    def __init__(self, in_channels, out_channels, metadata, heads=4, dropout=0.1):
        super().__init__()
        self.heads = heads
        self.out_channels = out_channels
        self.dropout = dropout
        self.k_linears = nn.ModuleDict()
        self.q_linears = nn.ModuleDict()
        self.v_linears = nn.ModuleDict()
        self.out_linears = nn.ModuleDict()
        self.norms = nn.ModuleDict()
        self.edge_type_weights = nn.ParameterDict()
        for (src, rel, dst) in metadata[1]:
            init_w = 1.0 if (src=='protein' and rel=='Protein-Disease' and dst=='disease') else 1.0
            key = f"{src}__{rel}__{dst}"
            self.edge_type_weights[key] = nn.Parameter(torch.tensor(init_w))
        for ntype in metadata[0]:
            self.k_linears[ntype] = nn.Linear(in_channels, out_channels)
            self.q_linears[ntype] = nn.Linear(in_channels, out_channels)
            self.v_linears[ntype] = nn.Linear(in_channels, out_channels)
            self.out_linears[ntype] = nn.Linear(out_channels, out_channels)
            self.norms[ntype] = nn.BatchNorm1d(out_channels)
    def forward(self, x_dict, edge_index_dict):
        out_dict = {k: [] for k in x_dict}
        for (src, rel, dst), edge_index in edge_index_dict.items():
            if edge_index is None or edge_index.numel() == 0:
                continue
            key = f"{src}__{rel}__{dst}"
            weight = self.edge_type_weights.get(key, torch.tensor(1.0, device=edge_index.device))
            Q = self.q_linears[dst](x_dict[dst])
            K = self.k_linears[src](x_dict[src])
            V = self.v_linears[src](x_dict[src])
            src_i, dst_i = edge_index
            attn = (Q[dst_i]*K[src_i]).sum(dim=-1)/(self.out_channels**0.5)
            attn = softmax(attn*weight, dst_i)
            msg = V[src_i]*attn.unsqueeze(-1)
            msg = F.dropout(msg, p=self.dropout, training=self.training)
            agg = torch.zeros_like(x_dict[dst])
            agg = agg.index_add(0, dst_i, msg)
            out_dict[dst].append(agg)
        result = {}
        for k, vs in out_dict.items():
            if vs:
                stacked = sum(vs)
                x_plus = x_dict[k] + stacked
                x_normed = self.norms[k](x_plus)
                result[k] = self.out_linears[k](x_normed)
            else:
                result[k] = self.out_linears[k](x_dict[k])
        return result

class HGTWithDiseaseSim(nn.Module):
    def __init__(self, hidden_channels, metadata, node_in_dims, num_layers=3, heads=4, dropout=0.1):
        super().__init__()
        self.input_proj = nn.ModuleDict({
            k: nn.Linear(v, hidden_channels)
            for k, v in node_in_dims.items()
        })
        self.convs = nn.ModuleList([
            HGTConvWithWeight(hidden_channels, hidden_channels, metadata, heads, dropout)
            for _ in range(num_layers)
        ])
        self.residuals = nn.ParameterDict({
            k: nn.Parameter(torch.zeros(hidden_channels))
            for k in node_in_dims
        })
    def forward(self, x_dict, edge_index_dicts, disease_sim_idx=None, disease_sim_wts=None, gate_c=None, ddi_train_edges=None):
        x = {}
        for k, v in x_dict.items():
            dev = next(self.input_proj[k].parameters()).device
            x[k] = self.input_proj[k](v.to(dev))
        for conv, edge_index_dict in zip(self.convs, edge_index_dicts):
            x = conv(x, edge_index_dict)
        out = {k: v + self.residuals[k] for k, v in x.items()}
        if (disease_sim_idx is not None and disease_sim_wts is not None and gate_c is not None):
            device_local = next(self.parameters()).device
            d_idx = disease_sim_idx.to(device_local)
            d_wts = disease_sim_wts.to(device_local)
            g_c   = gate_c.to(device_local).view(-1, 1)
            h_d = out['disease']
            neigh = h_d[d_idx]
            w_exp = d_wts.unsqueeze(-1)
            neigh_agg = (neigh * w_exp).sum(dim=1)
            fused = (1.0 - g_c) * h_d + g_c * neigh_agg
            out['disease'] = fused
        return out

class LinkPredictor(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(2*hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim//2), nn.ReLU(),
            nn.Linear(hidden_dim//2,1)
        )
    def forward(self, h_src, h_dst):
        return self.model(torch.cat([h_src,h_dst], dim=-1)).view(-1)
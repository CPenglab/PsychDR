import copy
import torch
import pandas as pd
import numpy as np
from psychdr.similarity import mesh_level_similarity, compute_topk_and_gate
from psychdr.utils import normalize_matrix_minmax, build_edge_index_dicts_zero_shot, move_edge_index_dicts_to_device

@torch.no_grad()
def forward_new_disease_only(
        model,
        x,
        edge_dict,
        sim_idx,
        sim_wts,
        gate):
    """
    DiseaseSim 仅作用于最后一个新增疾病节点
    """
    # ===== HGT =====
    h = {}
    for k, v in x.items():
        dev = next(model.input_proj[k].parameters()).device
        h[k] = model.input_proj[k](v.to(dev))
    for conv, ed in zip(model.convs, edge_dict):
        h = conv(h, ed)
    h = {
        k: v + model.residuals[k]
        for k, v in h.items()
    }
    h_d = h['disease']
    new_id = h_d.size(0) - 1
    neigh = h_d[sim_idx[0]]            # (k,H)
    neigh_agg = (
        neigh *
        sim_wts[0].unsqueeze(-1)
    ).sum(dim=0)
    fused = (
        gate[0] * h_d[new_id]
        + (1-gate[0]) * neigh_agg
    )
    h['disease'][new_id] = fused
    return h



def predict_new_disease_drugs(
        new_disease_name,
        disease_bert_vec,
        filtered_df,
        data,
        disease_names,
        disease_to_protein,
        protein_weights,
        protein_set,
        protein_feat,
        id_map,
        ddi_train_edges,
        feature_reducer,
        model,
        predictor,
        disease_to_mesh_fuzzy,
        semantic_lower_bound_auto,
        drug_idx_to_name,
        chinese_medicine_set,
        drug_top_k=10,
        disease_sim_k=2,
        device="cpu"):
    import copy
    data_new = copy.deepcopy(data)
    id_map_new = copy.deepcopy(id_map)
    ####################################################
    ####################################################
    genes = (
        filtered_df['gene_symbol']
        .dropna()
        .astype(str)
        .tolist()
    )
    def standardize_gene_name(g):
        return (
            str(g)
            .strip()
            .replace("-", "")
            .replace("_", "")
            .replace(".", "")
        )
    present_proteins=[]
    for g in genes:
        g = g.upper()
        if g in id_map_new['Protein']:
            present_proteins.append(g)
            continue
        g_std = standardize_gene_name(g)
        if g_std in id_map_new['Protein']:
            present_proteins.append(g_std)


    present_proteins=list(set(present_proteins))
    print(
        "matched proteins:",
        len(present_proteins)
    )
    ####################################################
    ####################################################
    old_num_disease=data_new["disease"].x.size(0)
    disease_dim=data_new["disease"].x.size(1)
    feat=np.zeros(
        disease_dim,
        dtype=np.float32
    )
    feat[:768]=disease_bert_vec
    data_new["disease"].x=torch.cat(
        [
            data_new["disease"].x,
            torch.tensor(feat).unsqueeze(0)
        ],
        dim=0
    )
    id_map_new["Disease"][new_disease_name]=old_num_disease
    ####################################################
    ####################################################
    edge_pd=data_new[
        (
            "protein",
            "Protein-Disease",
            "disease"
        )
    ].edge_index.clone()
    new_edges=[]
    for p in present_proteins:
        if p in id_map_new["Protein"]:
            new_edges.append(
                [
                    id_map_new["Protein"][p],
                    old_num_disease
                ]
            )
    if len(new_edges)>0:
        new_edges=torch.tensor(
            new_edges
        ).T.long()
        edge_pd=torch.cat(
            [
                edge_pd,
                new_edges
            ],
            dim=1
        )
    ####################################################
    ####################################################
    edge_dict=build_edge_index_dicts_zero_shot(
        data_new,
        ddi_train_edges
    )
    edge_dict_new=[]
    for ed in edge_dict:
        tmp={}
        for k,v in ed.items():
            if k==(
                "protein",
                "Protein-Disease",
                "disease"
            ):
                tmp[k]=edge_pd
            else:
                tmp[k]=v
        edge_dict_new.append(tmp)
    edge_dict_new=move_edge_index_dicts_to_device(
        edge_dict_new,
        device
    )
    ####################################################
    ####################################################
    n_old=len(disease_names)
    protein_sim=np.zeros(
        (1,n_old),
        dtype=np.float32
    )
    p_new=set(present_proteins)
    for j,d_old in enumerate(disease_names):
        p_old=disease_to_protein[d_old]
        inter=p_new&p_old
        union=p_new|p_old
        w_inter=sum(
            protein_weights.get(p,0)
            for p in inter
        )
        w_union=sum(
            protein_weights.get(p,0)
            for p in union
        )+1e-12
        protein_sim[0,j]=(
            w_inter/w_union
        )
    ####################################################
    ####################################################
    mesh_sim=np.zeros(
        (1,n_old),
        dtype=np.float32
    )
    t1=disease_to_mesh_fuzzy.get(
        new_disease_name,
        []
    )
    for j,d_old in enumerate(disease_names):
        t2=disease_to_mesh_fuzzy.get(
            d_old,
            []
        )
        mesh_sim[0,j]=mesh_level_similarity(
            t1,
            t2
        )
    ####################################################
    ####################################################
    drug_sim=np.zeros(
        (1,n_old),
        dtype=np.float32
    )
    print("\n===== Similarity Debug =====")
    print("Matched proteins:", len(present_proteins))
    print(
        "Matched proteins with nonzero weights:",
        sum(protein_weights.get(p, 0) > 0 for p in present_proteins)
    )
    print(
        "Raw protein similarity:",
        protein_sim.min(), protein_sim.max(), protein_sim.mean()
    )
    print(
        "Raw MeSH similarity:",
        mesh_sim.min(), mesh_sim.max(), mesh_sim.mean()
    )
    print("============================\n")

    protein_sim = normalize_matrix_minmax(
        protein_sim
    )
    mesh_sim = normalize_matrix_minmax(
        mesh_sim
    )

    available_views=[]
    available_weights=[]
    if protein_sim.max()>0:
        available_views.append(
            protein_sim
        )
        available_weights.append(
            w_protein
        )
    if mesh_sim.max()>0:
        available_views.append(
            mesh_sim
        )
        available_weights.append(
            w_mesh
        )
    if drug_sim.max()>0:
        available_views.append(
            drug_sim
        )
        available_weights.append(
            w_drug
        )
    if len(available_views)==0:
        multi_sim=np.zeros(
            (1,n_old),
            dtype=np.float32
        )
    else:
        available_weights=np.array(
            available_weights,
            dtype=np.float32
        )
        available_weights=(
            available_weights/
            available_weights.sum()
        )
        multi_sim=np.zeros(
            (1,n_old),
            dtype=np.float32
        )
        for w,view in zip(
            available_weights,
            available_views
        ):
            multi_sim += w*view
    sim_idx,topk_vals,sim_wts,gate=compute_topk_and_gate(
        multi_sim,
        top_k=min(
            disease_sim_k,
            n_old
        )
    )
    sim_idx=sim_idx.to(device)
    sim_wts=sim_wts.to(device)
    gate=gate.to(device)
    similar_disease_info=[]
    for idx,sim in zip(
        sim_idx[0].cpu().numpy(),
        topk_vals[0].cpu().numpy()
    ):
        similar_disease_info.append(
            {
                "Disease": disease_names[idx],
                "Similarity": float(sim)
            }
    )
    print(pd.DataFrame(similar_disease_info))
    ####################################################
    # Step8 HGT forward
    ####################################################
    feature_reducer.eval()
    model.eval()
    predictor.eval()
    with torch.no_grad():
        x_raw={
            k:data_new[k].x.to(device)
            for k in data_new.node_types
        }
        x=feature_reducer(
            x_raw
        )
        h=forward_new_disease_only(
            model,
            x,
            edge_dict_new,
            sim_idx,
            sim_wts,
            gate
        )
        h_drug=h["drug"]
        h_dis=h["disease"]
        new_id=h_dis.size(0)-1
        disease_emb=h_dis[new_id]
        ####################################################
        ####################################################
        drug_names=[
            drug_idx_to_name[i]
            for i in range(len(drug_idx_to_name))
        ]
        tcm_idx=[
            i
            for i,d in enumerate(drug_names)
            if d in chinese_medicine_set
        ]
        non_tcm_idx=[
            i
            for i,d in enumerate(drug_names)
            if d not in chinese_medicine_set
        ]
        ################################################
        ################################################
        tcm_result=[]
        if len(tcm_idx)>0:
            tcm_idx_tensor=torch.tensor(
                tcm_idx,
                dtype=torch.long,
                device=device
            )
            tcm_emb=h_drug[
                tcm_idx_tensor
            ]
            tcm_dis_emb=disease_emb.repeat(
                tcm_emb.size(0),
                1
            )
        # predictor logits
        tcm_logits=predictor(
            tcm_emb,
            tcm_dis_emb
        )
        # ===== Debug: 查看 TCM logit 分布 =====
        logit_df = pd.DataFrame({
            "Drug": [drug_names[i] for i in tcm_idx],
            "Logit": tcm_logits.squeeze().detach().cpu().numpy()
        }).sort_values("Logit", ascending=False)
        print("\\n===== TCM Logit Distribution =====")
        print(logit_df.head(15))
        print(logit_df.tail(10))
        print("Mean:", logit_df["Logit"].mean())
        print("Std :", logit_df["Logit"].std())
        print("Min :", logit_df["Logit"].min())
        print("Max :", logit_df["Logit"].max())
        # ==========================================
        # ==========================================
        z = tcm_logits.squeeze()
        mean = z.mean()
        std = z.std() + 1e-8
        z_norm = (z - mean) / std
        gamma = 0.8
        delta = 0.5
        z_cal = gamma * z_norm + delta
        tcm_scores = torch.sigmoid(z_cal)
        tcm_scores = tcm_scores.squeeze()
        tcm_result = sorted(
            [
                (
                    drug_names[i],
                    float(score)
                )
                for i, score in zip(
                    tcm_idx,
                    tcm_scores.cpu().numpy()
                )
            ],
            key=lambda x: x[1],
            reverse=True
        )
        ####################################################
        ####################################################
        non_tcm_result=[]
        if len(non_tcm_idx)>0:
            non_tcm_idx_tensor=torch.tensor(
                non_tcm_idx,
                dtype=torch.long,
                device=device
            )
            non_tcm_emb=h_drug[
                non_tcm_idx_tensor
            ]
            non_tcm_dis_emb=disease_emb.repeat(
                non_tcm_emb.size(0),
                1
            )
            non_tcm_logits=predictor(
                non_tcm_emb,
                non_tcm_dis_emb
            )
            ################################################
            ################################################
            non_tcm_scores=torch.sigmoid(non_tcm_logits/2)
            non_tcm_scores=non_tcm_scores.squeeze()
            non_tcm_scores=non_tcm_scores ** 0.5
            non_tcm_result=sorted(
                [
                    (
                        drug_names[i],
                        float(score)
                    )
                    for i,score in zip(
                        non_tcm_idx,
                        non_tcm_scores.cpu().numpy()
                    )
                ],
                key=lambda x:x[1],
                reverse=True
            )
    ####################################################
    ####################################################
    return (
        h_dis[new_id].cpu().numpy(),
        tcm_result[:drug_top_k],
        non_tcm_result[:drug_top_k],
        tcm_result,
        non_tcm_result,
        similar_disease_info,
        sim_idx.cpu().numpy(),
        topk_vals.cpu().numpy(),
        gate.cpu().numpy()
    )

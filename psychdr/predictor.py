import torch
import torch.nn.functional as F
import os
import numpy as np
import pandas as pd
import re
from collections import Counter, defaultdict, deque
from transformers import AutoTokenizer, AutoModel
from huggingface_hub import snapshot_download

# 导入包内的其他模块
from psychdr.data_loader import build_hetero_data
from psychdr.models import FeatureReducer, HGTWithDiseaseSim, LinkPredictor
from psychdr.similarity import (
    parse_mesh_with_synonyms,
    normalize_name,
    fuzzy_map_disease_to_mesh_tns,
    compute_protein_similarity,
    compute_mesh_similarity
)
from psychdr.inference import predict_new_disease_drugs
from psychdr.utils import normalize_matrix_minmax


def ensure_biobert_model(biobert_path):
    """
    检查本地 BioBERT 模型是否完整。
    如果缺少模型权重，则从 Hugging Face 自动下载。
    """
    model_id = "dmis-lab/biobert-base-cased-v1.1"

    required_files = [
        "config.json",
        "pytorch_model.bin",
        "vocab.txt"
    ]

    missing_files = [
        filename
        for filename in required_files
        if not os.path.isfile(os.path.join(biobert_path, filename))
    ]

    if missing_files:
        print("   - 本地 BioBERT 模型文件不完整，开始下载...")
        print(f"   - 缺失文件: {missing_files}")
        print(f"   - 模型来源: {model_id}")

        os.makedirs(biobert_path, exist_ok=True)

        snapshot_download(
            repo_id=model_id,
            local_dir=biobert_path
        )

        print("   - BioBERT 模型下载完成！")

    else:
        print("   - 检测到完整的本地 BioBERT 模型，直接加载。")


class PsychDRPredictor:
    def __init__(
        self,
        data_dir,
        weights_path,
        mesh_xml_path,
        gene_score_path,
        tcm_list_path,
        biobert_path,
        disease_bert_csv_path,
        device="cpu"
    ):
        """
        初始化 PsychDR 预测引擎
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        print(f"⚙️ 正在初始化 PsychDR 引擎 ({self.device})...")

        # ================= 1. 加载背景图谱 =================
        print("   - 加载异构图数据...")
        self.data, self.id_map, self.nodes, self.edges = build_hetero_data(
            data_dir=data_dir,
            drug_feat_file="drug_features.csv",
            protein_feat_file="protein_features.csv",
            disease_feat_file="disease_features.csv",
            nodes_file="node.csv",
            edges_file="edge.csv"
        )

        self.disease_names = list(self.nodes[self.nodes['type'] == "Disease"]["name"])
        self.protein_names = list(self.nodes[self.nodes['type'] == "Protein"]["name"])
        self.protein_set = set(self.protein_names)

        # ================= 2. 预计算全局相似度 =================
        print("   - 计算蛋白质相似度...")
        df_gene = pd.read_excel(gene_score_path)
        self.protein_sim = compute_protein_similarity(self.disease_names, self.protein_set, self.edges, df_gene)

        print("   - 计算 MeSH 相似度与映射...")
        # 提取疾病与蛋白映射、权重 (基于背景图)
        # 提取疾病与蛋白映射、药物靶点和 PPI 网络
        self.disease_to_protein = defaultdict(set)
        drug_target_proteins = set()
        ppi_graph = defaultdict(list)

        for _, row in self.edges.iterrows():
            p, d = row.get("from"), row.get("to")

            if row["edge_type"] == "Protein-Disease" and p in self.protein_set and d in self.disease_names:
                self.disease_to_protein[d].add(p)

            elif row["edge_type"] == "Protein-Drug" and p in self.protein_set:
                drug_target_proteins.add(p)

            elif row["edge_type"] == "Protein-Protein" and p in self.protein_set and d in self.protein_set:
                ppi_graph[p].append(d)
                ppi_graph[d].append(p)

        # Seed proteins
        seed_proteins = set(df_gene["gene_symbol"].dropna().tolist()) & self.protein_set

        # BFS: 计算药物靶点到 seed proteins 的最短距离
        def bfs_min_dist(start, targets, graph, max_steps=3):
            if start in targets:
                return 0

            visited = {start}
            queue = deque([(start, 0)])

            while queue:
                node, dist = queue.popleft()

                if dist >= max_steps:
                    continue

                for nxt in graph[node]:
                    if nxt in visited:
                        continue

                    if nxt in targets:
                        return dist + 1

                    visited.add(nxt)
                    queue.append((nxt, dist + 1))

            return None

        # 计算药物靶点到 seed proteins 的距离
        drug_target_dist = {}

        for p in drug_target_proteins:
            drug_target_dist[p] = bfs_min_dist(
                p,
                seed_proteins,
                ppi_graph,
                max_steps=3
            )

        # 按照原始 notebook 的规则计算蛋白质权重
        self.protein_weights = defaultdict(float)

        alpha = 1.0
        beta = 0.7
        gamma = 0.5

        for d in self.disease_names:
            for p in self.disease_to_protein[d]:

                # Rule 1: Disease protein is also a drug target
                if p in drug_target_proteins:
                    self.protein_weights[p] = max(
                        self.protein_weights[p], alpha
                    )

                # Rule 2: Disease protein directly interacts with a drug target
                elif any(
                    n in drug_target_proteins
                    for n in ppi_graph.get(p, [])
                ):
                    self.protein_weights[p] = max(
                        self.protein_weights[p], beta
                    )

        # Rule 3: Drug targets connected to seed proteins within 3 PPI steps
        for p in drug_target_proteins:
            dist = drug_target_dist[p]

            if dist is not None:
                self.protein_weights[p] = max(
                    self.protein_weights[p],
                    gamma / (dist + 1)
                )

        # 解析 MeSH
        self.mesh_info, self.mesh_name_index = parse_mesh_with_synonyms(mesh_xml_path)
        self.token_to_uis = {}
        for ui, info in self.mesh_info.items():
            names = [info.get("name","")] + info.get("synonyms",[])
            seen = set()
            for nm in names:
                for t in normalize_name(nm).split():
                    if t not in seen:
                        self.token_to_uis.setdefault(t, set()).add(ui)
                        seen.add(t)

        # 通过名字模糊匹配背景疾病得到 MeSH 字典
        self.disease_to_mesh_fuzzy = fuzzy_map_disease_to_mesh_tns(
            self.disease_names, self.mesh_name_index, self.mesh_info, self.token_to_uis
        )

        # 自动权重
        p_norm = normalize_matrix_minmax(self.protein_sim)
        self.mesh_sim_matrix, _ = compute_mesh_similarity(self.disease_names, mesh_xml_path)
        m_norm = normalize_matrix_minmax(self.mesh_sim_matrix)

        vars = np.array([np.var(p_norm), np.var(m_norm)], dtype=np.float32)
        weights = vars / vars.sum()
        self.w_protein, self.w_mesh = float(weights[0]), float(weights[1])

        # ================= 3. 加载模型与权重 =================
        print("   - 加载预训练模型...")
        node_in_dims = {ntype: self.data[ntype].x.size(1) for ntype in self.data.node_types}
        node_out_dims = {'drug':512, 'protein':256, 'disease':256}

        self.feature_reducer = FeatureReducer(node_in_dims, node_out_dims).to(self.device)
        self.model = HGTWithDiseaseSim(hidden_channels=256, metadata=self.data.metadata(), node_in_dims=node_out_dims).to(self.device)
        self.predictor = LinkPredictor(256).to(self.device)

        checkpoint = torch.load(weights_path, map_location=self.device)
        self.feature_reducer.load_state_dict(checkpoint['feature_reducer'])
        self.model.load_state_dict(checkpoint['model'])
        self.predictor.load_state_dict(checkpoint['predictor'])

        # ================= 4. 加载字典与边缘信息 =================
        self.chinese_medicine_set = set(pd.read_csv(tcm_list_path, index_col=0).index.tolist())
        self.drug_idx_to_name = {v: k for k, v in self.id_map['Drug'].items()}
        # 推理时使用全图边进行拓扑传播
        self.ddi_train_edges = self.data[('drug', 'Drug-Disease', 'disease')].edge_index

        # ================= 5. 加载 BioBERT 与背景向量 =================
        print("   - 加载 BioBERT 编码器与背景疾病向量库...")
        self.bg_bert_df = pd.read_csv(disease_bert_csv_path, index_col=0)
        self.bg_disease_names = self.bg_bert_df.index.tolist()
        self.bg_bert_tensor = torch.tensor(self.bg_bert_df.values, dtype=torch.float32).to(self.device)
        self.bg_bert_tensor = F.normalize(self.bg_bert_tensor, p=2, dim=1) # L2 归一化

        ensure_biobert_model(biobert_path)
        self.tokenizer = AutoTokenizer.from_pretrained(biobert_path,local_files_only=True)
        self.bert_model = AutoModel.from_pretrained(biobert_path,local_files_only=True).to(self.device)
        self.bert_model.eval()

        print("✅ 引擎初始化完毕！\n")

    @torch.no_grad()
    def _encode_description(self, text, max_length=256):
        """使用 BioBERT 将文本编码为 768 维 CLS 向量"""
        inputs = self.tokenizer([text], return_tensors="pt", truncation=True, padding=True, max_length=max_length)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.bert_model(**inputs)
        cls_emb = outputs.last_hidden_state[:, 0, :] # 提取 CLS 向量
        return F.normalize(cls_emb, p=2, dim=1)      # 归一化

    @torch.no_grad()
    def _infer_mesh_from_description(self, description):
        """【未知突变体专用】通过 BioBERT 相似度匹配最接近的已知疾病及其 MeSH"""
        desc_vec = self._encode_description(description)
        sim_scores = torch.matmul(desc_vec, self.bg_bert_tensor.T).squeeze(0)

        best_idx = torch.argmax(sim_scores).item()
        best_score = sim_scores[best_idx].item()
        best_disease = self.bg_disease_names[best_idx]

        print(f"🔍 [突变体] BioBERT 描述匹配: 最相似的已知疾病 -> {best_disease} (相似度: {best_score:.4f})")

        mesh_trees = self.disease_to_mesh_fuzzy.get(best_disease, [])
        print(f"   👉 自动获取对应的 MeSH Tree: {mesh_trees}")
        return mesh_trees

    def predict(self, new_disease_name, description, deg_df, user_mesh_tree=None, top_k=10):
        print(f"🔮 正在为 [{new_disease_name}] 生成推荐...")

        # 1. 生成 BioBERT 特征向量
        text_to_encode = f"{new_disease_name}: {description}" if description else new_disease_name
        desc_vec = self._encode_description(text_to_encode).cpu().numpy().flatten()

        # 2. 级联确定 MeSH Tree Number
        mesh_tree = []
        is_valid_disease_mesh = False

        if user_mesh_tree is not None:
            print(f"✅ 使用用户指定的 MeSH Tree: {user_mesh_tree}")
            mesh_tree = user_mesh_tree
        else:
            print(f"   - 尝试通过疾病名称 [{new_disease_name}] 匹配 MeSH...")
            mapped_dict = fuzzy_map_disease_to_mesh_tns(
                [new_disease_name], self.mesh_name_index, self.mesh_info, self.token_to_uis
            )
            mapped_trees = mapped_dict.get(new_disease_name, [])

            if mapped_trees:
                if any(t.startswith('C') or t.startswith('F') for t in mapped_trees):
                    print(f"✅ [已知疾病] 匹配到有效的疾病 MeSH Tree: {mapped_trees}")
                    mesh_tree = mapped_trees
                    is_valid_disease_mesh = True
                else:
                    print(f"⚠️ 名字 [{new_disease_name}] 匹配到了非疾病实体 (如物种): {mapped_trees}。已忽略！")

            # 如果没有匹配到有效的疾病树号，强制走 BioBERT 描述匹配
            if not is_valid_disease_mesh:
                if description:
                    print(f"🚀 启动 BioBERT 行为学/表型描述匹配...")
                    mesh_tree = self._infer_mesh_from_description(text_to_encode)
                else:
                    print("⚠️ 未提供描述，无法计算 MeSH 相似度。")

        # 将最终算好的 mesh_tree 直接塞进字典
        if mesh_tree:
            self.disease_to_mesh_fuzzy[new_disease_name] = mesh_tree

        import psychdr.inference as inference_module
        inference_module.w_protein = self.w_protein
        inference_module.w_mesh = self.w_mesh

        new_disease_emb,tcm, non_tcm,all_tcm,all_non_tcm,similar_disease_info,sim_idx,topk_vals,gate = predict_new_disease_drugs(
            new_disease_name=new_disease_name,
            disease_bert_vec=desc_vec,
            filtered_df=deg_df,
            data=self.data,
            id_map=self.id_map,
            disease_names=self.disease_names,
            disease_to_protein=self.disease_to_protein,
            protein_weights=self.protein_weights,
            protein_feat=self.data['protein'].x,
            protein_set=self.protein_set,
            disease_to_mesh_fuzzy=self.disease_to_mesh_fuzzy,
            semantic_lower_bound_auto=None,
            drug_idx_to_name=self.drug_idx_to_name,
            chinese_medicine_set=self.chinese_medicine_set,
            ddi_train_edges=self.ddi_train_edges,
            feature_reducer=self.feature_reducer,
            model=self.model,
            predictor=self.predictor,
            device=self.device,
            drug_top_k=top_k
        )

        return {"TCM": tcm, "Non_TCM": non_tcm,  "All_TCM": all_tcm, "All_Non_TCM": all_non_tcm}

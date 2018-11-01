import os
import pandas as pd
from psychdr.predictor import PsychDRPredictor

def main():
    # 1. 设置路径（请改成你实际的文件路径）
    DATA_DIR = "./data" 
    WEIGHTS = "../saved_weights/psychdr_pretrained.pth"
    MESH_XML = "./data/desc2025.xml"
    GENE_SCORE = "./data/NND_gene_score.xlsx"
    TCM_LIST = "./data/TCM_ingredient_smiles_all_1104.csv"
    BIOBERT_PATH = "dmis-lab/biobert-base-cased-v1.1" # 让 transformers 自动下载
    BERT_CSV = "./data/disease_nodes_biobert_cls_features_20261104.csv"    

    # 2. 初始化引擎
    engine = PsychDRPredictor(
        data_dir=DATA_DIR, weights_path=WEIGHTS, mesh_xml_path=MESH_XML, 
        gene_score_path=GENE_SCORE, tcm_list_path=TCM_LIST,
        biobert_path=BIOBERT_PATH, disease_bert_csv_path=BERT_CSV,
        device="cpu"
    )
    
    # ==========================================
    # 场景 A：测试已知疾病 (Parkinson Disease)
    # ==========================================
    print("\n" + "="*50)
    print("测试场景 A：已知疾病")
    print("="*50)
    
    # 读取差异基因
    pd_deg = pd.read_csv("/public/home/liyu/Disease_drug/TXGNN/indication/No repeat mental illness/new_kg/k2_256_0.1/中药重复采样5次/不同疾病的模型评估/GSE7621_PD_limma_all_results.csv", index_col=0)
    filtered_df = pd_deg[pd_deg['P.Value'] < 0.001].copy()
    filtered_df['GeneSymbol'] = filtered_df['GeneSymbol'].str.upper()
    filtered_df = filtered_df.rename(columns={'GeneSymbol': 'gene_symbol'})
    
    results_pd = engine.predict(
        new_disease_name="Parkinson Disease",
        description="A progressive neurodegenerative disorder characterized by resting tremor, bradykinesia, and rigidity.",
        deg_df=filtered_df,
        top_k=10
    )
    
    print("\n非中药推荐 Top 5:", results_pd["Non_TCM"][:5])
    print("中药推荐 Top 5:", results_pd["TCM"][:5])
    
    # ==========================================
    # 场景 B：测试未知突变体 (Tcf20 Mutant)
    # ==========================================
    print("\n" + "="*50)
    print("测试场景 B：未知突变体 (模拟)")
    print("="*50)
    
    # 为了演示，换成自闭症/神经发育相关的基因
    dummy_df = pd.DataFrame({'gene_symbol': ['TCF20', 'SHANK3', 'MECP2', 'PTEN', 'CHD8']})    
    results_tcf20 = engine.predict(
        new_disease_name="Tcf20_KO_Model_XYZ",
        description="The mouse exhibits social deficits, repetitive behavior, anxiety, and cognitive impairment, which are typical autism-like phenotypes.",
        deg_df=dummy_df,
        top_k=10
    )
    
    print("\n非中药推荐 Top 5:", results_tcf20["Non_TCM"][:5])
    print("中药推荐 Top 5:", results_tcf20["TCM"][:5])

if __name__ == "__main__":
    main()
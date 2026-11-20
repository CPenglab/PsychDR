
import os
import re
import pandas as pd

from psychdr.predictor import PsychDRPredictor


# ============================================================
# 1. User inputs: only modify these three variables
# ============================================================

DISEASE_NAME = ""

DISEASE_DESCRIPTION = (
    ""
)

DEG_FILE = ""


# ============================================================
# 2. Prediction settings
# ============================================================

TOP_K = 10
P_VALUE_THRESHOLD = 0.05

# Output directory
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "recommendation_results"
)


# ============================================================
# 3. Load DEG data
# ============================================================

def load_deg_data(deg_file, pvalue_threshold=0.05):

    if not os.path.isabs(deg_file):
        deg_file = os.path.join(PROJECT_ROOT, deg_file)

    if not os.path.exists(deg_file):
        raise FileNotFoundError(
            f"DEG file not found: {deg_file}"
        )

    print("\n" + "=" * 60)
    print("Loading differential expression data")
    print("=" * 60)

    df = pd.read_csv(deg_file)

    required_columns = ["gene_symbol", "pvalue"]

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}\n"
            f"Your DEG CSV must contain: "
            f"{required_columns}"
        )

    # Remove missing gene symbols and p-values
    df = df.dropna(
        subset=["gene_symbol", "pvalue"]
    ).copy()

    # Standardize gene symbols
    df["gene_symbol"] = (
        df["gene_symbol"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    # Convert p-values to numeric
    df["pvalue"] = pd.to_numeric(
        df["pvalue"],
        errors="coerce"
    )

    df = df.dropna(
        subset=["pvalue"]
    )

    # Filter significant DEGs
    df = df[
        df["pvalue"] < pvalue_threshold
    ].copy()

    # Remove empty gene symbols
    df = df[
        df["gene_symbol"] != ""
    ]

    # Remove duplicated genes, retaining the smallest p-value
    df = (
        df.sort_values("pvalue")
        .drop_duplicates(
            subset=["gene_symbol"],
            keep="first"
        )
        .reset_index(drop=True)
    )

    if df.empty:
        raise ValueError(
            "No significant DEGs were found. "
            "Please check your DEG file and p-value threshold."
        )

    print(f"DEG file: {deg_file}")
    print(f"P-value threshold: {pvalue_threshold}")
    print(f"Number of significant DEGs: {len(df)}")

    return df


# ============================================================
# 4. Save prediction results
# ============================================================

def save_drug_results(results, disease_name, output_dir):

    os.makedirs(output_dir, exist_ok=True)

    # Save all candidates and Top-K candidates
    save_items = {
        "all_non_tcm": results["All_Non_TCM"],
        "all_tcm": results["All_TCM"],
        "top_non_tcm": results["Non_TCM"],
        "top_tcm": results["TCM"]
    }

    # Make disease name suitable for filenames
    safe_name = re.sub(
        r'[\\/:*?"<>|]+',
        "_",
        disease_name.strip()
    ).replace(" ", "_")

    print("\n" + "=" * 60)
    print("Saving prediction results")
    print("=" * 60)

    for result_type, drugs in save_items.items():

        df = pd.DataFrame(
            drugs,
            columns=["Drug", "Score"]
        )

        df.insert(
            0,
            "Rank",
            range(1, len(df) + 1)
        )

        df.insert(
            0,
            "Disease",
            disease_name
        )

        filename = (
            f"{safe_name}_{result_type}_predictions.csv"
        )

        output_path = os.path.join(
            output_dir,
            filename
        )

        df.to_csv(
            output_path,
            index=False,
            encoding="utf-8-sig"
        )

        print(
            f"Saved: {output_path} "
            f"({len(df)} drugs)"
        )


# ============================================================
# 5. Main prediction workflow
# ============================================================

def main():

    print("\n" + "=" * 60)
    print("PsychDR: New Disease Drug Recommendation")
    print("=" * 60)

    print(f"Disease name: {DISEASE_NAME}")
    print(f"Disease description: {DISEASE_DESCRIPTION}")

    # --------------------------------------------------------
    # Step 1. Initialize PsychDR
    # --------------------------------------------------------

    print("\n[Step 1] Initializing PsychDR...")

    engine = PsychDRPredictor(

        data_dir=os.path.join(
            PROJECT_ROOT,
            "data"
        ),

        weights_path=os.path.join(
            PROJECT_ROOT,
            "saved_weights",
            "psychdr_pretrained.pth"
        ),

        mesh_xml_path=os.path.join(
            PROJECT_ROOT,
            "data",
            "desc2025.xml"
        ),

        gene_score_path=os.path.join(
            PROJECT_ROOT,
            "data",
            "NND_gene.xlsx"
        ),

        tcm_list_path=os.path.join(
            PROJECT_ROOT,
            "data",
            "TCM.csv"
        ),

        biobert_path=os.path.join(
            PROJECT_ROOT,
            "pretrained_models",
            "biobert-base-cased-v1.1"
        ),

        disease_bert_csv_path=os.path.join(
            PROJECT_ROOT,
            "data",
            "disease_features.csv"
        ),

        device="cpu"
    )

    # --------------------------------------------------------
    # Step 2. Load DEG data
    # --------------------------------------------------------

    print("\n[Step 2] Loading DEG data...")

    deg_df = load_deg_data(
        DEG_FILE,
        pvalue_threshold=P_VALUE_THRESHOLD
    )

    # --------------------------------------------------------
    # Step 3. Predict candidate drugs
    # --------------------------------------------------------

    print("\n[Step 3] Predicting candidate drugs...")

    results = engine.predict(

        new_disease_name=DISEASE_NAME,

        description=DISEASE_DESCRIPTION,

        deg_df=deg_df,

        top_k=TOP_K
    )

    # --------------------------------------------------------
    # Step 4. Display Top-K results
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("Top candidate drugs")
    print("=" * 60)

    print("\nTop non-TCM drugs:")

    for rank, (drug, score) in enumerate(
        results["Non_TCM"],
        start=1
    ):
        print(
            f"{rank:>3}. {drug:<35} Score: {score:.4f}"
        )

    print("\nTop TCM drugs:")

    for rank, (drug, score) in enumerate(
        results["TCM"],
        start=1
    ):
        print(
            f"{rank:>3}. {drug:<35} Score: {score:.4f}"
        )

    # --------------------------------------------------------
    # Step 5. Save all prediction results
    # --------------------------------------------------------

    print("\n[Step 4] Saving results...")

    save_drug_results(
        results=results,
        disease_name=DISEASE_NAME,
        output_dir=OUTPUT_DIR
    )

    print("\n" + "=" * 60)
    print("PsychDR prediction completed!")
    print(f"Results saved in: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()

# PsychDR: Knowledge Graph-Driven Similar Disease Search for Drug Repurposing of Psychiatric Disorders

## Overview

PsychDR is a computational approach of drug repurposing, by using expression and symptom data to search similar diseases in the constructed biomedical knowledge graph to explore drug recommendations for psychiatric disorders outside the knowledge graph.

PsychDR generates ranked candidate lists for both traditional Chinese medicine (TCM) and non-TCM compounds.

## Installation

### Requirements

- Python 3.10+
- PyTorch
- PyTorch Geometric
- Transformers
- Hugging Face Hub
- NumPy
- Pandas
- Scikit-learn
- SciPy
- OpenPyXL
- lxml
- tqdm

### Environment setup

Clone the repository:

```bash
git clone https://github.com/liyu-ynu/PsychDR.git
cd PsychDR
```

Create and activate the Python environment:

```bash
conda create -n psychdr python=3.10
conda activate psychdr
```

Install PyTorch according to your computing environment. For CPU-based installation, for example:

```bash
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
```

Then install the remaining project dependencies:

```bash
pip install -r requirements.txt
```

The requirements file installs PyTorch Geometric and the PsychDR package dependencies. The `-e .` entry in `requirements.txt` installs the local PsychDR package using the installation configuration defined in `setup.py`.

Please ensure that the installed PyTorch and PyTorch Geometric versions are compatible with each other and your computing environment. For GPU acceleration, select the appropriate PyTorch build according to your CUDA version and hardware configuration.

PsychDR supports CPU-based inference. GPU acceleration can be configured when compatible hardware and dependencies are available.

## Data Availability and Preparation

The required data files and pretrained resources should be placed in the following directory structure:

```text
PsychDR/
├── examples/
│   └── predict_new_disease.py
├── main_train.py
├── psychdr/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── inference.py
│   ├── models.py
│   ├── predictor.py
│   ├── similarity.py
│   ├── trainer.py
│   └── utils.py
├── saved_weights/
│   └── psychdr_pretrained.pth
├── pretrained_models/
│   └── biobert-base-cased-v1.1/
│       ├── config.json
│       ├── special_tokens_map.json
│       ├── tokenizer_config.json
│       └── vocab.txt
├── data/
│   ├── desc2025.xml
│   ├── disease_features.csv
│   ├── drug_features.csv
│   ├── protein_features.csv
│   ├── node.csv
│   ├── edge.csv
│   ├── NND_gene.xlsx
│   └── TCM.csv
├── requirements.txt
├── setup.py
└── README.md
```

**Important:** Due to GitHub file size limitations, `protein_features.csv`, `desc2025.xml`, and the BioBERT model weights are not included in the GitHub repository. These resources must be obtained separately or downloaded automatically as described below.

### Required data files

| File | Description | Distribution |
|---|---|---|
| `drug_features.csv` | Molecular feature representations of drug nodes | GitHub |
| `protein_features.csv` | Protein feature representations | External download |
| `disease_features.csv` | BioBERT-derived feature representations of background diseases | GitHub |
| `node.csv` | Node information for the heterogeneous knowledge graph | GitHub |
| `edge.csv` | Relationship information for the heterogeneous knowledge graph | GitHub |
| `NND_gene.xlsx` | Gene information used for protein-based disease similarity calculation | GitHub |
| `TCM.csv` | List of traditional Chinese medicine compounds | GitHub |
| `desc2025.xml` | MeSH descriptor data used for disease name matching and semantic similarity calculation | NLM |

#### 1. Protein features

The `protein_features.csv` file contains protein feature representations required for constructing the heterogeneous knowledge graph.

Download the file from the following location:

**Download:** [Protein feature data](<PROTEIN_FEATURES_DOWNLOAD_URL>)

After downloading, place it in:

```text
data/protein_features.csv
```

**Note:** Use the protein feature file distributed with this project. Replacing it with a differently generated or processed version may affect compatibility with the pretrained PsychDR model.

#### 2. MeSH descriptor data

PsychDR uses the 2025 Medical Subject Headings (MeSH) descriptor dataset.

Download `desc2025.xml` from the official National Library of Medicine (NLM) archive:

[Download MeSH 2025 Descriptor XML](https://nlmpubs.nlm.nih.gov/projects/mesh/MESH_FILES/NLM_only/desc2025.xml)

Alternatively, visit the [NLM MeSH Data Download page](https://www.nlm.nih.gov/databases/download/mesh.html).

Place the downloaded file in:

```text
data/desc2025.xml
```

#### 3. Pretrained PsychDR model

The pretrained PsychDR model weights are included in the GitHub repository.

The model weights should be located at:

```text
saved_weights/psychdr_pretrained.pth
```

No additional model download is required for the pretrained PsychDR model.

### Pretrained BioBERT

PsychDR uses the pretrained BioBERT model `dmis-lab/biobert-base-cased-v1.1` for phenotype description encoding.

The tokenizer and model configuration files are included in the repository. If the required pretrained model files are missing locally, PsychDR automatically downloads the model from Hugging Face during initialization.

The model is stored in:

```text
pretrained_models/biobert-base-cased-v1.1/
```

An active internet connection is required for the initial download. Subsequent runs can use the locally stored model.

Users do not need to manually download the BioBERT model weights.

## Drug Recommendation for New Diseases

PsychDR provides a unified prediction script for drug recommendation in new diseases and phenotype-based disease models.

Users only need to specify three input variables in:

```text
examples/predict_new_disease.py
```

### Step 1. Specify disease information

Open `examples/predict_new_disease.py` and modify the following three variables:

```python
DISEASE_NAME = ""

DISEASE_DESCRIPTION = (
    ""
)

DEG_FILE = ""
```

- `DISEASE_NAME`: Name of the disease or phenotype model.
- `DISEASE_DESCRIPTION`: Clinical description or major phenotypic characteristics.
- `DEG_FILE`: Path to the differential gene expression (DEG) file.

The same script supports both named diseases and unnamed mutant or phenotype models. The MeSH matching and BioBERT fallback procedures are handled automatically.

### Step 2. Prepare the DEG file

The DEG file must be provided in CSV format and contain at least the following two columns:

| Column | Description |
|---|---|
| `gene_symbol` | Gene symbol |
| `pvalue` | P-value from differential expression analysis |

Example:

```csv
gene_symbol,pvalue
SNCA,0.0001
LRRK2,0.002
PINK1,0.003
PARK7,0.01
```

Gene symbols are automatically converted to uppercase. Genes with missing values are removed, and duplicate gene symbols are handled by retaining the entry with the smallest P-value.

By default, genes with `pvalue < 0.05` are retained for prediction.

### Step 3. Run drug recommendation

From the project root directory, execute:

```bash
python -u examples/predict_new_disease.py
```

If the package cannot be imported from the project root, use:

```bash
PYTHONPATH=. python -u examples/predict_new_disease.py
```

### Step 4. Examine prediction results

The prediction results are automatically saved in:

```text
recommendation_results/
```

Four CSV files are generated:

| Output file | Description |
|---|---|
| `<Disease>_all_non_tcm_predictions.csv` | All ranked non-TCM drug candidates |
| `<Disease>_all_tcm_predictions.csv` | All ranked TCM drug candidates |
| `<Disease>_top_non_tcm_predictions.csv` | Top-ranked non-TCM drug candidates |
| `<Disease>_top_tcm_predictions.csv` | Top-ranked TCM drug candidates |

Each output file contains the following columns:

- `Disease`: Input disease name.
- `Rank`: Ranking of the candidate drug.
- `Drug`: Drug name.
- `Score`: Predicted drug–disease association score.

The Top-K lists and complete candidate lists are generated separately for TCM and non-TCM compounds.

## Prediction Parameters

The default prediction parameters are defined in `examples/predict_new_disease.py`.

| Parameter | Default value | Description |
|---|---|---|
| `TOP_K` | `10` | Number of top-ranked candidates returned for each drug category |
| `P_VALUE_THRESHOLD` | `0.05` | P-value threshold for DEG filtering |
| `device` | `"cpu"` | Computing device used for prediction |

Users can modify these parameters according to their requirements.

For example, to obtain the top 20 candidate drugs and use a more stringent DEG threshold:

```python
TOP_K = 20
P_VALUE_THRESHOLD = 0.01
```

The complete candidate lists are saved independently of the `TOP_K` setting.

---

For questions or issues regarding PsychDR, please open an issue in this GitHub repository.
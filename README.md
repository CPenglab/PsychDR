# PsychDR: Phenotype-Driven Drug Recommendation for Neuropsychiatric Diseases

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**PsychDR** is a novel end-to-end Heterogeneous Graph Transformer (HGT) framework designed for drug repurposing in neuropsychiatric diseases. It leverages transcriptomic data (DEGs) and phenotypic/behavioral descriptions to recommend potential therapeutic drugs, seamlessly separating **Traditional Chinese Medicine (TCM)** and **Non-TCM** candidates.

## 🌟 Key Features

- **Phenotype-Driven Matching:** Integrates BioBERT to automatically map unknown mutant models (e.g., Tcf20 KO mice) to known human diseases based on behavioral descriptions.
- **Cascaded MeSH Retrieval:** Automatically retrieves precise MeSH Tree Numbers through fuzzy string matching for known diseases, or semantic embedding matching for unknown models.
- **Dual-Track Recommendation:** Simultaneously provides ranked lists for both TCM and non-TCM drugs, enabling flexible drug discovery strategies.
- **Multi-view Similarity Fusion:** Combines protein-protein interaction (PPI) network similarity with MeSH semantic similarity for robust disease representations.

## 📦 Installation

1. Clone the repository:
```bash
git clone https://github.com/your_username/PsychDR.git
cd PsychDR
# Project A - Section B: Hybrid Search Engine

### Team Members
- Ori Hillel
- Harel Tzoran

## Pre-computed Artifacts (Git LFS)
To allow a seamless run during grading without needing an index rebuild, all pre-computed artifacts are tracked and saved directly in the repository via Git LFS:

- `artifacts/index_vectors.npy`: Pre-computed dense embedding matrix.
- `artifacts/index_meta.json`: Core mapping coordinates linking vector indices to document and chunk IDs.
- `artifacts/bm25_data.json`: Pre-computed inverted index database and document lengths used for BM25 calculation (1GB file handled via LFS).

## Setup & Execution

### 1. Clone the Repository
Open your terminal and run the following commands to clone the repository and navigate into the project directory:

```bash
git clone [https://github.com/ori-hillel/DataAnalysisLab_ProjectA_SecB.git](https://github.com/ori-hillel/DataAnalysisLab_ProjectA_SecB.git)
cd DataAnalysisLab_ProjectA_SecB
```

### 2. Dependencies
Install the required python libraries using the provided requirements file (in case you haven't already):

```bash
pip install -r requirements.txt
```

### 3. Running Evaluation
Since the pre-computed indices are stored inside the repository via Git LFS, the evaluation suite runs out-of-the-box on a fresh clone without any manual compilation or rebuilding steps.

Run the evaluation script from the root directory:

```bash
python scripts/eval_public.py
```
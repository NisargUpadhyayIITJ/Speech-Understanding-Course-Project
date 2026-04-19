# Speech Understanding course project

[![Hugging Face weights](https://img.shields.io/badge/Hugging%20Face-Model-blue)](https://huggingface.co/bappaiitj/deepfake_detection_audio)


This repository extends the original **AASIST3** implementation with a series of architectural experiments, replacing and augmenting key components to improve speech deepfake detection performance. Our best configuration — **Mamba + Fixed GraphSAGE + SupCon Loss** — outperforms all other explored variants on our evaluation subset.

---

## What We Changed (and Why)

The original AASIST3 uses KAN linear layers and Graph Attention Networks (GAT). We explored the following axes of modification:

| Component | Original | Our Variants |
|-----------|----------|--------------|
| Sequential modelling | — | **Mamba** (state-space model) |
| Graph layer | GAT | **GraphSAGE** (learnable & fixed aggregation) |
| Loss function | CrossEntropy | **SupCon** (Supervised Contrastive Loss) |

---

## Architectural Variants

### 1. Baseline (AASIST3)
The original AASIST3 architecture with KAN layers and GAT-based graph attention. Serves as our reference point.

### 2. KAN → Mamba
Replaces KAN linear transformation blocks with **Mamba** (Selective State Space Model) blocks. Mamba's selective scan mechanism is better suited to capturing long-range temporal dependencies in audio without the quadratic cost of attention.

### 3. GAT → GraphSAGE (Learnable)
Swaps the Graph Attention Networks for **GraphSAGE with learnable aggregation weights**. The aggregation function parameters are trained end-to-end, giving the model flexibility to learn node neighbourhood weighting from data.

### 4. GAT → GraphSAGE (Fixed)
Same as above, but with **fixed (non-learnable) aggregation** — mean pooling over neighbours. Surprisingly competitive; avoids overfitting to graph topology in limited-data regimes.

### 5. CE → SupCon Loss
Replaces CrossEntropy with **Supervised Contrastive Loss**, pulling embeddings of the same class together and pushing different classes apart in representation space before the final classifier head.
### 5. W2V2 → hubert
Replaced encoder which is more suitable of deepfake detection tasks.
---

##  Results on Evaluation Subset

> Metrics below are on our internal evaluation subset. Lower EER is better; higher min-tDCF is worse.

| Model Variant                                      | EER (%) ↓ | Notes                                   |
|---------------------------------------------------|----------|-----------------------------------------|
| Baseline (AASIST3)                                | 22.22    | Original KAN + GAT + CE                 |
| + Learnable GAT (reimpl.)                         | 76.33    | GAT with trained edge weights           |
| + Mamba (KAN replaced)                            | 12.23    | SSM-based temporal modelling            |
| + Learnable GraphSAGE                             | 73.33    | GraphSAGE, trainable aggregation        |
| **Final: Mamba + Fixed GraphSAGE + SupCon**       | **11.11**| **Best overall**                        |


---

## Overview

AASIST3+ retains the full pipeline of the original model while allowing plug-and-play swapping of graph and sequential components:

- **Wav2Vec2 Encoder** — SSL feature extraction from raw waveforms
- **KAN Bridge / Mamba Bridge** — Feature transformation (KAN in baseline; Mamba in best variant)
- **Residual Encoder** — Multi-scale residual blocks
- **Graph Layer** — GAT (baseline) or GraphSAGE (fixed/learnable) for relational modelling
- **Multi-branch Inference** — Four parallel branches with master tokens
- **Output Head** — KAN linear or standard linear + SupCon / CE loss

---

## Quick Start

### Installation

```bash
git clone https://github.com/NisargUpadhyayIITJ/Speech-Understanding-Course-Project
cd Speech-Understanding-Course-Project
pip install -r requirements.txt
```

### Load the Best Model

```python
from model import aasist3_mamba_sage

model = aasist3_mamba_sage.from_pretrained("MTUCI/AASIST3")
model.eval()
```

### Inference

```bash
python infer.py /path/to/audio.wav 

```

---

## Training Details

### Datasets

| Dataset | Split used |
|---------|-----------|
| ASVspoof 2019 LA | Train + Dev |
| ASVspoof 2024 (ASVspoof5) | Train |
| MLAAD | Train |


### Configuration

| Hyperparameter | Value |
|---|---|
| Epochs | 40 |
| Batch size (train) | 8 |
| Batch size (val) | 16 |
| Learning rate | 1e-4 |
| Optimizer | AdamW |
| Gradient accumulation | 2 steps |
| Loss (best model) | SupCon + CE |


### Training & Validation

```bash
bash train.sh      # train selected variant
bash validate.sh   # evaluate on test sets
```

Select the variant via `configs/train.yaml`:

```yaml
model:
  graph_layer: "sage_fixed"   # options: gat | sage_learnable | sage_fixed
  seq_module:  "mamba"        # options: kan | mamba
  loss:        "supcon"       # options: ce | supcon
```

---

## Key Findings

- **Mamba > KAN** for sequential modelling: state-space dynamics capture temporal spoofing artefacts more effectively than KAN transformations alone.
- **Fixed GraphSAGE ≥ Learnable GraphSAGE**: Fixed mean aggregation generalises better under limited graph-supervision, avoiding overfitting to spurious topology.
- **SupCon boosts separation**: Supervised contrastive loss produces better-separated bonafide/spoof clusters in embedding space, translating to lower EER at decision boundaries.
- **Combined effect is super-additive**: The three changes together outperform any single substitution.

---

## Citation



```bibtex
@inproceedings{borodin24_asvspoof,
  title     = {AASIST3: KAN-enhanced AASIST speech deepfake detection using SSL features
               and additional regularization for the ASVspoof 2024 Challenge},
  author    = {Kirill Borodin and Vasiliy Kudryavtsev and Dmitrii Korzh and Alexey Efimenko
               and Grach Mkrtchian and Mikhail Gorodnichev and Oleg Y. Rogov},
  year      = {2024},
  booktitle = {The Automatic Speaker Verification Spoofing Countermeasures Workshop (ASVspoof 2024)},
  pages     = {48--55},
  doi       = {10.21437/ASVspoof.2024-8},
}
```

---

## License

CC BY-NC-ND 4.0 — see [LICENSE](LICENSE).  
You may share with attribution. Commercial use and derivative distribution are not permitted.

**Disclaimer**: Research implementation. Model weights are for demonstration; exact paper numbers may differ.

# DAT305 Project 6: What does a pretrained ECG model actually see?

Group 27: Anders, Øystein and Håvard.

We run 1,100 ECGs from PTB-XL through the pretrained ECG model ECGFounder and get
1,024 numbers (an embedding) per ECG. We then examine which diagnoses the model mixes up,
using PCA, UMAP, k-NN agreement, silhouette score and a matrix showing which diagnoses are neighbors.

## Run in this order

| Step | File | What it does | Output |
|---|---|---|---|
| 1 | `1_utvalg.ipynb` | Selects ECGs with exactly one diagnosis, at least 50 % confidence, one per patient, 100 per class | `utvalg.csv` (1,100 ECGs, 11 classes) |
| 2 | `2_embeddings.ipynb` | Downloads the ECGs from PhysioNet and runs them through ECGFounder | `data/embeddings.npy` (1,100 × 1,024) |
| 3 | `3_analyse.ipynb` | PCA, UMAP, k-NN, silhouette, neighbor mixing and comparison across methods | Plots and numbers for the report |

`utvalg.csv` and `data/embeddings.npy` are included in the repository, so you can start at step 3
without a GPU or the model weights. The ECG plots at the end of step 3 need the signal copies in
`data/signaler/`, which are created in step 2.

## Setup

```
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Download the model weights (370 MB) from
https://huggingface.co/PKUDigitalHealth/ECGFounder and place the file here:
`checkpoint/12_lead_ECGFounder.pth`.

Step 2: open `2_embeddings.ipynb` and run all cells. The first run takes about half an hour,
because each ECG is downloaded from PhysioNet. The copies are saved in `data/signaler/`, so later runs are fast.

## Folders

| Folder | Contents | In git? |
|---|---|---|
| `ecgfounder/` | Model code from ECGFounder (`net1d.py`, `tasks.txt`, MIT license) | Yes |
| `utforsking/` | Early exploration: first embedding, Øystein's first run, the PTB-XL columns | Yes |
| `data/` | `embeddings.npy` (in git) and ECG copies in `data/signaler/` (not in git) | Partly |
| `checkpoint/` | Model weights | No |

The files in `utforsking/` show how we arrived at the setup. They are not part of the
solution, and the paths in them may be outdated.

## References

- Li et al. (2025). An electrocardiogram foundation model built on over 10 million recordings. *NEJM AI*, 2(7). https://doi.org/10.1056/AIoa2401033. Code: https://github.com/PKUDigitalHealth/ECGFounder
- Wagner et al. (2022). PTB-XL, a large publicly available electrocardiography dataset (version 1.0.3). PhysioNet. https://doi.org/10.13026/kfzx-aw45
- Wagner et al. (2020). PTB-XL, a large publicly available electrocardiography dataset. *Scientific Data*, 7, 154. https://doi.org/10.1038/s41597-020-0495-6
- Pollard et al. (2026). PhysioNet as a global platform for biomedical research. *Nature Health*, 1(8), 792–795. https://doi.org/10.1038/s44360-026-00096-z
- McInnes, Healy and Melville (2018). UMAP: Uniform manifold approximation and projection for dimension reduction. arXiv:1802.03426.
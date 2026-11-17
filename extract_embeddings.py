import numpy as np
import pandas as pd
import torch

from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm
from net1d import Net1D

# 1. Load ECG signals, IDs and metadata
X = np.load("ecg_signals.npy")
ecg_ids = np.load("ecg_ids.npy")
metadata = pd.read_csv("ptbxl_subset.csv")

print("Loaded ECG signals:", X.shape)
print("Loaded ECG IDs:", ecg_ids.shape)
print("Loaded metadata:", metadata.shape)

## Validate alignment
assert len(X) == len(ecg_ids)
assert len(X) == len(metadata)

assert np.array_equal(
    ecg_ids,
    metadata["ecg_id"].to_numpy()
), "ECG ID mismatch between signals and metadata!"

assert len(np.unique(ecg_ids)) == len(ecg_ids)
assert X.ndim == 3 and X.shape[1:] == (12, 5000)
assert np.isfinite(X).all()

print("Signal-to-metadata alignment verified!")

## Initialize ECGFounder
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)

model = Net1D(
    in_channels=12,
    base_filters=64,
    ratio=1,
    filter_list=[
        64, 160, 160, 400, 400, 1024, 1024
    ],
    m_blocks_list=[
        2, 2, 2, 3, 3, 4, 4
    ],
    kernel_size=16,
    stride=2,
    groups_width=16,
    n_classes=150,
    use_bn=False,
    use_do=False,
    return_features=True
)

checkpoint = torch.load(
    "checkpoint/12_lead_ECGFounder.pth",
    map_location="cpu",
    weights_only=False  # Trusted official checkpoint only
)

model.load_state_dict(
    checkpoint["state_dict"],
    strict=True
)

model = model.to(device).eval()

for param in model.parameters():
    param.requires_grad = False

## Create DataLoader with ECG IDs
dataset = TensorDataset(
    torch.from_numpy(X).float(),
    torch.from_numpy(ecg_ids)
)

loader = DataLoader(
    dataset,
    batch_size=16,
    shuffle=False
)
## Extract frozen embeddings
embedding_batches = []
id_batches = []

with torch.inference_mode():

    for batch, batch_ids in tqdm(
        loader,
        desc="Extracting embeddings"
    ):
        batch = batch.to(device)

        # ECGFounder returns logits and features
        logits, features = model(batch)

        embedding_batches.append(
            features.cpu().numpy()
        )

        id_batches.append(
            batch_ids.cpu().numpy()
        )

## Combine outputs
embeddings = np.concatenate(
    embedding_batches,
    axis=0
)

embedding_ids = np.concatenate(
    id_batches,
    axis=0
)

## Validate embedding alignment
assert len(embeddings) == len(embedding_ids)

assert np.array_equal(
    embedding_ids,
    ecg_ids
), "Embedding IDs do not match signal IDs!"

assert np.array_equal(
    embedding_ids,
    metadata["ecg_id"].to_numpy()
), "Embedding IDs do not match metadata!"

assert embeddings.ndim == 2
assert np.isfinite(embeddings).all()

print("\nEmbedding alignment verified!")
print("Embedding matrix:", embeddings.shape)
print("Corresponding ECG IDs:", embedding_ids.shape)

## Save embeddings and IDs

# Keep original standalone embedding file
np.save(
    "ecg_embeddings.npy",
    embeddings
)

# Save embeddings and IDs in the same file
np.savez_compressed(
    "ecg_embeddings_with_ids.npz",
    embeddings=embeddings,
    ecg_ids=embedding_ids
)

print("\nSaved files:")
print("  ecg_embeddings.npy")
print("  ecg_embeddings_with_ids.npz")
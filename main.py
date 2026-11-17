
import ast
from pathlib import PurePosixPath

import numpy as np
import pandas as pd
import wfdb
from tqdm.auto import tqdm

## Load metadata

df = pd.read_csv("ptbxl_database.csv")
scp = pd.read_csv("scp_statements.csv", index_col=0)

# Map diagnostic SCP codes to superclasses
diagnostic = scp[scp["diagnostic"] == 1]

code_to_class = (
    diagnostic["diagnostic_class"]
    .dropna()
    .to_dict()
)

df["scp_codes"] = df["scp_codes"].apply(ast.literal_eval)

def get_labels(codes):
    return sorted({
        code_to_class[code]
        for code in codes
        if code in code_to_class
    })


df["labels"] = df["scp_codes"].apply(get_labels)

## Select ECG recordings
# Exactly one diagnostic superclass
single = df[df["labels"].apply(len) == 1].copy()
single["label"] = single["labels"].str[0]

# Sample up to 400 ECGs per class
sample = pd.concat([
    group.sample(
        n=min(400, len(group)),
        random_state=42,
        replace=False
    )
    for _, group in single.groupby("label")
])

# Deterministic order
sample = (
    sample.sort_values("ecg_id")
    .reset_index(drop=True)
)

assert sample["ecg_id"].is_unique

print("Selected classes:")
print(sample["label"].value_counts())
print("Total selected recordings:", len(sample))

## Download ECG signals
signals = []
valid_indices = []
failed_records = []

for i, row in tqdm(
    sample.iterrows(),
    total=len(sample),
    desc="Downloading ECGs"
):
    try:
        # Example:
        # records500/14000/14706_hr
        record_path = PurePosixPath(
            str(row["filename_hr"])
        )

        record_name = record_path.name
        record_directory = record_path.parent.as_posix()

        signal, fields = wfdb.rdsamp(
            record_name=record_name,
            pn_dir=(
                f"ptb-xl/1.0.3/"
                f"{record_directory}"
            )
        )

        # Expected: 5000 samples, 12 leads
        if signal.shape != (5000, 12):
            raise ValueError(
                f"Unexpected signal shape: {signal.shape}"
            )

        # Convert to (12 leads, 5000 samples)
        signal = signal.T

        # Replace nonfinite values
        signal = np.nan_to_num(
            signal,
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        )

        # Global z-score normalization
        signal = (
            (signal - signal.mean()) /
            (signal.std() + 1e-8)
        )

        # Append signal and corresponding row index
        signals.append(signal.astype(np.float32))
        valid_indices.append(i)

    except Exception as e:
        failed_records.append(int(row["ecg_id"]))
        tqdm.write(
            f"Failed ECG {row['ecg_id']}: {e}"
        )

## Align signals and metadata
if not signals:
    raise RuntimeError(
        "No ECG signals were downloaded successfully."
    )

# Shape: (N, 12, 5000)
X = np.stack(signals)

# Keep metadata only for successful downloads
sample = (
    sample.iloc[valid_indices]
    .reset_index(drop=True)
)

## IDs in exactly the same order as X
ecg_ids = sample["ecg_id"].to_numpy(dtype=np.int64)

## Validate alignment
assert len(X) == len(sample)
assert len(X) == len(ecg_ids)
assert sample["ecg_id"].is_unique
assert sample["ecg_id"].is_monotonic_increasing

assert np.array_equal(
    ecg_ids,
    sample["ecg_id"].to_numpy()
)

assert X.shape[1:] == (12, 5000)
assert np.isfinite(X).all()

print("\nAlignment verification passed!")

## Save signals, IDs and metadata
np.save("ecg_signals.npy", X)
np.save("ecg_ids.npy", ecg_ids)

sample.to_csv(
    "ptbxl_subset.csv",
    index=False
)

sample.to_pickle("ecg_metadata.pkl")

print("\nDownload complete!")
print("Signal matrix:", X.shape)
print("ECG IDs:", ecg_ids.shape)
print("Metadata records:", len(sample))
print("Failed recordings:", len(failed_records))

if failed_records:
    print("Failed ECG IDs:", failed_records)

print("\nSaved files:")
print("  ecg_signals.npy")
print("  ecg_ids.npy")
print("  ptbxl_subset.csv")
print("  ecg_metadata.pkl")

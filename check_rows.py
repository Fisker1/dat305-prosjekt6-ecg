import numpy as np
import pandas as pd

data = np.load("ecg_embeddings_with_ids.npz")
metadata = pd.read_csv("ptbxl_subset.csv")

embeddings = data["embeddings"]
ecg_ids = data["ecg_ids"]

assert np.array_equal(
    ecg_ids,
    metadata["ecg_id"].to_numpy()
)

print("All records are correctly aligned!")
print("Number of embeddings:", len(embeddings))

for i in range(min(5, len(embeddings))):
    print(
        f"Row {i}: "
        f"ECG ID = {ecg_ids[i]}, "
        f"Label = {metadata.iloc[i]['label']}, "
        f"Embedding shape = {embeddings[i].shape}"
    )
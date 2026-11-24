
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import umap

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score

## Load embeddings and metadata
data = np.load("ecg_embeddings_with_ids.npz")

embeddings = data["embeddings"]
ecg_ids = data["ecg_ids"]

metadata = pd.read_csv("ptbxl_subset.csv")

assert len(embeddings) == len(metadata)
assert np.array_equal(
    ecg_ids,
    metadata["ecg_id"].to_numpy()
)
assert np.isfinite(embeddings).all()

labels = metadata["label"].to_numpy()
classes = sorted(np.unique(labels))

print("Embedding matrix:", embeddings.shape)
print("Class distribution:")
print(metadata["label"].value_counts())
print("ECG IDs and metadata are aligned!")

## Standardize embeddings
scaler = StandardScaler()
Z = scaler.fit_transform(embeddings)

## Dimensionality reduction
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(Z)

reducer = umap.UMAP(
    n_components=2,
    n_neighbors=15,
    min_dist=0.1,
    random_state=42
)
X_umap = reducer.fit_transform(Z)

print(
    "PCA explained variance:",
    pca.explained_variance_ratio_.sum()
)

## Visualize PCA and UMAP
fig, axes = plt.subplots(
    1, 2, figsize=(14, 6)
)

for ax, features, title in zip(
    axes,
    [X_pca, X_umap],
    ["PCA", "UMAP"]
):
    for label in classes:
        mask = labels == label

        ax.scatter(
            features[mask, 0],
            features[mask, 1],
            label=label,
            s=15,
            alpha=0.6
        )

    ax.set_title(title)
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")
    ax.legend()
    ax.grid(alpha=0.2)

plt.tight_layout()
plt.savefig("pca_umap.png", dpi=300)
plt.show()

## k-NN label agreement
def get_neighbors(features, k=10):
    if k >= len(features):
        raise ValueError("k must be smaller than N")

    model = NearestNeighbors(
        n_neighbors=k + 1
    )
    model.fit(features)

    # Exclude the query ECG itself, even
    # when identical vectors produce ties.
    indices = model.kneighbors(
        features,
        return_distance=False
    )

    neighbors = []
    for i, row in enumerate(indices):
        other = row[row != i][:k]

        # Rare case: self is not among k+1
        if len(other) != k:
            raise RuntimeError(
                "Could not retrieve k neighbors"
            )

        neighbors.append(other)

    return np.asarray(neighbors)


def knn_agreement(neighbors, labels):
    return np.mean(
        labels[neighbors] == labels[:, None]
    )

## Quantitative evaluation
methods = {
    "Original": Z,
    "PCA": X_pca,
    "UMAP": X_umap
}

results = []
neighbor_sets = {}

for name, features in methods.items():

    neighbors = get_neighbors(features, k=10)
    neighbor_sets[name] = neighbors

    agreement = knn_agreement(neighbors, labels)

    silhouette = silhouette_score(
        features, labels
    )

    results.append({
        "method": name,
        "knn_agreement": agreement,
        "silhouette_score": silhouette
    })

results_df = pd.DataFrame(results)

print("\nQuantitative results:")
print(results_df.to_string(index=False))

results_df.to_csv(
    "separation_metrics.csv",
    index=False
)

## Class-overlap matrices
def overlap_matrix(neighbors, labels, classes):

    matrix = pd.DataFrame(
        0.0,
        index=classes,
        columns=classes
    )

    for i, row in enumerate(neighbors):
        source = labels[i]

        for j in row:
            target = labels[j]
            matrix.loc[source, target] += 1

    # Normalize by source class
    matrix = matrix.div(
        matrix.sum(axis=1),
        axis=0
    )

    return matrix

overlap_matrices = {}

for name, neighbors in neighbor_sets.items():

    matrix = overlap_matrix(
        neighbors, labels, classes
    )

    overlap_matrices[name] = matrix

    print(f"\n{name} overlap matrix:")
    print(matrix.round(3))

    matrix.to_csv(
        f"overlap_{name.lower()}.csv"
    )

## Find strongest overlapping pairs
pair_results = []

for i, class_a in enumerate(classes):
    for class_b in classes[i + 1:]:

        row = {
            "class_a": class_a,
            "class_b": class_b
        }

        for name, matrix in overlap_matrices.items():

            # Symmetric mean of both directions
            overlap = (
                matrix.loc[class_a, class_b]
                + matrix.loc[class_b, class_a]
            ) / 2

            row[name] = overlap

        pair_results.append(row)


pairs_df = pd.DataFrame(pair_results)

# Rank pairs independently in each method.
# Rank 1 means strongest observed overlap.
for name in methods:
    pairs_df[f"{name}_rank"] = (
        pairs_df[name]
        .rank(ascending=False, method="min")
        .astype(int)
    )

pairs_df = pairs_df.sort_values(
    ["Original_rank", "PCA_rank", "UMAP_rank"]
)

print("\nDiagnostic pair overlap:")
print(pairs_df.to_string(index=False))

pairs_df.to_csv(
    "diagnostic_pair_overlap.csv",
    index=False
)

print("\nAnalysis complete!")
## N2D: (Not Too) Deep clustering via clustering the local manifold of an autoencoded embedding.

## Abstract
Deep clustering has increasingly been demonstrating superiority over conventional shallow clustering algorithms. 
Deep clustering algorithms usually combine representation learning with deep neural networks to achieve this performance, typically optimizing a clustering and non-clustering loss.
In such cases, an autoencoder is typically connected with a clustering network, and the final clustering is jointly learned by both the autoencoder and clustering network.
Instead, we propose to learn an autoencoded embedding and then search this further for the underlying manifold.
For simplicity, we then cluster this with a shallow clustering algorithm, rather than a deeper network.
We study a number of local and global manifold learning methods on both the raw data and autoencoded embedding, concluding that UMAP in our framework is able to find the best clusterable manifold of the embedding. This suggests that local manifold learning on an autoencoded embedding is effective for discovering higher quality clusters.
We quantitatively show across a range of image and time-series datasets that our method has competitive performance against the latest deep clustering algorithms, including out-performing current state-of-the-art on several.
We postulate that these results show a promising research direction for deep clustering.

## Paper
https://seis.bristol.ac.uk/~rm17770/publications/n2d.pdf

## Install

```sh
conda install tensorflow-gpu=1.13.1 cudatoolkit=9.0
pip install -f requirements.txt
```

## Reproduce Results
```sh
bash run.sh
```

## N2D: (Not Too) Deep Clustering via Clustering the Local Manifold of an Autoencoded Embedding.

## Abstract
Deep clustering has increasingly been demonstrating superiority over conventional shallow clustering algorithms. 
Deep clustering algorithms usually combine representation learning with deep neural networks to achieve this performance, typically optimizing a clustering and non-clustering loss.
In such cases, an autoencoder is typically connected with a clustering network, and the final clustering is jointly learned by both the autoencoder and clustering network.
Instead, we propose to learn an autoencoded embedding and then search this further for the underlying manifold.
For simplicity, we then cluster this with a shallow clustering algorithm, rather than a deeper network.
We study a number of local and global manifold learning methods on both the raw data and autoencoded embedding, concluding that UMAP in our framework is able to find the best clusterable manifold of the embedding. This suggests that local manifold learning on an autoencoded embedding is effective for discovering higher quality clusters.
We quantitatively show across a range of image and time-series datasets that our method has competitive performance against the latest deep clustering algorithms, including out-performing current state-of-the-art on several.
We postulate that these results show a promising research direction for deep clustering.

## Results
![N2D results](https://seis.bristol.ac.uk/~rm17770/publications/n2d-results.png)

## Paper

https://arxiv.org/abs/1908.05968

## Install

```sh
conda install tensorflow-gpu=1.13.1 cudatoolkit=9.0
pip install -r requirements.txt
```

## Reproduce Results
```sh
bash run.sh
```

## Visualization
If you would like to produce some plots for visualization purposes add the agument '--visualize'. I also reccomend setting the argument '--umap_dim' to be 2.

## Citation
```
@article{2019arXiv190805968M,
  title = {N2D:(Not Too) Deep Clustering via Clustering the Local Manifold of an Autoencoded Embedding},
  author = {{McConville}, Ryan and {Santos-Rodriguez}, Raul and {Piechocki}, Robert J and {Craddock}, Ian},
  journal = {arXiv preprint arXiv:1908.05968},
  year = "2019",
}
```

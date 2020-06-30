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

## Visualizations
### MNIST
<img src="https://seis.bristol.ac.uk/~rm17770/publications/mnist-n2d.png" width="600px">

### HAR (Human Activity Recognition)
<img src="https://seis.bristol.ac.uk/~rm17770/publications/har-n2d.png" width="600px">
Note: clusters 'look' better in higher dimensions (based on clustering metrics) than they do here in 2d. The intended use of n2d is for clustering. Visualized here are the first 5000 points.


## Paper

https://arxiv.org/abs/1908.05968

## Install

### Install Anaconda
```sh
wget https://repo.anaconda.com/archive/Anaconda3-2019.07-Linux-x86_64.sh
bash Anaconda3-2019.07-Linux-x86_64.sh
source anaconda3/bin/activate
```

### Create environment
```sh
conda create -n n2d python=3.7  
conda activate n2d
```
### Clone repo
```sh
git clone https://github.com/rymc/n2d.git
```
### Install packages
```sh
pip install -r requirements.txt
```

### Reproduce results
```sh
bash run.sh
```

### For training a new network
If you remove the --ae_weights argument when running n2d then it will train a new network, rather than load the pretrained weights.

For adding a new dataset you should add a load function to datasets.py (you can use the existing ones to understand how) and a function to call your data loading function from n2d.py

I used the following packages for training the networks using the GPU.
```sh
conda install tensorflow-gpu=1.13.1 cudatoolkit=9.0
```

### Visualization
If you would like to produce some plots for visualization purposes add the agument '--visualize'. I also reccomend setting the argument '--umap_dim' to be 2.

## Citation
```
@inproceedings{McConville2020,
  author = {Ryan McConville and Raul Santos-Rodriguez and Robert J Piechocki and Ian Craddock},
  title = {N2D:(Not Too) Deep Clustering via Clustering the Local Manifold of an Autoencoded Embedding},
  booktitle = {25th International Conference on Pattern Recognition, {ICPR} 2020},
  publisher = {{IEEE} Computer Society},
  year = {2020},
}
```

import argparse
import operator
import os
import random as rn

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import sys
import tensorflow as tf
import umap
from keras import backend as K
from keras.layers import Dense, Input
from keras.models import Model
from sklearn import metrics
from sklearn import mixture
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.manifold import Isomap
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.utils.linear_assignment_ import linear_assignment
from time import time

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(sys.argv[2])
os.environ['PYTHONHASHSEED'] = '0'
os.environ['TF_CUDNN_USE_AUTOTUNE'] = '0'

rn.seed(0)
tf.set_random_seed(0)
np.random.seed(0)

if len(K.tensorflow_backend._get_available_gpus()) > 0:
    print("Using GPU")
    session_conf = tf.ConfigProto(intra_op_parallelism_threads=1,
                                  inter_op_parallelism_threads=1,
                                )
    sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
    K.set_session(sess)

try:
    from MulticoreTSNE import MulticoreTSNE as TSNE
except BaseException:
    print("Missing MulticoreTSNE package.. Only important if evaluating other manifold learners.")

np.set_printoptions(threshold=sys.maxsize)

matplotlib.use('agg')

def eval_other_methods(x, y):
    gmm = mixture.GaussianMixture(
        covariance_type='full',
        n_components=args.n_clusters,
        random_state=0)
    gmm.fit(x)
    y_pred_prob = gmm.predict_proba(x)
    y_pred = y_pred_prob.argmax(1)
    acc = np.round(cluster_acc(y, y_pred), 5)
    nmi = np.round(metrics.normalized_mutual_info_score(y, y_pred), 5)
    ari = np.round(metrics.adjusted_rand_score(y, y_pred), 5)
    print(args.dataset + " | GMM clustering on raw data")
    print("======================")
    print(acc)
    print(nmi)
    print(ari)
    print("======================")

    y_pred = KMeans(
        n_clusters=args.n_clusters,
        random_state=0).fit_predict(x)
    acc = np.round(cluster_acc(y, y_pred), 5)
    nmi = np.round(metrics.normalized_mutual_info_score(y, y_pred), 5)
    ari = np.round(metrics.adjusted_rand_score(y, y_pred), 5)
    print(args.dataset + " | K-Means clustering on raw data")
    print("======================")
    print(acc)
    print(nmi)
    print(ari)
    print("======================")

    sc = SpectralClustering(
        n_clusters=args.n_clusters,
        random_state=0,
        affinity='nearest_neighbors')
    y_pred = sc.fit_predict(x)
    acc = np.round(cluster_acc(y, y_pred), 5)
    nmi = np.round(metrics.normalized_mutual_info_score(y, y_pred), 5)
    ari = np.round(metrics.adjusted_rand_score(y, y_pred), 5)
    print(args.dataset + " | Spectral Clustering on raw data")
    print("======================")
    print(acc)
    print(nmi)
    print(ari)
    print("======================")

    if args.manifold_learner == 'UMAP':
        md = float(args.umap_min_dist)
        hle = umap.UMAP(
            random_state=0,
            metric=args.umap_metric,
            n_components=args.umap_dim,
            n_neighbors=args.umap_neighbors,
            min_dist=md).fit_transform(x)
    elif args.manifold_learner == 'LLE':
        from sklearn.manifold import LocallyLinearEmbedding
        hle = LocallyLinearEmbedding(
            n_components=args.umap_dim,
            n_neighbors=args.umap_neighbors).fit_transform(x)
    elif args.manifold_learner == 'tSNE':
        method = 'exact'
        hle = TSNE(
            n_components=args.umap_dim,
            n_jobs=16,
            random_state=0,
            verbose=0).fit_transform(x)
    elif args.manifold_learner == 'isomap':
        hle = Isomap(
            n_components=args.umap_dim,
            n_neighbors=5,
        ).fit_transform(x)

    gmm = mixture.GaussianMixture(
        covariance_type='full',
        n_components=args.n_clusters,
        random_state=0)
    gmm.fit(hle)
    y_pred_prob = gmm.predict_proba(hle)
    y_pred = y_pred_prob.argmax(1)
    acc = np.round(cluster_acc(y, y_pred), 5)
    nmi = np.round(metrics.normalized_mutual_info_score(y, y_pred), 5)
    ari = np.round(metrics.adjusted_rand_score(y, y_pred), 5)
    print(args.dataset + " | GMM clustering on " +
          str(args.manifold_learner) + " embedding")
    print("======================")
    print(acc)
    print(nmi)
    print(ari)
    print("======================")

    plt.scatter(*zip(*hle[:1000, :2]), c=y[:1000], label=y[:1000])
    plt.savefig(args.save_dir + '/' + args.dataset +
                '-' + str(args.manifold_learner) + '.png')
    plt.clf()

    y_pred = KMeans(
        n_clusters=args.n_clusters,
        random_state=0).fit_predict(hle)
    acc = np.round(cluster_acc(y, y_pred), 5)
    nmi = np.round(metrics.normalized_mutual_info_score(y, y_pred), 5)
    ari = np.round(metrics.adjusted_rand_score(y, y_pred), 5)
    print(args.dataset + " | K-Means " +
          str(args.manifold_learner) + " embedding")
    print("======================")
    print(acc)
    print(nmi)
    print(ari)
    print("======================")

    sc = SpectralClustering(
        n_clusters=args.n_clusters,
        random_state=0,
        affinity='nearest_neighbors')
    y_pred = sc.fit_predict(hle)
    acc = np.round(cluster_acc(y, y_pred), 5)
    nmi = np.round(metrics.normalized_mutual_info_score(y, y_pred), 5)
    ari = np.round(metrics.adjusted_rand_score(y, y_pred), 5)
    print(args.dataset + " | Spectral Clustering on " +
          str(args.manifold_learner) + " embedding")
    print("======================")
    print(acc)
    print(nmi)
    print(ari)
    print("======================")


def cluster_manifold_in_embedding(hl, y, n_clusters, save_dir, visualize):
    # find manifold on autoencoded embedding
    if args.manifold_learner == 'UMAP':
        md = float(args.umap_min_dist)
        hle = umap.UMAP(
            random_state=0,
            metric=args.umap_metric,
            n_components=args.umap_dim,
            n_neighbors=args.umap_neighbors,
            min_dist=md).fit_transform(hl)
    elif args.manifold_learner == 'LLE':
        hle = LocallyLinearEmbedding(
            n_components=args.umap_dim,
            n_neighbors=args.umap_neighbors).fit_transform(hl)
    elif args.manifold_learner == 'tSNE':
        hle = TSNE(
            n_components=args.umap_dim,
            n_jobs=16,
            random_state=0,
            verbose=0).fit_transform(hl)
    elif args.manifold_learner == 'isomap':
        hle = Isomap(
            n_components=args.umap_dim,
            n_neighbors=5,
        ).fit_transform(hl)

    # clustering on new manifold of autoencoded embedding
    if args.cluster == 'GMM':
        gmm = mixture.GaussianMixture(
            covariance_type='full',
            n_components=n_clusters,
            random_state=0)
        gmm.fit(hle)
        y_pred_prob = gmm.predict_proba(hle)
        y_pred = y_pred_prob.argmax(1)
    elif args.cluster == 'KM':
        km = KMeans(
            init='k-means++',
            n_clusters=n_clusters,
            random_state=0,
            n_init=20)
        y_pred = km.fit_predict(hle)
    elif args.cluster == 'SC':
        sc = SpectralClustering(
            n_clusters=n_clusters,
            random_state=0,
            affinity='nearest_neighbors')
        y_pred = sc.fit_predict(hle)

    y_pred = np.asarray(y_pred)
    y_pred = y_pred.reshape(len(y_pred), )
    y = np.asarray(y)
    y = y.reshape(len(y), )
    acc = np.round(cluster_acc(y, y_pred), 5)
    nmi = np.round(metrics.normalized_mutual_info_score(y, y_pred), 5)
    ari = np.round(metrics.adjusted_rand_score(y, y_pred), 5)
    print(args.dataset + " | " + args.manifold_learner +
          " on autoencoded embedding with " + args.cluster + " - N2D")
    print("======================")
    print(acc)
    print(nmi)
    print(ari)
    print("======================")

    if visualize:
        plt.scatter(*zip(*hle[:1000, :2]), c=y[:1000], label=y[:1000])
        plt.savefig(save_dir + '/' + args.dataset + '-n2d.png')
        plt.clf()

    return y_pred, acc, nmi, ari


def cluster_acc(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = linear_assignment(w.max() - w)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size


def autoencoder(dims, act='relu'):
    n_stacks = len(dims) - 1
    x = Input(shape=(dims[0],), name='input')
    h = x
    for i in range(n_stacks - 1):
        h = Dense(dims[i + 1], activation=act, name='encoder_%d' % i)(h)
    h = Dense(dims[-1], name='encoder_%d' % (n_stacks - 1))(h)
    for i in range(n_stacks - 1, 0, -1):
        h = Dense(dims[i], activation=act, name='decoder_%d' % i)(h)
    h = Dense(dims[0], name='decoder_0')(h)

    return Model(inputs=x, outputs=h)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='(Not Too) Deep',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('dataset', default='mnist', )
    parser.add_argument('gpu', default=0, )
    parser.add_argument('--n_clusters', default=10, type=int)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--pretrain_epochs', default=1000, type=int)
    parser.add_argument('--ae_weights', default=None)
    parser.add_argument('--save_dir', default='results/n2d')
    parser.add_argument('--umap_dim', default=2, type=int)
    parser.add_argument('--umap_neighbors', default=10, type=int)
    parser.add_argument('--umap_min_dist', default="0.00", type=str)
    parser.add_argument('--umap_metric', default='euclidean', type=str)
    parser.add_argument('--cluster', default='GMM', type=str)
    parser.add_argument('--eval_all', default=False, action='store_true')
    parser.add_argument('--manifold_learner', default='UMAP', type=str)
    parser.add_argument('--visualize', default=False, action='store_true')
    args = parser.parse_args()
    print(args)

    optimizer = 'adam'
    from datasets import load_mnist, load_mnist_test, load_usps, load_pendigits, load_fashion, load_har

    if args.dataset == 'mnist':
        x, y = load_mnist()
    elif args.dataset == 'mnist-test':
        x, y = load_mnist_test()
    elif args.dataset == 'usps':
        x, y = load_usps()
    elif args.dataset == 'pendigits':
        x, y = load_pendigits()
    elif args.dataset == 'fashion':
        x, y = load_fashion()
    elif args.dataset == 'har':
        x, y = load_har()

    shape = [x.shape[-1], 500, 500, 2000, args.n_clusters]
    autoencoder = autoencoder(shape)

    hidden = autoencoder.get_layer(name='encoder_%d' % (len(shape) - 2)).output
    encoder = Model(inputs=autoencoder.input, outputs=hidden)

    pretrain_time = time()

    # Pretrain autoencoders before clustering
    if args.ae_weights is None:
        autoencoder.compile(loss='mse', optimizer=optimizer)
        batch_size = 256
        autoencoder.fit(
            x,
            x,
            batch_size=batch_size,
            epochs=args.pretrain_epochs,
            verbose=0)
        pretrain_time = time() - pretrain_time
        autoencoder.save_weights('weights/' +
                                 args.dataset +
                                 "-" +
                                 str(args.pretrain_epochs) +
                                 '-ae_weights.h5')
        print("Time to train the autoencoder: " + str(pretrain_time))
    else:
        autoencoder.load_weights('weights/' + args.ae_weights)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    with open(args.save_dir + '/args.txt', 'w') as f:
        f.write("\n".join(sys.argv))

    hl = encoder.predict(x)
    if args.eval_all == True:
        eval_other_methods(x, y)
    clusters, t_acc, t_nmi, t_ari = cluster_manifold_in_embedding(
        hl, y, args.n_clusters, args.save_dir, args.visualize)
    np.savetxt(args.save_dir+"/"+args.dataset+'-clusters.txt', clusters, fmt='%i', delimiter=',')
    

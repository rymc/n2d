# Evaluate only n2d
python n2d.py mnist 0 --ae_weights=mnist-1000-ae_weights.h5  --umap_dim=10 --umap_neighbors=20 --manifold_learner=UMAP  --save_dir=mnist-n2d --umap_min_dist=0.00
python n2d.py fashion 0 --ae_weights=fashion-1000-ae_weights.h5  --umap_dim=10 --umap_neighbors=20 --manifold_learner=UMAP  --save_dir=fashion-n2d --umap_min_dist=0.00
python n2d.py mnist-test 0 --ae_weights=mnist-test-1000-ae_weights.h5  --umap_dim=10 --umap_neighbors=20 --manifold_learner=UMAP  --save_dir=mnist-test-n2d --umap_min_dist=0.00
python n2d.py usps 0 --ae_weights=usps-1000-ae_weights.h5  --umap_dim=10 --umap_neighbors=20 --manifold_learner=UMAP  --save_dir=usps-n2d --umap_min_dist=0.00
python n2d.py pendigits 0 --ae_weights=pendigits-1000-ae_weights.h5  --umap_dim=10 --umap_neighbors=20 --manifold_learner=UMAP  --save_dir=pendigits-n2d --umap_min_dist=0.00
python n2d.py har 0 --ae_weights=har-1000-ae_weights.h5 --umap_dim=6 --umap_neighbors=20 --manifold_learner=UMAP  --save_dir=har-n2d --umap_min_dist=0.00 --n_clusters=6

# Evaluate a number of approaches (including baselines)
python n2d.py mnist 0 --ae_weights=mnist-1000-ae_weights.h5  --umap_dim=2 --umap_neighbors=20 --manifold_learner=UMAP  --save_dir=mnist-n2d-viz --umap_min_dist=0.00 --visualize --eval_all
python n2d.py fashion 0 --ae_weights=fashion-1000-ae_weights.h5  --umap_dim=2 --umap_neighbors=20 --manifold_learner=UMAP  --save_dir=fashion-n2d-viz --umap_min_dist=0.00 --visualize --eval_all
python n2d.py mnist-test 0 --ae_weights=mnist-test-1000-ae_weights.h5  --umap_dim=2 --umap_neighbors=20 --manifold_learner=UMAP  --save_dir=mnist-test-n2d-viz --umap_min_dist=0.00 --visualize --eval_all
python n2d.py usps 0 --ae_weights=usps-1000-ae_weights.h5  --umap_dim=2 --umap_neighbors=20 --manifold_learner=UMAP  --save_dir=usps-n2d-viz --umap_min_dist=0.00 --visualize --eval_all
python n2d.py pendigits 0 --ae_weights=pendigits-1000-ae_weights.h5  --umap_dim=2 --umap_neighbors=20 --manifold_learner=UMAP  --save_dir=pendigits-n2d-viz --umap_min_dist=0.00 --visualize --eval_all
python n2d.py har 0 --ae_weights=har-1000-ae_weights.h5 --umap_dim=2 --umap_neighbors=20 --manifold_learner=UMAP  --save_dir=har-n2d-viz --umap_min_dist=0.00 --n_clusters=6 --visualize --eval_all


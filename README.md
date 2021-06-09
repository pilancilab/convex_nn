# Convex optimization for two-layer ReLU neural networks

In this repository, we provide two distinct implementations to optimize two-layer ReLU neural networks. Particularly, we utilize the exact convex formulations introduced in [1]. Then, we optimize these equivalent architectures both via the interior point solvers in CVXPY and optimizers in PyTorch.

Run the following CVXPY based implementation to perform a binary classification task on a toy dataset:

```` 
python convex_nn.py 
````

Run the following PyTorch implementation to perform a ten class classification task on CIFAR-10 (see the plots folder for the training results):

````
python convexnn_pytorch_stepsize_fig.py --GD 0 --CVX 0 --n_epochs 100 100 --solver_cvx sgd
````

[1] M. Pilanci and T. Ergen. Neural Networks are Convex Regularizers: Exact Polynomial-time Convex Optimization Formulations for Two-layer Networks. ICML 2020 (http://proceedings.mlr.press/v119/pilanci20a.html)

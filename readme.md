### On the Importance of the Embedding Norms in Self-Supervised Learning Repository

This is the codebase which was used to run the experiments for the paper in question. To get started, clone the repository and run `pip install -e .` from the root directory so that python knows how to find all of the files.

A simple experiment has been set up in `CLA/experiments/model_comparison.py`.
Running `CLA/experiments/model_comparison.py` will train three SimCLR models with a resnet18 backbone on Cifar10. One will be the default one, one will be SimCLR with cut constant 3, and one will be SimCLR with GradScale training and the corresponding learning rate schedule.
The actual training loop can be found in the `training/training_loop.py` file.

For an example of how to run more involved experiment sweeps, refer to `weight_decay_ablation.ipynb`.   The simulations are produced in `Simulations.ipynb`. Example plots are given in `Norm as Accuracy.ipynb`.

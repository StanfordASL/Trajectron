**NOTE:** A new version of the Trajectron has been released! Check out [Trajectron++](https://github.com/StanfordASL/Trajectron-plus-plus)!

<p align="center"><img width="100%" src="img/Trajectron.png"/></p>

# The Trajectron: Probabilistic Multi-Agent Trajectory Modeling with Dynamic Spatiotemporal Graphs

This repository contains the code for [The Trajectron: Probabilistic Multi-Agent Trajectory Modeling with Dynamic Spatiotemporal Graphs](https://arxiv.org/abs/1810.05993) by Boris Ivanovic and Marco Pavone.

## Installation ##

First, we'll create a conda environment to hold the dependencies.
```
conda create --name dynstg python=3.6 -y
source activate dynstg
pip install -r requirements.txt
```

Then, since this project uses IPython notebooks, we'll install this conda environment as a kernel.
```
python -m ipykernel install --user --name dynstg --display-name "Python 3.6 (DynSTG)"
```

Now, you can start a Jupyter session and view/run all the notebooks with
```
jupyter notebook
```

When you're done, don't forget to deactivate the conda environment with
```
source deactivate
```

## Scripts ##
Run any of these with a `-h` or `--help` flag to see all available command arguments.
* `code/train.py` - Trains a new Trajectron.
* `code/test_online.py` - Replays a scene from a dataset and performs online inference with a trained Trajectron.
* `code/evaluate_alongside_sgan.py` - Evaluates the performance of the Trajectron against Social GAN. This script mainly collects evaluation data, which can be visualized with `sgan-dataset/Result Analyses.ipynb`.
* `code/compare_runtimes.py` - Evaluates the runtime of the Trajectron against Social GAN. This script mainly collects runtime data, which can be visualized with `sgan-dataset/Runtime Analysis.ipynb`.
* `sgan-dataset/Qualitative Plots.ipynb` - Can be used to visualize predictions from the Trajectron alone, or against those from Social GAN.

## Datasets ##

The preprocessed datasets are available in this repository, under `data/` folders (i.e. `sgan-dataset/data/`).

If you want the *original* ETH or UCY datasets, you can find them here: [ETH Dataset](http://www.vision.ee.ethz.ch/en/datasets/) and [UCY Dataset](https://graphics.cs.ucy.ac.cy/research/downloads/crowd-data).

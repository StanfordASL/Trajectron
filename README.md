# Modeling Dynamic Spatiotemporal Graphs

This repository contains the code for [Modeling Multimodal Dynamic Spatiotemporal Graphs](https://arxiv.org/abs/1810.05993) by Boris Ivanovic and Marco Pavone.

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

## Datasets ##

The preprocessed datasets are available in this repository, under `data/` folders (e.g. `nba-dataset/data/`).

If you want the *original* NBA or ETH datasets, I obtained them from here: [NBA Dataset](https://github.com/linouk23/NBA-Player-Movements) and [ETH Dataset](http://www.vision.ee.ethz.ch/en/datasets/).

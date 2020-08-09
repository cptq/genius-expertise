Code for analysis of a [genius.com](https://genius.com) dataset, as used in the paper:
> Derek Lim, Austin Benson. "Expertise and Dynamics within Crowdsourced Musical Knowledge Curation: A Case Study of the Genius Platform." 2020.

The paper is available as [arxiv:2006.08108](https://arxiv.org/abs/2006.08108).

### Data and Requirements
The data can be found [here](https://www.cs.cornell.edu/~arb/data/genius-expertise/). It must be placed in the `DATAPATH` folder specified in `constants.py`, which by default is `data/`. The data can be loaded with the methods in `load_data.py`. These methods tend to use generators, but the graph `G` (~2GB) and `annotation_info` (~3GB) are currently loaded directly into memory when they are needed (the graph is only used for Figure 10).

Required python3 dependencies for running the code can be found in `requirements.py`


### Figures
We have provided code to recreate figures in the paper. The figures will be saved to the `FIGPATH` folder specified in `constants.py`, which by default is `figures/`.


### Running the Code

For the *utility model* results, run `python utility.py`. This generates Figure 7.

For the *early prediction of expertise* results, run `python prediction.py`.

Code to compute the remaining figures can be found in `gen_plots.py`. For example, to recreate Figure 2, run `python gen_plots.py --figure 2`.

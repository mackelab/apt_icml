# Sequential Neural Likelihood

This is a convenience package wrapper of the code written by George Papamakarios found [here](https://github.com/gpapamak/snl)

for reproducing the experiments in the paper:

> G. Papamakarios, D. C. Sterratt, I. Murray. _Sequential Neural Likelihood: Fast Likelihood-free Inference with Autoregressive Flows_. arXiv:1805.07226. 2018.
> [[pdf]](https://arxiv.org/pdf/1805.07226.pdf) [[bibtex]](http://homepages.inf.ed.ac.uk/s1459647/bibtex/snl.bib)


## Content

This repo contains a python 3 port of the original python 2 [code](https://github.com/gpapamak/snl) obtained from a simple application of [2to3](http://python3porting.com/2to3.html) (with a few subsequent manual fixes).
A setup.py allows installing and calling the functionality (e.g. fitting MAFs to data with SNL) from outside.
Scripts and modules for reproducing the original [experiments](https://github.com/gpapamak/snl/tree/master/exps) of Papamakarios et al. (2018) were removed.

# APT

Code for the submission
> Automatic Posterior Transformation for Likelihood-free Inference, submitted to ICML 2019.

See jupyter notebooks in experiments/ for demos that recreate the central experiments of the paper. 

# Installation instruction 

```
git clone https://github.com/aptalg/apt_icml.git
cd code/apt
python setup.py install --user
cd code/snl_py3
python setup.py install --user
```

# Remark
The package in code/apt builds on the publicly availablle [delfi](https://github.com/mackelab/delfi) package. 
The included package code/snl_py3 is a Python3 port of the publicly available [snl](https://github.com/gpapamak/snl) repository (originally written for Python 2).


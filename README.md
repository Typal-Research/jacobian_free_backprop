[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Build Status - Travis](https://travis-ci.org/howardheaton/fixed_point_networks.svg?branch=master)](https://travis-ci.org/howardheaton/fixed_point_networks)
[![Build Status - GitHub](https://github.com/howardheaton/pytesting/workflows/pytesting/badge.svg)](https://github.com/howardheaton/fixed_point_networks/actions?query=workflow%3Apytesting)
[![Coverage Status](https://coveralls.io/repos/github/howardheaton/fixed_point_networks/badge.svg?branch=master)](https://coveralls.io/github/howardheaton/fixed_point_networks?branch=master)

# Fixed Point Networks


## Associated Publication

_Fixed Point Networks: Implicit Depth with Jacobian-Free Backprop_ (**[arXiv Link](https://arxiv.org/pdf/2103.12803.pdf)**)

    @article{WuFung2020learned,
        title={Fixed Point Networks: Implicit Depth Models with Jacobian-Free Backprop},
        author={Fung, Samy Wu and Heaton, Howard and Li, Qiuwei and McKenzie, Daniel and Osher, Stanley and Yin, Wotao},
        journal={arXiv preprint arXiv:2103.12803},
        year={2021}


## Set-up

Install all the requirements:
```
pip install -r requirements.txt 
```

## Training 

For each dataset, there are three types of training drivers: 
1) FPN with our proposed backprop:
```
	python train_CIFAR10.py
	python train_CIFAR10_Unaugmented.py
	python train_MNIST.py
	python train_SVHN.py
```
2) FPN with Jacobian-based backprop:
```
	python train_CIFAR10_Jacobian_Based.py
	python train_CIFAR10_Unaugmented_Jacobian_Based.py
	python train_MNIST_Jacobian_Based.py
	python train_SVHN_Jacobian_Based.py
```
3) Explicit models. 
```
	python train_CIFAR10_Explicit.py
	python train_CIFAR10_Unaugmented_Explicit.py
	python train_MNIST_Explicit.py
	python train_SVHN_Explicit.py
```

## Introduction
The AdversarialAttack library provides an implementation of the Fast Gradient Sign Method (FGSM) targeted attack.

The implementation is based on the PyTorch tutorial on [Adversarial Example Generation](https://pytorch.org/tutorials/beginner/fgsm_tutorial.html) 
and Yannick Merkli's [FGSM implementation](https://github.com/ymerkli/fgsm-attack).

## How to run
An example of using FGSM to generate targeted adversarial examples on a pretrained MNIST model is provided in `adversarial_attack_demo.ipynb`.

## `FGSMTargeted` class
The file `AdversarialAttack.py` contains the implementation of the abstract class `AdversarialAttack` and the concrete class `FGSMTargeted`.

The `FGSMTargeted` class implements the targeted FGSM targeted attack on a neural network model, which can be any PyTorch model.

You can add another attack method (e.g. PGD) by implementing the abstract method `generate_adv_example()` in new concrete class that implements `AdversarialAttack`.


# Reinforcement Learning Cryptography in [TensorFlow](https://github.com/tensorflow/tensorflow)

by Faris Sbahi -- final project for CS 590: Reinforcement Learning by [Prof. Parr](https://users.cs.duke.edu/~parr/)

A project inspired by Google Brain's work regarding using neural networks to protect communications. Here, we seek to use reinforcement learning as an alternative model.

> In this setting, we model attackers by neural networks; alternative models may perhaps
be enabled by reinforcement learning. [2]


## Setup

1.  Install pip and virtualenv

On MacOS

```bash
$ sudo easy_install pip
$ sudo pip install --upgrade virtualenv
```

2. Clone the repository and install the requirements

```bash
$ git clone https://github.com/brodykellish/rl-agents
$ cd RLFlow
$ mkdir ~/rl-cryp
$ virtualenv --system-site-packages ~/rl-cryp
$ source ~/rl-cryp/bin/activate
$ pip install -r requirements.txt
```

> `rl-env` is an arbitrary directory. You can replace `rl-env` with a directory of your choice.

## Run

To train the neural networks, run the `main.py` script.

```bash
$ python main.py --msg-len 32 --epochs 50
```

## Deactivate

Once finished deactivate the virtual environment

```bash
(rl-cryp)$ deactivate
```

## Cited Publications
* [Learning to Communicate with Deep Multi-Agent Reinforcement Learning](https://arxiv.org/pdf/1605.06676.pdf), Oxford [1]
* [Learning to Protect Communications with Adversarial Neural Cryptography (ANC)](https://arxiv.org/pdf/1610.06918v1.pdf)), Google Brain [2]

## Cited Projects
* ankeshanand's [implementation of ANC](https://github.com/ankeshanand/neural-cryptography-tensorflow) in TensorFlow
* Liam's [implementation of ANC](https://github.com/nlml/adversarial-neural-crypt) of Adversarial Neural Cryptography in Theano.

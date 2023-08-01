# Unified Transformer

This repository contains code for a multi-modal transformer that processes both text and images. Multiple datasets and data modules for pytorch lightning are included including Visual Genome and MNIST.

To install all necessary requirements, run:

```bash
$ pip install -r requirements.txt
```


To train the model, run the following command:

```bash
$ python3 main.py --image-embedding [ convolutional | pure-attention-based ]
```

This is the official repository of our paper [Interpretable Single-Cell Set Classification with Kernel Mean Embeddings](https://arxiv.org/abs/1909.09140).

```
@article{shan2022interpretable,
  title={Interpretable Single-Cell Set Classification with Kernel Mean Embeddings},
  author={Shan, Siyuan and Baskaran, Vishal and Yi, Haidong and Ranek, Jolene and Stanley, Natalie and Oliva, Junier},
  journal={arXiv preprint arXiv:2201.07322},
  year={2022}
}
```

Our codes are tested with Python 3.7.4, numpy 1.19.5, anndata 0.7, and scikit-learn 1.0.

Please follow train.py to prepare dataset, transform data into random fourier feature space, use kernel herding to subselect cells and finlly compute mean embedding vectors for classification. We also show how to compute a score for every cell in a sample-set.

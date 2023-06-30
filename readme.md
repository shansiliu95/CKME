This is the official repository of our paper [Transparent Single-Cell Set Classification with Kernel Mean Embeddings](https://arxiv.org/pdf/2201.07322.pdf).

```
@inproceedings{shan2022transparent,
  title={Transparent single-cell set classification with kernel mean embeddings},
  author={Shan, Siyuan and Baskaran, Vishal Athreya and Yi, Haidong and Ranek, Jolene and Stanley, Natalie and Oliva, Junier B},
  booktitle={Proceedings of the 13th ACM International Conference on Bioinformatics, Computational Biology and Health Informatics},
  pages={1--10},
  year={2022}
}
```

Our codes are tested with Python 3.7.4, numpy 1.19.5, anndata 0.7, and scikit-learn 1.0.

Please follow train.py to prepare dataset, transform data into random fourier feature space, use kernel herding to subselect cells and finlly compute mean embedding vectors for classification. We also show how to compute a score for every cell in a sample-set.

## Donwload the dataset

To run our codes, first download the HVTN dataset from [https://drive.google.com/file/d/1F21BFBcs9nFh4feoed1SEAJ0S89JPViA/view?usp=sharing](https://drive.google.com/file/d/1F21BFBcs9nFh4feoed1SEAJ0S89JPViA/view?usp=sharing). Then place the downloaded file in to a dicrectory.

Then change the line 12 of "train.py" to the location of your downloaded file.


## Run the codes 

Finally, run "train.py"

```
python train.py
```

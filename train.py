from utils import kernel_herding_main, random_feats
import anndata
import numpy as np
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn import svm
import pdb

##############################################################################################################################

dataset_path = '/playpen1/scribble/ssy/singlecell/hvtn_preprocessed.h5ad' # Change this to where your data locates
gamma = 1.0 # Hyperparameter gamma to compute random fourier features
m = 200 # number of cells subselected by Hernel Herding for each sample-set

##############################################################################################################################

# Load and prepare dataset 
# The following commands are used to read data in h5ad format, if you have conveted your data to numpy arrays you can directly load it using np.load()
data = anndata.read_h5ad(dataset_path) 
num_sample_sets = len(data.obs.FCS_File.values.unique()) # Find the number of sample-sets in the dataset
x = []
x_rff_kh = []
y = []
print("Transforming the data into Random Fourier Feature space and subselect using Kernel Herding, this may take several minutes.")
for i in tqdm(range(num_sample_sets)):
    fcs_file = data.obs.FCS_File.values.unique()[i]
    fcs_data = data[data.obs.FCS_File == fcs_file]
    fcs_X = fcs_data.X # This is the data for a single sample-set, the shape is [num_cells, num_features]
    label = fcs_data.obs.label.unique()[0] # This is the label of this sample-set
    phi, W = random_feats(fcs_X, gamma=gamma, frequency_seed=0) # transform the data into random fourier feature space
    kh_indices, kh_samples, kh_rf = kernel_herding_main(fcs_X, phi, num_subsamples=m) # get the subselected indices using kernel herding
    feature = np.mean(phi[kh_indices], 0) # first subselect using the subselected indices, then featurize the sample-set into a single feature vector via mean embedding
    x.append(feature[None, :])
    y.append(label)
    x_rff_kh.append(phi[kh_indices][None, ...])
x = np.concatenate(x, 0) # shape [num_sample_sets, D], where D (= 2000 by default) is the dimensionality of the raondom fourier feature space
y = np.array(y) # shape [num_sample_sets]
x_rff_kh = np.concatenate(x_rff_kh, 0) # shape [num_sample_sets, m, D]


##############################################################################################################################

# Train a linear regression classifier for classification 
clf = LogisticRegression(random_state=0)
scores = cross_val_score(clf, x, y, cv=5)
print(f"Logistic Regression, gamma {gamma}, m {m}: accuracy: {np.mean(scores)} +- {np.std(scores)}")
# Train a linear SVM classifier for classification
clf = svm.SVC(kernel='linear')
scores = cross_val_score(clf, x, y, cv=5)
print(f"SVM Linear, gamma {gamma}, m {m}: accuracy: {np.mean(scores)} +- {np.std(scores)}")

##############################################################################################################################

# If you want to assign a score to each individual cell:
# first, split the data into train/test set
x_train, x_test = x[:int(0.8*x.shape[0])], x[int(0.8*x.shape[0]):]
y_train, y_test = y[:int(0.8*x.shape[0])], y[int(0.8*x.shape[0]):]
x_rff_kh_train, x_rff_kh_test = x_rff_kh[:int(0.8*x.shape[0])], x_rff_kh[int(0.8*x.shape[0]):]
# Second, train a linear SVM classifier
clf = svm.SVC(kernel='linear')
clf.fit(x_train, y_train)
# Compute the scores for each individual cell
score_train = np.sum(clf.coef_ * x_rff_kh_train, -1) + clf.intercept_
score_test = np.sum(clf.coef_ * x_rff_kh_test, -1) + clf.intercept_
# Validate that we can get the same classification results by averaging all cell scores in a sample-set
print(f"prediction made by SVM: {clf.predict(x_test)}")
print(f"prediction made by averaging the cell scores: {(score_test.mean(1) > 0).astype(np.int16)}")


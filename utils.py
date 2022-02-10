import numpy as np

def random_feats(X, gamma=6, frequency_seed=None):
    scale = 1 / gamma
    if(frequency_seed is not None):
        np.random.seed(frequency_seed)
        W = np.random.normal(scale=scale, size=(X.shape[1], 1000))
    else:
        W = np.random.normal(scale=scale, size=(X.shape[1], 1000))
    XW = np.dot(X, W)
    sin_XW = np.sin(XW)
    cos_XW = np.cos(XW)
    Xnew = np.concatenate((cos_XW, sin_XW), axis=1)
    del sin_XW
    del cos_XW
    del XW
    #del W
    return Xnew, W

def kernel_herding(X, phi, num_samples):
    w_t = np.mean(phi, axis=0)
    w_0 = w_t
    subsample = []
    indices = []
    for i in range(1, num_samples + 1):
        new_ind = np.argmax(np.dot(phi, w_t))
        x_t = X[new_ind]
        w_t = w_t + w_0 - phi[new_ind]
        indices.append(new_ind)
        subsample.append(x_t)

    return indices, subsample


def kernel_herding_main(X, phi, num_subsamples):
    kh_indices, kh_samples = kernel_herding(X, phi, num_subsamples)
    kh_rf = phi[kh_indices]
    return kh_indices, kh_samples, kh_rf

import numpy as np
from sklearn import metrics
from sklearn.cluster import KMeans


def ACC(y_true, y_pred):
    Y_pred = y_pred
    Y = y_true
    from sklearn.utils.linear_assignment_ import linear_assignment
    assert Y_pred.size == Y.size
    D = max(Y_pred.max(), Y.max())+1
    w = np.zeros((D,D), dtype=np.int64)
    for i in range(Y_pred.size):
        w[Y_pred[i], Y[i]] += 1
    ind = linear_assignment(w.max() - w)
    return sum([w[i, j] for i, j in ind])*1.0/Y_pred.size, ind


def calc_cluster(latent, labels, num_classes):
    clustering_algo = KMeans(n_clusters=num_classes, precompute_distances=True, n_jobs=1)
    y_pred = clustering_algo.fit_predict(latent)
    results = {}
    results['ACC'] = ACC(labels, y_pred)[0]
    results['NMI'] = metrics.normalized_mutual_info_score(labels, y_pred, 'geometric')
    results['ARI'] = metrics.adjusted_rand_score(labels, y_pred)
    return results

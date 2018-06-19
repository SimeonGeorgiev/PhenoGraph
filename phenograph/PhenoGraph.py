from cluster import cluster
from sklearn.neighbors import KNeighborsClassifier

class PhenoGraph(object):
    """

    This is a wrapper over the phenograph.cluster function.
    It supports the scikit-learn API- fit, predict and fit_predict methods.
    
    """
    def __init__(self, k=50, directed=False, prune=False, min_cluster_size=100, jaccard=True,
            primary_metric="manhattan", n_jobs=-1, q_tol=1e-3, louvain_time_limit=3000,
            nn_method="brute", save_graph=False):

        self.metric = primary_metric
        self.k_ = k
        self.n_jobs = n_jobs
        self.save_graph = save_graph
        self.cluster_func = lambda X: cluster(X, k=k, directed=directed, 
                prune=prune, min_cluster_size=min_cluster_size, jaccard=jaccard,
                primary_metric=primary_metric, n_jobs=n_jobs, q_tol=q_tol,
                louvain_time_limit=louvain_time_limit, nn_method=nn_method)

    def fit(self, X):
        self.labels_, self.graph_, self.Q = self.cluster_func(X)
        if self.save_graph == False:
            del self.graph_
        # experiment with the effect of leaf_size on accuracy and speed.
        self.clf = KNeighborsClassifier(n_neighbors=self.k_, 
            metric=self.metric, weights="distance", n_jobs=self.n_jobs, leaf_size=self.k_)
        self.clf.fit(X, self.labels_)
        return self

    def predict(self, X):
        return self.clf.predict(X)

    def fit_predict(self, X):
        self.fit(X)
        return self.labels_

    def predict_proba(self, X):
        return self.clf.predict_proba(X)

if __name__ == "__main__":
    from sklearn.metrics import normalized_mutual_info_score as nmis
    from sklearn.datasets import make_blobs 
    from numpy import unique
    len_u = lambda x: len(unique(x))
    clusters = 20
    ds = make_blobs(n_samples=5000, n_features=30, centers=clusters, center_box=(0, 9))
    ph = PhenoGraph().fit(ds[0])
    predictions = ph.predict(ds[0])
    labels_2 = PhenoGraph().fit_predict(ds[0])

    assert len_u(ph.labels_) == clusters
    assert nmis(predictions, ph.labels_) > 0.99
    assert nmis(ph.labels_, ds[1]) > 0.99
    assert nmis(ds[1], predictions) > 0.99
    assert nmis(labels_2, ph.labels_) > 0.99

  


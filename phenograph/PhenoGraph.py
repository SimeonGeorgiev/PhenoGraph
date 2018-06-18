from cluster import cluster
from sklearn.neighbors import KNeighborsClassifier

class PhenoGraph(object):
  """

    This is a wrapper over the phenograph.cluster function.
    It supports the scikit-learn API- fit, predict and fit_predict methods.
    *there's no weird underscores at the end of the attributes -> labels instead of labels_
    This makes it possible to use bootstrapping to assess cluster significance.
    
  """
  def __init__(self, k=50, directed=False, prune=False, min_cluster_size=100, jaccard=True,
            primary_metric="manhattan", n_jobs=-1, q_tol=1e-3, louvain_time_limit=3000,
            nn_method="brute", save_graph=False):
    self.metric = primary_metric
    self.k = k
    self.n_jobs = n_jobs
    self.save_graph = save_graph
    self.k = k; self.directed = directed
    self.prune = prune; self.min_cluster_size = min_cluster_size
    self.jaccard = jaccard
    self.primary_metric = primary_metric
    self.n_jobs = n_jobs
    self.q_tol = q_tol; self.nn_method = nn_method
    self.louvain_time_limit = louvain_time_limit

  def fit(self, X):
    self.data = X
    self.clust_func = lambda: cluster(X, k=self.k, directed=self.directed, prune=self.prune, min_cluster_size=self.min_cluster_size, jaccard=self.jaccard,
            primary_metric=self.primary_metric, n_jobs=self.n_jobs, q_tol=self.q_tol, louvain_time_limit=self.louvain_time_limit, nn_method=self.nn_method)

    self.labels, self.graph, self.Q = self.clust_func()
    if self.save_graph == False:
      del self.graph
    # experiment with the effect of leaf_size on accuracy and speed.
    self.clf = KNeighborsClassifier(n_neighbors=self.k, 
      metric=self.metric, weights="distance", n_jobs=self.n_jobs, leaf_size=self.k)
    self.clf.fit(self.data, self.labels)
    return self

  def predict(self, X):
    return self.clf.predict(X)

  def fit_predict(self, X):
    self.fit(X)
    return self.labels

  def predict_proba(self, X):
    return self.clf.predict_proba(X)

if __name__ == "__main__":
  from sys import argv
  from sklearn.metrics import normalized_mutual_info_score as nmis
  from sklearn.datasets import make_blobs 
  from numpy import unique
  len_u = lambda x: len(unique(x))
  clusters = 20
  ds = make_blobs(n_samples=5000, n_features=30, centers=clusters, center_box=(0, 9))
  ph = PhenoGraph().fit(ds[0])
  predictions = ph.predict(ds[0])
  assert len_u(ph.labels) == clusters
  assert nmis(predictions, ph.labels) > 0.99
  assert nmis(ph.labels, ds[1]) > 0.99
  assert nmis(ds[1], predictions) > 0.99
  


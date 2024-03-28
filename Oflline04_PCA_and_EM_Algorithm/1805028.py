import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD as SVD
import matplotlib.patches as patches
import argparse
import imageio

# Parse command line arguments
parser = argparse.ArgumentParser(description='GMM with command line args')
parser.add_argument('--k_start', type=int, required=True, help='Start of K range')
parser.add_argument('--k_end', type=int, required=True, help='End of K range')
parser.add_argument('--dataset', type=str, required=True, help='Dataset file name')
args = parser.parse_args()

#=============== Task 1 Start ===============#
# Load the dataset
data = np.loadtxt(args.dataset, delimiter=',')

# PCA using SVD
if data.shape[1] > 2:
    pca = SVD(n_components=2)
    data = pca.fit_transform(data)

# Plot the data
plt.scatter(data[:, 0], data[:, 1])
plt.savefig('pca.png')
#=============== Task 1 End ===============#


#=============== Task 2 Start ===============#
class GMM:
    def __init__(self, K, iterations):
        self.K = K
        self.iterations = iterations

    def initialize(self, data):
        self.shape = data.shape
        self.n, self.m = self.shape

        self.phi = np.full(shape=self.K, fill_value=1/self.K)
        self.weights = np.full( shape=self.shape, fill_value=1/self.K)

        kmeans = KMeans(n_clusters=self.K, n_init=10).fit(data)
        self.mu = kmeans.cluster_centers_
        self.sigma = [np.cov(data[kmeans.labels_ == k].T) for k in range(self.K)]
    

    def e_step(self, data):
        self.weights = self.predict_proba(data)
        self.phi = self.weights.mean(axis=0)

    def m_step(self, data):
        for i in range(self.K):
            weight = self.weights[:, [i]]
            total_weight = weight.sum()
            self.mu[i] = (data * weight).sum(axis=0) / total_weight
            self.sigma[i] = np.cov(data.T, 
                aweights=(weight/total_weight).flatten(), 
                bias=True)

    def fit(self, data):
        self.initialize(data)

        fig, ax = plt.subplots()

        for i in range(self.iterations):
            self.e_step(data)
            self.m_step(data)

            # Clear the current plot
            ax.clear()

            # Plot the data points
            ax.scatter(data[:, 0], data[:, 1], c=self.predict(data))

            # Plot the Gaussian distributions
            for j in range(self.K):
                eigenvalues, eigenvectors = np.linalg.eigh(self.sigma[j])
                order = eigenvalues.argsort()[::-1]
                eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]
                vx, vy = eigenvectors[:,0][0], eigenvectors[:,0][1]
                theta = np.arctan2(vy, vx)

                color = plt.cm.jet(float(j) / np.max(self.K - 1))
                for cov_factor in range(1, 4):
                    ell = patches.Ellipse(self.mu[j], np.sqrt(eigenvalues[0]) * cov_factor * 2, np.sqrt(eigenvalues[1]) * cov_factor * 2,
                                        theta * 180 / np.pi, color=color)
                    ell.set_alpha(0.5 / cov_factor)
                    ax.add_artist(ell)

            # Show the plot
            plt.pause(0.1)

        plt.show()

        return self
            
    def predict_proba(self, data):
        likelihood = np.zeros( (self.n, self.K) )
        for i in range(self.K):
            distribution = multivariate_normal(
                mean=self.mu[i], 
                cov=self.sigma[i])
            likelihood[:,i] = distribution.pdf(data)+ 1e-16 # add a small value to avoid division by zero
        
        numerator = likelihood * self.phi
        denominator = numerator.sum(axis=1)[:, np.newaxis]
        weights = numerator / denominator
        return weights

    def predict(self, data):
        weights = self.predict_proba(data)
        return np.argmax(weights, axis=1)


K_values = range(args.k_start, args.k_end+1)
best_log_likelihood = float('-inf')
best_K = None
best_gmm = None
log_likelihoods = []

for K in K_values:
    gmm = GMM(K, 5).fit(data)
    log_likelihood = np.sum(np.log(gmm.predict_proba(data)+ 1e-16))
    log_likelihoods.append(log_likelihood)
    if log_likelihood > best_log_likelihood:
        best_log_likelihood = log_likelihood
        best_K = K
        best_gmm = gmm

plt.figure()
plt.plot(K_values, log_likelihoods)
plt.title('Best log-likelihood vs K')
plt.xlabel('K')
plt.ylabel('Best log-likelihood')
plt.savefig('log_likelihoods.png')

plt.figure()
plt.scatter(data[:, 0], data[:, 1], c=best_gmm.predict(data))
plt.title('Clustering using the best GMM')
plt.savefig('gmm.png')
#=============== Task 2 End ===============#
import numpy as np
from scipy.special import digamma, gammaln
import time

np.random.seed(42)

def normalize(ndarray, axis):
    return ndarray / ndarray.sum(axis = axis, keepdims = True)


def normalized_random_array(d0, d1):
    ndarray = np.random.rand(d0, d1)
    return normalize(ndarray, axis = 1)


def log_sum_exp(X):
    max_x = np.max(X, axis=1).reshape(-1, 1)
    return np.log(np.sum(np.exp(X - max_x), axis=1)).reshape(-1, 1) + max_x


if __name__ == "__main__":
    # initialize parameters

    """
    D = documents
    K = number of topics
    V = vocabularies
    """

    D, K, V = 1000, 2, 6
    alpha0, beta0 = 1.0, 1.0
    alpha = alpha0 + np.random.rand(D, K)
    beta = beta0 + np.random.rand(K, V)
    theta = normalized_random_array(D, K)
    phi = normalized_random_array(K, V)

    # for generate documents
    _theta = np.array([theta[:, :k+1].sum(axis=1) for k in range(K)]).T
    _phi = np.array([phi[:, :v+1].sum(axis=1) for v in range(V)]).T

    # generate documents
    W, Z = [], []
    N = np.random.randint(100, 300, D)
    for (d, N_d) in enumerate(N):
        Z.append((np.random.rand(N_d, 1) < _theta[d, :]).argmax(axis=1))
        W.append((np.random.rand(N_d, 1) < _phi[Z[-1], :]).argmax(axis=1))
    
    # estimate parameters
    T = 30
    start = time.time()
    for t in range(T):
        dig_alpha = digamma(alpha) - digamma(alpha.sum(axis=1, keepdims=True))
        dig_beta = digamma(beta) - digamma(beta.sum(axis=1, keepdims=True))

        # initialize parameter
        alpha_new = np.ones((D, K)) * alpha0
        beta_new = np.ones((K, V)) * beta0

        # stochasticにしたいならここをrandom choiceにする
        for (d, N_d) in enumerate(N):
            q = np.zeros((V, K))
            v, count = np.unique(W[d], return_counts=True)
            q[v, :] = (np.exp(dig_alpha[d, :].reshape(-1, 1) + dig_beta[:, v]) * count).
            # * は要素積．vの一要素にたいしてdig_alpha等を計算し，そのvの出現回数だけかける
            # q[v, :] = (np.exp(dig_alpha[d, :].reshape(-1, 1) + dig_beta[:, v]) * count).T
            # q[v, :] = log_sum_exp(q[v, :])
            q[v, :] /= q[v, :].sum(axis=1, keepdims=True)


            alpha_new[d, :] += count.dot(q[v])
            beta_new[:, v] += count * q[v].T


        alpha = alpha_new.copy()
        beta = beta_new.copy()

    end = time.time()
    theta_est = np.array([np.random.dirichlet(a) for a in alpha])
    phi_est = np.array([np.random.dirichlet(b) for b in beta])

    print(theta_est)
    print(phi_est)
    print(end - start)


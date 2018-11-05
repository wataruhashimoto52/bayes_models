# coding: utf-8

import numpy as np
import scipy.stats as stats
from scipy.special import digamma, gammaln, gamma
import matplotlib
import matplotlib.pyplot as plt



class PMM:
    """
    Poisson Mixture model.
    """
    def __init__(self, X, K, maxiter=200, update_params=False):
        self.N = int(X.shape[0])
        self.X = X
        self.K = K
        self.maxiter = maxiter
        self.update_params = update_params

        # hyperparameters
        self.init_a = np.ones((self.K, 1)) / K 
        self.init_b = np.ones((self.K, 1)) / K
        self.init_alpha = np.random.rand(K, 1)

        # variational parameters
        self.a = self.init_a.copy()
        self.b = self.init_b.copy()
        self.alpha = self.init_alpha.copy()
        self.eta = np.zeros((self.N, self.K))


    def variational_inference(self, epsilon=1e-4):
        """
        run variational inference
        """

        elbos = []

        for i in range(self.maxiter):

            # update variational params and calculate elbo
            elbo = self.learn_vb(update_params=self.update_params)
            elbos.append(elbo)

            print("iteration: %d,\tlower bound: %.2f" % (i + 1, elbo))

            """
            if i == 0:
                continue
            else:
                if (abs(lls[i-1]- ll) <= epsilon):
                    break
            """

        print("Finished.")


        return elbos


    def learn_vb(self, update_params=False):
        """
        update variational parameters
        """
        ex_lmd = self.a / self.b
        ex_lnlmd = digamma(self.a) - np.log(self.b)
        ex_lnpi = digamma(self.alpha) - digamma(np.sum(self.alpha))

        # update eta 
        for n in range(self.N):
            eta_k = np.zeros(self.K)
            for k in range(self.K):
                eta_k[k] = np.exp(self.X[n] * ex_lnlmd[k] - ex_lmd[k] + ex_lnpi[k])
            self.eta[n, :] = eta_k / np.sum(eta_k)

        
        # update a, b, alpha
        for k in range(self.K):

            self.a[k] = self.eta[:, k].T.dot(X) + self.init_a[k]

            self.b[k] = np.sum(self.eta[:, k]) + self.init_b[k]

            self.alpha[k] = np.sum(self.eta[:, k]) + self.init_alpha[k]

        
        # update hyperparameters
        if update_params is True:
            for k in range(self.K):
                self.init_alpha[k] = (digamma(np.sum(self.eta[:, k] + self.alpha[k])) - digamma(np.sum(self.alpha))) \
                                    / self.alpha[k] * (digamma(np.sum(self.eta[:, k]) + self.alpha[k]) - digamma(self.alpha[k]))
                """
                self.init_a[k] = self.a[k] * (digamma(self.eta[:, k].T.dot(X) + self.a[k]) - digamma(self.a[k])) \
                                / (np.log(self.eta[:, k].T.dot(X) + self.init_b[k]) - np.log(self.init_b[k]))
                
                self.init_b[k] = np.sum(self.eta[:, k] * self.init_a[k]) \
                                / (self.a[k] * np.sum(self.eta[:, k]) - self.init_a[k])
                """
        
        elbo = self.calculate_elbo()
    
        return elbo


    def calculate_elbo(self):
        """
        calculate evidence lower bound
        """

        first_term = 0.
        second_term = 0.
        third_term = 0.
        fourth_term = 0.
        fifth_term = 0.
        sixth_term = 0.
        seventh_term = 0.

        for n in range(self.N):
            for k in range(self.K):
                first_term += self.eta[n, k] * (self.X[n] * ((digamma(self.a[k]) - np.log(self.b[k])) - (self.a[k] / self.b[k])))

                second_term += self.eta[n, k] * (digamma(self.alpha[k]) - digamma(np.sum(self.alpha)))

                third_term -= self.eta[n, k] * np.log(self.eta[n, k])


        for k in range(self.K):
    
            fourth_term -= (self.init_a[k] - 1) * (digamma(self.a[k]) - np.log(self.b[k])) \
                           - self.init_b[k] * (self.a[k] / self.b[k])
            
            fifth_term += (self.a[k] - 1) * (digamma(self.a[k]) - np.log(self.b[k])) \
                          - self.a[k]
            
            sixth_term -= (self.init_alpha[k] - 1) * (digamma(self.alpha[k]) - digamma(np.sum(self.alpha)))
            seventh_term += (self.alpha[k] - 1) * (digamma(self.alpha[k]) - digamma(np.sum(self.alpha)))




        elbo = first_term + second_term + third_term + fourth_term \
               + fifth_term + sixth_term + seventh_term
        
        # normalize terms from Gamma dist & Dirichlet dist
        norm_terms = gammaln(np.sum(self.init_alpha)) - np.sum(gammaln(self.init_alpha)) \
               - gammaln(np.sum(self.alpha)) + np.sum(gammaln(self.alpha)) \
               + np.sum(self.init_a * np.log(self.init_b) - gammaln(self.init_a)) \
               - np.sum(self.a * np.log(self.b) - gammaln(self.a))
        
        elbo += -1 * norm_terms

        return elbo



if __name__ == "__main__":
    
    tmp1 = stats.poisson.rvs(15, size=300)
    tmp2 = stats.poisson.rvs(30, size=200)
    X = np.hstack((tmp1, tmp2)).reshape(-1, 1)
    K = 2
    maxiter = 100

    model = PMM(X, K, maxiter=maxiter, update_params=False)

    # run variational inference
    losses = model.variational_inference()

    fig, (axloss,axplot) = plt.subplots(ncols=2, figsize=(10, 4))

    # plot evidence lower bounds
    axloss.plot(losses)
    axloss.set_xlabel("iterations")
    axloss.set_ylabel("elbo")
    axloss.set_title("convergence elbo")


    # plot lambdas
    lmd_x = np.linspace(10, 60, 1000)
    lmd_1_dist = stats.gamma(a=model.a[0], scale=1/model.b[0]).pdf(lmd_x)
    lmd_2_dist = stats.gamma(a=model.a[1], scale=1/model.b[1]).pdf(lmd_x)
    axplot.plot(lmd_x, lmd_1_dist, label="lambda_1")
    axplot.plot(lmd_x, lmd_2_dist, label="lambda_2")
    axplot.set_xlabel("lmd")
    axplot.set_ylabel("density")
    axplot.set_title("posterior lmd")
    axplot.legend()

    plt.show()

import numpy as np
from handwriting_gen.distributions import bivariate_normal_sample


def sample_stroke(pi, e, mixture_bias, bv_params, std_bias):
    """
    Sample a stroke from mixture model parameters including bias terms
    to make handwriting neater
    """

    # add bias to mixture probabilities
    pi = np.exp(np.log(pi[0]) * (1 + mixture_bias))
    pi = pi / pi.sum()

    # sample a mixture component
    mixture_idx = np.random.choice(pi.shape[0], 1, p=pi)[0]
    for idx in range(len(bv_params)):
        bv_params[idx] = bv_params[idx][0][mixture_idx]

    mu1, mu2, sigma1, sigma2, rho = bv_params

    # add bias to std of bivariate gaussian ouputs
    sigma1 /= std_bias
    sigma2 /= std_bias

    # sample the stroke point
    x2, x3 = bivariate_normal_sample(
        mu1, mu2, sigma1, sigma2, rho)[0]
    x1 = np.random.binomial(1, e[0])[0]

    return np.array([[[x1, x2, x3]]])

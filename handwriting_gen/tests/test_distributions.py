import numpy as np
import tensorflow as tf
from handwriting_gen.distributions import bivariate_normal_likelihood


def test_bivariate_normal_likelihood():
    from scipy.stats import multivariate_normal
    mu1, mu2 = -0.5, 0.22
    sigma1, sigma2 = 0.3, 0.9
    rho = -0.15
    x1, x2 = -1.0, 2.3

    cov_off_diag = rho * sigma1 * sigma2
    p = multivariate_normal(
        [mu1, mu2], [[sigma1**2, cov_off_diag], [cov_off_diag, sigma2**2]]
    ).pdf([x1, x2])

    sess = tf.Session()
    assert np.allclose(p, sess.run(
        bivariate_normal_likelihood(x1, x2, mu1, mu2, sigma1, sigma2, rho)))

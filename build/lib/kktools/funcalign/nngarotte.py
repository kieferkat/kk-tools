"""
Non-Negative Garotte implementation with the scikit-learn
"""

# Author: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#         Jaques Grobler (__main__ script) <jaques.grobler@inria.fr>
#
# License: BSD Style.
#
# KK: added in ridge estimate alternative to the Linear Regression ols 
# estimate 
#

import numpy as np


from sklearn.linear_model.base import LinearModel
from sklearn.linear_model import LinearRegression, Lasso, lasso_path, RidgeCV
from sklearn.utils import check_random_state
from sklearn.linear_model import lars_path


def non_negative_garotte(X, y, alpha, tol=0.001, estimate_method='ols'):

    if estimate_method == 'ols':
        coef_est = LinearRegression(fit_intercept=False).fit(X, y).coef_
    elif estimate_method == 'ridge':
        coef_est = RidgeCV(alphas=[0.01,0.1,1.,10.], fit_intercept=False).fit(X, y).coef_


    X = X * coef_est[np.newaxis, :]
    shrink_coef = Lasso(alpha=alpha, fit_intercept=False,
                        positive=True, normalize=False,
                        tol=tol).fit(X, y).coef_

    # Shrunken betas
    coef = coef_est * shrink_coef

    # Residual Sum of Squares
    rss = np.sum((y - np.dot(X, coef)) ** 2)
    return coef, shrink_coef, rss


def non_negative_garotte_path(X, y, eps=1e-3, n_alphas=100, alphas=None,
                   precompute='auto', estimate_method='ols', **params):

    if estimate_method == 'ols':
        coef_est = LinearRegression(fit_intercept=False).fit(X, y).coef_
    elif estimate_method == 'ridge':
        coef_est = RidgeCV(alphas=[.01,.1,1.,10], fit_intercept=False).fit(X, y).coef_

    X = X * coef_est[np.newaxis, :]

    # Use lars_path even if it does not support positivity (much faster)
    _, _, shrink_coef_path = lars_path(X, y, method='lasso')

    # models = lasso_path(X, y, eps, n_alphas, alphas=None,
    #            precompute=precompute, Xy=None, fit_intercept=False,
    #            normalize=False, copy_X=True, verbose=False,
    #            **params)
    #
    # shrink_coef_path = np.array([m.coef_ for m in models]).T

    coef_path = shrink_coef_path * coef_est[:, None]

    # Residual Sum of Squares
    rss_path = np.sum((y[:, None] - np.dot(X, coef_path)) ** 2, axis=0)

    return coef_path, shrink_coef_path, rss_path


class NonNegativeGarrote(LinearModel):
    """NonNegativeGarrote

    Ref:
    Breiman, L. (1995), "Better Subset Regression Using the Nonnegative
    Garrote," Technometrics, 37, 373-384. [349,351]
    """
    def __init__(self, alpha, fit_intercept=True, tol=1e-4, normalize=False,
                 copy_X=True, estimate_method='ols'):
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.tol = tol
        self.normalize = normalize
        self.copy_X = copy_X
        self.estimate_method = estimate_method

    def fit(self, X, y):

        X, y, X_mean, y_mean, X_std = self._center_data(X, y,
                self.fit_intercept, self.normalize, self.copy_X)

        self.coef_, self.shrink_coef_, self.rss_ = \
                                    non_negative_garotte(X, y, self.alpha, 
                                                         estimate_method=self.estimate_method)
        self._set_intercept(X_mean, y_mean, X_std)


    



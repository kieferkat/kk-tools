
import pylab, pandas
import statsmodels.api
import numpy as np



# returns the fit model of the logistic regression:
def logistic(X, Y, intercept=True, verbose=True):
    n, p = X.shape
    
    if intercept:
        X = np.hstack([np.ones(n,1), X])
        
    if verbose:
        print 'Y shape: ', Y.shape
        print 'X shape: ', X.shape
        
    model = statsmodels.api.Logit(Y, X)
    if verbose:
        print 'Fitting logistic model...'
        
    results = model.fit()
    return results


# returns the accuracy of a given logistic regression on predicting a test set Y:
def logistic_predict(X, Y, train_indices=None, test_indices=None):
    
    if not train_indices and not test_indices:
        trainX, testX = X, X
        trainY, testY = Y, Y
        
    elif not test_indices:
        train_indices = train_indices or [t for t in range(len(Y)) if t not in test_indices]
        test_indices = test_indices or [t for t in range(len(Y)) if t not in train_indices]
        trainX, trainY = [X[t] for t in train_indices], [Y[t] for t in train_indices]
        testX, testY = [X[t] for t in test_indices], [Y[t] for t in test_indices]

    
    training_fit = logistic(trainX, trainY)
    training_betas = training_fit.params
    
    testing_eta = np.dot(testX, training_betas)
    testing_p = np.exp(testing_eta) / (1. + np.exp(testing_eta))
    
    test_predict = (testing_p > 0.5)
    return (testY==test_predict).sum()*1. / testY.shape[0]
    
    
    
    
    
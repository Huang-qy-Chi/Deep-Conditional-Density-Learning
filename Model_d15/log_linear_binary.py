
import numpy as np
from numpy.linalg import inv
from numpy.linalg import pinv
import scipy.optimize as spo
#%%----------Input:(R,A,X)-- Output:E(R|A,X) --------------
def E_ols_bin(train_data, val_data, test_data):
    # Linear model with least square estimation
    ## Step 1: data arrangement: with intercept
    X_train = train_data['X']
    X_val = val_data['X']
    X_train = np.vstack((X_train,X_val))  #least square with linear model does not need validation
    A_train = train_data['A']
    A_val = val_data['A']
    A_train = np.vstack((A_train,A_val))

    n = A_train.shape[0]
    A0 = np.ones(n)
    # X_train1 = np.c_[A0,X_train]  #with intercept 

    ## Step 2: consider influence of A and data reorganization
    d = X_train.shape[1]
    A1 = np.tile(A_train[:,0],(d,1)).T  #untreat
    A2 = np.tile(A_train[:,1],(d,1)).T  #treated
    X_tilde = np.hstack((A1*X_train,A2*X_train)) #intercept,beta,gamma
    X_tilde = np.c_[np.ones(n),X_tilde]
    p = (X_tilde.shape)[1]

    ## Step 3: estimate the parameter
    R_0_train = train_data['R']
    R_0_val = val_data['R']
    R_0_train = np.r_[R_0_train,R_0_val]
    logR_train = np.log(R_0_train)
    def TF(*args):   
        Loss_F = np.mean((logR_train-X_tilde@args[0])**2)
        return Loss_F
    result = spo.minimize(TF,np.zeros(p),method='SLSQP') #
    para = result['x']
    # para = pinv(X_tilde.T@X_tilde+regu)@(X_tilde.T@logR_train)  #least square
    # zeta = para[0:(d)]
    intercept = para[0]
    beta = para[1:(d+1)]
    gamma = para[(d+1):]

    ## Step 4: estimate the conditional mean on test data
    X_test = test_data['X']
    n1 = X_test.shape[0]
    At1 = np.ones(n1)
    # X_test1 = np.c_[At1, X_test]
    X_test_un = np.hstack((X_test,np.zeros(X_test.shape)))  #untreat
    X_test_un = np.c_[np.ones(n1),X_test_un]
    X_test_tr = np.hstack((np.zeros(X_test.shape),X_test))  #treated
    X_test_tr = np.c_[np.ones(n1),X_test_tr]
    Cmean_R_X_treated=np.exp(X_test_tr@para)  #conditional mean for treated
    Cmean_R_X_untreat=np.exp(X_test_un@para)  #conditional mean for untreat


    return{
        'Cmean_R_X_untreat': Cmean_R_X_untreat, # 500*1
        'Cmean_R_X_treated': Cmean_R_X_treated,
        'intercept': intercept,
        'beta': beta, 
        'gamma': gamma  
    }


# Qian and Murphy (2011), linear basis
# OLS may suffer multicollinearity, using variable selection may help
from sklearn.linear_model import Lasso
from sklearn.model_selection import cross_val_score
def E_lasso_bin(train_data,val_data,test_data,lam_min=-4,lam_max=-1,K=5): 
   # Linear model with LASSO estimation
    ## Step 1: data arrangement: with intercept
    X_train = train_data['X']
    X_val = val_data['X']
    X_train = np.vstack((X_train,X_val))  #least square with linear model does not need validation
    A_train = train_data['A']
    A_val = val_data['A']
    A_train = np.vstack((A_train,A_val))
    n = A_train.shape[0]
    R_0_train = train_data['R']
    R_0_val = val_data['R']
    R_0_train = np.r_[R_0_train,R_0_val]
    logR_train = np.log(R_0_train)


    ## Step 2: consider influence of A and data reorganization
    d = X_train.shape[1]
    A1 = np.tile(A_train[:,0],(d,1)).T  #untreat
    A2 = np.tile(A_train[:,1],(d,1)).T  #treated
    X_tilde = np.hstack((A1*X_train,A2*X_train)) #intercept,beta,gamma
    X_tilde = np.c_[np.ones(n),X_tilde]
    p = (X_tilde.shape)[1]
    
    ## Step 3: K-fold cross validation: alpha selection
    alphas = np.logspace(lam_min, lam_max, 10)  # default: 1e-4 to 1e-1
    cv_scores = []
    for alpha in alphas:
        lasso_cv = Lasso(alpha=alpha, fit_intercept=False)
        scores = cross_val_score(lasso_cv, X_tilde, logR_train, cv=K, scoring='neg_mean_squared_error')
        cv_scores.append(-scores.mean())
    best_alpha = alphas[np.argmin(cv_scores)]

    ## Step 4: Estimate again using the best alpha
    lasso_best = Lasso(alpha=best_alpha, fit_intercept=False)
    lasso_best.fit(X_tilde, logR_train)
    para = lasso_best.coef_
    intercept = para[0]
    beta = para[1:d+1]
    gamma = para[d+1:]

    ## Step 5: Conditional mean on test data
    X_test = test_data['X']
    n1 = X_test.shape[0]
    X_test_un = np.hstack((X_test,np.zeros(X_test.shape)))  #untreat
    X_test_un = np.c_[np.ones(n1),X_test_un]
    X_test_tr = np.hstack((np.zeros(X_test.shape),X_test))  #treated
    X_test_tr = np.c_[np.ones(n1),X_test_tr]
    Cmean_R_X_treated = np.exp(X_test_tr@para)  #conditional mean for treated
    Cmean_R_X_untreat = np.exp(X_test_un@para)  #conditional mean for untreat

    return{
        'Cmean_R_X_untreat': Cmean_R_X_untreat, # 500*1
        'Cmean_R_X_treated': Cmean_R_X_treated,
        'intercept': intercept,
        'beta': beta, 
        'gamma': gamma  
    }


























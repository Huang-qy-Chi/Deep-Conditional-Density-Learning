import numpy as np
import torch


def optITR_lin_bin(test_data,Cmean_R_X_treated, Cmean_R_X_untreat, beta,gamma):
    d_est = (Cmean_R_X_treated-Cmean_R_X_untreat)>0
    d_est = d_est.astype(int)  #estimated decision
    X_test = test_data['X']
    d_opt = X_test@(gamma-beta)>0  #optimal decision
    d_opt = d_opt.astype(int)
    #
    d_est = np.vstack((1-d_est,d_est)) #untreat-treated
    d_opt = np.vstack((1-d_opt,d_opt))

    #misspecification rate
    misspecification = np.mean(np.abs(d_opt-d_est))

    #quality value 
    theta = np.vstack((beta,gamma))
    quality = X_test@theta.T
    mu_test = test_data['mu0']
    epsilon_test = test_data['epsilon']

    V_est = np.mean(np.exp(np.sum(d_est.T*quality,axis=1)+mu_test+epsilon_test))

    V_opt = np.mean(np.exp(np.sum(d_opt.T*quality,axis=1)+mu_test+epsilon_test))
    
    Regret = V_opt - V_est
    
    #relative error
    # C_mean = 
    # Delta = 
    return{
        'mis_rate': misspecification,
        'Regret': Regret
    }





def optITR_deep_bin(test_data,Cmean_R_X_treated, Cmean_R_X_untreat):
    #estimated decision
    d_est = (Cmean_R_X_treated-Cmean_R_X_untreat)>0
    d_est = d_est.astype(int)  #estimated decision
    X = test_data['X']
    
    #optimal decision
    #A=0, influence g_X
    g_X = test_data['g_X']
    #A=1, influence h_X
    h_X = test_data['h_X']
    
    d_opt = (h_X-g_X)>0  #optimal decision
    d_opt = d_opt.astype(int)
    #
    d_est = np.vstack((1-d_est,d_est)) #untreat-treated
    d_opt = np.vstack((1-d_opt,d_opt))

    #misspecification rate
    misspecification = np.mean(np.abs(d_opt-d_est))

    #quality value 
    quality = np.vstack((g_X , h_X)) 
    quality = quality.T
    mu_test = test_data['mu0']
    epsilon_test = test_data['epsilon']

    V_est = np.mean(np.exp(np.sum(d_est.T*quality,axis=1)+mu_test+epsilon_test)) #logR -> R

    V_opt = np.mean(np.exp(np.sum(d_opt.T*quality,axis=1)+mu_test+epsilon_test))
    
    Regret = V_opt - V_est
    
    #relative error
    # C_mean = 
    # Delta = 
    return{
        'mis_rate': misspecification,
        'Regret': Regret
    }












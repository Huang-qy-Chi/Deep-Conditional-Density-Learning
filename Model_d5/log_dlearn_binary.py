
import numpy as np
import sklearn
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import RandomForestClassifier  # propensity score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso, LassoCV
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_score
from xgboost import XGBRegressor
import warnings

#%%---------------------------Competitive 1: AD Learning-------------------------------------

def ADL_bin(train_data,val_data,test_data):
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
    X_train1 = np.c_[np.ones(n),X_train]



    ## Step 2: estimate propensity score and conditional expectation via linear regression

    ###Step 2.1 standardizetion, without intercept
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    ### Step 2.2 estimate propensity score pi(X) = P(A=1|X)
    lr = LogisticRegression()
    lr.fit(X_scaled, A_train[:,1])
    pi_X = lr.predict_proba(X_scaled)[:, 1]
    pi_X = np.clip(pi_X, 0.01, 0.99)  # avoid 0

    ### Step 2.3 estimate conditional expectation E(R|A=0,X) and E(R|A=1,X) via linear regression
    reg_A0 = LinearRegression()
    reg_A1 = LinearRegression()
    reg_A0.fit(X_scaled[A_train[:,1] == 0], logR_train[A_train[:,1] == 0])
    reg_A1.fit(X_scaled[A_train[:,1] == 1], logR_train[A_train[:,1] == 1])
    E_logR_A0 = (reg_A0.predict(X_scaled))
    E_logR_A1 = (reg_A1.predict(X_scaled))



    ## Step 3: calculate the AIPW score on test data 
    phi_A0 = np.zeros(n)
    phi_A1 = np.zeros(n)
    phi_A0 = logR_train*(A_train[:,1]==0)/(1-pi_X)+E_logR_A0*(1-(A_train[:,1]==0)/(1-pi_X))
    phi_A1 = logR_train*(A_train[:,1]==1)/(pi_X)+E_logR_A1*(1-(A_train[:,1]==0)/(pi_X))
    CATE = phi_A1-phi_A0


    ## Step 4: ITR learning 
    reg_itr = LinearRegression()
    reg_itr.fit(X_scaled, CATE)
    cate_para = reg_itr.coef_
    tau_hat = reg_itr.predict(X_scaled)
    itr_adl_train = (tau_hat > 0).astype(int)

    X_test = test_data['X'] #least square with linear model does not need validation
    A_test = test_data['A']
    n1 = A_test.shape[0]
    R_0_test = test_data['R']
    logR_test = np.log(R_0_test)
    X_scaled_test = scaler.fit_transform(X_test)

    tau_hat_test = reg_itr.predict(X_scaled_test)
    itr_adl_test = (tau_hat_test>0).astype(int)



    ## output: AIPW score, CATE estimation, estimation of propensity score and conditional mean
    return{
        'itr_test': itr_adl_test,
        'itr_train': itr_adl_train,
        'CATE': cate_para, 
        'pi_train': pi_X
    }




#%%------------------------Competetive method 2: Robust Direct Learning--------------------------
def RDL_bin(train_data,val_data,test_data,alphas=None,cv=5):
    if alphas is None:
        alphas = np.logspace(-2, 0, 10)  #1e-2 to 1
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
    
    X_test = test_data['X'] #least square with linear model does not need validation
    A_test = test_data['A']
    n1 = A_test.shape[0]
    R_0_test = test_data['R']
    logR_test = np.log(R_0_test)


    # ## Step 2: estimate the mu function propensity score via kernel method
    # mu_model = KernelRidge(kernel='rbf', alpha=alpha, gamma=gamma)
    # mu_model.fit(X_train, logR_train)
    # mu_pred_train = mu_model.predict(X_train)
    # mu_pred_test = mu_model.predict(X_test)
    
    ## Step 2: estimate the mu function propensity score via LASSO
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)
    mu_model = LassoCV(alphas=alphas, cv = cv, random_state=42, n_jobs=-1)
    mu_model.fit(X_train, logR_train)
    mu_pred_train = mu_model.predict(X_train)
    mu_pred_test = mu_model.predict(X_test)


    # ## Step 3: estimate the propensity score via random forest
    # pi_model = RandomForestClassifier(n_estimators=100, random_state=42)
    # pi_model.fit(X_train, A_train)
    # pi_pred_train = pi_model.predict_proba(X_train)[:, 1]
    # pi_pred_test = pi_model.predict_proba(X_test)[:, 1]

    ## Step 3: estimate the propensity score via Logistic regression
    pi_model = LogisticRegression(random_state=42, max_iter=1000)
    pi_model.fit(X_train, A_train[:,1])
    pi_pred_train = pi_model.predict_proba(X_train)[:, 1]
    pi_pred_test = pi_model.predict_proba(X_test)[:, 1]



    # estimate the CATE tau(X)
    # pseudo_outcome = (logR_train - mu_pred_train) / (A_train - pi_pred_train + 1e-10)
    # tau_model = KernelRidge(kernel=kernel, alpha=alpha, gamma=gamma)
    # tau_model.fit(X_train, pseudo_outcome)
    # tau_pred_train = tau_model.predict(X_train)
    # tau_pred_test = tau_model.predict(X_test)
    pseudo_outcome = (logR_train - mu_pred_train) / (A_train[:,1] - pi_pred_train + 1e-10)
    tau_model_cv = LassoCV(alphas=alphas, cv=cv, random_state=42, n_jobs=-1)
    tau_model_cv.fit(X_train, pseudo_outcome)
    tau_pred_test_cv = tau_model_cv.predict(X_test)
    tau_pred_train_cv = tau_model_cv.predict(X_train)


    ## Step 4: ITR decision on test data
    itr_rdl_test = (tau_pred_test_cv>0).astype(int)
    itr_rdl_train = (tau_pred_train_cv>0).astype(int)

    ## output: para of tau, propensity score, decision on test data
    ## It does not use the fuction optITR
    return{
        'itr_train': itr_rdl_train,
        'itr_test': itr_rdl_test,
        'pi_train': pi_pred_train,
        'pi_test': pi_pred_test,
        'tau_train': tau_pred_train_cv,
        'tau_test': tau_pred_test_cv

    }


def RDLITR_lin_bin(test_data):


    return{

    }








#%%-----------------Competetive method 3: SD Learning---------------------------------------------
def SDL_bin(train_data,val_data,test_data,alphas=None,cv=5,precompute=True):
    if alphas is None:
        alphas = np.logspace(-2, 0, 10)  #1e-2 to 1
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
    X_test = test_data['X'] #least square with linear model does not need validation
    A_test = test_data['A']
    n1 = A_test.shape[0]
    R_0_test = test_data['R']
    logR_test = np.log(R_0_test)
    A_train1 = 2*A_train[:,1]-1 #adjust to A in {-1,1}

    ## Step 2: D-Learning via least square 
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)

    pi_model = LogisticRegression(random_state=42, max_iter=1000)
    pi_model.fit(X_train, A_train1)
    pi_pred_train = pi_model.predict_proba(X_train)
    pi_pred_test = pi_model.predict_proba(X_test)
    propensity_weights = np.zeros_like(A_train1, dtype=float)
    for i in range(len(A_train1)):
        if (A_train1)[i] == 1:
            propensity_weights[i] = 1 / (pi_pred_train[i, 1] + 1e-6)  # 1 / P(A=1|X)，避免除以零
        else:
            propensity_weights[i] = 1 / (pi_pred_train[i, 0] + 1e-6)  # 1 / P(A=0|X)
    ### IPW estimation between 2AlogR and X via lasso
    Z_train = 2*logR_train*(A_train1)   
    f_model = LassoCV(alphas=alphas, cv=cv, fit_intercept=True, random_state=42,precompute=precompute)
    f_model.fit(X_train, Z_train, sample_weight=propensity_weights)
    f_hat = f_model.predict(X_train)
    residuals = Z_train - f_hat

    def estimate_variance_with_ml(X_A, residuals, model_types=['xgboost', 'rf'], cv=5):
        residual_squared = residuals**2  # res_square
        kf = KFold(n_splits=cv, shuffle=True, random_state=42)
        best_score = -np.inf
        best_model = None
        best_model_type = None
        
        # compare boost and randomforest
        for model_type in model_types:
            if model_type == 'xgboost':
                model = XGBRegressor(n_estimators=100, random_state=42)
            else:  # 'rf'
                model = RandomForestRegressor(n_estimators=100, random_state=42)
            
            # cross validation, -MSE
            scores = cross_val_score(model, X_A, residual_squared, cv=kf, scoring='neg_mean_squared_error')
            mean_score = np.mean(scores)
            
            if mean_score > best_score:
                best_score = mean_score
                best_model_type = model_type
                best_model = model
        
        # use best model to approximate
        best_model.fit(X_A, residual_squared)
        sigma_squared_hat = best_model.predict(X_A)
        # avoid singular
        sigma_squared_hat = np.maximum(sigma_squared_hat, 1e-6)
        return best_model, sigma_squared_hat, best_model_type


    ## Step 3: residual reweighting via lasso (or other machine learning)
    residual_squared = residuals**2
    X_A = np.hstack([X_train, A_train1.reshape(-1, 1)])
    variance_model = LassoCV(alphas=alphas, cv=cv, fit_intercept=True, random_state=42,precompute=precompute)
    variance_model.fit(X_A, residual_squared)
    sigma_squared_hat = variance_model.predict(X_A)
    sigma_squared_hat = np.maximum(sigma_squared_hat, 1e-6)
    # variance_model, sigma_squared_hat, varmodel_type =\
    #       estimate_variance_with_ml(X_A, residuals, model_types=['xgboost', 'rf'], cv=cv)
    ## Step 4: adjustment of D-Learning estimation 
    residual_weights = 1 / sigma_squared_hat
    final_model = LassoCV(alphas=alphas, cv=cv, fit_intercept=True, random_state=42,precompute=precompute)
    final_model.fit(X_train, Z_train, sample_weight=residual_weights)
    coef = final_model.coef_

    ## Step 5: decision on test data
    ## rule: X*coef>0, d(X)=1, else d(X)=0
    itr_sdl_test = ((np.sign(np.dot(X_test,coef))+1)/2).astype(int)
    itr_sdl_train = ((np.sign(np.dot(X_train,coef))+1)/2).astype(int)

    ## output: decision, parameter estimation for SD learning
    return{
        'itr_train': itr_sdl_train,
        'itr_test': itr_sdl_test,
        'pi_train': pi_pred_train,
        'pi_test': pi_pred_test #,
        # 'model': variance_model,
        # 'type': varmodel_type
    }



#%%-----------------Competetive method 3: SD Learning---------------------------------------------
def DL_bin(train_data,val_data,test_data,alphas=None,cv=5,precompute=True):
    if alphas is None:
        alphas = np.logspace(-2, 0, 10)  #1e-2 to 1
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
    X_test = test_data['X'] #least square with linear model does not need validation
    A_test = test_data['A']
    n1 = A_test.shape[0]
    R_0_test = test_data['R']
    logR_test = np.log(R_0_test)
    A_train1 = 2*A_train[:,1]-1 #adjust to A in {-1,1}

    ## Step 2: D-Learning via least square 
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)

    pi_model = LogisticRegression(random_state=42, max_iter=1000)
    pi_model.fit(X_train, A_train1)
    pi_pred_train = pi_model.predict_proba(X_train)
    pi_pred_test = pi_model.predict_proba(X_test)
    propensity_weights = np.zeros_like(A_train1, dtype=float)
    for i in range(len(A_train1)):
        if (A_train1)[i] == 1:
            propensity_weights[i] = 1 / (pi_pred_train[i, 1] + 1e-6)  # 1 / P(A=1|X)，避免除以零
        else:
            propensity_weights[i] = 1 / (pi_pred_train[i, 0] + 1e-6)  # 1 / P(A=0|X)
    ### IPW estimation between 2AlogR and X via lasso
    Z_train = 2*logR_train*(A_train1)   
    f_model = LassoCV(alphas=alphas, cv=cv, fit_intercept=True, random_state=42,precompute=precompute)
    f_model.fit(X_train, Z_train, sample_weight=propensity_weights)
    f_hat = f_model.predict(X_train)
    coef = f_model.coef_

   

    ## Step 5: decision on test data
    ## rule: X*coef>0, d(X)=1, else d(X)=0
    itr_sdl_test = ((np.sign(np.dot(X_test,coef))+1)/2).astype(int)
    itr_sdl_train = ((np.sign(np.dot(X_train,coef))+1)/2).astype(int)

    ## output: decision, parameter estimation for SD learning
    return{
        'itr_train': itr_sdl_train,
        'itr_test': itr_sdl_test,
        'pi_train': pi_pred_train,
        'pi_test': pi_pred_test #,
        # 'model': variance_model,
        # 'type': varmodel_type
    }






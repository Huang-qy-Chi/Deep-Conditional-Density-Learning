
import numpy as np
import numpy.random as ndm
from scipy.special import expit


#%%----------------------------Generate Function Linear Binary------------------------------------
def gendata_Linear(n,corr,beta,gamma,pat=3):
    d = len(beta)
    #treatment A
    A = ndm.binomial(1, 0.5, n) #parametric
    A = np.vstack((A,1-A))

    

    #parametric X
    mean = np.zeros(d)
    cov = np.identity(d)*(1-corr) + np.ones((d, d))*corr
    X = np.random.multivariate_normal(mean,cov,n)
    X = np.clip(X, -1, 1) #t-distributed with [-1,1]

    #baseline mu0
    if pat==1:
        mu0 = np.zeros(n)
    elif pat==2:
        mu0 = 0.5 + 0.25*X[:,0] - 0.25*X[:,1]
    else:
        mu0 = -0.5 + 0.25*X[:,0] - 0.25*X[:,1] + 0.5*X[:,2] - 0.5*X[:,3] -0.25*X[:,4]
    
    # 
    # mu0 = -0.5 + 0.25*X[:,0] - 0.25*X[:,1] + 0.5*X[:,2] - 0.5*X[:,3] -0.25*X[:,4]

    X_quality = np.vstack((X @ beta , X @ gamma))  #For A=0 and A=1
    #error epsilon
    sigma = 1
    epsilon = np.random.normal(loc=0, scale=sigma, size=n)
    epsilon = np.clip(epsilon, -1, 1)
    
    #Reward R
    R = np.exp(mu0 + (X_quality*A).sum(axis=0)+epsilon) #log R=beta*X+gamma*X*A+epsilon
    return {
        'A': np.array(A.T, dtype='float32'),
        'X': np.array(X, dtype='float32'),
        'R': np.array(R, dtype='float32'),
        'mu0': np.array(mu0, dtype='float32'),
        'epsilon': np.array(epsilon, dtype='float32')
    }

def gendata_Linear_ax(n,corr,beta,gamma,rho):
    d = len(beta)
    #treatment A
    A = ndm.binomial(1, 0.5, n) #parametric
    A = np.vstack((A,1-A))

    

    #parametric X
    mean = np.zeros(d)
    cov = np.identity(d)*(1-corr) + np.ones((d, d))*corr
    X = np.random.multivariate_normal(mean,cov,n)
    X = np.clip(X, -1, 1) #t-distributed with [-1,1]

    #baseline mu0
    # if pat==1:
    #     mu0 = np.zeros(n)
    # elif pat==2:
    #     mu0 = 0.5 + 0.25*X[:,0] - 0.25*X[:,1]
    # else:
    mu0 = -0.5 + 0.25*X[:,0] - 0.25*X[:,1] + 0.5*X[:,2] - 0.5*X[:,3] -0.25*X[:,4]
    
    # 
    # mu0 = -0.5 + 0.25*X[:,0] - 0.25*X[:,1] + 0.5*X[:,2] - 0.5*X[:,3] -0.25*X[:,4]

    X_quality = rho*np.vstack((X @ beta , X @ gamma))  #For A=0 and A=1
    #error epsilon
    sigma = 1
    epsilon = np.random.normal(loc=0, scale=sigma, size=n)
    epsilon = np.clip(epsilon, -1, 1)
    
    #Reward R
    R = np.exp(mu0 + (X_quality*A).sum(axis=0)+epsilon) #log R=beta*X+gamma*X*A+epsilon
    return {
        'A': np.array(A.T, dtype='float32'),
        'X': np.array(X, dtype='float32'),
        'R': np.array(R, dtype='float32'),
        'mu0': np.array(mu0, dtype='float32'),
        'epsilon': np.array(epsilon, dtype='float32')
    }



#%%----------------------------Generate Function Linear Binary------------------------------------
def gendata_Linear_hetero(n,corr,beta,gamma,pat=3):
    d = len(beta)
    #treatment A
    A = ndm.binomial(1, 0.5, n) #parametric
    A = np.vstack((A,1-A))

    

    #parametric X
    mean = np.zeros(d)
    cov = np.identity(d)*(1-corr) + np.ones((d, d))*corr
    X = np.random.multivariate_normal(mean,cov,n)
    X = np.clip(X, -1, 1) #t-distributed with [-1,1]

    #baseline mu0
    if pat==1:
        mu0 = np.zeros(n)
    elif pat==2:
        mu0 = 0.5 + 0.25*X[:,0] - 0.25*X[:,1]
    else:
        mu0 = -0.5 + 0.25*X[:,0] - 0.25*X[:,1] + 0.5*X[:,2] - 0.5*X[:,3] -0.25*X[:,4]
    
    # 
    # mu0 = -0.5 + 0.25*X[:,0] - 0.25*X[:,1] + 0.5*X[:,2] - 0.5*X[:,3] -0.25*X[:,4]

    X_quality = np.vstack((X @ beta , X @ gamma))  #For A=0 and A=1
    #error epsilon
    sigma = np.sqrt(((X[:,0]+X[:,1])**2+0.1))
    epsilon = np.random.normal(loc=0, scale=sigma, size=n)
    epsilon = np.clip(epsilon, -1, 1)
    
    #Reward R
    R = np.exp(mu0 + (X_quality*A).sum(axis=0)+epsilon) #log R=beta*X+gamma*X*A+epsilon
    return {
        'A': np.array(A.T, dtype='float32'),
        'X': np.array(X, dtype='float32'),
        'R': np.array(R, dtype='float32'),
        'mu0': np.array(mu0, dtype='float32'),
        'epsilon': np.array(epsilon, dtype='float32')
    }






#%%-----------------------------Generate Funtion Deep Binary------------------------------------------
def gendata_Deep(n,corr,pat=3):
    #treatment A
    A = ndm.binomial(1, 0.5, n) #parametric
    A = np.vstack((A,1-A))

    #parametric X
    mean = np.zeros(5)
    cov = np.identity(5)*(1-corr) + np.ones((5, 5))*corr
    X = np.random.multivariate_normal(mean,cov,n)
    X = np.clip(X, -1, 1) #t-distributed with [-1,1]
    #A=0, influence g_X
    g_X = np.cos(X[:,0]**2+2*X[:,1]**2+X[:,2]**3+np.sqrt(X[:,3]+1)*np.log(X[:,4]+2)/20)
    #A=1, influence h_X
    h_X = np.sin(X[:,0]/3 + np.exp(X[:,1])/4 + np.cos(X[:,2]* X[:,3])-(np.log(X[:,4]+2)) -0.45)
    X_quality = np.vstack((g_X , h_X))  #For A=0 and A=1
    #error epsilon
    sigma = 1
    epsilon = np.random.normal(loc=0, scale=sigma,size=n)
    epsilon = np.clip(epsilon, -1, 1)
    
    #baseline mu0
    if pat==1:
        mu0 = np.zeros(n)
    elif pat==2:
        mu0 =  0.5 + 0.25*X[:,0] *X[:,1] +np.exp(X[:,2])/5 -np.sqrt(X[:,3]+1)*np.log(X[:,4]+2)
    else:
        mu0 = -0.5 + 0.25*X[:,0] *X[:,1]+np.sin(X[:,2])/3-np.cos(X[:,3]*np.log(X[:,4]+2))/2
    # mu0 = np.zeros(n)
    # mu0 = 0.5 + 0.25*X[:,0] *X[:,1] +np.exp(X[:,2])/5 -np.sqrt(X[:,3]+1)*np.log(X[:,4]+2)
    # mu0 = -0.5 + 0.25*X[:,0] *X[:,1]+np.sin(X[:,2])/3-np.cos(X[:,3]*np.log(X[:,4]+2))/2

    #Reward R
    R = np.exp(mu0+(X_quality*A).sum(axis=0)+epsilon) #log R=beta*X+gamma*X*A+epsilon
    return {
        'A': np.array(A.T, dtype='float32'),
        'X': np.array(X, dtype='float32'),
        'R': np.array(R, dtype='float32'),
        'g_X': np.array(g_X, dtype='float32'),
        'h_X': np.array(h_X, dtype='float32'),
        'mu0': np.array(mu0, dtype='float32'),
        'epsilon': np.array(epsilon, dtype='float32')
    }




def gendata_Deep_ax(n,corr,rho):
    #treatment A
    A = ndm.binomial(1, 0.5, n) #parametric
    A = np.vstack((A,1-A))

    #parametric X
    mean = np.zeros(5)
    cov = np.identity(5)*(1-corr) + np.ones((5, 5))*corr
    X = np.random.multivariate_normal(mean,cov,n)
    X = np.clip(X, -1, 1) #t-distributed with [-1,1]
    #A=0, influence g_X
    g_X = np.cos(X[:,0]**2+2*X[:,1]**2+X[:,2]**3+np.sqrt(X[:,3]+1)*np.log(X[:,4]+2)/20)
    #A=1, influence h_X
    h_X = np.sin(X[:,0]/3 + np.exp(X[:,1])/4 + np.cos(X[:,2]* X[:,3])-(np.log(X[:,4]+2)) -0.45)
    # h_X = g_X
    X_quality = rho*np.vstack((g_X , h_X))  #For A=0 and A=1
    #error epsilon
    sigma = 1
    epsilon = np.random.normal(loc=0, scale=sigma,size=n)
    epsilon = np.clip(epsilon, -1, 1)
    
    #baseline mu0
    # if pat==1:
    #     mu0 = np.zeros(n)
    # elif pat==2:
    #     mu0 =  0.5 + 0.25*X[:,0] *X[:,1] +np.exp(X[:,2])/5 -np.sqrt(X[:,3]+1)*np.log(X[:,4]+2)
    # else:
    mu0 = -0.5 + 0.25*X[:,0] *X[:,1]+np.sin(X[:,2])/3-np.cos(X[:,3]*np.log(X[:,4]+2))/2
    # mu0 = np.zeros(n)
    # mu0 = 0.5 + 0.25*X[:,0] *X[:,1] +np.exp(X[:,2])/5 -np.sqrt(X[:,3]+1)*np.log(X[:,4]+2)
    # mu0 = -0.5 + 0.25*X[:,0] *X[:,1]+np.sin(X[:,2])/3-np.cos(X[:,3]*np.log(X[:,4]+2))/2

    #Reward R
    R = np.exp(mu0+(X_quality*A).sum(axis=0)+epsilon) #log R=beta*X+gamma*X*A+epsilon
    return {
        'A': np.array(A.T, dtype='float32'),
        'X': np.array(X, dtype='float32'),
        'R': np.array(R, dtype='float32'),
        'g_X': np.array(g_X, dtype='float32'),
        'h_X': np.array(h_X, dtype='float32'),
        'mu0': np.array(mu0, dtype='float32'),
        'epsilon': np.array(epsilon, dtype='float32')
    }










#%%-----------------------------Generate Funtion Deep Binary------------------------------------------
def gendata_Deep_hetero(n,corr,pat=3):
    #treatment A
    A = ndm.binomial(1, 0.5, n) #parametric
    A = np.vstack((A,1-A))

    #parametric X
    mean = np.zeros(5)
    cov = np.identity(5)*(1-corr) + np.ones((5, 5))*corr
    X = np.random.multivariate_normal(mean,cov,n)
    X = np.clip(X, -1, 1) #t-distributed with [-1,1]
    #A=0, influence g_X
    g_X = np.cos(X[:,0]**2+2*X[:,1]**2+X[:,2]**3+np.sqrt(X[:,3]+1)*np.log(X[:,4]+2)/20)
    #A=1, influence h_X
    h_X = np.sin(X[:,0]/3 + np.exp(X[:,1])/4 + np.cos(X[:,2]* X[:,3])-(np.log(X[:,4]+2)) -0.45)
    X_quality = np.vstack((g_X , h_X))  #For A=0 and A=1
    #error epsilon
    sigma = np.sqrt(((X[:,0]+X[:,1])**2+0.1))
    epsilon = np.random.normal(loc=0, scale=sigma,size=n)
    epsilon = np.clip(epsilon, -1, 1)
    
    #baseline mu0
    if pat==1:
        mu0 = np.zeros(n)
    elif pat==2:
        mu0 =  0.5 + 0.25*X[:,0] *X[:,1] +np.exp(X[:,2])/5 -np.sqrt(X[:,3]+1)*np.log(X[:,4]+2)
    else:
        mu0 = -0.5 + 0.25*X[:,0] *X[:,1]+np.sin(X[:,2])/3-np.cos(X[:,3]*np.log(X[:,4]+2))/2
    # mu0 = np.zeros(n)
    # mu0 = 0.5 + 0.25*X[:,0] *X[:,1] +np.exp(X[:,2])/5 -np.sqrt(X[:,3]+1)*np.log(X[:,4]+2)
    # mu0 = -0.5 + 0.25*X[:,0] *X[:,1]+np.sin(X[:,2])/3-np.cos(X[:,3]*np.log(X[:,4]+2))/2

    #Reward R
    R = np.exp(mu0+(X_quality*A).sum(axis=0)+epsilon) #log R=beta*X+gamma*X*A+epsilon
    return {
        'A': np.array(A.T, dtype='float32'),
        'X': np.array(X, dtype='float32'),
        'R': np.array(R, dtype='float32'),
        'g_X': np.array(g_X, dtype='float32'),
        'h_X': np.array(h_X, dtype='float32'),
        'mu0': np.array(mu0, dtype='float32'),
        'epsilon': np.array(epsilon, dtype='float32')
    }


#%%-----------------------------Generate Funtion Deep Binary------------------------------------------
def gendata_Deep_mixG(n,corr,pat=3):
    #treatment A
    A = ndm.binomial(1, 0.5, n) #parametric
    A = np.vstack((A,1-A))

    #parametric X
    mean = np.zeros(5)
    cov = np.identity(5)*(1-corr) + np.ones((5, 5))*corr
    X = np.random.multivariate_normal(mean,cov,n)
    X = np.clip(X, -1, 1) #t-distributed with [-1,1]
    #A=0, influence g_X
    g_X = np.cos(X[:,0]**2+2*X[:,1]**2+X[:,2]**3+np.sqrt(X[:,3]+1)*np.log(X[:,4]+2)/20)
    #A=1, influence h_X
    h_X = np.sin(X[:,0]/3 + np.exp(X[:,1])/4 + np.cos(X[:,2]* X[:,3])-(np.log(X[:,4]+2)) -0.45)
    X_quality = np.vstack((g_X , h_X))  #For A=0 and A=1
    #error epsilon: mixture of Guassian
    
    epsilon = np.random.normal(loc=0, scale=np.sqrt((X[:,0]**2+0.5)),size=n)
    epsilon = np.clip(epsilon, -1, 1)
    
    #baseline mu0
    if pat==1:
        mu0 = np.zeros(n)
    elif pat==2:
        mu0 =  0.5 + 0.25*X[:,0] *X[:,1] +np.exp(X[:,2])/5 -np.sqrt(X[:,3]+1)*np.log(X[:,4]+2)
    else:
        mu0 = -0.5 + 0.25*X[:,0] *X[:,1]+np.sin(X[:,2])/3-np.cos(X[:,3]*np.log(X[:,4]+2))/2
    # mu0 = np.zeros(n)
    # mu0 = 0.5 + 0.25*X[:,0] *X[:,1] +np.exp(X[:,2])/5 -np.sqrt(X[:,3]+1)*np.log(X[:,4]+2)
    # mu0 = -0.5 + 0.25*X[:,0] *X[:,1]+np.sin(X[:,2])/3-np.cos(X[:,3]*np.log(X[:,4]+2))/2

    #Reward R
    R = np.exp(mu0+(X_quality*A).sum(axis=0)+epsilon) #log R=beta*X+gamma*X*A+epsilon
    return {
        'A': np.array(A.T, dtype='float32'),
        'X': np.array(X, dtype='float32'),
        'R': np.array(R, dtype='float32'),
        'g_X': np.array(g_X, dtype='float32'),
        'h_X': np.array(h_X, dtype='float32'),
        'mu0': np.array(mu0, dtype='float32'),
        'epsilon': np.array(epsilon, dtype='float32')
    }





#%%-----------------------------Generate Funtion Deep Binary------------------------------------------
def gendata_Deep_con(n,corr,pat=3):
    #treatment A
    A = ndm.binomial(1, 0.5, n) #parametric
    A = np.vstack((A,1-A))

    #parametric X
    mean = np.zeros(5)
    cov = np.identity(5)*(1-corr) + np.ones((5, 5))*corr
    X = np.random.multivariate_normal(mean,cov,n)
    X = np.clip(X, -1, 1) #t-distributed with [-1,1]
    #A=0, influence g_X
    g_X = np.cos(X[:,0]**2+2*X[:,1]**2+X[:,2]**3+np.sqrt(X[:,3]+1)*np.log(X[:,4]+2)/20)
    #A=1, influence h_X
    h_X = np.sin(X[:,0]/3 + np.exp(X[:,1])/4 + np.cos(X[:,2]* X[:,3])-(np.log(X[:,4]+2)) -0.45)
    X_quality = np.vstack((g_X , h_X))  #For A=0 and A=1
    # Confounder setting: U, both infect A and R
    U = ndm.normal(loc=1.5, scale=0.25, size = n)
    logit_A = 0.5 * U - 0.25 * X[:,0]
    prob_A = expit(logit_A) # logit to probability
    A = np.random.binomial(1, prob_A)
    A = np.vstack((A,1-A))


    U_value = A.T[:,1]*U
    
    epsilon = np.random.normal(loc=0, scale=np.sqrt((X[:,0]**2+0.5)),size=n)
    epsilon = np.clip(epsilon, -1, 1)
    
    #baseline mu0
    if pat==1:
        mu0 = np.zeros(n)
    elif pat==2:
        mu0 =  0.5 + 0.25*X[:,0] *X[:,1] +np.exp(X[:,2])/5 -np.sqrt(X[:,3]+1)*np.log(X[:,4]+2)
    else:
        mu0 = -0.5 + 0.25*X[:,0] *X[:,1]+np.sin(X[:,2])/3-np.cos(X[:,3]*np.log(X[:,4]+2))/2
    # mu0 = np.zeros(n)
    # mu0 = 0.5 + 0.25*X[:,0] *X[:,1] +np.exp(X[:,2])/5 -np.sqrt(X[:,3]+1)*np.log(X[:,4]+2)
    # mu0 = -0.5 + 0.25*X[:,0] *X[:,1]+np.sin(X[:,2])/3-np.cos(X[:,3]*np.log(X[:,4]+2))/2

    #Reward R
    R = np.exp(mu0+(X_quality*A).sum(axis=0)+U_value+epsilon) #log R=beta*X+gamma*X*A+epsilon
    return {
        'A': np.array(A.T, dtype='float32'),
        'X': np.array(X, dtype='float32'),
        'R': np.array(R, dtype='float32'),
        'g_X': np.array(g_X, dtype='float32'),
        'h_X': np.array(h_X, dtype='float32'),
        'mu0': np.array(mu0, dtype='float32'),
        'epsilon': np.array(epsilon, dtype='float32')
    }


#%%---------------------------------------
def gendata_Deep_size(n,corr,pat=3):
    #treatment A
    # A = ndm.binomial(1, 0.5, n) #parametric
    # A = np.vstack((A,1-A))

    #parametric X
    mean = np.zeros(5)
    cov = np.identity(5)*(1-corr) + np.ones((5, 5))*corr
    X = np.random.multivariate_normal(mean,cov,n)
    X = np.clip(X, -1, 1) #t-distributed with [-1,1]
    #A=0, influence g_X
    g_X = np.cos(X[:,0]**2+2*X[:,1]**2+X[:,2]**3+np.sqrt(X[:,3]+1)*np.log(X[:,4]+2)/20)
    #A=1, influence h_X
    h_X = np.sin(X[:,0]/3 + np.exp(X[:,1])/4 + np.cos(X[:,2]* X[:,3])-(np.log(X[:,4]+2)) -0.45)
    X_quality = np.vstack((g_X , h_X))  #For A=0 and A=1
    variance = 0.5
    epsilon = np.random.normal(loc=0, scale=np.sqrt(variance+0.5),size=n)
    epsilon = np.clip(epsilon, -1, 1)

    # Confounder setting: U, both infect A and R
    # U = ndm.normal(loc=1.5, scale=0.25, size = n)
    # logit_A = 0.5 * U - 0.25 * X[:,0]
    # prob_A = expit(logit_A)  # logit to probability
    # A = np.random.binomial(1, 0.5, n)
    A = np.zeros(n)
    A = np.vstack((A,1-A))
    # A1 = np.vstack((np.zeros(n),np.ones(n)))


    # U_value = A.T[:,1]*U
    #baseline mu0
    if pat==1:
        mu0 = np.zeros(n)
    elif pat==2:
        mu0 =  0.5 + 0.25*X[:,0] *X[:,1] +np.exp(X[:,2])/5 -np.sqrt(X[:,3]+1)*np.log(X[:,4]+2)
    else:
        mu0 = -0.5 + 0.25*X[:,0] *X[:,1]+np.sin(X[:,2])/3-np.cos(X[:,3]*np.log(X[:,4]+2))/2
    # mu0 = np.zeros(n)
    # mu0 = 0.5 + 0.25*X[:,0] *X[:,1] +np.exp(X[:,2])/5 -np.sqrt(X[:,3]+1)*np.log(X[:,4]+2)
    # mu0 = -0.5 + 0.25*X[:,0] *X[:,1]+np.sin(X[:,2])/3-np.cos(X[:,3]*np.log(X[:,4]+2))/2

    #Reward R
    R = np.exp(mu0+(X_quality*A).sum(axis=0)+epsilon) #log R=beta*X+gamma*X*A+epsilon
    return {
        'A': np.array(A.T, dtype='float32'),
        'X': np.array(X, dtype='float32'),
        'R': np.array(R, dtype='float32'),
        'g_X': np.array(g_X, dtype='float32'),
        'h_X': np.array(h_X, dtype='float32'),
        'mu0': np.array(mu0, dtype='float32'),
        'epsilon': np.array(epsilon, dtype='float32')
    }


























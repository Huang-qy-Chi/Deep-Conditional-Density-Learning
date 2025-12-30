import numpy as np
from log_hazard_rx_binary import grx_dnn_bin, grax_dnn_bin, grax1_dnn_bin
from sklearn.model_selection import train_test_split
import torch
import scipy.stats as stats

#%%----------------------------------Unconfoundedness Test--------------------------------------------
def confound_test_bin(train_data, val_data, test_data,\
                       t_nodes, t_fig, s_k, n_layer, n_node, n_lr, n_epoch,\
                          patiences,rho1=1,rho2=0,seed=42,\
                              show_val = True, n_points=100, alpha=0.05):
    ## Step 1: Data splitting
    X1, X2, R1, R2, A1, A2 = train_test_split(train_data['X'],\
                                              train_data['R'], train_data['A'], \
                                                test_size=0.5, random_state=seed) 
    train_data1 = {'X':X1, 'A':A1, 'R':R1}
    train_data2 = {'X':X2, 'A':A2, 'R':R2}
    
    Xv1, Xv2, Rv1, Rv2, Av1, Av2 = train_test_split(val_data['X'],\
                                              val_data['R'], val_data['A'], \
                                                test_size=0.5, random_state=seed) 
    val_data1 = {'X':Xv1, 'A':Av1, 'R':Rv1}
    val_data2 = {'X':Xv2, 'A':Av2, 'R':Rv2}
    ### sample size
    n = len(R1)

    ## Step 2: Estimate g(r,a,x) and g(r,x)
    ### train with A
    n_layer1 = n_layer[0]; n_node1 = n_node[0]; n_lr1 = n_lr[0]; n_epoch1 = n_epoch[0]; patiences1 = patiences[0]
    Srax = grax_dnn_bin(train_data1, val_data1, test_data, t_nodes,\
                         t_fig, s_k, n_layer1, n_node1, n_lr1, n_epoch1, patiences1,show_val=show_val)
    S_R_A_X = Srax['S_R_X']   ## S(r|a,x)
    lam_R_A_X = Srax['Lambda_R_X']   ##lambda(r|a,x)
    model_R_A_X = Srax['model']
    S_R_A_X_train = Srax['S_R_X_train']
    ### train without A
    n_layer2 = n_layer[1]; n_node2 = n_node[1]; n_lr2 = n_lr[1]; n_epoch2 = n_epoch[1]; patiences2 = patiences[1]
    Srx = grx_dnn_bin(train_data2, val_data2, test_data, t_nodes,\
                       t_fig, s_k, n_layer2, n_node2, n_lr2, n_epoch2, patiences2,show_val=show_val)
    S_R_X = Srx['S_R_X']    ## S(r|x)
    lam_R_X = Srx['Lambda_R_X']    ## lambda(r|x)
    model_R_X = Srx['model']
    S_R_X_train = Srx['S_R_X_train']

    ## Step 3:  Calculate the direction function h(r,a,x) and h(r,x)
    h_R_A_X = (S_R_A_X**rho1)*((1-S_R_A_X)**rho2)
    h_R_X = (S_R_X**rho1)*((1-S_R_X)**rho2)
    h_R_A_X_train = (S_R_A_X_train**rho1)*((1-S_R_A_X_train)**rho2)
    h_R_X_train = (S_R_X_train**rho1)*((1-S_R_X_train)**rho2)

    ## Step 4: Test statistic and variance
    ## test statistic
    Tn = np.sqrt(n)*(h_R_A_X*(np.log(lam_R_A_X)-np.log(lam_R_X))).mean()
    
    ## variance
    X_train1 = torch.tensor(train_data1['X'], dtype=torch.float32)
    A_train1 = torch.tensor(train_data1['A'], dtype=torch.float32)
    R_0_train1 = torch.tensor(train_data1['R'], dtype=torch.float32)
    X_train2 = torch.tensor(train_data2['X'], dtype=torch.float32)
    A_train2 = torch.tensor(train_data2['A'], dtype=torch.float32)
    R_0_train2 = torch.tensor(train_data2['R'], dtype=torch.float32)

    ### integration via Gaussian-Legendre method: use troch.tensor
    def batch_func_train_R_X(t):
        # t: [batch_size, n_points]
        # A_train = A
        batch_size, n_points = t.shape
        t = t.reshape(-1, 1)  # [batch_size * n_points, 1]
        X_repeated = X_train2.repeat_interleave(n_points, dim=0)  # [batch_size * n_points, d_X]
        inputs = torch.cat((t, X_repeated), dim=1)  # [batch_size * n_points, d_X + 1]
        outputs = model_R_X(inputs).squeeze()  # [batch_size * n_points]
        return torch.exp(outputs).reshape(batch_size, n_points)  # 恢复为 [batch_size, n_points]
    
    def batch_func_train_R_A_X(t,A):
        # t: [batch_size, n_points]
        A_train1 = A
        batch_size, n_points = t.shape
        t = t.reshape(-1, 1)  # [batch_size * n_points, 1]
        X_repeated = X_train1.repeat_interleave(n_points, dim=0)  # [batch_size * n_points, d_X]
        A_repeated = A_train1.repeat_interleave(n_points, dim=0)  # [batch_size * n_points, 2]
        inputs = torch.cat((t, X_repeated), dim=1)  # [batch_size * n_points, d_X + 1]
        outputs = model_R_A_X(inputs).squeeze()  # [batch_size * n_points]
        outputs1 = torch.sum(outputs*A_repeated, dim = 1)
        return torch.exp(outputs1).reshape(batch_size, n_points)  # 恢复为 [batch_size, n_points]

    nodes, weights = np.polynomial.legendre.leggauss(n_points)
    nodes = torch.tensor(nodes, dtype=torch.float32) 
    weights = torch.tensor(weights, dtype=torch.float32) 
    b = R_0_train2
    a = torch.zeros_like(R_0_train2)
    nodes = 0.5 * (nodes + 1) * (b - a).unsqueeze(1) + a.unsqueeze(1)
    ### the log-hazard results
    values_R_X = batch_func_train_R_X(nodes) # n*n_points
    
    nodes1, weights1 = np.polynomial.legendre.leggauss(n_points)
    nodes1 = torch.tensor(nodes1, dtype=torch.float32) 
    weights1 = torch.tensor(weights1, dtype=torch.float32) 
    b1 = R_0_train1
    a1 = torch.zeros_like(R_0_train1)
    nodes1 = 0.5 * (nodes1 + 1) * (b1 - a1).unsqueeze(1) + a1.unsqueeze(1)
    values_R_A_X = batch_func_train_R_A_X(nodes1,A=A_train1)  # n*n_points


    ### the survival-based direction: h_R_X_train, construct the integral int_0^R exp(g(r,X))h(r,X)dr 
    h_R_X_train = torch.tensor(h_R_X_train, dtype=torch.float32) 
    h_R_A_X_train = torch.tensor(h_R_A_X_train, dtype=torch.float32) 
    intgral_R_X_index = torch.exp(values_R_X)*h_R_X_train   # n*n_points
    int_expRX_hX = torch.sum(weights*intgral_R_X_index,dim=1)  # n*1
    intgral_R_A_X_index = torch.exp(values_R_A_X)*h_R_X_train   # n*n_points
    int_expRAX_hX = torch.sum(weights*intgral_R_A_X_index,dim=1)  # n*1

    ### score function and variance
    h_R_X = torch.tensor(h_R_X, dtype=torch.float32) 
    h_R_A_X = torch.tensor(h_R_A_X, dtype=torch.float32) 
    Phi_R_X_h = h_R_X - int_expRX_hX #n*1
    Phi_R_A_X_h = h_R_X - int_expRAX_hX #n*1
    sigmasq_R_X_h = torch.mean(Phi_R_X_h**2)
    sigmasq_R_A_X_h = torch.mean(Phi_R_A_X_h**2)
    sigmasq_R_X_h = sigmasq_R_X_h.detach().numpy()
    sigmasq_R_A_X_h = sigmasq_R_A_X_h.detach().numpy()
    sigmasq_Tn_h = 2*(sigmasq_R_X_h+sigmasq_R_X_h)
    sigma_Tn_h = np.sqrt(sigmasq_Tn_h)


    # Step 5: Decision Making and p-value calculation
    Un = np.abs(Tn/sigma_Tn_h)
    norm_dist = stats.norm(loc=0, scale=1)  #N(0,1)
    criteria = norm_dist.ppf(1 - alpha/2)
    p_value = 2 * norm_dist.sf(abs(Un))
    decision = (Un>criteria).astype(int)
    Phi_R_A_X_h = Phi_R_A_X_h.detach().numpy()
    Phi_R_X_h = Phi_R_X_h.detach().numpy()
    Phi_A_mean = Phi_R_A_X_h.mean()
    Phi_mean = Phi_R_X_h.mean()


    return{
        'statistic': Tn, 
        'sigma_Tn_h': sigma_Tn_h,
        'U_n': Un,
        'criteria': criteria,
        'p_value': p_value,
        'decision': decision, 
        'Phi_A_mean': Phi_A_mean, 
        'Phi_mean': Phi_mean
    }


#%%---------------------------------------------------------------------------------------------
def confound_test1_bin(train_data, val_data,test_data,\
                       t_nodes, t_fig, s_k, n_layer, n_node, n_lr, n_epoch,\
                          patiences,rho1=1,rho2=0,seed=42,\
                              show_val = True, n_points=100, alpha=0.05):
    X1, X2, R1, R2, A1, A2 = train_test_split(train_data['X'],\
                                                train_data['R'], train_data['A'], \
                                                    test_size=0.5, random_state=seed) 
    train_data1 = {'X':X1, 'A':A1, 'R':R1}
    train_data2 = {'X':X2, 'A':A2, 'R':R2}

    Xv1, Xv2, Rv1, Rv2, Av1, Av2 = train_test_split(val_data['X'],\
                                                val_data['R'], val_data['A'], \
                                                test_size=0.5, random_state=seed) 
    val_data1 = {'X':Xv1, 'A':Av1, 'R':Rv1}
    val_data2 = {'X':Xv2, 'A':Av2, 'R':Rv2}
    ### sample size
    # n = len(R1)
    ### train with A: g(r,a,x)
    n_layer1 = n_layer[0]; n_node1 = n_node[0]; n_lr1 = n_lr[0]; n_epoch1 = n_epoch[0]; patiences1 = patiences[0]
    Srax = grax1_dnn_bin(train_data1, val_data1, test_data, t_nodes,\
                            t_fig, s_k, n_layer1, n_node1, n_lr1, n_epoch1, patiences1,show_val=show_val)
    S_R_A_X = Srax['S_R_X']   ## S(r|a,x)
    lam_R_A_X = Srax['Lambda_R_X']   ##lambda(r|a,x)
    model_R_A_X = Srax['model']
    ### train without A: g(r,x)
    n_layer2 = n_layer[1]; n_node2 = n_node[1]; n_lr2 = n_lr[1]; n_epoch2 = n_epoch[1]; patiences2 = patiences[1]
    Srx = grx_dnn_bin(train_data2, val_data2, test_data, t_nodes,\
                        t_fig, s_k, n_layer2, n_node2, n_lr2, n_epoch2, patiences2,show_val=show_val)
    S_R_X = Srx['S_R_X']    ## S(r|x)
    lam_R_X = Srx['Lambda_R_X']    ## lambda(r|x)
    model_R_X = Srx['model']
    def W_n(r,R,X,model,n_points,rho1,rho2):
        """
        Function that calculate the weight W_n(R,X|rho1, rho2)
        Input: 
            data (R,X) of size n, location r of n-dim, n_points, rho1, rho2, log-hazard function g(r|x): model
        Output: 
            W_n(r,R,X|rho1, rho2), size n*1
        """
        # r  #size n, all elements are r
        input = torch.cat((r.unsqueeze(1), X), dim=1)
        # part 1: conditional log hazard g(r,X), size n
        output = model(input).squeeze()  
        # part 2: conditional hazard lambda(r|X), size n
        hazard = torch.exp(output) 
        # part 3: conditional survival S(r|X): integral, size n
        nodes, weights = np.polynomial.legendre.leggauss(n_points)
        nodes = torch.tensor(nodes, dtype=torch.float32) 
        weights = torch.tensor(weights, dtype=torch.float32) 
        b = r
        a = torch.zeros_like(r)
        nodes = 0.5 * (nodes + 1) * (b - a).unsqueeze(1) + a.unsqueeze(1)
        weights = weights * 0.5 * (b - a).unsqueeze(1)  # [batch_size, n_points], len(a) = len(b) = batch_size
        def batch_func_train_R_X(t,X,model):
            # t: [batch_size, n_points]
            # A_train = A
            batch_size, n_points = t.shape
            t = t.reshape(-1, 1)  # [batch_size * n_points, 1]
            X_repeated = X.repeat_interleave(n_points, dim=0)  # [batch_size * n_points, d_X]
            inputs = torch.cat((t, X_repeated), dim=1)  # [batch_size * n_points, d_X + 1]
            outputs = model(inputs).squeeze()  # [batch_size * n_points]
            return torch.exp(outputs).reshape(batch_size, n_points)  # 恢复为 [batch_size, n_points]
        values_r_X = batch_func_train_R_X(nodes,X,model=model)  #[batch_size, n_points], n*n_points
        intgral_r_X_index = torch.exp(values_r_X)  # n*n_points
        ## cumulative hazard function Lambda(r|X)
        Lambda_r_X = torch.sum(weights*intgral_r_X_index,dim=1)  # n*1: int_0^r lambda(s|X)ds, for every x_i
        ## survival function S(r|X)
        S_r_X = torch.exp(-Lambda_r_X)  # n*1, for every x_i
        
        # part 4: weight (1/n)*sum_{i=1}^n I{R_i>=r}
        def count_greater_equal(R, r):
            """
            calculate  M_{k} = sum_{i=1}^n I{R_i >= r_{k}}, output len(R)*1 vector (k=1,...,len(r))
            input:
                R: torch.tensor,shape [n]
                r: torch.tensor,shape [K, J],K=n, 
            output:
                M: torch.tensor,shape [n, 1], M[k] = sum_{i=1}^n I{R_i >= r_{k}}
            """
            n = R.shape[0]
            
            # sort R (small-large)
            R_sorted, _ = torch.sort(R)  # O(n log n)
            
            # use searchsorted to find the location r_{kj} inserted in R_sorted（>=）
            # side='left' , ensuring R_i >= r_{kj}
            indices = torch.searchsorted(R_sorted, r, right=False)  # O(n*q*log n)
            
            # number of >= r_{kj}= length - insert location
            M = n - indices
            
            return M.float()/n
            
        
        Rr_index = count_greater_equal(R, r)  #size n

        # part 5: Wn(r,x)=
        Wn_r = ((Rr_index))*((S_r_X)**rho1)*((1-S_r_X)**rho2)

        return Wn_r  # output n dim vector for different X_i 

    def W_n_vec(s,R,X,model,n_points,rho1,rho2):
        """
        Function to calculate the weight for the integral
        Input: s for n_points (e.g. n_points=100), R: reward, n*1, X:covariate, n*d
        Output: W_vec: weight function, n*n_point matrix for integral
        """
        W_vec = torch.ones(len(R),len(s))
        lens = len(s)
        for k in range(lens):
            r = s[k]*torch.ones_like(R)
            W_vec[:,k] = W_n(r,R,X,model = model,n_points=n_points,rho1=rho1,rho2=rho2)

        return W_vec    # n*n_points

    def lamSX(s,R,X,model):
        """
        Calculate the lambda(s_j|X_i)
        """
        lam_S_X = torch.ones(len(R),len(s))
        lens = len(s)
        for k in range(lens):
            r = s[k]*torch.ones_like(R)
            input = torch.cat((r.unsqueeze(1), X), dim=1)
            output = model(input).squeeze()  
            lam_S_X[:,k] = torch.exp(output) 

        return lam_S_X   # n*n_points

    def lamSAX(s,R,A,X,model):
        """
        Calculate the lambda(s_j|X_i)
        """
        lam_S_A_X = torch.ones(len(R),len(s))
        lens = len(s)
        for k in range(lens):
            r = s[k]*torch.ones_like(R)
            input = torch.cat((r.unsqueeze(1),A.unsqueeze(1), X), dim=1)
            output = model(input).squeeze()  #n*1
            # output1 = torch.sum(output*A,dim=1)
            lam_S_A_X[:,k] = torch.exp(output) 

        return lam_S_A_X   # n*n_points

    def indicator_greater_equal(R, S):
        R_expanded = R.unsqueeze(1)  # shape(n, 1)
        S_expanded = S.unsqueeze(0)  # shape (1, n_points)
        result = (R_expanded >= S_expanded).float()  #shape(n, n_points)
        return result    # n*n_points

    X_train = torch.tensor(train_data['X'], dtype=torch.float32)
    # A_train = torch.tensor(train_data['A'], dtype=torch.float32)
    A_train = (train_data['A'])[:,1]
    A_train =  torch.tensor(A_train, dtype=torch.float32)
    R_0_train = torch.tensor(train_data['R'], dtype=torch.float32)

    X_train1 = torch.tensor(train_data1['X'], dtype=torch.float32)
    # A_train = torch.tensor(train_data['A'], dtype=torch.float32)
    A_train1 = (train_data1['A'])[:,1]
    A_train1 =  torch.tensor(A_train1, dtype=torch.float32)
    R_0_train1 = torch.tensor(train_data1['R'], dtype=torch.float32)

    X_train2 = torch.tensor(train_data2['X'], dtype=torch.float32)
    # A_train = torch.tensor(train_data['A'], dtype=torch.float32)
    A_train2 = (train_data2['A'])[:,1]
    A_train2 =  torch.tensor(A_train2, dtype=torch.float32)
    R_0_train2 = torch.tensor(train_data2['R'], dtype=torch.float32)
    ### calculate the integral
    s_k = torch.tensor(s_k, dtype=torch.float32)
    s_max = torch.max(s_k)
    gap = s_max/(len(s_k)-1)
    # weg = W_n_vec(s=s_k,R=R_0_train,X=X_train,\
    #                 model = model_R_X,n_points=n_points,rho1=rho1,rho2=rho2)  #n*n_points
    weg1 = W_n_vec(s=s_k,R=R_0_train1,X=X_train1,\
                    model = model_R_X,n_points=n_points,rho1=rho1,rho2=rho2)  #n/2*n_points
    weg2 = W_n_vec(s=s_k,R=R_0_train2,X=X_train2,\
                    model = model_R_X,n_points=n_points,rho1=rho1,rho2=rho2)  #n/2*n_points
    lam_S_X = lamSX(s=s_k,R=R_0_train2,X=X_train2,model = model_R_X)   #n/2*n_points
    lam_S_A_X = lamSAX(s=s_k,R=R_0_train1,A=A_train1,X=X_train1,model = model_R_A_X) #n/2*npoints
    # ind_RS = indicator_greater_equal(R_0_train,s_k)   #n*n_points
    ind_RS1 = indicator_greater_equal(R_0_train1,s_k)   #n/2*n_points
    ind_RS2 = indicator_greater_equal(R_0_train2,s_k)   #n/2*n_points
    gap = gap*torch.ones_like(ind_RS1)
    int_egrx_W = torch.sum(gap*ind_RS2*lam_S_X*weg2,dim=1)  #n/2*1
    int_egrax_W = torch.sum(gap*ind_RS1*lam_S_A_X*weg1,dim=1)  #n/2*1

    ### calculate W_n(R_i,X_i)
    W_R_X = W_n(r=R_0_train,R=R_0_train,X=X_train,model=model_R_X,n_points=n_points,rho1=rho1,rho2=rho2)
    W_R_X1 = W_n(r=R_0_train1,R=R_0_train1,X=X_train1,model=model_R_X,n_points=n_points,rho1=rho1,rho2=rho2)
    W_R_X2 = W_n(r=R_0_train2,R=R_0_train2,X=X_train2,model=model_R_X,n_points=n_points,rho1=rho1,rho2=rho2)

    ### calculate the variance
    Phi_R_X = W_R_X2 - int_egrx_W
    Phi_R_A_X = W_R_X1 - int_egrax_W
    sigmasq_R_X_h = torch.mean(Phi_R_X**2)
    sigmasq_R_A_X_h = torch.mean(Phi_R_A_X**2)
    sigmasq_R_X_h = sigmasq_R_X_h.detach().numpy()
    sigmasq_R_A_X_h = sigmasq_R_A_X_h.detach().numpy()
    sigmasq_Tn_h = 2*(sigmasq_R_X_h+sigmasq_R_A_X_h)
    sigma_Tn_h = np.sqrt(sigmasq_Tn_h)

    ### calculate the statistics Tn
    g_RAX = model_R_A_X(torch.cat((R_0_train.unsqueeze(1), A_train.unsqueeze(1), X_train), dim=1)).squeeze()
    g_RX = model_R_X(torch.cat((R_0_train.unsqueeze(1), X_train), dim=1)).squeeze()
    g_RAX = g_RAX.detach().numpy()
    g_RX = g_RX.detach().numpy()
    W_R_X = W_R_X.detach().numpy()
    n_total = len(R_0_train)
    Tn = np.sqrt(n_total)*np.mean((W_R_X)*(g_RAX-g_RX))
    
    ### decision making
    Un = np.abs(Tn/sigma_Tn_h)
    norm_dist = stats.norm(loc=0, scale=1)  #N(0,1)
    criteria = norm_dist.ppf(1 - alpha/2)
    p_value = 2 * norm_dist.sf(abs(Un))
    decision = (Un>criteria).astype(int)
    Phi_R_A_X = Phi_R_A_X.detach().numpy()
    Phi_R_X = Phi_R_X.detach().numpy()
    Phi_A_mean = Phi_R_A_X.mean()
    Phi_mean = Phi_R_X.mean()

    return{
        'statistic': Tn, 
        'sigma_Tn_h': sigma_Tn_h,
        'U_n': Un,
        'criteria': criteria,
        'p_value': p_value,
        'decision': decision, 
        'Phi_A_mean': Phi_A_mean, 
        'Phi_mean': Phi_mean
    }




#%%---------------------------------------------------------------------------------------------
def hetero_test1_bin(train_data, val_data,test_data,\
                       t_nodes, t_fig, s_k, n_layer, n_node, n_lr, n_epoch,\
                          patiences,rho1=0,rho2=0,seed=42,\
                              show_val = True, n_points=100, alpha=0.05):
    # X1, X2, R1, R2, A1, A2 = train_test_split(train_data['X'],\
    #                                             train_data['R'], train_data['A'], \
    #                                                 test_size=0.5, random_state=seed) 
    
    # train_data1 = {'X':X1, 'A':A1, 'R':R1}
    # train_data2 = {'X':X2, 'A':A2, 'R':R2}
    # Xv1, Xv2, Rv1, Rv2, Av1, Av2 = train_test_split(val_data['X'],\
    #                                             val_data['R'], val_data['A'], \
    #                                             test_size=0.5, random_state=seed) 
    
    # val_data1 = {'X':Xv1, 'A':Av1, 'R':Rv1}
    # val_data2 = {'X':Xv2, 'A':Av2, 'R':Rv2}

    # Data splittig: two groups of A=1 and A=0
    # df = pd.DataFrame(train_data)

    # split to A=1 anf A=0
    indices_A1 = [i for i, a in enumerate((train_data['A'])[:,1]) if a == 1]
    indices_A0 = [i for i, a in enumerate((train_data['A'])[:,1]) if a == 0]
    R1 = [train_data ['R'][i] for i in indices_A1]
    A1 = [train_data ['A'][i] for i in indices_A1]
    X1 = [train_data ['X'][i] for i in indices_A1]
    R2 = [train_data ['R'][i] for i in indices_A0]
    A2 = [train_data ['A'][i] for i in indices_A0]
    X2 = [train_data ['X'][i] for i in indices_A0]

    R1 = np.array(R1) 
    A1 = np.array([list(t) for t in A1]) 
    X1 = np.array([list(t) for t in X1]) 
    R2 = np.array(R2) 
    A2 = np.array([list(t) for t in A2]) 
    X2 = np.array([list(t) for t in X2]) 
    train_data1 = {'X':X1, 'A':A1, 'R':R1}
    train_data2 = {'X':X2, 'A':A2, 'R':R2}

    indices_A1v = [i for i, a in enumerate((val_data['A'])[:,1]) if a == 1]
    indices_A0v = [i for i, a in enumerate((val_data['A'])[:,1]) if a == 0]
    Rv1 = [val_data ['R'][i] for i in indices_A1v]
    Av1 = [val_data ['A'][i] for i in indices_A1v]
    Xv1 = [val_data ['X'][i] for i in indices_A1v]
    Rv2 = [val_data ['R'][i] for i in indices_A0v]
    Av2 = [val_data ['A'][i] for i in indices_A0v]
    Xv2 = [val_data ['X'][i] for i in indices_A0v]
    Rv1 = np.array(Rv1) 
    Av1 = np.array([list(t) for t in Av1]) 
    Xv1 = np.array([list(t) for t in Xv1]) 
    Rv2 = np.array(Rv2) 
    Av2 = np.array([list(t) for t in Av2]) 
    Xv2 = np.array([list(t) for t in Xv2]) 
    val_data1 = {'X':Xv1, 'A':Av1, 'R':Rv1}
    val_data2 = {'X':Xv2, 'A':Av2, 'R':Rv2}
    ### sample size
    # n = len(R1)
    ### train with A: g(r,a,x)
    n_layer1 = n_layer[0]; n_node1 = n_node[0]; n_lr1 = n_lr[0]; n_epoch1 = n_epoch[0]; patiences1 = patiences[0]
    Srax = grax1_dnn_bin(train_data1, val_data1, test_data, t_nodes,\
                            t_fig, s_k, n_layer1, n_node1, n_lr1, n_epoch1, patiences1,show_val=show_val)
    S_R_A_X = Srax['S_R_X']   ## S(r|a,x)
    lam_R_A_X = Srax['Lambda_R_X']   ##lambda(r|a,x)
    model_R_A_X = Srax['model']
    ### train without A: g(r,x)
    n_layer2 = n_layer[1]; n_node2 = n_node[1]; n_lr2 = n_lr[1]; n_epoch2 = n_epoch[1]; patiences2 = patiences[1]
    Srx = grx_dnn_bin(train_data2, val_data2, test_data, t_nodes,\
                        t_fig, s_k, n_layer2, n_node2, n_lr2, n_epoch2, patiences2,show_val=show_val)
    S_R_X = Srx['S_R_X']    ## S(r|x)
    lam_R_X = Srx['Lambda_R_X']    ## lambda(r|x)
    model_R_X = Srx['model']
    def W_n(r,R,X,model,n_points,rho1,rho2):
        """
        Function that calculate the weight W_n(R,X|rho1, rho2)
        Input: 
            data (R,X) of size n, location r of n-dim, n_points, rho1, rho2, log-hazard function g(r|x): model
        Output: 
            W_n(r,R,X|rho1, rho2), size n*1
        """
        # r  #size n, all elements are r
        input = torch.cat((r.unsqueeze(1), X), dim=1)
        # part 1: conditional log hazard g(r,X), size n
        output = model(input).squeeze()  
        # part 2: conditional hazard lambda(r|X), size n
        hazard = torch.exp(output) 
        # part 3: conditional survival S(r|X): integral, size n
        nodes, weights = np.polynomial.legendre.leggauss(n_points)
        nodes = torch.tensor(nodes, dtype=torch.float32) 
        weights = torch.tensor(weights, dtype=torch.float32) 
        b = r
        a = torch.zeros_like(r)
        nodes = 0.5 * (nodes + 1) * (b - a).unsqueeze(1) + a.unsqueeze(1)
        weights = weights * 0.5 * (b - a).unsqueeze(1)  # [batch_size, n_points], len(a) = len(b) = batch_size
        def batch_func_train_R_X(t,X,model):
            # t: [batch_size, n_points]
            # A_train = A
            batch_size, n_points = t.shape
            t = t.reshape(-1, 1)  # [batch_size * n_points, 1]
            X_repeated = X.repeat_interleave(n_points, dim=0)  # [batch_size * n_points, d_X]
            inputs = torch.cat((t, X_repeated), dim=1)  # [batch_size * n_points, d_X + 1]
            outputs = model(inputs).squeeze()  # [batch_size * n_points]
            return torch.exp(outputs).reshape(batch_size, n_points)  # 恢复为 [batch_size, n_points]
        values_r_X = batch_func_train_R_X(nodes,X,model=model)  #[batch_size, n_points], n*n_points
        intgral_r_X_index = torch.exp(values_r_X)  # n*n_points
        ## cumulative hazard function Lambda(r|X)
        Lambda_r_X = torch.sum(weights*intgral_r_X_index,dim=1)  # n*1: int_0^r lambda(s|X)ds, for every x_i
        ## survival function S(r|X)
        S_r_X = torch.exp(-Lambda_r_X)  # n*1, for every x_i
        
        # part 4: weight (1/n)*sum_{i=1}^n I{R_i>=r}
        def count_greater_equal(R, r):
            """
            calculate  M_{k} = sum_{i=1}^n I{R_i >= r_{k}}, output len(R)*1 vector (k=1,...,len(r))
            input:
                R: torch.tensor,shape [n]
                r: torch.tensor,shape [K, J],K=n, 
            output:
                M: torch.tensor,shape [n, 1], M[k] = sum_{i=1}^n I{R_i >= r_{k}}
            """
            n = R.shape[0]
            
            # sort R (small-large)
            R_sorted, _ = torch.sort(R)  # O(n log n)
            
            # use searchsorted to find the location r_{kj} inserted in R_sorted（>=）
            # side='left' , ensuring R_i >= r_{kj}
            indices = torch.searchsorted(R_sorted, r, right=False)  # O(n*q*log n)
            
            # number of >= r_{kj}= length - insert location
            M = n - indices
            
            return M.float()/n
            
        
        Rr_index = count_greater_equal(R, r)  #size n

        # part 5: Wn(r,x)=
        Wn_r = ((Rr_index))*((S_r_X)**rho1)*((1-S_r_X)**rho2)

        return Wn_r  # output n dim vector for different X_i 

    def W_n_vec(s,R,X,model,n_points,rho1,rho2):
        """
        Function to calculate the weight for the integral
        Input: s for n_points (e.g. n_points=100), R: reward, n*1, X:covariate, n*d
        Output: W_vec: weight function, n*n_point matrix for integral
        """
        W_vec = torch.ones(len(R),len(s))
        lens = len(s)
        for k in range(lens):
            r = s[k]*torch.ones_like(R)
            W_vec[:,k] = W_n(r,R,X,model = model,n_points=n_points,rho1=rho1,rho2=rho2)

        return W_vec    # n*n_points

    def lamSX(s,R,X,model):
        """
        Calculate the lambda(s_j|X_i)
        """
        lam_S_X = torch.ones(len(R),len(s))
        lens = len(s)
        for k in range(lens):
            r = s[k]*torch.ones_like(R)
            input = torch.cat((r.unsqueeze(1), X), dim=1)
            output = model(input).squeeze()  
            lam_S_X[:,k] = torch.exp(output) 

        return lam_S_X   # n*n_points

    def lamSAX(s,R,A,X,model):
        """
        Calculate the lambda(s_j|X_i)
        """
        lam_S_A_X = torch.ones(len(R),len(s))
        lens = len(s)
        for k in range(lens):
            r = s[k]*torch.ones_like(R)
            input = torch.cat((r.unsqueeze(1),A.unsqueeze(1), X), dim=1)
            output = model(input).squeeze()  #n*1
            # output1 = torch.sum(output*A,dim=1)
            lam_S_A_X[:,k] = torch.exp(output) 

        return lam_S_A_X   # n*n_points

    def indicator_greater_equal(R, S):
        R_expanded = R.unsqueeze(1)  # shape(n, 1)
        S_expanded = S.unsqueeze(0)  # shape (1, n_points)
        result = (R_expanded >= S_expanded).float()  #shape(n, n_points)
        return result    # n*n_points

    X_train = torch.tensor(train_data['X'], dtype=torch.float32)
    # A_train = torch.tensor(train_data['A'], dtype=torch.float32)
    A_train = (train_data['A'])[:,1]
    A_train =  torch.tensor(A_train, dtype=torch.float32)
    R_0_train = torch.tensor(train_data['R'], dtype=torch.float32)

    X_train1 = torch.tensor(train_data1['X'], dtype=torch.float32)
    # A_train = torch.tensor(train_data['A'], dtype=torch.float32)
    A_train1 = (train_data1['A'])[:,1]
    A_train1 =  torch.tensor(A_train1, dtype=torch.float32)
    R_0_train1 = torch.tensor(train_data1['R'], dtype=torch.float32)

    X_train2 = torch.tensor(train_data2['X'], dtype=torch.float32)
    # A_train = torch.tensor(train_data['A'], dtype=torch.float32)
    A_train2 = (train_data2['A'])[:,1]
    A_train2 =  torch.tensor(A_train2, dtype=torch.float32)
    R_0_train2 = torch.tensor(train_data2['R'], dtype=torch.float32)

    ### calculate the integral
    s_k = torch.tensor(s_k, dtype=torch.float32)
    s_max = torch.max(s_k)
    gap = s_max/(len(s_k)-1)
    # weg = W_n_vec(s=s_k,R=R_0_train,X=X_train,\
                    # model = model_R_X,n_points=n_points,rho1=rho1,rho2=rho2)  #n*n_points
    weg1 = W_n_vec(s=s_k,R=R_0_train1,X=X_train1,\
                model = model_R_X,n_points=n_points,rho1=rho1,rho2=rho2)  #n/2*n_points
    weg2 = W_n_vec(s=s_k,R=R_0_train2,X=X_train2,\
                model = model_R_X,n_points=n_points,rho1=rho1,rho2=rho2)  #n/2*n_points
    lam_S_X = lamSX(s=s_k,R=R_0_train2,X=X_train2,model = model_R_X)   #n*n_points
    lam_S_A_X = lamSAX(s=s_k,R=R_0_train1,A=A_train1,X=X_train1,model = model_R_A_X) #n*npoints
    # ind_RS = indicator_greater_equal(R_0_train,s_k)   #n*n_points
    ind_RS1 = indicator_greater_equal(R_0_train1,s_k)   #n/2*n_points
    ind_RS2 = indicator_greater_equal(R_0_train2,s_k)   #n/2*n_points
    gap1 = gap*torch.ones_like(ind_RS1)
    gap2 = gap*torch.ones_like(ind_RS2)
    int_egrx_W = torch.sum(gap2*ind_RS2*lam_S_X*weg2,dim=1)  #n*1
    int_egrax_W = torch.sum(gap1*ind_RS1*lam_S_A_X*weg1,dim=1)  #n*1

    ### calculate W_n(R_i,X_i)
    W_R_X = W_n(r=R_0_train,R=R_0_train,X=X_train,model=model_R_X,n_points=n_points,rho1=rho1,rho2=rho2)
    W_R_X1 = W_n(r=R_0_train1,R=R_0_train1,X=X_train1,model=model_R_X,n_points=n_points,rho1=rho1,rho2=rho2)
    W_R_X2 = W_n(r=R_0_train2,R=R_0_train2,X=X_train2,model=model_R_X,n_points=n_points,rho1=rho1,rho2=rho2)
    Phi_R_X = W_R_X2 - int_egrx_W
    Phi_R_A_X = W_R_X1 - int_egrax_W

    sigmasq_R_X_h = torch.mean(Phi_R_X**2)
    sigmasq_R_A_X_h = torch.mean(Phi_R_A_X**2)
    sigmasq_R_X_h = sigmasq_R_X_h.detach().numpy()
    sigmasq_R_A_X_h = sigmasq_R_A_X_h.detach().numpy()
    sigmasq_Tn_h = 2*(sigmasq_R_X_h+sigmasq_R_A_X_h)
    # sigmasq_Tn_h = 4*(sigmasq_R_X_h)
    sigma_Tn_h = np.sqrt(sigmasq_Tn_h)

    g_RAX = model_R_A_X(torch.cat((R_0_train.unsqueeze(1), A_train.unsqueeze(1), X_train), dim=1)).squeeze()
    g_RX = model_R_X(torch.cat((R_0_train.unsqueeze(1), X_train), dim=1)).squeeze()
    g_RAX = g_RAX.detach().numpy()
    g_RX = g_RX.detach().numpy()
    W_R_X = W_R_X.detach().numpy()
    n_total = len(R_0_train)
    Tn = np.sqrt(n_total)*np.mean((W_R_X)*(g_RAX-g_RX))

    Un = np.abs(Tn/sigma_Tn_h)
    norm_dist = stats.norm(loc=0, scale=1)  #N(0,1)
    criteria = norm_dist.ppf(1 - alpha/2)
    p_value = 2 * norm_dist.sf(abs(Un))
    decision = (Un>criteria).astype(int)
    Phi_R_A_X = Phi_R_A_X.detach().numpy()
    Phi_R_X = Phi_R_X.detach().numpy()
    Phi_A_mean = Phi_R_A_X.mean()
    Phi_mean = Phi_R_X.mean()

    return{
        'statistic': Tn, 
        'sigma_Tn_h': sigma_Tn_h,
        'U_n': Un,
        'criteria': criteria,
        'p_value': p_value,
        'decision': decision, 
        'Phi_A_mean': Phi_A_mean, 
        'Phi_mean': Phi_mean
    }
















#%%---------------------------------------------------------------------------------------------
def hetero_test_bin(train_data, val_data,test_data,\
                       t_nodes, t_fig, s_k, n_layer, n_node, n_lr, n_epoch,\
                          patiences,rho1=0,rho2=0,seed=42,\
                              show_val = True, n_points=100, alpha=0.05):
    # X1, X2, R1, R2, A1, A2 = train_test_split(train_data['X'],\
    #                                             train_data['R'], train_data['A'], \
    #                                                 test_size=0.5, random_state=seed) 
    
    # train_data1 = {'X':X1, 'A':A1, 'R':R1}
    # train_data2 = {'X':X2, 'A':A2, 'R':R2}
    # Xv1, Xv2, Rv1, Rv2, Av1, Av2 = train_test_split(val_data['X'],\
    #                                             val_data['R'], val_data['A'], \
    #                                             test_size=0.5, random_state=seed) 
    
    # val_data1 = {'X':Xv1, 'A':Av1, 'R':Rv1}
    # val_data2 = {'X':Xv2, 'A':Av2, 'R':Rv2}

    # Data splittig: two groups of A=1 and A=0
    # df = pd.DataFrame(train_data)

    # split to A=1 anf A=0
    indices_A1 = [i for i, a in enumerate((train_data['A'])[:,1]) if a == 1]
    indices_A0 = [i for i, a in enumerate((train_data['A'])[:,1]) if a == 0]
    R1 = [train_data ['R'][i] for i in indices_A1]
    A1 = [train_data ['A'][i] for i in indices_A1]
    X1 = [train_data ['X'][i] for i in indices_A1]
    R2 = [train_data ['R'][i] for i in indices_A0]
    A2 = [train_data ['A'][i] for i in indices_A0]
    X2 = [train_data ['X'][i] for i in indices_A0]

    R1 = np.array(R1) 
    A1 = np.array([list(t) for t in A1]) 
    X1 = np.array([list(t) for t in X1]) 
    R2 = np.array(R2) 
    A2 = np.array([list(t) for t in A2]) 
    X2 = np.array([list(t) for t in X2]) 
    train_data1 = {'X':X1, 'A':A1, 'R':R1}
    train_data2 = {'X':X2, 'A':A2, 'R':R2}

    indices_A1v = [i for i, a in enumerate((val_data['A'])[:,1]) if a == 1]
    indices_A0v = [i for i, a in enumerate((val_data['A'])[:,1]) if a == 0]
    Rv1 = [val_data ['R'][i] for i in indices_A1v]
    Av1 = [val_data ['A'][i] for i in indices_A1v]
    Xv1 = [val_data ['X'][i] for i in indices_A1v]
    Rv2 = [val_data ['R'][i] for i in indices_A0v]
    Av2 = [val_data ['A'][i] for i in indices_A0v]
    Xv2 = [val_data ['X'][i] for i in indices_A0v]
    Rv1 = np.array(Rv1) 
    Av1 = np.array([list(t) for t in Av1]) 
    Xv1 = np.array([list(t) for t in Xv1]) 
    Rv2 = np.array(Rv2) 
    Av2 = np.array([list(t) for t in Av2]) 
    Xv2 = np.array([list(t) for t in Xv2]) 
    val_data1 = {'X':Xv1, 'A':Av1, 'R':Rv1}
    val_data2 = {'X':Xv2, 'A':Av2, 'R':Rv2}
    ### sample size
    # n = len(R1)
    ### train with A: g(r,a,x)
    n_layer1 = n_layer[0]; n_node1 = n_node[0]; n_lr1 = n_lr[0]; n_epoch1 = n_epoch[0]; patiences1 = patiences[0]
    Srax = grx_dnn_bin(train_data1, val_data1, test_data, t_nodes,\
                            t_fig, s_k, n_layer1, n_node1, n_lr1, n_epoch1, patiences1,show_val=show_val)
    S_R_A_X = Srax['S_R_X']   ## S(r|a,x)
    lam_R_A_X = Srax['Lambda_R_X']   ##lambda(r|a,x)
    model_R_A_X = Srax['model']
    ### train without A: g(r,x)
    n_layer2 = n_layer[1]; n_node2 = n_node[1]; n_lr2 = n_lr[1]; n_epoch2 = n_epoch[1]; patiences2 = patiences[1]
    Srx = grx_dnn_bin(train_data2, val_data2, test_data, t_nodes,\
                        t_fig, s_k, n_layer2, n_node2, n_lr2, n_epoch2, patiences2,show_val=show_val)
    S_R_X = Srx['S_R_X']    ## S(r|x)
    lam_R_X = Srx['Lambda_R_X']    ## lambda(r|x)
    model_R_X = Srx['model']
    def W_n(r,R,X,model,n_points,rho1,rho2):
        """
        Function that calculate the weight W_n(R,X|rho1, rho2)
        Input: 
            data (R,X) of size n, location r of n-dim, n_points, rho1, rho2, log-hazard function g(r|x): model
        Output: 
            W_n(r,R,X|rho1, rho2), size n*1
        """
        # r  #size n, all elements are r
        input = torch.cat((r.unsqueeze(1), X), dim=1)
        # part 1: conditional log hazard g(r,X), size n
        output = model(input).squeeze()  
        # part 2: conditional hazard lambda(r|X), size n
        hazard = torch.exp(output) 
        # part 3: conditional survival S(r|X): integral, size n
        nodes, weights = np.polynomial.legendre.leggauss(n_points)
        nodes = torch.tensor(nodes, dtype=torch.float32) 
        weights = torch.tensor(weights, dtype=torch.float32) 
        b = r
        a = torch.zeros_like(r)
        nodes = 0.5 * (nodes + 1) * (b - a).unsqueeze(1) + a.unsqueeze(1)
        weights = weights * 0.5 * (b - a).unsqueeze(1)  # [batch_size, n_points], len(a) = len(b) = batch_size
        def batch_func_train_R_X(t,X,model):
            # t: [batch_size, n_points]
            # A_train = A
            batch_size, n_points = t.shape
            t = t.reshape(-1, 1)  # [batch_size * n_points, 1]
            X_repeated = X.repeat_interleave(n_points, dim=0)  # [batch_size * n_points, d_X]
            inputs = torch.cat((t, X_repeated), dim=1)  # [batch_size * n_points, d_X + 1]
            outputs = model(inputs).squeeze()  # [batch_size * n_points]
            return torch.exp(outputs).reshape(batch_size, n_points)  # 恢复为 [batch_size, n_points]
        values_r_X = batch_func_train_R_X(nodes,X,model=model)  #[batch_size, n_points], n*n_points
        intgral_r_X_index = torch.exp(values_r_X)  # n*n_points
        ## cumulative hazard function Lambda(r|X)
        Lambda_r_X = torch.sum(weights*intgral_r_X_index,dim=1)  # n*1: int_0^r lambda(s|X)ds, for every x_i
        ## survival function S(r|X)
        S_r_X = torch.exp(-Lambda_r_X)  # n*1, for every x_i
        
        # part 4: weight (1/n)*sum_{i=1}^n I{R_i>=r}
        def count_greater_equal(R, r):
            """
            calculate  M_{k} = sum_{i=1}^n I{R_i >= r_{k}}, output len(R)*1 vector (k=1,...,len(r))
            input:
                R: torch.tensor,shape [n]
                r: torch.tensor,shape [K, J],K=n, 
            output:
                M: torch.tensor,shape [n, 1], M[k] = sum_{i=1}^n I{R_i >= r_{k}}
            """
            n = R.shape[0]
            
            # sort R (small-large)
            R_sorted, _ = torch.sort(R)  # O(n log n)
            
            # use searchsorted to find the location r_{kj} inserted in R_sorted（>=）
            # side='left' , ensuring R_i >= r_{kj}
            indices = torch.searchsorted(R_sorted, r, right=False)  # O(n*q*log n)
            
            # number of >= r_{kj}= length - insert location
            M = n - indices
            
            return M.float()/n
            
        
        Rr_index = count_greater_equal(R, r)  #size n

        # part 5: Wn(r,x)=
        Wn_r = ((Rr_index))*((S_r_X)**rho1)*((1-S_r_X)**rho2)

        return Wn_r  # output n dim vector for different X_i 

    def W_n_vec(s,R,X,model,n_points,rho1,rho2):
        """
        Function to calculate the weight for the integral
        Input: s for n_points (e.g. n_points=100), R: reward, n*1, X:covariate, n*d
        Output: W_vec: weight function, n*n_point matrix for integral
        """
        W_vec = torch.ones(len(R),len(s))
        lens = len(s)
        for k in range(lens):
            r = s[k]*torch.ones_like(R)
            W_vec[:,k] = W_n(r,R,X,model = model,n_points=n_points,rho1=rho1,rho2=rho2)

        return W_vec    # n*n_points

    def lamSX(s,R,X,model):
        """
        Calculate the lambda(s_j|X_i)
        """
        lam_S_X = torch.ones(len(R),len(s))
        lens = len(s)
        for k in range(lens):
            r = s[k]*torch.ones_like(R)
            input = torch.cat((r.unsqueeze(1), X), dim=1)
            output = model(input).squeeze()  
            lam_S_X[:,k] = torch.exp(output) 

        return lam_S_X   # n*n_points

    def lamSAX(s,R,A,X,model):
        """
        Calculate the lambda(s_j|X_i)
        """
        lam_S_A_X = torch.ones(len(R),len(s))
        lens = len(s)
        for k in range(lens):
            r = s[k]*torch.ones_like(R)
            input = torch.cat((r.unsqueeze(1),A.unsqueeze(1), X), dim=1)
            output = model(input).squeeze()  #n*1
            # output1 = torch.sum(output*A,dim=1)
            lam_S_A_X[:,k] = torch.exp(output) 

        return lam_S_A_X   # n*n_points

    def indicator_greater_equal(R, S):
        R_expanded = R.unsqueeze(1)  # shape(n, 1)
        S_expanded = S.unsqueeze(0)  # shape (1, n_points)
        result = (R_expanded >= S_expanded).float()  #shape(n, n_points)
        return result    # n*n_points

    X_train = torch.tensor(train_data['X'], dtype=torch.float32)
    # A_train = torch.tensor(train_data['A'], dtype=torch.float32)
    A_train = (train_data['A'])[:,1]
    A_train =  torch.tensor(A_train, dtype=torch.float32)
    R_0_train = torch.tensor(train_data['R'], dtype=torch.float32)

    X_train1 = torch.tensor(train_data1['X'], dtype=torch.float32)
    # A_train = torch.tensor(train_data['A'], dtype=torch.float32)
    A_train1 = (train_data1['A'])[:,1]
    A_train1 =  torch.tensor(A_train1, dtype=torch.float32)
    R_0_train1 = torch.tensor(train_data1['R'], dtype=torch.float32)

    X_train2 = torch.tensor(train_data2['X'], dtype=torch.float32)
    # A_train = torch.tensor(train_data['A'], dtype=torch.float32)
    A_train2 = (train_data2['A'])[:,1]
    A_train2 =  torch.tensor(A_train2, dtype=torch.float32)
    R_0_train2 = torch.tensor(train_data2['R'], dtype=torch.float32)

    ### calculate the integral
    s_k = torch.tensor(s_k, dtype=torch.float32)
    s_max = torch.max(s_k)
    gap = s_max/(len(s_k)-1)
    # weg = W_n_vec(s=s_k,R=R_0_train,X=X_train,\
                    # model = model_R_X,n_points=n_points,rho1=rho1,rho2=rho2)  #n*n_points
    weg1 = W_n_vec(s=s_k,R=R_0_train1,X=X_train1,\
                model = model_R_X,n_points=n_points,rho1=rho1,rho2=rho2)  #n/2*n_points
    weg2 = W_n_vec(s=s_k,R=R_0_train2,X=X_train2,\
                model = model_R_X,n_points=n_points,rho1=rho1,rho2=rho2)  #n/2*n_points
    lam_S_X = lamSX(s=s_k,R=R_0_train2,X=X_train2,model = model_R_X)   #n*n_points
    lam_S_A_X = lamSX(s=s_k,R=R_0_train1,X=X_train1,model = model_R_A_X) #n*npoints
    # ind_RS = indicator_greater_equal(R_0_train,s_k)   #n*n_points
    ind_RS1 = indicator_greater_equal(R_0_train1,s_k)   #n/2*n_points
    ind_RS2 = indicator_greater_equal(R_0_train2,s_k)   #n/2*n_points
    gap1 = gap*torch.ones_like(ind_RS1)
    gap2 = gap*torch.ones_like(ind_RS2)
    int_egrx_W = torch.sum(gap2*ind_RS2*lam_S_X*weg2,dim=1)  #n*1
    int_egrax_W = torch.sum(gap1*ind_RS1*lam_S_A_X*weg1,dim=1)  #n*1

    ### calculate W_n(R_i,X_i)
    W_R_X = W_n(r=R_0_train,R=R_0_train,X=X_train,model=model_R_X,n_points=n_points,rho1=rho1,rho2=rho2)
    W_R_X1 = W_n(r=R_0_train1,R=R_0_train1,X=X_train1,model=model_R_X,n_points=n_points,rho1=rho1,rho2=rho2)
    W_R_X2 = W_n(r=R_0_train2,R=R_0_train2,X=X_train2,model=model_R_X,n_points=n_points,rho1=rho1,rho2=rho2)
    Phi_R_X = W_R_X2 - int_egrx_W
    Phi_R_A_X = W_R_X1 - int_egrax_W

    sigmasq_R_X_h = torch.mean(Phi_R_X**2)
    sigmasq_R_A_X_h = torch.mean(Phi_R_A_X**2)
    sigmasq_R_X_h = sigmasq_R_X_h.detach().numpy()
    sigmasq_R_A_X_h = sigmasq_R_A_X_h.detach().numpy()
    sigmasq_Tn_h = 2*(sigmasq_R_X_h+sigmasq_R_A_X_h)
    # sigmasq_Tn_h = 4*(sigmasq_R_X_h)
    sigma_Tn_h = np.sqrt(sigmasq_Tn_h)

    g_RAX = model_R_A_X(torch.cat((R_0_train.unsqueeze(1), X_train), dim=1)).squeeze()
    g_RX = model_R_X(torch.cat((R_0_train.unsqueeze(1), X_train), dim=1)).squeeze()
    g_RAX = g_RAX.detach().numpy()
    g_RX = g_RX.detach().numpy()
    W_R_X = W_R_X.detach().numpy()
    n_total = len(R_0_train)
    Tn = np.sqrt(n_total)*np.mean((W_R_X)*(g_RAX-g_RX))

    Un = np.abs(Tn/sigma_Tn_h)
    norm_dist = stats.norm(loc=0, scale=1)  #N(0,1)
    criteria = norm_dist.ppf(1 - alpha/2)
    p_value = 2 * norm_dist.sf(abs(Un))
    decision = (Un>criteria).astype(int)
    Phi_R_A_X = Phi_R_A_X.detach().numpy()
    Phi_R_X = Phi_R_X.detach().numpy()
    Phi_A_mean = Phi_R_A_X.mean()
    Phi_mean = Phi_R_X.mean()

    return{
        'statistic': Tn, 
        'sigma_Tn_h': sigma_Tn_h,
        'U_n': Un,
        'criteria': criteria,
        'p_value': p_value,
        'decision': decision, 
        'Phi_A_mean': Phi_A_mean, 
        'Phi_mean': Phi_mean
    }
































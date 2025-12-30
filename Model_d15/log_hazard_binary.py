import torch
import numpy as np

#%%----------Input:(R,A,X)-- Output:g(R,A,X) --------------
def g_dnn_bin(train_data, val_data, test_data, t_nodes, t_fig, s_k, n_layer, n_node, n_lr, n_epoch, patiences,show_val=True):
    #Assume g(R,A,X)=Ag_1(R,1,X)+(1-A)g_0(R,0,X)
    if show_val == True:
        print('DNN_iteration')
    # Convert training and test data to tensors
    X_train = torch.tensor(train_data['X'], dtype=torch.float32)
    A_train = torch.tensor(train_data['A'], dtype=torch.float32)
    R_O_train = torch.tensor(train_data['R'], dtype=torch.float32)
    #A_tr = (A_train,1-A_train)

    X_val = torch.tensor(val_data['X'], dtype=torch.float32)
    A_val = torch.tensor(val_data['A'], dtype=torch.float32)
    R_O_val = torch.tensor(val_data['R'], dtype=torch.float32)

    X_test = torch.tensor(test_data['X'], dtype=torch.float32)
    A_test = torch.tensor(test_data['A'], dtype=torch.float32)
    R_O_test = torch.tensor(test_data['R'], dtype=torch.float32)
    t_nodes = torch.tensor(t_nodes, dtype=torch.float32)
    t_fig = torch.tensor(t_fig, dtype=torch.float32)
    d_X = X_train.size()[1]

    # Define the DNN model : outfeature=2, g_1(r,x) and g_0(r,x) respectively
    class DNNModel(torch.nn.Module):
        def __init__(self, in_features=d_X + 1, out_features=2, hidden_nodes=n_node, hidden_layers=n_layer, drop_rate=0):
            super(DNNModel, self).__init__()
            layers = []
            # Input layer
            layers.append(torch.nn.Linear(in_features, hidden_nodes))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(drop_rate))
            # Hidden layers
            for _ in range(hidden_layers):
                layers.append(torch.nn.Linear(hidden_nodes, hidden_nodes))
                layers.append(torch.nn.ReLU())
                layers.append(torch.nn.Dropout(drop_rate))
            # Output layer
            layers.append(torch.nn.Linear(hidden_nodes, out_features))
            self.linear_relu_stack = torch.nn.Sequential(*layers)
        
        def forward(self, x):
            return self.linear_relu_stack(x)

    # Initialize model and optimizer
    model = DNNModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=n_lr)

    # Custom loss function for binary decision: g(Tx)=Ag_1(R,X)+(1-A)g_0(R,X)
    
    def my_loss(g_TX, int_exp_g_TX, A):
        g_TX1 = torch.sum(g_TX*A, dim = 1)
        loss_fun = g_TX1 - int_exp_g_TX
        return -loss_fun.mean()

    
        #simultaneously calculate the intrgral for A=0 and A=1 (need to be corrected)
    def vectorized_gaussian_quadrature_integral(func, a, b, n_points,A):
        nodes, weights = np.polynomial.legendre.leggauss(n_points)
        nodes = torch.tensor(nodes, dtype=torch.float32) 
        weights = torch.tensor(weights, dtype=torch.float32) 

        nodes = 0.5 * (nodes + 1) * (b - a).unsqueeze(1) + a.unsqueeze(1)  # [batch_size, n_points]， len(a) = len(b) = batch_size
        weights = weights * 0.5 * (b - a).unsqueeze(1)  # [batch_size, n_points], len(a) = len(b) = batch_size

        values = func(nodes,A)  # [batch_size, n_points]

        return torch.sum(weights * values, dim=1)  # [batch_size] #sum by column

    def batch_func_train(t,A):  #output the log-hazard for train data
        # t: [batch_size, n_points]
        A_train = A
        batch_size, n_points = t.shape
        t = t.reshape(-1, 1)  # [batch_size * n_points, 1]
        X_repeated = X_train.repeat_interleave(n_points, dim=0)  # [batch_size * n_points, d_X]
        A_repeated = A_train.repeat_interleave(n_points, dim=0)  # [batch_size * n_points, 2]
        inputs = torch.cat((t, X_repeated), dim=1)  # [batch_size * n_points, d_X + 1]
        outputs = model(inputs).squeeze()  # [batch_size * n_points] #output the log-hazard
        outputs1 = torch.sum(outputs*A_repeated, dim = 1)
        return torch.exp(outputs1).reshape(batch_size, n_points)  # 恢复为 [batch_size, n_points]

    def batch_func_val(t,A):
        # t: [batch_size, n_points]
        A_val = A
        batch_size, n_points = t.shape
        t = t.reshape(-1, 1)  # [batch_size * n_points, 1]
        X_repeated = X_val.repeat_interleave(n_points, dim=0)  # [batch_size * n_points, d_X]
        A_repeated = A_val.repeat_interleave(n_points, dim=0)
        inputs = torch.cat((t, X_repeated), dim=1)  # [batch_size * n_points, d_X + 1]
        outputs = model(inputs).squeeze()  # [batch_size * n_points]
        outputs1 = torch.sum(outputs*A_repeated, dim = 1)
        return torch.exp(outputs1).reshape(batch_size, n_points)  # 恢复为 [batch_size, n_points]

    def batch_func_test(t,A):
        # t: [batch_size, n_points]
        A_test = A
        batch_size, n_points = t.shape
        t = t.reshape(-1, 1)  # [batch_size * n_points, 1]
        X_repeated = X_test.repeat_interleave(n_points, dim=0)  # [batch_size * n_points, d_X]
        A_repeated = A_test.repeat_interleave(n_points, dim=0)
        inputs = torch.cat((t, X_repeated), dim=1)  # [batch_size * n_points, d_X + 1]
        outputs = model(inputs).squeeze()  # [batch_size * n_points]
        outputs1 = torch.sum(outputs*A_repeated, dim = 1)
        return torch.exp(outputs1).reshape(batch_size, n_points)  # 恢复为 [batch_size, n_points]
    
    # Treat and Untreat
    A_untreat = torch.zeros(A_test.shape, dtype=torch.float32)
    A_untreat[:,0]=1   #untreat
    A_treated = torch.zeros(A_test.shape, dtype=torch.float32)
    A_treated[:,1]=1   #treated


    # Early stopping parameters
    best_val_loss = float('inf')
    patience_counter = 0

    # Training loop
    for epoch in range(n_epoch):
        model.train()  # Set model to training mode
            
        g_TX = model(torch.cat((R_O_train.unsqueeze(1), X_train), dim=1)).squeeze()

        int_exp_g_TX = vectorized_gaussian_quadrature_integral(batch_func_train, torch.zeros_like(R_O_train), R_O_train, n_points=100,A=A_train) # batch_size = 0.8n

        loss = my_loss(g_TX, int_exp_g_TX, A_train)
        # print('epoch=', epoch, 'loss=', loss.detach().numpy())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Validation step
        model.eval()  # Set model to evaluation mode
        with torch.no_grad():
            g_TX_val = model(torch.cat((R_O_val.unsqueeze(1), X_val), dim=1)).squeeze()
            int_exp_g_TX_val = vectorized_gaussian_quadrature_integral(batch_func_val, torch.zeros_like(R_O_val), R_O_val, n_points=100,A=A_val) # batch_size = 0.2n
            
            val_loss = my_loss(g_TX_val, int_exp_g_TX_val, A_val)
            if show_val == True:
                print('epoch=', epoch, 'val_loss=', val_loss.detach().numpy())
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict()  # Save the best model state
        else:
            patience_counter += 1
            if show_val == True:
                print('patience_counter =', patience_counter)
            
        if patience_counter >= patiences:
            if show_val == True:
                print(f'Early stopping at epoch {epoch + 1}, ', 'validation—loss=', val_loss.detach().numpy())
            break

    # Restore best model if needed
    model.load_state_dict(best_model_state)

    # Evaluation and other code remains unchanged...
    # Test survival under A_test
    model.eval()
    with torch.no_grad():
        # Survival, untreated
        S_T_X_ibs = torch.ones(len(s_k), len(R_O_test), dtype=torch.float32) # 100 * 500
        for k in range(len(s_k)):
            s_k_k_repeat = torch.full((len(R_O_test),), s_k[k], dtype=torch.float32)
            int_exp_g_TX = vectorized_gaussian_quadrature_integral(batch_func_test, torch.zeros_like(s_k_k_repeat), s_k_k_repeat, n_points=100,A=A_untreat)
            S_T_X_ibs[k] = torch.exp(-int_exp_g_TX)
    with torch.no_grad():
        # Survival, treated
        S_T_X_ibs1 = torch.ones(len(s_k), len(R_O_test), dtype=torch.float32) # 100 * 500
        for k in range(len(s_k)):
            s_k_k_repeat1 = torch.full((len(R_O_test),), s_k[k], dtype=torch.float32)
            int_exp_g_TX1 = vectorized_gaussian_quadrature_integral(batch_func_test, torch.zeros_like(s_k_k_repeat1), s_k_k_repeat1, n_points=100,A = A_treated)
            S_T_X_ibs1[k] = torch.exp(-int_exp_g_TX1)
    with torch.no_grad():
        #hazard function
        s_k_k_repeat = torch.full((len(R_O_test),), s_k[k], dtype=torch.float32)
        nodes, weights = np.polynomial.legendre.leggauss(100)
        nodes = torch.tensor(nodes, dtype=torch.float32) 
        weights = torch.tensor(weights, dtype=torch.float32) 
        a = torch.zeros_like(s_k_k_repeat)
        b = s_k_k_repeat
        nodes = 0.5 * (nodes + 1) * (b - a).unsqueeze(1) + a.unsqueeze(1)  # [batch_size, n_points]， len(a) = len(b) = batch_size
        weights = weights * 0.5 * (b - a).unsqueeze(1)  # [batch_size, n_points], len(a) = len(b) = batch_size

        exp_g_TX_untreat = batch_func_test(nodes, A_untreat) #hazard 500*100
        exp_g_TX_treated = batch_func_test(nodes, A_treated) #hazard
    
    with torch.no_grad():
        #conditional mean
        R_0_test = torch.tensor(test_data['R'], dtype=torch.float32)
        s_k1 = torch.tensor(s_k, dtype = torch.float32)
        s_k_k1 = torch.tile(s_k1,(len(R_0_test),1))
        Cmean_untreat = torch.trapz(y=S_T_X_ibs.T,x=s_k_k1)
        Cmean_treated = torch.trapz(y=S_T_X_ibs1.T,x=s_k_k1)
    
    return {
        'S_R_X_untreat': S_T_X_ibs.T.detach().numpy(),  #  500 * 100
        'S_R_X_treated': S_T_X_ibs1.T.detach().numpy(),
        'Lambda_R_X_untreat': exp_g_TX_untreat.detach().numpy(), # 500*100
        'Lambda_R_X_treated': exp_g_TX_treated.detach().numpy(),
        'Cmean_R_X_untreat': Cmean_untreat.detach().numpy(), # 500*1
        'Cmean_R_X_treated': Cmean_treated.detach().numpy()
    }















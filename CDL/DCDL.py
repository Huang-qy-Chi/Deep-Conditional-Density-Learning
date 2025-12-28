import numpy as np 
import torch 
#%% ----------Input:(t,X)-- Output:g(t,X) --------------
def DCDL(train_data, val_data, test_data, t_nodes, t_fig, s_k, n_layer,\
          n_node, n_lr, n_epoch, patiences, show_val=True):
    if show_val == True:
        print('DNN_iteration')
    # Convert training and test data to tensors
    X_train = torch.tensor(train_data['X'], dtype=torch.float32)
    # De_train = torch.tensor(train_data['De'], dtype=torch.float32)
    R_0_train = torch.tensor(train_data['R'], dtype=torch.float32)
    A_train = (train_data['A'])[:,1]
    A_train = torch.tensor(A_train, dtype=torch.float32)

    X_val = torch.tensor(val_data['X'], dtype=torch.float32)
    # De_val = torch.tensor(val_data['De'], dtype=torch.float32)
    R_0_val = torch.tensor(val_data['R'], dtype=torch.float32)
    A_val = (val_data['A'])[:,1]
    A_val = torch.tensor(A_val, dtype=torch.float32)

    X_test = torch.tensor(test_data['X'], dtype=torch.float32)
    R_0_test = torch.tensor(test_data['R'], dtype=torch.float32)
    A_test = (test_data['A'])[:,1]
    A_test = torch.tensor(A_test, dtype=torch.float32)
    t_nodes = torch.tensor(t_nodes, dtype=torch.float32)
    t_fig = torch.tensor(t_fig, dtype=torch.float32)
    d_X = X_train.size()[1]

    # Define the DNN model
    class DNNModel(torch.nn.Module):
        def __init__(self, in_features=d_X + 2, out_features=1, hidden_nodes=n_node, hidden_layers=n_layer, drop_rate=0):
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

    # Custom loss function
    def my_loss(g_TX, int_exp_g_TX):
        loss_fun = g_TX - int_exp_g_TX
        return -loss_fun.mean()

    def vectorized_gaussian_quadrature_integral(func, a, b, n_points):
        nodes, weights = np.polynomial.legendre.leggauss(n_points)
        nodes = torch.tensor(nodes, dtype=torch.float32) 
        weights = torch.tensor(weights, dtype=torch.float32) 

        nodes = 0.5 * (nodes + 1) * (b - a).unsqueeze(1) + a.unsqueeze(1)  # [batch_size, n_points]， len(a) = len(b) = batch_size
        weights = weights * 0.5 * (b - a).unsqueeze(1)  # [batch_size, n_points], len(a) = len(b) = batch_size

        values = func(nodes)  # [batch_size, n_points]

        return torch.sum(weights * values, dim=1)  # [batch_size]

    def batch_func_train(t):
        # t: [batch_size, n_points]
        batch_size, n_points = t.shape
        t = t.reshape(-1, 1)  # [batch_size * n_points, 1]
        X_repeated = X_train.repeat_interleave(n_points, dim=0)  # [batch_size * n_points, d_X]
        A_repeated = A_train.repeat_interleave(n_points, dim=0)
        inputs = torch.cat((t,A_repeated.unsqueeze(1), X_repeated), dim=1)  # [batch_size * n_points, d_X + 1]
        outputs = model(inputs).squeeze()  # [batch_size * n_points]
        return torch.exp(outputs).reshape(batch_size, n_points)  # 恢复为 [batch_size, n_points]
    
    def batch_func_val(t):
        # t: [batch_size, n_points]
        batch_size, n_points = t.shape
        t = t.reshape(-1, 1)  # [batch_size * n_points, 1]
        X_repeated = X_val.repeat_interleave(n_points, dim=0)  # [batch_size * n_points, d_X]
        A_repeated = A_val.repeat_interleave(n_points, dim=0)
        inputs = torch.cat((t,A_repeated.unsqueeze(1), X_repeated), dim=1)  # [batch_size * n_points, d_X + 1]
        outputs = model(inputs).squeeze()  # [batch_size * n_points]
        return torch.exp(outputs).reshape(batch_size, n_points)  # 恢复为 [batch_size, n_points]

    def batch_func_test(t):
        # t: [batch_size, n_points]
        batch_size, n_points = t.shape
        t = t.reshape(-1, 1)  # [batch_size * n_points, 1]
        X_repeated = X_test.repeat_interleave(n_points, dim=0)  # [batch_size * n_points, d_X]
        A_repeated = A_test.repeat_interleave(n_points, dim=0)
        inputs = torch.cat((t,A_repeated.unsqueeze(1), X_repeated), dim=1)  # [batch_size * n_points, d_X + 1]
        outputs = model(inputs).squeeze()  # [batch_size * n_points]
        return torch.exp(outputs).reshape(batch_size, n_points)  # 恢复为 [batch_size, n_points]
    
    # Early stopping parameters
    best_val_loss = float('inf')
    patience_counter = 0

    # Training loop
    for epoch in range(n_epoch):
        model.train()  # Set model to training mode
            
        g_TX = model(torch.cat((R_0_train.unsqueeze(1),A_train.unsqueeze(1), X_train), dim=1)).squeeze()

        int_exp_g_TX = vectorized_gaussian_quadrature_integral(batch_func_train, torch.zeros_like(R_0_train), R_0_train, n_points=100) # batch_size = 0.8n

        loss = my_loss(g_TX, int_exp_g_TX)
        # print('epoch=', epoch, 'loss=', loss.detach().numpy())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Validation step
        model.eval()  # Set model to evaluation mode
        with torch.no_grad():
            g_TX_val = model(torch.cat((R_0_val.unsqueeze(1),A_val.unsqueeze(1), X_val), dim=1)).squeeze()
            int_exp_g_TX_val = vectorized_gaussian_quadrature_integral(batch_func_val, torch.zeros_like(R_0_val), R_0_val, n_points=100) # batch_size = 0.2n
            
            val_loss = my_loss(g_TX_val, int_exp_g_TX_val)
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
    # Test
    model.eval()
    





    with torch.no_grad():
        # IBS
        S_T_X_ibs = torch.ones(len(s_k), len(R_0_test), dtype=torch.float32) # 100 * 500
        for k in range(len(s_k)):
            s_k_k_repeat = torch.full((len(R_0_test),), s_k[k], dtype=torch.float32)
            int_exp_g_TX = vectorized_gaussian_quadrature_integral(batch_func_test, torch.zeros_like(s_k_k_repeat), s_k_k_repeat, n_points=100)
            S_T_X_ibs[k] = torch.exp(-int_exp_g_TX)
    with torch.no_grad():
        #hazard function
        s_k_k_repeat = torch.full((len(R_0_test),), s_k[k], dtype=torch.float32)
        nodes, weights = np.polynomial.legendre.leggauss(100)
        nodes = torch.tensor(nodes, dtype=torch.float32) 
        weights = torch.tensor(weights, dtype=torch.float32) 
        a = torch.zeros_like(s_k_k_repeat)
        b = s_k_k_repeat
        nodes = 0.5 * (nodes + 1) * (b - a).unsqueeze(1) + a.unsqueeze(1)  # [batch_size, n_points]， len(a) = len(b) = batch_size
        weights = weights * 0.5 * (b - a).unsqueeze(1)  # [batch_size, n_points], len(a) = len(b) = batch_size

        exp_g_TX_test = batch_func_test(nodes) #hazard n*100
        g_RX = model(torch.cat((R_0_test.unsqueeze(1),A_test.unsqueeze(1), X_test), dim=1)).squeeze()
        g_R_X_test = g_RX
        int_exp_g_RX_test = vectorized_gaussian_quadrature_integral(batch_func_test, torch.zeros_like(R_0_test), R_0_test, n_points=100) # batch_size = 0.2n
        S_R_X = torch.exp(-int_exp_g_RX_test)
        lambda_R_X = torch.exp(g_R_X_test)
        # conditional density
        f_R_X_test = exp_g_TX_test*S_T_X_ibs.T
        # log- conditional density
        log_f_R_test = torch.log(lambda_R_X*S_R_X)
    
    return {
        'S_R_X_test': S_T_X_ibs.T.detach().numpy(),  #  n_test * 100
        'lambda_R_X_test': exp_g_TX_test.detach().numpy(), #  n_test * 100
        'pdf_grid_from_test': f_R_X_test.detach().numpy(), #  n_test * 100
        'S_R_X': S_R_X.detach().numpy(),  #(n_test, )
        'lambda_R_X': lambda_R_X.detach().numpy(), #(n_test, )
        "logpdf_from_test": log_f_R_test.detach().numpy(), #(n_test, )
        'model': model
    }








if __name__ == '__main__':
    from seed import set_seed
    from data_generator import gendata_Deep
    # set_seed(1145)
    # train_data = gendata_Deep(n = 500, corr = 0.5)
    set_seed(4514)
    test_data = gendata_Deep(n = 200, corr = 0.5)

    # Hyperparameter
    patiences = 30
    n_node = 64
    n_layer = 1
    t_nodes = 100
    t_fig = 50
    n_lr = 1e-4
    n_epoch = 1000
    r_grid = np.linspace(0.1, 5.0, 100)
    s_k = r_grid
    set_seed(1145)
    val_data = gendata_Deep(100, 0.5)
    train_data = gendata_Deep(400, 0.5)
    St = DCDL(train_data, val_data, test_data, t_nodes, t_fig,\
               s_k, n_layer, n_node, n_lr, n_epoch, patiences,show_val=False)
    # 选 test 中的全部条件点输出 pdf 曲线
    r_grid = np.linspace(0.1, 5.0, 100)
    pdf_all = St["pdf_grid_from_test"]    # (n_test, n_grid)

    # 或者只抽 m_eval 个点（这就是你之前问的 m）
    # r_grid = np.linspace(0.1, 5.0, 400)
    # rng = np.random.default_rng(0)
    # idx = rng.choice(len(test_data["R"]), size=200, replace=False)
    # pdf_200 = kcde["pdf_grid_from_test"](r_grid, test_data, idx=idx)    # (n_test, n_grid)

    # NLL 用点密度（每个 test 样本的自身 r）
    logp = St["logpdf_from_test"] # (n_test, )
    nll = -float(np.mean(logp))
    print("NLL:", nll)
    print(pdf_all.shape)
    print(type(pdf_all))
    print(logp.shape)
    print(type(logp))
    from L2_error import integrated_l2
    from L2_error import normalize_pdf
    from true_pdf import true_pdf_grid_fn
    pdf_all = np.asarray(pdf_all, dtype=float)
    A_test = np.asarray(test_data["A"])
    X_test = np.asarray(test_data["X"], dtype=float)
    if A_test.ndim == 2 and A_test.shape[1] == 2:
        A_scalar = A_test[:, 1].astype(int)
    else:
        A_scalar = A_test.astype(int)

    X_eval = X_test

    f_hat = normalize_pdf(pdf_all, r_grid)
    f_true = true_pdf_grid_fn(r_grid, A_scalar, X_eval)
    f_true = np.asarray(f_true, dtype=float)
    f_true = normalize_pdf(f_true, r_grid)

    ise = integrated_l2(f_hat, f_true, r_grid)
    print("KCDE Integrated L2 error (ISE):", ise)
    








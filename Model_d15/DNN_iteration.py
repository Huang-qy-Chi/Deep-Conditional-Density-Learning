import torch
import numpy as np

#%% ----------Input:(t,X)-- Output:g(t,X) --------------
def g_dnn(train_data, val_data, test_data, t_nodes, t_fig, s_k, n_layer, n_node, n_lr, n_epoch, patiences):
    print('DNN_iteration')
    # Convert training and test data to tensors
    X_train = torch.tensor(train_data['X'], dtype=torch.float32)
    De_train = torch.tensor(train_data['De'], dtype=torch.float32)
    T_O_train = torch.tensor(train_data['T_O'], dtype=torch.float32)

    X_val = torch.tensor(val_data['X'], dtype=torch.float32)
    De_val = torch.tensor(val_data['De'], dtype=torch.float32)
    T_O_val = torch.tensor(val_data['T_O'], dtype=torch.float32)

    X_test = torch.tensor(test_data['X'], dtype=torch.float32)
    T_O_test = torch.tensor(test_data['T_O'], dtype=torch.float32)
    t_nodes = torch.tensor(t_nodes, dtype=torch.float32)
    t_fig = torch.tensor(t_fig, dtype=torch.float32)
    d_X = X_train.size()[1]

    # Define the DNN model
    class DNNModel(torch.nn.Module):
        def __init__(self, in_features=d_X + 1, out_features=1, hidden_nodes=n_node, hidden_layers=n_layer, drop_rate=0):
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
    def my_loss(De, g_TX, int_exp_g_TX):
        loss_fun = De * g_TX - int_exp_g_TX
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
        inputs = torch.cat((t, X_repeated), dim=1)  # [batch_size * n_points, d_X + 1]
        outputs = model(inputs).squeeze()  # [batch_size * n_points]
        return torch.exp(outputs).reshape(batch_size, n_points)  # 恢复为 [batch_size, n_points]
    
    def batch_func_val(t):
        # t: [batch_size, n_points]
        batch_size, n_points = t.shape
        t = t.reshape(-1, 1)  # [batch_size * n_points, 1]
        X_repeated = X_val.repeat_interleave(n_points, dim=0)  # [batch_size * n_points, d_X]
        inputs = torch.cat((t, X_repeated), dim=1)  # [batch_size * n_points, d_X + 1]
        outputs = model(inputs).squeeze()  # [batch_size * n_points]
        return torch.exp(outputs).reshape(batch_size, n_points)  # 恢复为 [batch_size, n_points]

    def batch_func_test(t):
        # t: [batch_size, n_points]
        batch_size, n_points = t.shape
        t = t.reshape(-1, 1)  # [batch_size * n_points, 1]
        X_repeated = X_test.repeat_interleave(n_points, dim=0)  # [batch_size * n_points, d_X]
        inputs = torch.cat((t, X_repeated), dim=1)  # [batch_size * n_points, d_X + 1]
        outputs = model(inputs).squeeze()  # [batch_size * n_points]
        return torch.exp(outputs).reshape(batch_size, n_points)  # 恢复为 [batch_size, n_points]
    
    # Early stopping parameters
    best_val_loss = float('inf')
    patience_counter = 0

    # Training loop
    for epoch in range(n_epoch):
        model.train()  # Set model to training mode
            
        g_TX = model(torch.cat((T_O_train.unsqueeze(1), X_train), dim=1)).squeeze()

        int_exp_g_TX = vectorized_gaussian_quadrature_integral(batch_func_train, torch.zeros_like(T_O_train), T_O_train, n_points=100) # batch_size = 0.8n

        loss = my_loss(De_train, g_TX, int_exp_g_TX)
        # print('epoch=', epoch, 'loss=', loss.detach().numpy())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Validation step
        model.eval()  # Set model to evaluation mode
        with torch.no_grad():
            g_TX_val = model(torch.cat((T_O_val.unsqueeze(1), X_val), dim=1)).squeeze()
            int_exp_g_TX_val = vectorized_gaussian_quadrature_integral(batch_func_val, torch.zeros_like(T_O_val), T_O_val, n_points=100) # batch_size = 0.2n
            
            val_loss = my_loss(De_val, g_TX_val, int_exp_g_TX_val)
            print('epoch=', epoch, 'val_loss=', val_loss.detach().numpy())
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict()  # Save the best model state
        else:
            patience_counter += 1
            print('patience_counter =', patience_counter)
            
        if patience_counter >= patiences:
            print(f'Early stopping at epoch {epoch + 1}', 'validation—loss=', val_loss.detach().numpy())
            break

    # Restore best model if needed
    model.load_state_dict(best_model_state)

    # Evaluation and other code remains unchanged...
    # Test
    model.eval()
    with torch.no_grad():
        # IBS
        S_T_X_ibs = torch.ones(len(s_k), len(T_O_test), dtype=torch.float32) # 100 * 500
        for k in range(len(s_k)):
            s_k_k_repeat = torch.full((len(T_O_test),), s_k[k], dtype=torch.float32)
            int_exp_g_TX = vectorized_gaussian_quadrature_integral(batch_func_test, torch.zeros_like(s_k_k_repeat), s_k_k_repeat, n_points=100)
            S_T_X_ibs[k] = torch.exp(-int_exp_g_TX)

    
    
    return {
        'S_T_X_ibs': S_T_X_ibs.T.detach().numpy(),  #  500 * 100
    }


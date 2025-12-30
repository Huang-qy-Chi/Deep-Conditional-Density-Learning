import torch
import numpy as np

#%%----------Input:(R,A,X)-- Output:g(R,A,X) --------------
def E_dls_bin(train_data, val_data, test_data, n_layer, n_node, n_lr, n_epoch, patiences,show_val=True):
    #Assume g(R,A,X)=Ag_1(R,X)+(1-A)g_0(R,X)
    if show_val == True:
        print('DNN_iteration')
    # Convert training and test data to tensors
    X_train = torch.tensor(train_data['X'], dtype=torch.float32)
    A_train = torch.tensor(train_data['A'], dtype=torch.float32)
    R_0_train = torch.tensor(train_data['R'], dtype=torch.float32)
    logR_0_train = torch.log(R_0_train)
    

    X_val = torch.tensor(val_data['X'], dtype=torch.float32)
    A_val = torch.tensor(val_data['A'], dtype=torch.float32)
    R_0_val = torch.tensor(val_data['R'], dtype=torch.float32)
    logR_0_val = torch.log(R_0_val)

    X_test = torch.tensor(test_data['X'], dtype=torch.float32)
    A_test = torch.tensor(test_data['A'], dtype=torch.float32)
    R_0_test = torch.tensor(test_data['R'], dtype=torch.float32)
    logR_0_test = torch.log(R_0_test)
    d_X = X_train.size()[1]

    # Define the DNN model : outfeature=3, mu_0(x) baseline, mu_1(x) untreat and mu_2(x) treated respectively
    class DNNModel(torch.nn.Module):
        def __init__(self, in_features=d_X, out_features=3, hidden_nodes=n_node, hidden_layers=n_layer, drop_rate=0):
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

    # Custom loss function for binary decision: via MSEloss
    # my_loss = torch.nn.MSELoss(reduction='mean')
    def my_loss(mu_TX, A, logR):  #mu_TX is 3 dim, A = (1,A) is 3 dim, logR is 1 dim
        A1 = torch.ones(A.shape[0])
        A1 = A1.unsqueeze(1)
        A = torch.cat((A1,A),dim = 1) # add the baseline mu0
        mu_TX1 = torch.sum(mu_TX*A, dim = 1)
        loss_fun = (logR-mu_TX1)**2  #least square
        return loss_fun.mean()

    
        #simultaneously calculate the intrgral for A=0 and A=1 (need to be corrected)
    
    
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
            
        # mu_TX = model(torch.cat((logR_0_train.unsqueeze(1), X_train), dim=1)).squeeze()
        mu_TX = model(X_train).squeeze()
        # int_exp_g_TX = vectorized_gaussian_quadrature_integral(batch_func_train, torch.zeros_like(R_O_train), R_O_train, n_points=100,A=A_train) # batch_size = 0.8n

        loss = my_loss(mu_TX, A_train, logR_0_train)
        # print('epoch=', epoch, 'loss=', loss.detach().numpy())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Validation step
        model.eval()  # Set model to evaluation mode
        with torch.no_grad():
            # g_TX_val = model(torch.cat((R_O_val.unsqueeze(1), X_val), dim=1)).squeeze()
            mu_TX_val = model(X_val).squeeze()
            # int_exp_g_TX_val = vectorized_gaussian_quadrature_integral(batch_func_val, torch.zeros_like(R_O_val), R_O_val, n_points=100,A=A_val) # batch_size = 0.2n
            
            # val_loss = my_loss(g_TX_val, int_exp_g_TX_val, A_val)
            val_loss = my_loss(mu_TX_val, A_val, logR_0_val)
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
    # Test prediction under A_test
    model.eval()
    with torch.no_grad():
        # prediction of mu0(x) for baseline, mu1(x) for untreat and mu2(x) for treated
        model.eval()
        mu_TX_test = model(X_test).squeeze()
        # print(mu_TX_test)
        mu_base_test = mu_TX_test[:,0]
        mu_un_test = mu_TX_test[:,1]
        mu_tr_test = mu_TX_test[:,2]
        At1 = torch.ones(A_test.shape[0])
        At1 = At1.unsqueeze(1)
        A_untreat = torch.zeros(A_test.shape, dtype=torch.float32)
        A_untreat[:,0]=1   #untreat
        A_treated = torch.zeros(A_test.shape, dtype=torch.float32)
        A_treated[:,1]=1   #treated
        A_untreat = torch.cat((At1,A_untreat),dim=1)
        A_treated = torch.cat((At1,A_treated),dim=1)
        Cmean_untreat = torch.exp(torch.sum(A_untreat*mu_TX_test,dim=1))
        Cmean_treated = torch.exp(torch.sum(A_treated*mu_TX_test,dim=1))
    
    return {
        'mu_base_test': mu_base_test.detach().numpy(),  #  500 * 1
        'mu_un_test': mu_un_test.detach().numpy(),
        'mu_tr_test': mu_tr_test.detach().numpy(), # 500*1
        'Cmean_R_X_untreat': Cmean_untreat.detach().numpy(), # 500*1
        'Cmean_R_X_treated': Cmean_treated.detach().numpy()
    }








































































































import os
from multiprocessing import get_context, cpu_count, Manager
# from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import time
from seed import set_seed
from KCDE import kcde1
from QRF import qrf1
from FlexCode import flexcode1
from DCDL import DCDL
from L2_error import integrated_l2, normalize_pdf
from true_pdf import true_pdf_grid_fn
from data_generator import gendata_Linear, gendata_Deep
import warnings
warnings.filterwarnings('ignore')


#%%-------------------------------Linear---------------------------------------------------------


# --- 并行任务：单次 Monte Carlo 重复 ---
def _one_run_linear(i, Seed, n, n_v, corr, beta, gamma, r_grid, test_data, t_nodes, t_fig,\
               s_k, n_layer, n_node, n_lr, n_epoch, patiences):
    set_seed(Seed + i)
    warnings.filterwarnings("ignore")

    train_data = gendata_Linear(n=n, corr=corr, beta=beta, gamma=gamma)
    val_data   = gendata_Linear(n=n_v, corr=corr, beta=beta, gamma=gamma)
    St = DCDL(train_data, val_data, test_data, t_nodes, t_fig,\
               s_k, n_layer, n_node, n_lr, n_epoch, patiences,show_val=False)
    
    # data combination
    X_train = train_data['X']
    X_val   = val_data['X']
    X_train_val = np.vstack((X_train, X_val))

    A_train = train_data['A']
    A_val   = val_data['A']
    A_train_val = np.vstack((A_train, A_val))

    R_0_train = train_data['R']
    R_0_val   = val_data['R']
    R_0_train_val = np.r_[R_0_train, R_0_val]

    train_val_data = {'R': R_0_train_val, 'A': A_train_val, 'X': X_train_val}

    # competetive mwthods: KCDE, QRF, FLEXCODE
    kcde = kcde1(train_val_data, bw="normal_reference")
    qrf  = qrf1(train_val_data, n_estimators=300, min_samples_leaf=10, random_state=Seed + i)
    r_lo = 0
    r_hi = np.max(train_data['R'])
    flex = flexcode1(train_val_data, z_min=r_lo, z_max=r_hi, max_basis=30)

    # pdf
    pdf_all_dcdl = St["pdf_grid_from_test"]    # (n_test, n_grid)
    pdf_all_kcde = kcde["pdf_grid_from_test"](r_grid, test_data)
    pdf_all_qrf  = qrf["pdf_grid_from_test"](r_grid, test_data)
    pdf_all_flex = flex["pdf_grid_from_test"](r_grid, test_data)

    A_test = np.asarray(test_data["A"])
    X_test = np.asarray(test_data["X"], dtype=float)
    if A_test.ndim == 2 and A_test.shape[1] == 2:
        A_scalar = A_test[:, 1].astype(int)
    else:
        A_scalar = A_test.astype(int)

    f_true = true_pdf_grid_fn(r_grid, A_scalar, X_test, model='linear')
    f_true = np.asarray(f_true, dtype=float)
    f_true = normalize_pdf(f_true, r_grid)
    
    f_hat_dcdl = normalize_pdf(pdf_all_dcdl, r_grid)
    f_hat_kcde = normalize_pdf(pdf_all_kcde, r_grid)
    f_hat_qrf  = normalize_pdf(pdf_all_qrf, r_grid)
    f_hat_flex = normalize_pdf(pdf_all_flex, r_grid)

    ise_dcdl = integrated_l2(f_hat_dcdl, f_true, r_grid)
    ise_kcde = integrated_l2(f_hat_kcde, f_true, r_grid)
    ise_qrf  = integrated_l2(f_hat_qrf,  f_true, r_grid)
    ise_flex = integrated_l2(f_hat_flex, f_true, r_grid)

    # logpdf
    logp_dcdl = St["logpdf_from_test"]
    logp_kcde = kcde["logpdf_from_test"](test_data)
    logp_qrf  = qrf["logpdf_from_test"](test_data)
    logp_flex = flex["logpdf_from_test"](test_data)

    nll_dcdl = -float(np.mean(logp_dcdl))
    nll_kcde = -float(np.mean(logp_kcde))
    nll_qrf  = -float(np.mean(logp_qrf))
    nll_flex = -float(np.mean(logp_flex))

    # 返回一轮的 6 个指标
    return ise_dcdl, ise_kcde, ise_qrf, ise_flex, nll_dcdl, nll_kcde, nll_qrf, nll_flex



def _one_run_deep(i, Seed, n, n_v, corr, r_grid, test_data, t_nodes, t_fig,\
               s_k, n_layer, n_node, n_lr, n_epoch, patiences):
    set_seed(Seed + i)
    warnings.filterwarnings("ignore")

    train_data = gendata_Deep(n=n, corr=corr)
    val_data   = gendata_Deep(n=n_v, corr=corr)
    St = DCDL(train_data, val_data, test_data, t_nodes, t_fig,\
               s_k, n_layer, n_node, n_lr, n_epoch, patiences,show_val=False)
    
    # data combination
    X_train = train_data['X']
    X_val   = val_data['X']
    X_train_val = np.vstack((X_train, X_val))

    A_train = train_data['A']
    A_val   = val_data['A']
    A_train_val = np.vstack((A_train, A_val))

    R_0_train = train_data['R']
    R_0_val   = val_data['R']
    R_0_train_val = np.r_[R_0_train, R_0_val]

    train_val_data = {'R': R_0_train_val, 'A': A_train_val, 'X': X_train_val}

    # competetive mwthods: KCDE, QRF, FLEXCODE
    kcde = kcde1(train_val_data, bw="normal_reference")
    qrf  = qrf1(train_val_data, n_estimators=300, min_samples_leaf=10, random_state=Seed + i)
    r_lo = 0
    r_hi = np.max(train_data['R'])
    flex = flexcode1(train_val_data, z_min=r_lo, z_max=r_hi, max_basis=30)

    # pdf
    pdf_all_dcdl = St["pdf_grid_from_test"]    # (n_test, n_grid)
    pdf_all_kcde = kcde["pdf_grid_from_test"](r_grid, test_data)
    pdf_all_qrf  = qrf["pdf_grid_from_test"](r_grid, test_data)
    pdf_all_flex = flex["pdf_grid_from_test"](r_grid, test_data)

    A_test = np.asarray(test_data["A"])
    X_test = np.asarray(test_data["X"], dtype=float)
    if A_test.ndim == 2 and A_test.shape[1] == 2:
        A_scalar = A_test[:, 1].astype(int)
    else:
        A_scalar = A_test.astype(int)

    f_true = true_pdf_grid_fn(r_grid, A_scalar, X_test, model='deep')
    f_true = np.asarray(f_true, dtype=float)
    f_true = normalize_pdf(f_true, r_grid)
    
    f_hat_dcdl = normalize_pdf(pdf_all_dcdl, r_grid)
    f_hat_kcde = normalize_pdf(pdf_all_kcde, r_grid)
    f_hat_qrf  = normalize_pdf(pdf_all_qrf, r_grid)
    f_hat_flex = normalize_pdf(pdf_all_flex, r_grid)

    ise_dcdl = integrated_l2(f_hat_dcdl, f_true, r_grid)
    ise_kcde = integrated_l2(f_hat_kcde, f_true, r_grid)
    ise_qrf  = integrated_l2(f_hat_qrf,  f_true, r_grid)
    ise_flex = integrated_l2(f_hat_flex, f_true, r_grid)

    # logpdf
    logp_dcdl = St["logpdf_from_test"]
    logp_kcde = kcde["logpdf_from_test"](test_data)
    logp_qrf  = qrf["logpdf_from_test"](test_data)
    logp_flex = flex["logpdf_from_test"](test_data)

    nll_dcdl = -float(np.mean(logp_dcdl))
    nll_kcde = -float(np.mean(logp_kcde))
    nll_qrf  = -float(np.mean(logp_qrf))
    nll_flex = -float(np.mean(logp_flex))

    # 返回一轮的 6 个指标
    return ise_dcdl, ise_kcde, ise_qrf, ise_flex, nll_dcdl, nll_kcde, nll_qrf, nll_flex



# ================== 原先的 for 循环改为并行 ==================
if __name__ == '__main__':
    import warnings
    print('-----Case: n = 500, d = 5-----')
    # Linear
    warnings.filterwarnings("ignore")
    nll_dcdl_linear = []; nll_kcde_linear = []; nll_flex_linear = []; nll_qrf_linear = []
    ise_dcdl_linear = []; ise_kcde_linear = []; ise_flex_linear = []; ise_qrf_linear = []
    patiences = 30; n_node = 64; n_layer = 1; t_nodes = 100; t_fig = 50
    n_lr = 1e-4; n_epoch = 1000
    B = 200; n = 400; n_v = 100; n1 = 500; corr = 0.5
    beta = np.array([0,-0.5,0.5,-0.25,0.5])
    gamma = np.array([0.5,-0.25,0,0.5,-0.25])
    r_grid = np.linspace(0.1, 5.0, 100)
    s_k = r_grid
    Seed = 1145
    set_seed(4514)
    test_data = gendata_Linear(n = n1, corr = corr,beta=beta,gamma=gamma)
    start_time = time.time()

    # 建议：留 1 个核给系统（你也可以改成 cpu_count()）
    n_jobs = 7

    ctx = get_context("spawn")  # 跨平台更稳
    with ctx.Pool(processes=n_jobs) as pool:
        args_iter = [(i, Seed, n, n_v, corr, beta, gamma, r_grid, test_data, t_nodes, t_fig,\
               s_k, n_layer, n_node, n_lr, n_epoch, patiences) for i in range(B)]
        results = pool.starmap(_one_run_linear, args_iter)

    # 拆回原来的 list
    ise_dcdl_linear, ise_kcde_linear, ise_qrf_linear, ise_flex_linear,\
          nll_dcdl_linear, nll_kcde_linear, nll_qrf_linear, nll_flex_linear = map(list, zip(*results))
    
    ISE_DCDL = np.mean(np.array(ise_dcdl_linear)); SE_ISE_DCDL = np.std(np.array(ise_dcdl_linear))
    ISE_KCDE = np.mean(np.array(ise_kcde_linear)); SE_ISE_KCDE = np.std(np.array(ise_kcde_linear))
    ISE_QRF = np.mean(np.array(ise_qrf_linear)); SE_ISE_QRF = np.std(np.array(ise_qrf_linear))
    ISE_FLEX = np.mean(np.array(ise_flex_linear)); SE_ISE_FLEX = np.std(np.array(ise_flex_linear))

    NLL_DCDL = np.mean(np.array(nll_dcdl_linear)); SE_NLL_DCDL = np.std(np.array(nll_dcdl_linear))
    NLL_KCDE = np.mean(np.array(nll_kcde_linear)); SE_NLL_KCDE = np.std(np.array(nll_kcde_linear))
    NLL_QRF = np.mean(np.array(nll_qrf_linear)); SE_NLL_QRF = np.std(np.array(nll_qrf_linear))
    NLL_FLEX = np.mean(np.array(nll_flex_linear)); SE_NLL_FLEX = np.std(np.array(nll_flex_linear))

    print(f"{"Est_d"}_{5}_{"n"}_{n}_{"case"}_{"Linear"}")
    print(f"ISE_DCDL: {ISE_DCDL:.3f}, SE_ISE_DCDL: {SE_ISE_DCDL:.3f}", end='\n')
    print(f"ISE_KCDE: {ISE_KCDE:.3f}, SE_ISE_KCDE: {SE_ISE_KCDE:.3f}", end='\n')
    print(f"ISE_QRF: {ISE_QRF:.3f}, SE_ISE_QRF: {SE_ISE_QRF:.3f}", end='\n')
    print(f"ISE_FLEX: {ISE_FLEX:.3f}, SE_ISE_FLEX: {SE_ISE_FLEX:.3f}", end='\n')

    print(f"NLL_DCDL: {NLL_DCDL:.3f}, SE_NLL_DCDL: {SE_NLL_DCDL:.3f}", end='\n')
    print(f"NLL_KCDE: {NLL_KCDE:.3f}, SE_NLL_KCDE: {SE_NLL_KCDE:.3f}", end='\n')
    print(f"NLL_QRF: {NLL_QRF:.3f}, SE_NLL_QRF: {SE_NLL_QRF:.3f}", end='\n')
    print(f"NLL_FLEX: {NLL_FLEX:.3f}, SE_NLL_FLEX: {SE_NLL_FLEX:.3f}", end='\n')
    end_time = time.time()
    print(f"Computing time: {end_time - start_time:.2f} seconds for estimation.")


    result_linear5 = {
        'ISE_DCDL': ise_dcdl_linear,
        'ISE_KCDE': ise_kcde_linear,
        'ISE_QRF': ise_qrf_linear,
        'ISE_FLEX': ise_flex_linear,
        'NLL_DCDL': nll_dcdl_linear,
        'NLL_KCDE': nll_kcde_linear, 
        'NLL_QRF': nll_qrf_linear,
        'NLL_FLEX': nll_flex_linear
    }
    filename = f"{"Est_d"}_{5}_{"n"}_{n}_{"pr"}_{"Linear_2"}.npy"
    np.save(filename, result_linear5, allow_pickle=True)

    print("System pause!")
    time.sleep(120)
    print('System continue!')

    # Deep
    warnings.filterwarnings("ignore")
    nll_dcdl_deep = []; nll_kcde_deep = []; nll_flex_deep = []; nll_qrf_deep = []
    ise_dcdl_deep = []; ise_kcde_deep = []; ise_flex_deep = []; ise_qrf_deep = []
    patiences = 30; n_node = 64; n_layer = 1; t_nodes = 100; t_fig = 50
    n_lr = 1e-4; n_epoch = 1000
    B = 200; n = 400; n_v = 100; n1 = 500; corr = 0.5
    r_grid = np.linspace(0.1, 5.0, 100)
    s_k = r_grid
    Seed = 1145
    set_seed(4514)
    test_data = gendata_Deep(n = n1, corr = corr)
    start_time = time.time()

    # 建议：留 1 个核给系统（你也可以改成 cpu_count()）
    n_jobs = 7

    ctx = get_context("spawn")  # 跨平台更稳
    with ctx.Pool(processes=n_jobs) as pool:
        args_iter = [(i, Seed, n, n_v, corr, r_grid, test_data, t_nodes, t_fig,\
               s_k, n_layer, n_node, n_lr, n_epoch, patiences) for i in range(B)]
        results = pool.starmap(_one_run_deep, args_iter)

    # 拆回原来的 list
    ise_dcdl_deep, ise_kcde_deep, ise_qrf_deep, ise_flex_deep,\
          nll_dcdl_deep, nll_kcde_deep, nll_qrf_deep, nll_flex_deep = map(list, zip(*results))
    
    ISE_DCDL = np.mean(np.array(ise_dcdl_deep)); SE_ISE_DCDL = np.std(np.array(ise_dcdl_deep))
    ISE_KCDE = np.mean(np.array(ise_kcde_deep)); SE_ISE_KCDE = np.std(np.array(ise_kcde_deep))
    ISE_QRF = np.mean(np.array(ise_qrf_deep)); SE_ISE_QRF = np.std(np.array(ise_qrf_deep))
    ISE_FLEX = np.mean(np.array(ise_flex_deep)); SE_ISE_FLEX = np.std(np.array(ise_flex_deep))

    NLL_DCDL = np.mean(np.array(nll_dcdl_deep)); SE_NLL_DCDL = np.std(np.array(nll_dcdl_deep))
    NLL_KCDE = np.mean(np.array(nll_kcde_deep)); SE_NLL_KCDE = np.std(np.array(nll_kcde_deep))
    NLL_QRF = np.mean(np.array(nll_qrf_deep)); SE_NLL_QRF = np.std(np.array(nll_qrf_deep))
    NLL_FLEX = np.mean(np.array(nll_flex_deep)); SE_NLL_FLEX = np.std(np.array(nll_flex_deep))

    print(f"{"Est_d"}_{5}_{"n"}_{n}_{"case"}_{"Deep"}")
    print(f"ISE_DCDL: {ISE_DCDL:.3f}, SE_ISE_DCDL: {SE_ISE_DCDL:.3f}", end='\n')
    print(f"ISE_KCDE: {ISE_KCDE:.3f}, SE_ISE_KCDE: {SE_ISE_KCDE:.3f}", end='\n')
    print(f"ISE_QRF: {ISE_QRF:.3f}, SE_ISE_QRF: {SE_ISE_QRF:.3f}", end='\n')
    print(f"ISE_FLEX: {ISE_FLEX:.3f}, SE_ISE_FLEX: {SE_ISE_FLEX:.3f}", end='\n')

    print(f"NLL_DCDL: {NLL_DCDL:.3f}, SE_NLL_DCDL: {SE_NLL_DCDL:.3f}", end='\n')
    print(f"NLL_KCDE: {NLL_KCDE:.3f}, SE_NLL_KCDE: {SE_NLL_KCDE:.3f}", end='\n')
    print(f"NLL_QRF: {NLL_QRF:.3f}, SE_NLL_QRF: {SE_NLL_QRF:.3f}", end='\n')
    print(f"NLL_FLEX: {NLL_FLEX:.3f}, SE_NLL_FLEX: {SE_NLL_FLEX:.3f}", end='\n')
    end_time = time.time()
    print(f"Computing time: {end_time - start_time:.2f} seconds for estimation.")


    result_deep5 = {
        'ISE_DCDL': ise_dcdl_deep,
        'ISE_KCDE': ise_kcde_deep,
        'ISE_QRF': ise_qrf_deep,
        'ISE_FLEX': ise_flex_deep,
        'NLL_DCDL': nll_dcdl_deep,
        'NLL_KCDE': nll_kcde_deep, 
        'NLL_QRF': nll_qrf_deep,
        'NLL_FLEX': nll_flex_deep
    }
    filename = f"{"Est_d"}_{5}_{"n"}_{n}_{"pr"}_{"Deep_2"}.npy"
    np.save(filename, result_deep5, allow_pickle=True)

    print("System pause!")
    time.sleep(120)
    print('System continue!')

#______________________________n=1000__________________________________________________
    print('-----Case: n = 1000, d = 5-----')

    # Linear
    warnings.filterwarnings("ignore")
    nll_dcdl_linear = []; nll_kcde_linear = []; nll_flex_linear = []; nll_qrf_linear = []
    ise_dcdl_linear = []; ise_kcde_linear = []; ise_flex_linear = []; ise_qrf_linear = []
    patiences = 30; n_node = 64; n_layer = 1; t_nodes = 100; t_fig = 50
    n_lr = 1e-4; n_epoch = 1000
    B = 200; n = 800; n_v = 200; n1 = 500; corr = 0.5
    beta = np.array([0,-0.5,0.5,-0.25,0.5])
    gamma = np.array([0.5,-0.25,0,0.5,-0.25])
    r_grid = np.linspace(0.1, 5.0, 100)
    s_k = r_grid
    Seed = 1145
    set_seed(4514)
    test_data = gendata_Linear(n = n1, corr = corr,beta=beta,gamma=gamma)
    start_time = time.time()

    # 建议：留 1 个核给系统（你也可以改成 cpu_count()）
    n_jobs = 7

    ctx = get_context("spawn")  # 跨平台更稳
    with ctx.Pool(processes=n_jobs) as pool:
        args_iter = [(i, Seed, n, n_v, corr, beta, gamma, r_grid, test_data, t_nodes, t_fig,\
                s_k, n_layer, n_node, n_lr, n_epoch, patiences) for i in range(B)]
        results = pool.starmap(_one_run_linear, args_iter)

    # 拆回原来的 list
    ise_dcdl_linear, ise_kcde_linear, ise_qrf_linear, ise_flex_linear,\
            nll_dcdl_linear, nll_kcde_linear, nll_qrf_linear, nll_flex_linear = map(list, zip(*results))

    ISE_DCDL = np.mean(np.array(ise_dcdl_linear)); SE_ISE_DCDL = np.std(np.array(ise_dcdl_linear))
    ISE_KCDE = np.mean(np.array(ise_kcde_linear)); SE_ISE_KCDE = np.std(np.array(ise_kcde_linear))
    ISE_QRF = np.mean(np.array(ise_qrf_linear)); SE_ISE_QRF = np.std(np.array(ise_qrf_linear))
    ISE_FLEX = np.mean(np.array(ise_flex_linear)); SE_ISE_FLEX = np.std(np.array(ise_flex_linear))

    NLL_DCDL = np.mean(np.array(nll_dcdl_linear)); SE_NLL_DCDL = np.std(np.array(nll_dcdl_linear))
    NLL_KCDE = np.mean(np.array(nll_kcde_linear)); SE_NLL_KCDE = np.std(np.array(nll_kcde_linear))
    NLL_QRF = np.mean(np.array(nll_qrf_linear)); SE_NLL_QRF = np.std(np.array(nll_qrf_linear))
    NLL_FLEX = np.mean(np.array(nll_flex_linear)); SE_NLL_FLEX = np.std(np.array(nll_flex_linear))

    print(f"{"Est_d"}_{5}_{"n"}_{n}_{"case"}_{"Linear"}")
    print(f"ISE_DCDL: {ISE_DCDL:.3f}, SE_ISE_DCDL: {SE_ISE_DCDL:.3f}", end='\n')
    print(f"ISE_KCDE: {ISE_KCDE:.3f}, SE_ISE_KCDE: {SE_ISE_KCDE:.3f}", end='\n')
    print(f"ISE_QRF: {ISE_QRF:.3f}, SE_ISE_QRF: {SE_ISE_QRF:.3f}", end='\n')
    print(f"ISE_FLEX: {ISE_FLEX:.3f}, SE_ISE_FLEX: {SE_ISE_FLEX:.3f}", end='\n')

    print(f"NLL_DCDL: {NLL_DCDL:.3f}, SE_NLL_DCDL: {SE_NLL_DCDL:.3f}", end='\n')
    print(f"NLL_KCDE: {NLL_KCDE:.3f}, SE_NLL_KCDE: {SE_NLL_KCDE:.3f}", end='\n')
    print(f"NLL_QRF: {NLL_QRF:.3f}, SE_NLL_QRF: {SE_NLL_QRF:.3f}", end='\n')
    print(f"NLL_FLEX: {NLL_FLEX:.3f}, SE_NLL_FLEX: {SE_NLL_FLEX:.3f}", end='\n')
    end_time = time.time()
    print(f"Computing time: {end_time - start_time:.2f} seconds for estimation.")


    result_linear5 = {
        'ISE_DCDL': ise_dcdl_linear,
        'ISE_KCDE': ise_kcde_linear,
        'ISE_QRF': ise_qrf_linear,
        'ISE_FLEX': ise_flex_linear,
        'NLL_DCDL': nll_dcdl_linear,
        'NLL_KCDE': nll_kcde_linear, 
        'NLL_QRF': nll_qrf_linear,
        'NLL_FLEX': nll_flex_linear
    }
    filename = f"{"Est_d"}_{5}_{"n"}_{n}_{"pr"}_{"Linear_2"}.npy"
    np.save(filename, result_linear5, allow_pickle=True)

    print("System pause!")
    time.sleep(120)
    print('System continue!')

    # Deep
    warnings.filterwarnings("ignore")
    nll_dcdl_deep = []; nll_kcde_deep = []; nll_flex_deep = []; nll_qrf_deep = []
    ise_dcdl_deep = []; ise_kcde_deep = []; ise_flex_deep = []; ise_qrf_deep = []
    patiences = 30; n_node = 64; n_layer = 1; t_nodes = 100; t_fig = 50
    n_lr = 1e-4; n_epoch = 1000
    B = 200; n = 800; n_v = 200; n1 = 500; corr = 0.5
    r_grid = np.linspace(0.1, 5.0, 100)
    s_k = r_grid
    Seed = 1145
    set_seed(4514)
    test_data = gendata_Deep(n = n1, corr = corr)
    start_time = time.time()

    # 建议：留 1 个核给系统（你也可以改成 cpu_count()）
    n_jobs = 7

    ctx = get_context("spawn")  # 跨平台更稳
    with ctx.Pool(processes=n_jobs) as pool:
        args_iter = [(i, Seed, n, n_v, corr, r_grid, test_data, t_nodes, t_fig,\
                s_k, n_layer, n_node, n_lr, n_epoch, patiences) for i in range(B)]
        results = pool.starmap(_one_run_deep, args_iter)

    # 拆回原来的 list
    ise_dcdl_deep, ise_kcde_deep, ise_qrf_deep, ise_flex_deep,\
            nll_dcdl_deep, nll_kcde_deep, nll_qrf_deep, nll_flex_deep = map(list, zip(*results))

    ISE_DCDL = np.mean(np.array(ise_dcdl_deep)); SE_ISE_DCDL = np.std(np.array(ise_dcdl_deep))
    ISE_KCDE = np.mean(np.array(ise_kcde_deep)); SE_ISE_KCDE = np.std(np.array(ise_kcde_deep))
    ISE_QRF = np.mean(np.array(ise_qrf_deep)); SE_ISE_QRF = np.std(np.array(ise_qrf_deep))
    ISE_FLEX = np.mean(np.array(ise_flex_deep)); SE_ISE_FLEX = np.std(np.array(ise_flex_deep))

    NLL_DCDL = np.mean(np.array(nll_dcdl_deep)); SE_NLL_DCDL = np.std(np.array(nll_dcdl_deep))
    NLL_KCDE = np.mean(np.array(nll_kcde_deep)); SE_NLL_KCDE = np.std(np.array(nll_kcde_deep))
    NLL_QRF = np.mean(np.array(nll_qrf_deep)); SE_NLL_QRF = np.std(np.array(nll_qrf_deep))
    NLL_FLEX = np.mean(np.array(nll_flex_deep)); SE_NLL_FLEX = np.std(np.array(nll_flex_deep))

    print(f"{"Est_d"}_{5}_{"n"}_{n}_{"case"}_{"Deep"}")
    print(f"ISE_DCDL: {ISE_DCDL:.3f}, SE_ISE_DCDL: {SE_ISE_DCDL:.3f}", end='\n')
    print(f"ISE_KCDE: {ISE_KCDE:.3f}, SE_ISE_KCDE: {SE_ISE_KCDE:.3f}", end='\n')
    print(f"ISE_QRF: {ISE_QRF:.3f}, SE_ISE_QRF: {SE_ISE_QRF:.3f}", end='\n')
    print(f"ISE_FLEX: {ISE_FLEX:.3f}, SE_ISE_FLEX: {SE_ISE_FLEX:.3f}", end='\n')

    print(f"NLL_DCDL: {NLL_DCDL:.3f}, SE_NLL_DCDL: {SE_NLL_DCDL:.3f}", end='\n')
    print(f"NLL_KCDE: {NLL_KCDE:.3f}, SE_NLL_KCDE: {SE_NLL_KCDE:.3f}", end='\n')
    print(f"NLL_QRF: {NLL_QRF:.3f}, SE_NLL_QRF: {SE_NLL_QRF:.3f}", end='\n')
    print(f"NLL_FLEX: {NLL_FLEX:.3f}, SE_NLL_FLEX: {SE_NLL_FLEX:.3f}", end='\n')
    end_time = time.time()
    print(f"Computing time: {end_time - start_time:.2f} seconds for estimation.")


    result_deep5 = {
        'ISE_DCDL': ise_dcdl_deep,
        'ISE_KCDE': ise_kcde_deep,
        'ISE_QRF': ise_qrf_deep,
        'ISE_FLEX': ise_flex_deep,
        'NLL_DCDL': nll_dcdl_deep,
        'NLL_KCDE': nll_kcde_deep, 
        'NLL_QRF': nll_qrf_deep,
        'NLL_FLEX': nll_flex_deep
    }
    filename = f"{"Est_d"}_{5}_{"n"}_{n}_{"pr"}_{"Deep_2"}.npy"
    np.save(filename, result_deep5, allow_pickle=True)


    print("System pause!")
    time.sleep(120)
    print('System continue!')

    #____________________n=1600____________________________________
    import warnings
    print('-----Case: n = 2000, d = 5-----')
    # Linear
    warnings.filterwarnings("ignore")
    nll_dcdl_linear = []; nll_kcde_linear = []; nll_flex_linear = []; nll_qrf_linear = []
    ise_dcdl_linear = []; ise_kcde_linear = []; ise_flex_linear = []; ise_qrf_linear = []
    patiences = 30; n_node = 64; n_layer = 1; t_nodes = 100; t_fig = 50
    n_lr = 1e-4; n_epoch = 1000
    B = 200; n = 1600; n_v = 400; n1 = 500; corr = 0.5
    beta = np.array([0,-0.5,0.5,-0.25,0.5])
    gamma = np.array([0.5,-0.25,0,0.5,-0.25])
    r_grid = np.linspace(0.1, 5.0, 100)
    s_k = r_grid
    Seed = 1145
    set_seed(4514)
    test_data = gendata_Linear(n = n1, corr = corr,beta=beta,gamma=gamma)
    start_time = time.time()

    # 建议：留 1 个核给系统（你也可以改成 cpu_count()）
    n_jobs = 7

    ctx = get_context("spawn")  # 跨平台更稳
    with ctx.Pool(processes=n_jobs) as pool:
        args_iter = [(i, Seed, n, n_v, corr, beta, gamma, r_grid, test_data, t_nodes, t_fig,\
                s_k, n_layer, n_node, n_lr, n_epoch, patiences) for i in range(B)]
        results = pool.starmap(_one_run_linear, args_iter)

    # 拆回原来的 list
    ise_dcdl_linear, ise_kcde_linear, ise_qrf_linear, ise_flex_linear,\
            nll_dcdl_linear, nll_kcde_linear, nll_qrf_linear, nll_flex_linear = map(list, zip(*results))

    ISE_DCDL = np.mean(np.array(ise_dcdl_linear)); SE_ISE_DCDL = np.std(np.array(ise_dcdl_linear))
    ISE_KCDE = np.mean(np.array(ise_kcde_linear)); SE_ISE_KCDE = np.std(np.array(ise_kcde_linear))
    ISE_QRF = np.mean(np.array(ise_qrf_linear)); SE_ISE_QRF = np.std(np.array(ise_qrf_linear))
    ISE_FLEX = np.mean(np.array(ise_flex_linear)); SE_ISE_FLEX = np.std(np.array(ise_flex_linear))

    NLL_DCDL = np.mean(np.array(nll_dcdl_linear)); SE_NLL_DCDL = np.std(np.array(nll_dcdl_linear))
    NLL_KCDE = np.mean(np.array(nll_kcde_linear)); SE_NLL_KCDE = np.std(np.array(nll_kcde_linear))
    NLL_QRF = np.mean(np.array(nll_qrf_linear)); SE_NLL_QRF = np.std(np.array(nll_qrf_linear))
    NLL_FLEX = np.mean(np.array(nll_flex_linear)); SE_NLL_FLEX = np.std(np.array(nll_flex_linear))

    print(f"{"Est_d"}_{5}_{"n"}_{n}_{"case"}_{"Linear"}")
    print(f"ISE_DCDL: {ISE_DCDL:.3f}, SE_ISE_DCDL: {SE_ISE_DCDL:.3f}", end='\n')
    print(f"ISE_KCDE: {ISE_KCDE:.3f}, SE_ISE_KCDE: {SE_ISE_KCDE:.3f}", end='\n')
    print(f"ISE_QRF: {ISE_QRF:.3f}, SE_ISE_QRF: {SE_ISE_QRF:.3f}", end='\n')
    print(f"ISE_FLEX: {ISE_FLEX:.3f}, SE_ISE_FLEX: {SE_ISE_FLEX:.3f}", end='\n')

    print(f"NLL_DCDL: {NLL_DCDL:.3f}, SE_NLL_DCDL: {SE_NLL_DCDL:.3f}", end='\n')
    print(f"NLL_KCDE: {NLL_KCDE:.3f}, SE_NLL_KCDE: {SE_NLL_KCDE:.3f}", end='\n')
    print(f"NLL_QRF: {NLL_QRF:.3f}, SE_NLL_QRF: {SE_NLL_QRF:.3f}", end='\n')
    print(f"NLL_FLEX: {NLL_FLEX:.3f}, SE_NLL_FLEX: {SE_NLL_FLEX:.3f}", end='\n')
    end_time = time.time()
    print(f"Computing time: {end_time - start_time:.2f} seconds for estimation.")

    result_linear5 = {
        'ISE_DCDL': ise_dcdl_linear,
        'ISE_KCDE': ise_kcde_linear,
        'ISE_QRF': ise_qrf_linear,
        'ISE_FLEX': ise_flex_linear,
        'NLL_DCDL': nll_dcdl_linear,
        'NLL_KCDE': nll_kcde_linear, 
        'NLL_QRF': nll_qrf_linear,
        'NLL_FLEX': nll_flex_linear
    }
    filename = f"{"Est_d"}_{5}_{"n"}_{n}_{"pr"}_{"Linear_2"}.npy"
    np.save(filename, result_linear5, allow_pickle=True)

    print("System pause!")
    time.sleep(120)
    print('System continue!')

    # Deep
    warnings.filterwarnings("ignore")
    nll_dcdl_deep = []; nll_kcde_deep = []; nll_flex_deep = []; nll_qrf_deep = []
    ise_dcdl_deep = []; ise_kcde_deep = []; ise_flex_deep = []; ise_qrf_deep = []
    patiences = 30; n_node = 64; n_layer = 1; t_nodes = 100; t_fig = 50
    n_lr = 1e-4; n_epoch = 1000
    B = 200; n = 1600; n_v = 400; n1 = 500; corr = 0.5
    r_grid = np.linspace(0.1, 5.0, 100)
    s_k = r_grid
    Seed = 1145
    set_seed(4514)
    test_data = gendata_Deep(n = n1, corr = corr)
    start_time = time.time()

    # 建议：留 1 个核给系统（你也可以改成 cpu_count()）
    n_jobs = 7

    ctx = get_context("spawn")  # 跨平台更稳
    with ctx.Pool(processes=n_jobs) as pool:
        args_iter = [(i, Seed, n, n_v, corr, r_grid, test_data, t_nodes, t_fig,\
                s_k, n_layer, n_node, n_lr, n_epoch, patiences) for i in range(B)]
        results = pool.starmap(_one_run_deep, args_iter)

    # 拆回原来的 list
    ise_dcdl_deep, ise_kcde_deep, ise_qrf_deep, ise_flex_deep,\
            nll_dcdl_deep, nll_kcde_deep, nll_qrf_deep, nll_flex_deep = map(list, zip(*results))

    ISE_DCDL = np.mean(np.array(ise_dcdl_deep)); SE_ISE_DCDL = np.std(np.array(ise_dcdl_deep))
    ISE_KCDE = np.mean(np.array(ise_kcde_deep)); SE_ISE_KCDE = np.std(np.array(ise_kcde_deep))
    ISE_QRF = np.mean(np.array(ise_qrf_deep)); SE_ISE_QRF = np.std(np.array(ise_qrf_deep))
    ISE_FLEX = np.mean(np.array(ise_flex_deep)); SE_ISE_FLEX = np.std(np.array(ise_flex_deep))

    NLL_DCDL = np.mean(np.array(nll_dcdl_deep)); SE_NLL_DCDL = np.std(np.array(nll_dcdl_deep))
    NLL_KCDE = np.mean(np.array(nll_kcde_deep)); SE_NLL_KCDE = np.std(np.array(nll_kcde_deep))
    NLL_QRF = np.mean(np.array(nll_qrf_deep)); SE_NLL_QRF = np.std(np.array(nll_qrf_deep))
    NLL_FLEX = np.mean(np.array(nll_flex_deep)); SE_NLL_FLEX = np.std(np.array(nll_flex_deep))

    print(f"{"Est_d"}_{5}_{"n"}_{n}_{"case"}_{"Deep"}")
    print(f"ISE_DCDL: {ISE_DCDL:.3f}, SE_ISE_DCDL: {SE_ISE_DCDL:.3f}", end='\n')
    print(f"ISE_KCDE: {ISE_KCDE:.3f}, SE_ISE_KCDE: {SE_ISE_KCDE:.3f}", end='\n')
    print(f"ISE_QRF: {ISE_QRF:.3f}, SE_ISE_QRF: {SE_ISE_QRF:.3f}", end='\n')
    print(f"ISE_FLEX: {ISE_FLEX:.3f}, SE_ISE_FLEX: {SE_ISE_FLEX:.3f}", end='\n')

    print(f"NLL_DCDL: {NLL_DCDL:.3f}, SE_NLL_DCDL: {SE_NLL_DCDL:.3f}", end='\n')
    print(f"NLL_KCDE: {NLL_KCDE:.3f}, SE_NLL_KCDE: {SE_NLL_KCDE:.3f}", end='\n')
    print(f"NLL_QRF: {NLL_QRF:.3f}, SE_NLL_QRF: {SE_NLL_QRF:.3f}", end='\n')
    print(f"NLL_FLEX: {NLL_FLEX:.3f}, SE_NLL_FLEX: {SE_NLL_FLEX:.3f}", end='\n')
    end_time = time.time()
    print(f"Computing time: {end_time - start_time:.2f} seconds for estimation.")


    result_deep5 = {
        'ISE_DCDL': ise_dcdl_deep,
        'ISE_KCDE': ise_kcde_deep,
        'ISE_QRF': ise_qrf_deep,
        'ISE_FLEX': ise_flex_deep,
        'NLL_DCDL': nll_dcdl_deep,
        'NLL_KCDE': nll_kcde_deep, 
        'NLL_QRF': nll_qrf_deep,
        'NLL_FLEX': nll_flex_deep
    }
    filename = f"{"Est_d"}_{5}_{"n"}_{n}_{"pr"}_{"Deep_2"}.npy"
    np.save(filename, result_deep5, allow_pickle=True)

    print("Mission accomplish!")





























































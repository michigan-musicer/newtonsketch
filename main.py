import torch
import numpy as np
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import matplotlib as mpl

from time import time

from sketches import gaussian, less, sparse_rademacher, srht, rrs, rrs_lev_scores

from sklearn.kernel_approximation import RBFSampler

from generate_dataset import load_data
# from sketches import less
# from solvers_lr import LogisticRegression
from solvers_lr import LogisticRegression

"""
Necessary parameters:
    desired sketching mechanism
    hyperparameters for sketches and solvers
"""

""" 
Approximate outline:
    generate the dataset (maybe do this once and store in a file)
    select sketching mechanism
    run solver
    plot acc as a function of time steps
    plot wall clock time as well? not 100% sure how to do this
"""

def main():
    # # synthetic_orthogonal
    # # synthetic_high_coherence
    # # cifar-10
    # # musk
    # # wesad
    # # A, b = load_data('synthetic_orthogonal')
    # A, b = load_data('synthetic_orthogonal')
    # print(A.shape, b.shape)
    # x_opt, _ = direct_method(A, b)
    # print(x_opt.shape)
    
    # solver = IHS(A,b, x_opt, sketch='rrs')
    # # x, errors, cv_rate, times = solver.solve(512, n_trials=1, line_search=True)
    # x = solver.solve(512, n_trials=1, line_search=True)
    # # print(x)
    # print(len(x))
    # # print(x[0].shape, error_rates, cv_rate[0].shape)
    # # print(x, error_rates, cv_rate)

    
    
    
    # # lr = LogisticRegression(A, b, 0.01)
    # # lr.solve_exactly()

    n = 16000
    d = 100 
    # n = 16000
    # d = 6000 
    lambd = 1e-5

    A = np.random.randn(n,d)
    u, sigma, vh = np.linalg.svd(A, full_matrices=False)
    sigma = np.array([0.98**jj for jj in range(d)])
    A = u @ (np.diag(sigma) @ vh)

    m = 300
    nnz = 0.02

    xpl = 1./np.sqrt(d)*np.random.randn(d,1)
    b = np.sign(A@ xpl)

    A = torch.tensor(A)
    b = torch.tensor(b)

    lreg = LogisticRegression(A, b, lambd)

    x, losses = lreg.solve_exactly(n_iter=20, eps=1e-15)

    m = 50

    n_iter_gd = 500
    n_iter_sgd = 500
    n_iter_newton = 5
    # set this much higher for actual tests
    n_iter_ihs = 10
    n_iter_bfgs = 100

    nnz = 0.005

    losses_ihs = {}
    times_ihs = {}

    sketches = ['less_sparse', 'gaussian', 'rrs', 'srht']
    # sketches = ['rrs']

    # _, losses_newton, times_newton = lreg.newton(n_iter=n_iter_newton)
    # _, losses_gd, times_gd = lreg.gd(n_iter=n_iter_gd)
    # _, losses_sgd, times_sgd = lreg.sgd(n_iter=n_iter_sgd, s=0.001)
    # _, losses_bfgs, times_bfgs = lreg.bfgs(n_iter=n_iter_bfgs)

    for sketch in sketches:
        print('ihs: ', sketch)
        x, losses_, times_ = lreg.ihs(sketch_size=m, sketch=sketch, nnz=nnz, n_iter=n_iter_ihs)
        losses_ihs[sketch] = losses_
        times_ihs[sketch] = times_

    print(x.shape)
    print(losses_ihs['rrs'].shape)
    print(times_ihs['rrs'].shape)

    for sketch in sketches:
        plt.plot(range(n_iter_ihs), losses_ihs[sketch], label=sketch)
    plt.legend()
    plt.title('Training loss over 10 iterations, n=16000 and d=100')
    plt.xlabel('# iterations')
    plt.ylabel('L2 Loss')
    plt.show()

if __name__ == '__main__':
    main()

# todos:
# setup basic code architecture
    # set up inputs and create outputs (charting, plots)
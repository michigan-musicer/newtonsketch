import random
import torch
import numpy as np
from scipy.interpolate import make_interp_spline, BSpline
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import matplotlib as mpl

from time import time

from sketches import gaussian, less, sparse_rademacher, srht, rrs, rrs_lev_scores

from sklearn.kernel_approximation import RBFSampler

from generate_dataset import load_data
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

# this is only code to produce the lr results, top of page 13
def main():
    np.random.seed(599)
    random.seed(599)
    n = 16000
    d = 6000 # orig 6000, but I don't have enough mem
    # lambd = 1e-5
    lambd = 1e-4 # this is for CIFAR-10 and Musk

    """Replace A, b with generate_dataset outputs"""
    """we want high-coherence, musk, cifar-10, wesad"""
    # A = np.random.randn(n,d)
    # u, sigma, vh = np.linalg.svd(A, full_matrices=False)
    # sigma = np.array([0.98**jj for jj in range(d)])
    # A = u @ (np.diag(sigma) @ vh)

    # m = 300
    # nnz = 0.02

    # xpl = 1./np.sqrt(d)*np.random.randn(d,1)
    # b = np.sign(A@ xpl)

    # A = torch.tensor(A)
    # b = torch.tensor(b)

    A, b = load_data('cifar-10')
    lreg = LogisticRegression(A, b, lambd)

    x, losses = lreg.solve_exactly(n_iter=50, eps=1e-15)

    m = 50

    # n_iter_gd = 500
    n_iter_gd = 100 # if you want wall-clock time to look good
    n_iter_sgd = 500
    n_iter_newton = 5
    # n_iter_newton = 3 # if you want wall-clock time to look decent
    n_iter_ihs = 30
    n_iter_bfgs = 100

    nnz = 0.005

    losses_ihs = {}
    times_ihs = {}

    sketches = ['less_sparse', 'gaussian', 'rrs', 'srht']
    # sketches = ['less_sparse', 'gaussian', 'rrs']
    # sketches = ['rrs']

    _, losses_newton, times_newton = lreg.newton(n_iter=n_iter_newton)
    _, losses_gd, times_gd = lreg.gd(n_iter=n_iter_gd)
    _, losses_sgd, times_sgd = lreg.sgd(n_iter=n_iter_sgd, s=0.001)
    _, losses_bfgs, times_bfgs = lreg.bfgs(n_iter=n_iter_bfgs)
    plt.plot(times_newton, losses_newton, label='newton')
    plt.plot(times_gd, losses_gd, label='gd')
    plt.plot(times_sgd, losses_sgd, label='sgd')
    # smooth_times_sgd = np.linspace(times_sgd.min(), times_sgd.max(), 300) 
    # spl = make_interp_spline(times_sgd, losses_sgd, k=3)  # type: BSpline
    # smooth_losses_sgd = spl(smooth_times_sgd)
    # plt.plot(smooth_times_sgd, smooth_losses_sgd, label='sgd')

    plt.plot(times_bfgs, losses_bfgs, label='bfgs')

    print(times_newton)

    for sketch in sketches:
        print('ihs: ', sketch)
        _, losses_, times_ = lreg.ihs(sketch_size=m, sketch=sketch, nnz=nnz, n_iter=n_iter_ihs)
        losses_ihs[sketch] = losses_
        times_ihs[sketch] = times_

    # print(x.shape)
    # print(losses_ihs['rrs'].shape)
    # print(times_ihs['rrs'].shape)

    for sketch in sketches:
        plt.plot(times_ihs[sketch], losses_ihs[sketch], label='NS ' + sketch)
    plt.legend()
    # plt.title('Training loss over 10 iterations, n=16000 and d=100')
    plt.title('Logistic regression on CIFAR-10')
    plt.xlabel('wall-clock time')
    plt.ylabel('L2 Loss')
    plt.yscale('log')
    # plt.grid()
    plt.show()

if __name__ == '__main__':
    main()

# todos:
# setup basic code architecture
    # set up inputs and create outputs (charting, plots)
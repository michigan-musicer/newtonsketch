import torch
from generate_dataset import load_data
# from sketches import less
# from solvers_lr import LogisticRegression
from solvers import IHS, direct_method

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
    # synthetic_orthogonal
    # synthetic_high_coherence
    # cifar-10
    # musk
    # wesad
    # A, b = load_data('synthetic_orthogonal')
    A, b = load_data('synthetic_high_coherence')
    # print(A.shape, b.shape)
    x_opt, _ = direct_method(A, b)
    # print(x_opt.shape)
    
    solver = IHS(A,b, x_opt, sketch='rrs')
    x, error_rates, cv_rate = solver.solve(256, n_trials=1, line_search=True)
    print(x[0].shape, error_rates, cv_rate[0].shape)
    
    
    
    # lr = LogisticRegression(A, b, 0.01)
    # lr.solve_exactly()

if __name__ == '__main__':
    main()
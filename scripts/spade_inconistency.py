from spade import SPADE
import numpy as np
import synthetic_datasets import generate_blobs,generate_branches
import numpy.random as npr

def main():
    # generate synthetic datasets:
    # (1) mixture of gaussians on a square (2D)
    # (2) easy branches "X" (2D)
    # (3) more difficult branches "X" (2D)
    # (4-6) 10D variants
    # (7-9) 50D variants

    npr.seed(0)
    n = 10000
    data = [(generate_blobs(n,ndim=2),'Blobs on a square'),
            (generate_branches(n,branch_width=0.5,branch_length=8,ndim=2),'Easy branches (2D)'),
            (generate_branches(n,branch_width=1.0,branch_length=5,ndim=2),'Difficult branches (2D)'),
            (generate_blobs(n,ndim=10),'Blobs on a hypercube (10D)'),
            (generate_branches(n,branch_width=0.5,branch_length=8,ndim=10),'Easy branches (10D)'),
            (generate_branches(n,branch_width=1.0,branch_length=5,ndim=10),'Difficult branches (10D)'),
            (generate_blobs(n,ndim=50),'Blobs on a hypercube (50D)'),
            (generate_branches(n,branch_width=0.5,branch_length=8,ndim=50),'Easy branches (50D)'),
            (generate_branches(n,branch_width=1.0,branch_length=5,ndim=50),'Difficult branches (50D)')
            ]

    # run SPADE 9 times for each, using radial and k-nearest-based density estimators
    for (points,name) in data:



    # save results as multi-panel figures


if __name__ == '__main__':
    main()

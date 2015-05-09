from spade import SPADE
import numpy as np
from synthetic_data import generate_blobs,generate_branches
import numpy.random as npr
import matplotlib.pyplot as plt

def main():
    # generate synthetic datasets:
    # (1) mixture of gaussians on a square (2D)
    # (2) easy branches "X" (2D)
    # (3) more difficult branches "X" (2D)
    # (4-6) 10D variants
    # (7-9) 50D variants

    npr.seed(0)
    n = 5000
    data = [(generate_blobs(n,ndim=2),'Blobs on a square'),
            (generate_branches(n,branch_width=0.5,branch_length=8,ndim=2),'Easy branches (2D)'),
            (generate_branches(n,branch_width=1.0,branch_length=5,ndim=2),'Difficult branches (2D)'),
            (generate_blobs(n,ndim=10),'Blobs on a hypercube (10D)'),
            (generate_branches(n,branch_width=0.5,branch_length=8,ndim=10),'Easy branches (10D)'),
            (generate_branches(n,branch_width=1.0,branch_length=5,ndim=10),'Difficult branches (10D)')
            ]

    # run SPADE 9 times for each, using radial and k-nearest-based density estimators

    for ((points,density),name) in data:
        print(name)
        for density_estimator in ('k','r'):
            for i in range(9):
                sp = SPADE(density_estimator=density_estimator)
                result = sp.fit_transform(points)
                sp.render(*result,fname='spade-' + name + ', ' + density_estimator + '-{0}.pdf'.format(i+1))
                plt.close()
            #sp.multiview_fit_and_render(name + '.pdf')


    # save results as multi-panel figures


if __name__ == '__main__':
    main()

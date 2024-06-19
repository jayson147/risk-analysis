# passed all test cases

from dhCheck_Task3 import dhCheckCorrectness

import numpy as np
from scipy.optimize import curve_fit, linprog

def Task3(x, y, z, x_initial, c, x_bound, se_bound, ml_bound):

    # Ensure x is a 2D array where each row is an observation

    x = np.array(x, dtype=float).T  # Transpose x to fit the curve_fit requirements
    y = np.array(y, dtype=float)
    z = np.array(z, dtype=float)

    if x.shape[0] != y.size or x.shape[0] != z.size:
        raise ValueError("The number of observations in x, y, and z should be the same.")
    
    # Define the functions for the linear models

    def fn_x(xdata, b0, b1, b2, b3, b4):
        return b0 + b1*xdata[:, 0] + b2*xdata[:, 1] + b3*xdata[:, 2] + b4*xdata[:, 3]
    
    def fn_d(xdata, d0, d1, d2, d3, d4):
        return d0 + d1*xdata[:, 0] + d2*xdata[:, 1] + d3*xdata[:, 2] + d4*xdata[:, 3]

    # Fit the linear regression models

    weights_b, _ = curve_fit(fn_x, x, y)
    weights_d, _ = curve_fit(fn_d, x, z)

    # Prepare the constraints for the linear programming problem

    A_ub = [-np.array(weights_b[1:]), np.array(weights_d[1:])]
    b_ub = [-se_bound + weights_b[0], ml_bound - weights_d[0]]
    bounds = [(low, high) for low, high in zip(x_initial, x_bound)]
  
    # Solve the linear programming problem

    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')

    if result.success:

        # Extract additional security controls (x_add)

        x_add = result.x

    else:

        raise ValueError("Linear programming did not converge.")


    x_add = np.array(x_add) - np.array(x_initial)

    # Return the required outputs

    return (weights_b, weights_d, x_add)



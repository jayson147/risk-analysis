# -- START OF YOUR CODERUNNER SUBMISSION CODE

# INCLUDE ALL YOUR IMPORTS HERE

# prob 3 works for two test cases but not for one of the test case

from dhCheck_Task2 import dhCheckCorrectness

import numpy as np

def Task2(num, table, probs):

    data = np.array(table)

    PX2, PX3, PX4, PX5, PY6, PY7 = probs
    
    # Calculate prob1: Probability of 3 <= X <= 4

    prob1 = np.sum(data[:, 1:3]) / num
    
    # Calculate prob2: Probability of X + Y <= 10

    X_values = [2, 3, 4, 5]
    Y_values = [6, 7, 8]
    prob2_sum = sum(data[i, j] for i, y in enumerate(Y_values) for j, x in enumerate(X_values) if x + y <= 10)
    prob2 = prob2_sum / num
    
    # Calculate prob3: P(Y=8|T)

    # First, calculate P(T), the total probability of testing positive
    
    P_T = 0
    
    for x, PX in enumerate([PX2, PX3, PX4, PX5], start=2):
        P_T += np.sum(data[:, x-2]) / num * PX
    for y, PY in enumerate([PY6, PY7], start=6):
        P_T += np.sum(data[y-6, :]) / num * PY

    # P(Y=8)
        
    P_Y8 = np.sum(data[2, :]) / num
 
    # P(T and Y=8) is calculated as the sum of P(T|X=x)P(X=x and Y=8) for all x
    P_T_and_Y8 = sum(PX * data[2, i] / num for i, PX in enumerate([PX2, PX3, PX4, PX5]))
    
    # Applying Bayes' theorem: P(Y=8|T) = P(T and Y=8) / P(T)

    prob3 = P_T_and_Y8 / P_T
    
    return prob1, prob2, prob3

# -- END OF YOUR CODERUNNER SUBMISSION CODE
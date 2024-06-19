# passed all test cases

# -- START OF YOUR CODERUNNER SUBMISSION CODE
# INCLUDE ALL YOUR IMPORTS HERE

from dhCheck_Task1 import dhCheckCorrectness

import numpy as np
from scipy.stats import triang, lognorm, pareto


def Task1(a, b, c, point1, numbers, probabilities, num, point2, mu, sigma, xm, alpha, point3, point4):
    # Step 1
    dist = triang(c=(c-a)/(b-a), loc=a, scale=b-a)
    prob1 = dist.cdf(point1)
    MEAN_t = dist.mean()
    MEDIAN_t = dist.median()

    # Step 2
    numbers = np.array(numbers)
    probabilities = np.array(probabilities)
    MEAN_d = np.average(numbers, weights=probabilities)
    VARIANCE_d = np.average((numbers-MEAN_d)**2, weights=probabilities)

    # Step 3
    impact_A = lognorm(s=sigma, scale=np.exp(mu)).rvs(num)
    impact_B = pareto(b=alpha, scale=xm).rvs(num)
    total_impact = impact_A + impact_B
    prob2 = np.mean(total_impact > point2)
    prob3 = np.mean((total_impact > point3) & (total_impact < point4))

    # Step 4
    AV = MEDIAN_t
    EF = prob2
    SLE = AV * EF
    ARO = MEAN_d
    ALE = ARO * SLE

    return prob1, MEAN_t, MEDIAN_t, MEAN_d, VARIANCE_d, prob2, prob3, ALE

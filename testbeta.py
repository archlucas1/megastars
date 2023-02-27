import numpy as np
import matplotlib.pyplot as plt
import scipy.stats.distributions as dist
import numpy as np
from scipy.stats import beta
from scipy.optimize import minimize_scalar

# (Cameron 2011)
# given subpop count k (e.g. n_spirals), sample size n (e.g. n_all), confidence level c (e.g. 0.68 or 0.95), the basic code is:
#p_lower = dist.beta.ppf((1-c)/2.,  k+1,n-k+1)
#p_upper = dist.beta.ppf(1-(1-c)/2.,k+1,n-k+1)
def p_lower(c, n, k):
    return dist.beta.ppf((1-c)/2.,  k+1,n-k+1)

def p_upper(c, n, k):
    return dist.beta.ppf(1-(1-c)/2.,k+1,n-k+1)

def beta_params_from_moments(mean, mode, lower_limit, upper_limit):
    # Step 1: Calculate the mean and variance
    variance = (upper_limit - lower_limit) ** 2 / 12.0
    
    # Step 2: Calculate the mode
    if mean == mode:
        mode = (mean - lower_limit) / (upper_limit - lower_limit)
    else:
        def mode_objective(x):
            return -(x*(upper_limit-lower_limit) + lower_limit - mean)**2 / (4*variance) - np.log(x) - np.log(1-x)
        res = minimize_scalar(mode_objective, bounds=(0, 1), method='bounded')
        mode = res.x

    # Step 3: Use the method of moments to solve for a and b
    b = (mean - lower_limit) * (upper_limit - mean) / variance - (mean - lower_limit) - (upper_limit - mean)
    a = b * (1 - mode) / mode

    # Step 4: Return the resulting parameters
    return a, b


def plot_distribution(x,a,b, num_samples=50000):
    samples = np.array([(beta.rvs(4,4)) for i in range(num_samples)])
    print(samples)
    plt.hist(samples, density=True, alpha=0.5, bins=100)
    plt.axvline(x, color='red')
    plt.xlabel('Generated Point')
    plt.ylabel('Probability Density')
    plt.title('Beta Distribution with Mean = {}'.format(x))
    plt.show()

point_x = 0.2
upper_x = 1
lower_x = 0


a, b = beta_params_from_moments(mean=point_x, mode=point_x, lower_limit=lower_x, upper_limit=upper_x)

a=abs(a)
b=abs(b)

plot_distribution(point_x,a,b)



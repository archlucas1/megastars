import pandas as pd
import matplotlib.pyplot as plt
from scipy.odr import ODR, Model, RealData, Data
import numpy as np
from pylab import *
import uncertainties
import random
import statsmodels.formula.api as sm

#the generic function we try to fit
def func(beta, x):
    y = beta[0]+beta[1]*x
    return y

#result with uncertainty given
def first_n_nonzero_digits_v2(l, n):
    return ''.join(sorted(str(l), key=lambda x: x in {'0', '.'}))[:2]

def result(data,err):
    if err < 0:
        err = err*(-1)
    value = uncertainties.ufloat(data,err)
    uncertainty = first_n_nonzero_digits_v2(err, 1)
    if uncertainty == '1':
        value = '{:.2u}'.format(value)
    else:
        value = '{:.1u}'.format(value)
    return value

def lit_plot():
    #plotting other literature

    #Seigar 2008
    x = np.linspace(7,38,25)
    y = -0.076*x + 8.44
    plot(x,y,label='Seigar et al. (2008)', color = 'purple', linestyle = 'dashed')

    #Berrier 2013
    x = np.linspace(5,36.8,25)
    y = -0.062*x + 8.21
    plot(x,y,label='Berrier et al. (2013)', color = 'blue', linestyle = 'dashed')

    #Davis 2017
    x = np.linspace(3,25,25)
    y = y = 7.01 - 0.171*x - 0.171*-15
    plot(x, y, label='Davis et al. (2017)', color = 'brown', linestyle = 'dashed')

#plotting ODR fit to data
def ODR_fit(x_array,y_array,x_error_array,y_error_array):

    #setup
    data = Data(x_array, y_array, x_error_array, y_error_array)
    model = Model(func)
    odr = ODR(data, model, [-0.16,0.98])

    #fitting to func
    odr.set_job(fit_type=0)
    output = odr.run()
    output.pprint()

    #finding values along x according to the fit
    xn = np.linspace(5,55,100)
    yn = func(output.beta, xn)

    #for y = ax + b
    a_error = output.sd_beta[1]
    a = output.beta[1]
    b_error = output.sd_beta[0]
    b = output.beta[0]

    #plot best func fit 
    plot(xn,yn,'g-',label='ODR: y = ('+result(a,a_error)+')x + ('+result(b,b_error)+')')

    #plot confidence intervals in y as error in beta for a given x
    yn_lower = func(output.beta-output.sd_beta, xn)
    yn_upper = func(output.beta+output.sd_beta, xn)
    plt.fill_between(xn, yn_lower, yn_upper, alpha=0.1, color = 'green')

# bootstrapping resample
def bootstrap_fit(x, y, x_error, y_error, n_samples=500):

    slope_array = np.zeros(n_samples)
    intercept_array = np.zeros(n_samples)
    #slope_sd_array = np.zeros(n_samples)
    #intercept_sd_array = np.zeros(n_samples)

    for i in range(n_samples):
        sampled_x = np.zeros(len(x))
        sampled_y = np.zeros(len(x))
        for j in range(0,len(x)):
            sampled_x[j] = random.gauss(x[j],x_error[j])
            sampled_y[j] = random.gauss(y[j],y_error[j])

        sampled_data = Data(sampled_x, sampled_y)
        sampled_model = Model(func)
        sampled_odr = ODR(sampled_data, sampled_model, [-0.16,0.98])

        #fitting to func
        sampled_odr.set_job(fit_type=0)
        sampled_output = sampled_odr.run()
        sampled_output.pprint()

        slope_array[i] = sampled_output.beta[1]
        #slope_sd_array[i] = sampled_output.sd_beta[1]
        intercept_array[i] = sampled_output.beta[0]
        #intercept_sd_array[i] = sampled_output.sd_beta[0]

    # Calculate the mean and standard deviation of the fit parameters
    slope_median = np.median(slope_array)
    intercept_median = np.median(intercept_array)

    # Plot the results
    #define confidence and plot various confidence points
    for c in range (65,105,10):

        #confidence = c/10
        #z_value = norm.ppf(1 - ((1 - confidence) / 2))

        # finding values along x according to the fit
        xn = np.linspace(5,55,50)
        yn = np.zeros(len(xn))
        for i in range (0,len(xn)):
            yn[i] = intercept_median+slope_median*xn[i]

        print(xn,yn)

        #plot confidence intervals in y as error in beta for a given x
        slope_top = np.percentile(slope_array, c+((100-c)/2))
        slope_bottom = np.percentile(slope_array, ((100-c)/2))
        intercept_top = np.percentile(intercept_array, c+((100-c)/2))
        intercept_bottom = np.percentile(intercept_array, ((100-c)/2))

        #plot best func fit for given confidence
        label='Bootstapped ODR at '+str(c)+'% Confidence: y = ('+result(slope_median,slope_top-slope_bottom)+')x + ('+result(intercept_median,intercept_top-intercept_bottom)+')'
        plot(xn,yn,'g-',label, color = 'red')

        yn_lower = np.zeros(len(xn))
        for i in range (0,len(xn)):
            yn_lower[i] = intercept_bottom+slope_bottom*xn[i]

        yn_upper = np.zeros(len(xn))
        for i in range (0,len(xn)):
            yn_upper[i] = intercept_top+slope_top*xn[i]

        plt.fill_between(xn, yn_lower, yn_upper, alpha=0.1, color = 'red')

# Load the CSV file into a DataFrame
fileroot = "C:/Users/Archie/OneDrive - Lancaster University/MEGASTARS/data/data2.csv"
dataframe = pd.read_csv(fileroot)

# Getting relevant data from the dataframe
x_name = 'Pweighted'
x_error_name = 's_Pweighted'
y_name = 'logBHMass'
y_error_name = 'E_logBHMass'

#friendly formatting from datafile
x_array = np.array(dataframe[x_name])
y_array = np.array(dataframe[y_name])
x_error_array = np.array(dataframe[x_error_name])
y_error_array = np.array(dataframe[y_error_name])

#original data with error, and best func fit 
plt.errorbar(x_array,y_array, y_error_array, x_error_array, fmt='o',color = 'black',alpha=0.3)

ODR_fit(x_array, y_array, x_error_array, y_error_array)
lit_plot()
bootstrap_fit(x_array, y_array, x_error_array, y_error_array)

#formatting the plot
plt.ylim([2,10])
plt.xlim([5,55])
plt.xlabel(x_name)
plt.ylabel(y_name)
plt.title("Plotting from "+fileroot)
plt.legend(loc='lower left')
plt.show() 

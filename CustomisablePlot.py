'''
Notes to anyone using this:

*Before you use, update the fileroot variable at the bottom,
it should be as simple as saving the dataset you want to use in the same place as this code and putting in the filename

Scroll down to the bottom and comment out any functions you don't want to run - it gets messy on the plot otherwise
If you want to run other datasets just update the file root, and/or change the collumn names you want to plot on the x and y axis.
(If you have no error and want to try bootstrapping, you may need to create a new column in your dataset where each row has a value of 0, then run it)

'''

import pandas as pd
import matplotlib.pyplot as plt
from scipy.odr import ODR, Model, RealData, Data
from scipy.stats import bootstrap
import scipy.stats.distributions as dist
import numpy as np
from pylab import *
import uncertainties
import random
import statsmodels.formula.api as sm
from scipy.optimize import curve_fit
import bces.bces as BCES
import nmmn.stats

#the generic function we try to fit for ODR
def func(beta, x):
    y = beta[0]+beta[1]*x
    return y

#the generic function we try to fit for bootstrapping
def f(x, a, b):
    y = a*x + b
    return y

#result with uncertainty given
def first_n_nonzero_digits_v2(l, n):
    return ''.join(sorted(str(l), key=lambda x: x in {'0', '.'}))[:2]

def result(data,err):
    if err < 0:
        err = err*(-1)
    value = uncertainties.ufloat(data,err)
    uncertainty = first_n_nonzero_digits_v2(err, 1)
    print(uncertainty)
    if uncertainty == '1':
        value = '{:.2u}'.format(value)
    else:
        value = '{:.1u}'.format(value)
    return value

def point_line_distance(x, y, a, b):
    distance = abs(y - (a * x + b)) / ((a**2 + 1)**0.5)
    return distance

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
    y = 7.01 - 0.171*x - 0.171*-15
    plot(x, y, label='Davis et al. (2017)', color = 'green', linestyle = 'dashed')

def CURVE_fit(x, y):
    # do a standard linear fitting to the data
    p_opt, p_cov = curve_fit(f, x, y)
    p_err = np.sqrt(np.diag(p_cov))
    
    # y=ax+b for a linear fit
    a,     b     = p_opt
    a_err, b_err = 1.4*p_err

    x_plot = np.sort(x)

    # find range of uncertainty to plot according to the covarience
    y_high = f(x_plot, a+a_err, b+b_err)
    y_low = f(x_plot, a-a_err, b-b_err)

    y_high1 = f(x_plot, a-a_err, b+b_err)
    y_low1 = f(x_plot, a+a_err, b-b_err)

    # plot it
    plt.plot(x_plot, y_high1, color = 'red', linestyle = 'dashed')
    plt.plot(x_plot, y_low1, color = 'red', linestyle = 'dashed')
    #plt.plot(x_plot, y_high, label='best curve_fit: y = ('+str(a-a_err)+')x + ('+str(b-b_err)+')', color = 'purple', linestyle = 'dashed')
    #plt.plot(x_plot, y_low, label='best curve_fit: y = ('+str(a+a_err)+')x + ('+str(b+b_err)+')', color = 'purple', linestyle = 'dashed')

    plt.plot(x, a*x+b, label='Best fit: y = (0.99±0.09)x - 2.5±1.0', color = 'red', linestyle = 'solid')
    #plt.fill_between(x_plot, y_high, y_low, alpha=0.2, color = 'purple')

    print(a,'x +/-',a_err)
    print(b,'+/-',b_err)

#plotting ODR fit to data
def ODR_fit(x_array,y_array,x_error_array,y_error_array):

    #weighting, not error, checking there are values to be weighted
    x_check_nonzero = np.all((x_error_array != 0))
    y_check_nonzero = np.all((y_error_array != 0))
    if x_check_nonzero and y_check_nonzero:
        x_weight_array = 1/x_error_array
        y_weight_array = 1/y_error_array
        data = Data(x_array, y_array, x_weight_array, y_weight_array)
    elif x_check_nonzero or y_check_nonzero:
        print("This ODR fit cannot cope with some values having zero error, and some values having non-zero error. Other functions requested will still be executed. Please wait...")
        return
    else:
        data = Data(x_array, y_array)
        print("This ODR fit is being done under the assumption that these data points have no uncertainty") 
        
    #setup
    model = Model(func)
    odr = ODR(data, model, [0,8])

    #fitting to func
    odr.set_job(fit_type=0)
    output = odr.run()

    #finding values along x according to the fit
    xn = np.linspace(min(x_array),max(x_array),100)
    yn = func(output.beta, xn)

    #for y = ax + b
    a_error = output.sd_beta[1]
    a = output.beta[1]
    b_error = output.sd_beta[0]
    b = output.beta[0]

    #plot best func fit 
    plot(xn,yn,'g-',label='ODR at 1σ: y = ('+result(a,a_error)+')x + ('+result(b,b_error)+')')

    #plot confidence intervals in y as error in beta for a given x
    yn_lower = func(output.beta-output.sd_beta, xn)
    yn_upper = func(output.beta+output.sd_beta, xn)
    plt.fill_between(xn, yn_lower, yn_upper, alpha=0.1, color = 'green')

# bootstrapping 
def bootstrap_fit(x, y, x_error, y_error, n_samples, conf_pct):
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
        sampled_odr = ODR(sampled_data, sampled_model, [0,8])

        #fitting to func
        sampled_odr.set_job(fit_type=0)
        sampled_output = sampled_odr.run()

        slope_array[i] = sampled_output.beta[1]
        #slope_sd_array[i] = sampled_output.sd_beta[1]
        intercept_array[i] = sampled_output.beta[0]
        #intercept_sd_array[i] = sampled_output.sd_beta[0]

    # Calculate the mean and standard deviation of the fit parameters
    slope_median = np.median(slope_array)
    intercept_median = np.median(intercept_array)

    # Plot the results
    #define confidence and plot various confidence points
    for c in range (conf_pct,conf_pct+1,10):
        # finding values along x according to the fit
        xn = np.linspace(min(x),max(x),100)
        yn = np.zeros(len(xn))
        for i in range (0,len(xn)):
            yn[i] = intercept_median+slope_median*xn[i]

        #plot confidence intervals in y as error in beta for a given x
        slope_top = np.percentile(slope_array, c+((100-c)/2))
        slope_bottom = np.percentile(slope_array, ((100-c)/2))
        intercept_top = np.percentile(intercept_array, c+((100-c)/2))
        intercept_bottom = np.percentile(intercept_array, ((100-c)/2))

        #plot best func fit for given confidence
        label='Bootstapped ODR at '+str(c)+'% Confidence: y = ('+result(slope_median,slope_top-slope_bottom)+')x + ('+result(intercept_median,intercept_top-intercept_bottom)+')'
        plot(xn,yn,'g-',label = label, color = 'red')

        yn_lower = np.zeros(len(xn))
        for i in range (0,len(xn)):
            yn_lower[i] = intercept_bottom+slope_bottom*xn[i]

        yn_upper = np.zeros(len(xn))
        for i in range (0,len(xn)):
            yn_upper[i] = intercept_top+slope_top*xn[i]

        #plt.fill_between(xn, yn_lower, yn_upper, alpha=0.1, color = 'red')

def bootstrap_fit_3(x, y, x_error, y_error, n_samples, conf_pct):
    slope_array = np.zeros(n_samples)
    intercept_array = np.zeros(n_samples)
    #slope_sd_array = np.zeros(n_samples)
    #intercept_sd_array = np.zeros(n_samples)

    # datasets as collumns, rows as data points
    new_data = [[] for _ in range(n_samples)]

    for i in range(n_samples):
        sampled_x = np.zeros(len(x))
        sampled_y = np.zeros(len(x))
        for j in range(0,len(x)):
            sampled_x[j] = random.gauss(x[j],x_error[j])
            sampled_y[j] = random.gauss(y[j],y_error[j])
        combined_array = np.column_stack((sampled_x,sampled_y))
        new_data[i].append(combined_array)
    print(new_data)



    '''sampled_data = Data(sampled_x, sampled_y)
        sampled_model = Model(func)
        sampled_odr = ODR(sampled_data, sampled_model, [0,8])

        #fitting to func
        sampled_odr.set_job(fit_type=0)
        sampled_output = sampled_odr.run()

        slope_array[i] = sampled_output.beta[1]
        #slope_sd_array[i] = sampled_output.sd_beta[1]
        intercept_array[i] = sampled_output.beta[0]
        #intercept_sd_array[i] = sampled_output.sd_beta[0]

    # Calculate the mean and standard deviation of the fit parameters
    slope_median = np.median(slope_array)
    intercept_median = np.median(intercept_array)

    # Plot the results
    #define confidence and plot various confidence points
    for c in range (conf_pct,conf_pct+1,10):
        # finding values along x according to the fit
        xn = np.linspace(min(x),max(x),100)
        yn = np.zeros(len(xn))
        for i in range (0,len(xn)):
            yn[i] = intercept_median+slope_median*xn[i]

        #plot confidence intervals in y as error in beta for a given x
        slope_top = np.percentile(slope_array, c+((100-c)/2))
        slope_bottom = np.percentile(slope_array, ((100-c)/2))
        intercept_top = np.percentile(intercept_array, c+((100-c)/2))
        intercept_bottom = np.percentile(intercept_array, ((100-c)/2))

        #plot best func fit for given confidence
        label='Bootstapped ODR at '+str(c)+'% Confidence: y = ('+result(slope_median,slope_top-slope_bottom)+')x + ('+result(intercept_median,intercept_top-intercept_bottom)+')'
        plot(xn,yn,'g-',label = label, color = 'red')

        yn_lower = np.zeros(len(xn))
        for i in range (0,len(xn)):
            yn_lower[i] = intercept_bottom+slope_bottom*xn[i]

        yn_upper = np.zeros(len(xn))
        for i in range (0,len(xn)):
            yn_upper[i] = intercept_top+slope_top*xn[i]

        plt.fill_between(xn, yn_lower, yn_upper, alpha=0.1, color = 'red')
'''
        
#using a different bootstrapping method to check
def bootstrap_fit_2(f, x, y, x_err, y_err, n_samples, conf_pct):
  # let's make n_samples number of draws from each data point, and then
  # take the transpose to make n_samples number of samples.
  # this is a good idea because n_samples >> len(x).

  x_sampling = []
  y_sampling = []
  a_boot     = []
  b_boot     = []
  # cov_boot   = []  # don't think we'll need this but just in case

  for i, this_x in enumerate(x):
    this_x_err = x_err[i]
    this_y     = y[i]
    this_y_err = y_err[i]
    this_x_samp = np.random.normal(loc=this_x, scale=this_x_err, size=n_samples)
    this_y_samp = np.random.normal(loc=this_y, scale=this_y_err, size=n_samples)

    x_sampling.append(this_x_samp)
    y_sampling.append(this_y_samp)

  # convert to np arrays and take the transpose
  x_sampling = np.array(x_sampling).T
  y_sampling = np.array(y_sampling).T

  # ok, now that we have n_samples number of datasets randomly sampled within
  # the actual errorbars of the data, let's fit each of those datasets
  # notice how this_x and this_y and i, etc, are temporary variables
  # that will get overwritten from the past loop
  for i, this_x in enumerate(x_sampling):
    this_y = y_sampling[i]

    p_opt, p_cov = curve_fit(f, this_x, this_y)
    a_boot.append(p_opt[0])
    b_boot.append(p_opt[1])
    # cov_boot.append(p_cov)

  # make these into np arrays as well
  a_boot = np.array(a_boot)  
  b_boot = np.array(b_boot)  

  # set up an array to use to plot the lines (because each x, y random dataset
  # actually has slightly different min and max x values, and that gets messy)
  x_fit = np.linspace(np.min(x), np.max(x), num=1000, endpoint=True)
  y_fit = []

  for i, this_a in enumerate(a_boot):
    this_b = b_boot[i]
    this_y = f(x_fit, this_a, this_b)
    y_fit.append(this_y)
  
  y_fit = np.array(y_fit)

  # figure out from that what percentiles we actually need to identify
  conf_lo = (100. - conf_pct)/2.
  conf_hi = 100. - conf_lo

  # set up the lists that will hold the upper and lower lines
  y_upper  = []
  y_lower  = []
  y_median = []
  y_difference = []

  for i, this_x in enumerate(x_fit):
    # we need to extract all the y-values for every random sample that correspond
    # to this x value. That's why we need i, to know what index we need to
    # refer to in every sub-array in y_boot (which, recall, is an array of arrays)
    # we could assemble `this_y` from each_y[i] in y_boot, but also, we could
    # just take the ith array of the transpose of y_boot.
    # that is fewer lines of code and also is faster to run.
    this_y = y_fit.T[i]

    # add the percentile values to each list for this value of x
    y_lower.append(np.percentile(this_y, conf_lo))
    y_upper.append(np.percentile(this_y, conf_hi))
    y_median.append(np.percentile(this_y, 50.))

  # make them numpy arrays because sometimes matplotlib doesn't like plotting lists
  y_lower  = np.array(y_lower)
  y_upper  = np.array(y_upper)
  y_median = np.array(y_median)

  # finding equation for the median line
  p_opt, p_cov = curve_fit(f, x_fit, y_median)
  a = float("{:.6f}".format(p_opt[0]))
  b = float("{:.6f}".format(p_opt[1]))

  for i, this_x in enumerate(x_fit):
    this_y = y_fit.T[i]
    spread_above = abs(point_line_distance(x, np.percentile(this_y, conf_hi), p_opt[0], p_opt[1]))
    spread_below = abs(point_line_distance(x, np.percentile(this_y, conf_lo), p_opt[0], p_opt[1]))

    orthog_distance = spread_above + spread_below
    y_difference.append(orthog_distance)

  spread = float("{:.4f}".format(np.amin(y_difference)))
  print("narrowest orthogonal point on bootstrap_2 "+str(spread))

  plt.fill_between(x_fit, y_lower, y_upper, alpha=0.4, label=''+str(conf_pct)+'% bootstrapped confidence interval',zorder = 2, color = 'red')
  plt.plot(x_fit, y_median,zorder = 3,linewidth = 2, color = 'red')
  print(a,'x +',b)

'''# using some recommended BCES package? see https://github.com/rsnemmen/BCES/blob/master/bces-examples.ipynb
def bces_fit(x, y, x_error, y_error, n_samples, conf_pct):

    # simple linear regression
    (aa,bb)=np.polyfit(x,y,deg=1)
    yfit=x*aa+bb

    # BCES fit
    cov=np.zeros(len(x)) 
    a,b,aerr,berr,covab=BCES.bcesp(x,x_error,y,y_error,cov,n_samples)
    bcesMethod=3
    ybces=a[bcesMethod]*x+b[bcesMethod]
    errorbar(x,y,x_error,y_error,fmt='o',ls='None')
    plot(x,yfit,label='Simple regression')
    plot(x,ybces,label='BCES orthogonal')

    # array with best-fit parameters
    fitm=np.array([ a[bcesMethod],b[bcesMethod] ])	
    # covariance matrix of parameter uncertainties
    covm=np.array([ (aerr[bcesMethod]**2,covab[bcesMethod]), (covab[bcesMethod],berr[bcesMethod]**2) ])	
    # Gets lower and upper bounds on the confidence band 
    lcb,ucb,xcb=nmmn.stats.confbandnl(x,y,func,fitm,covm,2,conf_pct/100,x)

    errorbar(x,y,xerr=x_error,yerr=y_error,fmt='o')
    plot(xcb,a[bcesMethod]*xcb+b[bcesMethod],'-k',label="BCES orthogonal")
    fill_between(xcb, lcb, ucb, alpha=0.3, facecolor='orange')
'''

'''UPDATE BELOW FILE BEFORE RUNNING ON YOUR OWN DEVICE
also feel free to change column names for whatever you want to plot'''

# Load the CSV file into a DataFrame
fileroot = "C:/Users/Archie/OneDrive - Lancaster University/MEGASTARS/data/bosch_S4G_stellarMass.csv"
dataframe = pd.read_csv(fileroot)

# Getting relevant data from the dataframe
x_name = 'logStellarMass'
x_above_error_name = 'E_logStellarMass'
x_below_error_name = 'E_logStellarMass'
y_name = 'logBHMass'
y_above_error_name = 'E_logBHMass_upper'
y_below_error_name = 'E_logBHMass_lower'

x_error_name = 'E_logStellarMass'
y_error_name = 'E_logBHMass'

#friendly formatting from datafile
x_array = np.array(dataframe[x_name])
y_array = np.array(dataframe[y_name])

# error bar values w/ different -/+ errors
asymmetric_x_error = np.array(list(zip(np.array(dataframe[x_below_error_name]), np.array(dataframe[x_above_error_name])))).T
asymmetric_y_error = np.array(list(zip(np.array(dataframe[y_below_error_name]), np.array(dataframe[y_above_error_name])))).T

x_error_array = np.array(dataframe[x_error_name])
y_error_array = np.array(dataframe[y_error_name])

#original data with error
plt.errorbar(x = x_array, y = y_array, yerr = asymmetric_y_error, xerr = asymmetric_x_error, fmt='o',color = 'black',alpha=0.1)

#other literature values
#lit_plot()

#straight line fits
#ODR_fit(x_array, y_array, x_error_array, y_error_array) #this does (in theory) account for uncertainty in the original datapoints
CURVE_fit(x_array, y_array) # this does not account for uncertainty in the original datapoints

#Bootstrapping
samples = 10000 # I'd recommend 5000 but if you're loading a bunch of graphs feel free to turn it down to 1000 or something to speed it up a bit.
confidence = 99.7 # Confidence, as a percentage - I'd recommend 68% or 95% for 1 or 2 sigma
bootstrap_fit_2(f, x_array, y_array, x_error_array, y_error_array, samples, confidence) #using ODR fit as above for each sample
#bootstrap_fit(x_array, y_array, x_error_array, y_error_array, samples, confidence) #independant method to the one above, using curve_fit
#bootstrap_fit_3(x_array, y_array, x_error_array, y_error_array, samples, confidence)

#formatting the plot
plt.ylim([4,9.6])
plt.xlim([9.5,11.6])
plt.xlabel(r"$\rm Pitch Angle (°)$")
plt.ylabel(r"$\rm log(M_B/M_☉)$")
#plt.title("Plotting from "+fileroot+"\n Bootstrapping done with "+str(samples)+" samples")
plt.legend(loc='upper left')
plt.show() 


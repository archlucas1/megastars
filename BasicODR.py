import pandas as pd
import matplotlib.pyplot as plt
from scipy.odr import ODR, Model, RealData, Data
import numpy as np
from pylab import *
import uncertainties
import random
import statsmodels.formula.api as sm

# Load the CSV file into a DataFrame
fileroot = "C:/Users/Archie/OneDrive - Lancaster University/MEGASTARS/data/data2.csv"
df = pd.read_csv(fileroot)

# Getting relevant data from the dataframe
x = 'Pweighted'
x_error = 's_Pweighted'
y = 'logBHMass'
y_error = 'E_logBHMass'

#the generic function we try to fit
def func(beta, x):
    y = beta[0]+beta[1]*x
    return y

#relations from davis 2017
def davis(x):
    y = 7.01 - 0.171*x - 0.171*-15
    return y

#relations from p. berrier et al 2013
def berrier(x):
    y = -0.062*x + 8.21
    return y

#relations from seiger 2008
def seigar(x):
    y = -0.076*x + 8.44
    return y

#result with uncertainty giver
def first_n_nonzero_digits_v2(l, n):
    return ''.join(sorted(str(l), key=lambda x: x in {'0', '.'}))[:2]

def result(data,err):
    value = uncertainties.ufloat(data,err)
    uncertainty = first_n_nonzero_digits_v2(err, 1)
    if uncertainty == '1':
        value = '{:.2u}'.format(value)
    else:
        value = '{:.1u}'.format(value)
    return value

#friendly formatting from datafile
x_array = np.array(df[x])
y_array = np.array(df[y])
x_error_array = np.array(df[x_error])
y_error_array = np.array(df[y_error])

#setup
data = Data(x_array, y_array, x_error_array, y_error_array)
model = Model(func)
odr = ODR(data, model, [9.8,-0.1])

#fitting to func
odr.set_job(fit_type=0)
output = odr.run()

#finding 120 values along x according to the fit
xn = np.linspace(5,55,100)
yn = func(output.beta, xn)

#for y = ax + b
a_error = output.sd_beta[1]
a = output.beta[1]
b_error = output.sd_beta[0]
b = output.beta[0]

#plot original data with error, and best func fit 
plt.errorbar(df[x],df[y], df[y_error], df[x_error], fmt='o',color = 'black',alpha=0.3)
plot(xn,yn,'g-',label='ODR: y = ('+result(a,a_error)+')x + ('+result(b,b_error)+')')

#plot confidence intervals in y as error in beta for a given x
yn_lower = func(output.beta-output.sd_beta, xn)
yn_upper = func(output.beta+output.sd_beta, xn)
plt.fill_between(xn, yn_lower, yn_upper, alpha=0.1, color = 'green')

#plotting other literature
x_seigar = np.linspace(7,38,25)
y_seigar = seigar(x_seigar)
plot(x_seigar,y_seigar,label='Seigar et al. (2008)', color = 'purple', linestyle = 'dashed')

x_berrier = np.linspace(5,36.8,25)
y_berrier = berrier(x_berrier)
plot(x_berrier,y_berrier,label='Berrier et al. (2013)', color = 'blue', linestyle = 'dashed')

x_davis = np.linspace(3,25,25)
y_davis = davis(x_davis)
plot(x_davis, y_davis, label='Davis et al. (2017)', color = 'brown', linestyle = 'dashed')

#formatting the plot
plt.ylim([2,10])
#plt.xlim([5,55])
plt.xlabel(x)
plt.ylabel(y)
plt.title("Plotting from "+fileroot)
plt.legend(loc='lower left')

output.pprint()
plt.show() 

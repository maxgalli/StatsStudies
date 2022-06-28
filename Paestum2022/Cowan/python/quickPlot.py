import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import chi2

def func(n, C):
    return C/np.sqrt(n)

numVal   = np.array([100., 200., 400., 800.])
sigma_thetaHat   = np.array([0.071219, 0.052736, 0.036985, 0.026129])

# Set up plot
matplotlib.rcParams.update({'font.size':18})     # set all font sizes
plt.clf()
fig, ax = plt.subplots(1,1)
plt.gcf().subplots_adjust(bottom=0.15)
plt.gcf().subplots_adjust(left=0.15)
plt.plot(numVal, sigma_thetaHat, color='black', linestyle='None', marker='o')
plt.xlabel(r'$n$')
plt.ylabel(r'$\sigma_{\hat{\theta}}$', labelpad=5)
xMin = 0
xMax = 1000
yMin = 0
yMax = 0.1
plt.xlim(xMin, xMax)
plt.ylim(yMin, yMax)
xPlot = np.linspace(xMin, xMax, 100)        # enough points for a smooth curve
C = sigma_thetaHat[0] * np.sqrt(numVal[0])
f = func(xPlot, C)
plotLabel = r'$\sim 1/\sqrt{n}$'
plt.plot(xPlot, f, 'dodgerblue', linewidth=2, label=plotLabel)

# Tweak legend
handles, labels = ax.get_legend_handles_labels()
#handles = [handles[1], handles[0]]
#labels = [labels[1], labels[0]]
#handles = [handles[0][0], handles[1]]      # turn off error bar for data in legend
plt.legend(handles, labels, loc='upper right', fontsize=14, frameon=False)
plt.show()

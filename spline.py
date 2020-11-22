#test scipy rbf thin plate splines..

from scipy.interpolate import Rbf
import numpy as np
import scipy as sp
from ipdb import set_trace as st
import utils
from common import *
from scipy.interpolate import griddata

#let's just generate some random points in 1D
np.random.seed(0)

N = 100
T = 10
margins = 2
x = np.linspace(margins, T-margins, N)
y = np.random.randn(N) + ((x-T/2)/2)**2 #some fake trend

#and now let's fit an Rbf spline
def fit_and_plot(x,y,s=0.0):
    K = 500
    x_ = np.linspace(0, T, K)
    
    #out = Rbf(x, y, function='thin-plate', smooth = s)
    Q = 1000
    ys = 0
    for i in range(Q):
        mask = np.random.random(x.shape) < 0.2
        #ys += np.interp(x_, x[mask], y[mask])
        ys += griddata(x[mask], y[mask], x_, method='nearest')
    y_ = ys/Q

    # y_ = out(x_)
    plt.plot(x_, y_, linewidth=1)

# for s in [0.0, 0.01, 0.1, 0.2]:
#     fit_and_plot(x,y,s=s)
fit_and_plot(x,y,1.0)
#thin plate splines with large smoothness can blow up

#is there a better alternative?
#a better regularized approach?
    

plt.scatter(x, y)
plt.axes().set_ylim(-5,5)
plt.show()
st()

#!/usr/bin/env python
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm as CM
from matplotlib.ticker import NullFormatter
import sys

from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

def chi2(X):
  '''
  Calculated the Chi2 value
  X[0] => Obs
  X[1] => Err
  X[2] => Model
  '''
  return np.sum(((X[0] - X[2]) / X[1])**2.)

def chi2_red(X,P):
  ''' Calculates the reduced chi2 '''
  return chi2(X)/(len(X[0])-len(P))
  
def Merit(X):
  ''' Given a Chi2 value
  we calculate a likelihood as our merit function
  '''
  return np.exp(-chi2(X)/2.)

def Func_lin(x,Par):
  ''' linear function y = ax + b '''
  return Par[0]*x + Par[1]

def Func_cos(x,Par):
  ''' linear function y = ax + b '''
  return Par[0]*np.cos(x*Par[1])+Par[2]

def Gaussian_Dist(mean,stddev,npts):
  ''' Creates a Gaussian distribution of data '''
  return np.random.normal(mean,stddev,npts)

def Distrib(x):
   '''Finds median and 68% interval of array x.'''
   y=sorted(x)
   up=y[int(0.8413*len(y))]
   down=y[int(0.1587*len(y))]
   med=y[int(0.5*len(y))]
   return med,up,down   

def Plot_3D(chain,P):
  ''' 3D representation of the MCMC parameter space '''
  fig = plt.figure()
  ax = fig.gca(projection='3d')
  
  p1_min = chain[:,P[0]].min()
  p2_min = chain[:,P[1]].min()
  p1_max = chain[:,P[0]].max()
  p2_max = chain[:,P[1]].max()
  
  x_dir = np.arange(p1_min,p1_max,(p1_max-p1_min)/20.)
  y_dir = np.arange(p2_min,p2_max,(p2_max-p2_min)/20.)
  X, Y = np.meshgrid(x_dir, y_dir)
  Z =  np.histogram2d(chain[:,P[0]],chain[:,P[1]],bins=20)[0]
  
  ax.plot_surface(X, Y, Z, rstride=1, cstride=1, alpha=1.0,cmap=cm.coolwarm)
  cset = ax.contourf(X, Y, Z, zdir='x', offset=p1_min,cmap=cm.coolwarm)
  cset = ax.contourf(X, Y, Z, zdir='y', offset=p2_max, cmap=cm.coolwarm)

  ax.set_xlabel('P[0]')
  ax.set_ylabel('P[1]')
  ax.set_zlabel('Number')
  ax.zaxis.set_major_locator(LinearLocator(5))
  ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

def Plot_Chain(chain,P,S):
  fig = plt.figure()
  plt.clf()
  
  # Top plot
  top = plt.subplot2grid((3,3), (0, 0), colspan=2)
  top.hist(chain[:,P[0]],bins=30)
  plt.axvline(Distrib(chain[:,P[0]])[0],color="red",lw=2)
  plt.axvline(Distrib(chain[:,P[0]])[1],color="red",lw=2,linestyle='--')
  plt.axvline(Distrib(chain[:,P[0]])[2],color="red",lw=2,linestyle='--')
  top.get_xaxis().set_ticklabels([])
  plt.minorticks_on()
  
  # Right hand side plot
  right = plt.subplot2grid((3,3), (1, 2), rowspan=2)
  right.hist(chain[:,P[1]],orientation='horizontal',bins=30)
  plt.axhline(Distrib(chain[:,P[1]])[0],color="red",lw=2)
  plt.axhline(Distrib(chain[:,P[1]])[1],color="red",lw=2,linestyle='--')
  plt.axhline(Distrib(chain[:,P[1]])[2],color="red",lw=2,linestyle='--')
  right.get_yaxis().set_ticklabels([])
  right.xaxis.set_major_locator(LinearLocator(5))
  plt.minorticks_on()
  
  # Center plot
  center = plt.subplot2grid((3,3), (1, 0), rowspan=2, colspan=2)
  center.hist2d(chain[:,P[0]],chain[:,P[1]],bins=30)
  plt.minorticks_on()
  
  # Corner plot
  corner = plt.subplot2grid((3,3), (0, 2))
  corner.get_xaxis().set_ticklabels([])
  corner.get_yaxis().set_ticklabels([])
  corner.plot(chain[:,P[0]],chain[:,P[1]],'-k')
  plt.minorticks_on()

def Print_Uncertainties(P,S):
  ''' Prints the median parameter values
  with their associated uncertainties '''
  print "===================================="
  for i in range(len(P)):
    if S[i] != 0:
      print "\n\t\t+ ",Distrib(chain[:,P[i]])[1]
      print P[i],Distrib(chain[:,P[i]])[0]
      print "\t\t+ ",Distrib(chain[:,P[i]])[2],"\n"
  print "===================================="

def Plot_Func(X,P,F):
  fig = plt.figure()
  plt.clf()
  plt.errorbar(X[0],X[1],yerr=X[2],fmt='.',elinewidth=1.3, capsize=0,markersize=10,color="#4cbd38",markeredgecolor='black',ecolor='k')
  plt.plot(X[0],F(X[0],P),'-r')

def Gen_Fake_Data(N,F,P,E,mean_noise):
  '''
  N => Number of points
  F => Function used
  P => Paramters
  E => Average error on y-axis points
  mean_noise => Mean value of noise (Typically 0)
  '''
  noise = np.random.normal(mean_noise,E,N)
  x = np.arange(N)
  y = F(x,P)+noise
  yerr = np.ones(len(x))*E
  return x,y,yerr,Func_lin(x,P)


def MCMC(x,X,F,P,S,C):
  '''
  X => Data (y,yerr,model)
  F => Function used
  P => Parameters
  S => Scale
  C => Chain length
  '''
  L=Merit(X)
  moves = 0
  chain = np.zeros(shape=(C,len(P)))
  L_chain = np.zeros(shape=(C,1))
  check_pts = np.arange(0,C,100)
  for i in range(int(C)):
    jump = np.random.normal(0.,1.,len(S)) * S
    P = P + jump
    X = X[0],X[1],F(x,P)
    L_new = Merit(X)
    L_chain[i] = L_new
    ratio = L_new/L
    if (np.random.random() > ratio):
      P = P - jump
      moved = 0
    else:
      L=L_new
      moved = 1
    moves += moved
    chain[i,:] = np.array(P)
  print "\nAccepted steps: ",100.*(moves/C),"%"
  return chain, moves

# Initial Parameters
''' Linear Function Example '''
Pin = [0.018,.0]
P_fake = [0.015,0.01]
step = [0.001,0.016]
F = Func_lin

''' Cose Function Example '''
'''
Pin = [1.,0.1,0.0]
P_fake = [1.2,0.1,0.0]
step = [0.01,0.001,0.001] # steps
F = Func_cos
'''

X = Gen_Fake_Data(100,F,P_fake,0.1,0.)

# Perform MCMC cgain
chain, moves = MCMC(X[0],X[-3:],F,Pin,step,10000.)
Pout = chain[moves,:]

P_plot = [0,1] #Choose which paramters to plot

# Print Reduced Chi^2 of best fit
print 'Reduced Chi^2:',round(chi2_red(X[-3:],Pout),2)
Print_Uncertainties(P_plot,step)

#Plot_3D(chain,P_plot)
Plot_Chain(chain,P_plot,step)
Plot_Func(X,Pout,Func_lin)
#Plot_Func(X,P,Func_cos)
#Plot_Func(X,Pout,Func_cos)
plt.show()
"""
Created on Mon Jun 23 12:10:12 2025

@author: Jules Malavieille
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize 
from scipy.interpolate import interp1d
from scipy.signal import argrelextrema

r = 1.2
a = 0.5
b = 0.5
e = 0.7
m = 0.2
sig = 0.01

# x* = m*b/(e*a-m) = 0.533...  Initial equilibrium
# it becames unstable if K > (2*m*b)/(e*a-m) + b = Kcrit
# So the tipping point to the limit cycle is when K > Kcrit

def cut_L(L, every):
    return L[::int(every)]


def bruit(nbval, dt):
    return np.sqrt(dt) * np.random.normal(0, 1, nbval)


def Milstein(dt, B1, B2):
    nbval = len(B1)
    X = np.zeros(nbval)
    Y = np.zeros(nbval)
    K = 0.5
    X[0] = K
    Y[0] = 1
    g = 0
    for j in range(nbval - 1):
        
        if K < 2:
            K = K + 0.0019*dt
            
            Kcrit = b + (2*m*b)/(e*a-m)
            
            if K > Kcrit and g == 0:
                g = j*dt
        
        detX = dt * (r*X[j]*(1-X[j]/K) - a*X[j]*Y[j]/(b+X[j]))
        stoX = sig * B1[j]
        X[j+1] = max(X[j] + detX + stoX, 1e-6)
        
        detY = dt*(e*a*X[j]*Y[j]/(b+X[j]) - m*Y[j])
        stoY = sig * B2[j]
        Y[j+1] = max(Y[j] + detY + stoY, 1E-6)
        
    return X, Y, g



dt = 0.01
nbval = 100000
tmax = nbval*dt
tmin = 0
t_obs = np.linspace(tmin, tmax, nbval)

B1 = bruit(nbval, dt)
B2 = bruit(nbval, dt)

X, Y, t_crit = Milstein(dt, B1, B2)

plt.plot(t_obs, X, label="Modèle")
plt.plot([t_crit for i in range(3)], [i for i in range(3)], "--", color="red", label="Point de bifurcation")
plt.xlabel("Temps", fontsize=20)
plt.ylabel("X", fontsize=20)
plt.title("Proie du modèle de Ronsenzweig-McArthur", fontsize=30)

plt.legend()
plt.grid()

#np.savetxt("data_perio_sans.txt", X[30000:])  






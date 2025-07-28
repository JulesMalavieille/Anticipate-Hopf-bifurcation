"""
Created on Thu Jun 19 10:32:58 2025

@author: Jules Malavieille
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import minimize 
from scipy.interpolate import interp1d
from scipy.signal import argrelextrema
from scipy.interpolate import UnivariateSpline


def bruit(nbval, dt):   # Génération du bruit
    return np.sqrt(dt) * np.random.normal(0, 1, nbval)


def cut_L(L, every):   # Produire un échantillonage
    return L[::int(every)]


def Milstein_test(dt, B, u, m, sig):  # Simulation modèle test
    nbval = len(B)
    X = np.zeros(nbval)
    X[0] = 0
    for j in range(nbval - 1):
        
        ut = max(u + m * j * dt, 1e-6)
    
        phy = max(np.sqrt(ut), 1E-6)
        
        det = dt * np.sqrt(ut) * (phy - X[j])
        sto = sig * np.sqrt(phy) * B[j]
        X[j + 1] = max(X[j] + det + sto, 1e-6)
        
    return X


def Milstein_null(dt, B, u, sig):   # Simulation du modèle null
    nbval = len(B)
    X = np.zeros(nbval)
    X[0] = 0
    phy = 1E-6
    for j in range(nbval - 1):
        det = 0
        sto = sig * B[j]
        X[j + 1] = max(X[j] + det + sto, 1e-6)
    return X


def func(t, Y, u, m, sig):   # Esperance et variance modèle test
    E, V = Y
    ut = max(u + m * t, 1E-6)

    dE = 2*np.sqrt(ut) * (np.sqrt(ut) - E) 
    dV = -2 * np.sqrt(ut) * V + sig**2 * np.sqrt(ut)
    return dE, dV


def traj(u, m, sig, data_func, t_eval):
    E0 = data_func(0)
    sol = solve_ivp(func, [0, t_eval[-1]], [E0, 1e-6], t_eval=t_eval, args=(u, m, sig),method="LSODA")
    return sol.y[0], sol.y[1]


def test_model(params, data_func, t_eval, sig):  # Calcul log-vraisemblance modèle test
    u, m = params
    E, V = traj(u, m, sig, data_func, t_eval)
    logV = 0
    a = 1
    for i in range(len(t_eval)):
        w = t_eval[i] / t_eval[-1]
        Vi = max(V[i], 1E-6)
        obs = data_func(t_eval[i])
        log = w * (-0.5 * np.log(2 * np.pi) - np.log(np.sqrt(Vi)) - 0.5 * ((obs - E[i])**2 / Vi))
        logV += log
    penalty = (1 / (m + 1e-6)) 
    return -logV + a * penalty


def null_model(params, data, ti, sig):   # Calcul log-vraisemblance du modèle null
    u = params
    logV = 0
    X_L = []
    if u <= 0:
        u = 1E-6
        
    for i in range(len(data)):
        E = 0
        V = (sig**2) / (2*u) * (1 - np.exp(-2*u*ti))

        if V < 1E-6:
            V = 1E-6 
        
        log = -0.5 * np.log(2 * np.pi) - np.log(np.sqrt(V)) - 0.5 * ((data[i] - E)**2 / V)
        logV += log
        
    return -logV


def amplitude(data, t):   # Génération de l'amplitude à partir de données 
    data_c = data - data[0]  

    maxima = argrelextrema(data_c, np.greater)[0]
    minima = argrelextrema(data_c, np.less)[0]

    n_points = min(len(maxima), len(minima))
    maxima = maxima[:n_points]
    minima = minima[:n_points]

    t_mid = t[(maxima + minima) // 2]
    amp_vals = np.abs(data_c[maxima] - data_c[minima])
    
    interp_amp = interp1d(t_mid, amp_vals, kind='linear', fill_value='extrapolate')

    return interp_amp(t)


def sig_test(data_func, t_eval, u, m, sigma_grid, dt): # Méthode Monte-Carlo pour trouver sigma modèle test
    nbval = len(t_eval)
    logL_list = []
    t_sim = np.arange(0, nbval * dt, dt)
    
    for sig in sigma_grid:
        simulations = []
        for i in range(50):
                B = np.random.normal(0, np.sqrt(dt), nbval)
                X = Milstein_test(dt, B, u, m, sig)
                simulations.append(X)
                
        sims = np.array(simulations)
        mean_sim = np.mean(sims, axis=0)
        var_sim = np.var(sims, axis=0) + 1e-6
        mean_interp = interp1d(t_sim, mean_sim, kind="linear", fill_value="extrapolate")
        var_interp = interp1d(t_sim, var_sim, kind="linear", fill_value="extrapolate")
        
        logL = 0
        for i in range(nbval):
            obs = data_func(t_eval[i])
            mean = mean_interp(t_eval[i])
            var = var_interp(t_eval[i])
            log = -0.5 * np.log(2 * np.pi * var) - 0.5 * ((obs - mean) ** 2 / var)
            logL += log
    
        logL_list.append(logL)

    sigma_opt = sigma_grid[np.argmax(logL_list)]
    return sigma_opt, logL_list
    

def sig_null(params, dt, B, u, m, sig):  # Calibration du sigma du modèle null 
    sig0 = params
    null = Milstein_null(dt, B, u, sig0) 
    test = Milstein_test(dt, B, u, m, sig) 
    distT = 0
    for i in range(len(B)):   
        dist = (null[i] - test[i])**2
        distT += dist
    return distT 


size = 98   # point de bascule : size=144
dt = 0.005
nbval = 10000
n_sim = 100
ti = nbval*dt/size
B = bruit(nbval, dt)

data_test = np.genfromtxt("data_perio.txt")
data_t = cut_L(data_test, len(data_test)/250)
data_t = data_t[:size]

t_obs = np.linspace(0, nbval*dt, size)
t_tr = np.linspace(0, nbval*dt, nbval)

data = amplitude(data_t, t_obs)

x = np.arange(len(data)) 
y = data                  

spline = UnivariateSpline(x, data, s=0.1)  # Interpolation de l'amplitude -> Fonction lisse et exploitable 

data_func = interp1d(t_obs, spline(x), kind='linear', fill_value='extrapolate')

uL = (-10, 5)  # Gamme de u à tester 
mL = (0, 2)   # Gamme de m à tester 

x0 = [-2, 0.4]
bornes = [uL, mL]

param_T = minimize(test_model, x0, args=(data_func, t_obs, 0), bounds=bornes, method="L-BFGS-B") # Paramètre u et m du modèle test sur les données
param_N = minimize(null_model, -1, args=(data, ti, 0), bounds=[(-10, 5)], method="L-BFGS-B")  # Paramètre u du modèle null sur les données

sigL = np.linspace(0, 0.5, 50)
sig, sig_V = sig_test(data_func, t_obs, *param_T.x, sigL, dt)  # Calcul du sigma modèle test
sig_N = minimize(sig_null, 0, args=(dt, B, -1, 0, sig), bounds=[(sigL[0], sigL[-1])], method="L-BFGS-B")  #Calcul sigma modèle null

ampl_est, V = traj(*param_T.x, sig, data_func, t_obs)   # Calcul de l'esperance de l'amplitude 
ampl_null = Milstein_null(dt, B, param_N.x, sig_N.x)   # Simulation modèle null
ampl_test = Milstein_test(dt, B, *param_T.x, sig)   # Simulation du modèle test

plt.plot(t_obs, data_func(t_obs), label="Amplitude mesuré")   # Graphique permettant de valider le fonctionnement des modèles 
plt.plot(t_obs, ampl_est, label="Amplitude estimé")           # Et il permet de calibrer le bruit du modèle null sur le modèle test
plt.plot(t_tr, ampl_null, label="Modèle Null")
plt.plot(t_tr, ampl_test, label="Amplitude simulé par modèle test")
plt.xlabel("Temps")
plt.ylabel("Amplitude")
plt.title("Comparaison amplitude mesuré et estimé")
plt.legend()
plt.grid()

print("Les paramètres du modèle test sont :")
print("u =", param_T.x[0])
print("m =", param_T.x[1])
print("sig =", sig)

null = np.zeros([size, n_sim])
test = np.zeros([size, n_sim])

print("Boucle 1/3")  # la génération des 100 simulations du modèle null et du modèle test avec les paramètres estimé plus tôt
for i in range(n_sim): 
    B1 = bruit(nbval, dt)
    B2 = bruit(nbval, dt)
    Mnull = Milstein_null(dt, B1, param_N.x, sig_N.x)
    Mtest = Milstein_test(dt, B2, *param_T.x, sig)
    test[:,i] = cut_L(Mtest, int(len(Mtest)/size)+1)
    null[:,i] = cut_L(Mnull, int(len(Mnull)/size)+1)
    if i%10 == 0:
        print(i,"sur",n_sim,"simulations")
    if i == n_sim-1:
        print(n_sim,"sur",n_sim,"simulations")

print()
print("Boucle 2/3")   # Calcul de la D-stat pour les données du modèle null 
D0 = []
for i in range(n_sim):
    data_func0 = interp1d(t_obs, null[:,i], kind='linear', fill_value="extrapolate")
    null_opt = minimize(null_model, -1, args=(null[:,i], ti, sig_N.x), bounds=[(-10, 5)], method="L-BFGS-B")
    test_opt = minimize(test_model, x0, args=(data_func0, t_obs, sig), bounds=bornes, method="L-BFGS-B")
    D = 2*(test_opt.fun - null_opt.fun)
    D0.append(D)
    if i%10 == 0:
        print(i,"sur",n_sim,"simulations")
    if i == n_sim-1:
        print(n_sim,"sur",n_sim,"simulations")
        
print()
print("Boucle 3/3")  # Calcul de la D-stat pour les données du modèle test
D1 = []
for i in range(n_sim):
    data_func1 = interp1d(t_obs, test[:,i], kind='linear', fill_value="extrapolate")
    null_opt = minimize(null_model, -1, args=(test[:,i], ti, sig_N.x), bounds=[(-10, 5)], method="L-BFGS-B")
    test_opt = minimize(test_model, x0, args=(data_func1, t_obs, sig), bounds=bornes, method="L-BFGS-B")
    D = 2*(test_opt.fun - null_opt.fun)
    D1.append(D)
    if i%10 == 0:
        print(i,"sur",n_sim,"simulations")
    if i == n_sim-1:
        print(n_sim,"sur",n_sim,"simulations")


# Récupération dans des fichiers des distributions D0 et D1
np.savetxt("D0_Hopf.txt", D0)
np.savetxt("D1_Hopf.txt", D1)
    
    
    
























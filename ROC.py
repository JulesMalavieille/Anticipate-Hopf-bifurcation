"""
Created on Wed Jun  4 16:50:24 2025

@author: Jules Malavieille
"""

import matplotlib.pyplot as plt
import numpy as np
import sys


def gauss(u):  # Génère un nombre qui suit Normal(u, 1)
    gauss = 1/np.sqrt(2*np.pi) * np.exp(-u**2/2)
    return gauss


def KDE(x, xest, h):   # Estimation des lois de proba par KDE 
    x = np.asarray(x)[:, np.newaxis]      
    xest = np.asarray(xest)[np.newaxis, :]
    z = (xest - x) / h                     
    kernel_vals = gauss(z)                
    f = np.mean(kernel_vals, axis=0) / h 
    return f  


def k_fold(data, h_L, k):   # Validation croisé pour trouvé le h optimal pour le KDE
    data = np.array(data)
    np.random.shuffle(data)
    split = np.split(data, k)
    log_L = []
    for h in h_L:
        log = 0
        for i in range(k):
            test = split[i]
            train = np.concatenate([split[j] for j in range(k) if j != i])
            est = KDE(train, test, h)
            est[est < 1e-12] = 1e-12
            log += np.sum(np.log(est))
        log_L.append(log/len(data))
    
    idxmax = np.argmax(log_L)
    h_opt = h_L[idxmax]
    return h_opt, log_L


# Permet de générer la courbe ROC à partir de 2 distributions 
def ROC(f, g, xf, xg, n):    # f = fonction "négative" et g = fonction "positive" (du test)
    a = 1
    
    ROCf = []
    ROCg = []
    
    idxmaxf = np.argmax(f)
    idxmaxg = np.argmax(g)
    
    reverse = False
    if xg[idxmaxg] < xf[idxmaxf]:
        reverse = True 
    
    f_min = min(xf)
    f_max = max(xf)
    g_min = min(xg)
    g_max = max(xg)
    
    if f_max < g_min or g_max < f_min:
        print("Les deux distributions ne se chevauchent pas, le test est valide 100% du temps")
        print()
        a = 0
        ROCf.extend([0, 0, 100])
        ROCg.extend([0, 100, 100])
        return ROCf, ROCg, f_max, a
    
    start = max(f_min, g_min)
    end = min(f_max, g_max)
    seuils = np.linspace(start, end, n)
    
    for s in seuils:
        idxf = np.searchsorted(xf, s)
        idxg = np.searchsorted(xg, s)

        if reverse:
            proba_f = np.trapz(f[:idxf], xf[:idxf])
            proba_g = np.trapz(g[:idxg], xg[:idxg])
        else:
            proba_f = np.trapz(f[idxf:], xf[idxf:])
            proba_g = np.trapz(g[idxg:], xg[idxg:])

        ROCf.append(proba_f)
        ROCg.append(proba_g)

    return ROCf, ROCg, seuils, a


# Toutes les données générées 

"""Bifurcation pli"""

"""Données Gillepsie sans bascule"""   # Système sans bascule
# D0 = np.genfromtxt("D0_sansbascule.txt")
# D1 = np.genfromtxt("D1_sansbascule.txt")

"""Données chaotique sans bascule"""  # système sans bascule
# D0 = np.genfromtxt("D0_chaotic.txt")
# D1 = np.genfromtxt("D1_chaotic.txt")




"""Données complète """

"""Données Gillepsie"""    # Système avec legère décroissance puis bascule
# D0 = np.genfromtxt("D0_gillepsie.txt")
# D1 = np.genfromtxt("D1_gillepsie.txt")

"""Données ver épicéa"""   # Système stable puis bascule brutale
# D0 = np.genfromtxt("D0_ver.txt")
# D1 = np.genfromtxt("D1_ver.txt")

"""Données chaotique"""    # Système de chaos détérministe puis bascule brutale
# D0 = np.genfromtxt("D0_chaotic_shift.txt")
# D1 = np.genfromtxt("D1_chaotic_shift.txt")

"""Données réelle"""
# D0 = np.genfromtxt("D0_lake.txt")
# D1 = np.genfromtxt("D1_lake.txt")




"""Données tronqué avant la bascule"""

"""Données Gillepsie"""    # Système avec legère décroissance puis bascule
# D0 = np.genfromtxt("D0_gillepsie_t.txt")
# D1 = np.genfromtxt("D1_gillepsie_t.txt")

"""Données ver épicéa"""   # Système stable puis bascule brutale
# D0 = np.genfromtxt("D0_ver_t.txt")
# D1 = np.genfromtxt("D1_ver_t.txt")

"""Données chaotique"""    # Système de chaos détérministe puis bascule brutale
# D0 = np.genfromtxt("D0_chaotic_shift_t.txt")
# D1 = np.genfromtxt("D1_chaotic_shift_t.txt")

"""Données réelle"""
# D0 = np.genfromtxt("D0_lake_t.txt")
# D1 = np.genfromtxt("D1_lake_t.txt")



"""Bifurcation de Hopf"""

"""Test fonctionnement algorithme Hopf"""
# D0 = np.genfromtxt("D0_Hopf_t.txt")
# D1 = np.genfromtxt("D1_Hopf_t.txt")

"""Test algorithme Hopf conditions réelle"""
D0 = np.genfromtxt("D0_Hopf.txt")
D1 = np.genfromtxt("D1_Hopf.txt")

"""Test sans bascule Hopf"""
# D0 = np.genfromtxt("D0_Hopf_sans.txt")
# D1 = np.genfromtxt("D1_Hopf_sans.txt")



min0 = min(D0)
max0 = max(D0)

min1 = min(D1)
max1 = max(D1)

# Permet de rallonger les lois de proba pour qu'elles arrivent sufisament proche de 0
tail0 = (max(D0)-min(D0))*0.2 
tail1 = (max(D1)-min(D1))*0.2

x0 = np.linspace(min0-tail0, max0+tail0, 100)
x1 = np.linspace(min1-tail1, max1+tail1, 100)

var_D0 = np.std(D0)
var_D1 = np.std(D1)

h1_init = 1.059 * var_D1 * len(D1)**(-1/5)  # Permet d'initialiser les valeurs de h1
h0_init = 1.059 * var_D0 * len(D0)**(-1/5)  # Permet d'initialiser les valeurs de h0

h0_values = np.linspace(h0_init/2, h0_init*2, 30)
h1_values = np.linspace(h1_init/3, h1_init*3, 30)    

h0, log0 = k_fold(D0, h0_values, k=5)
h1, log1 = k_fold(D1, h1_values, k=5)

D0_law = KDE(D0, x0, h0)
D1_law = KDE(D1, x1, h1)
    
# Vérifie de le calcul des KDE soit bon, si intégrale de D0_law ou D1_law =! 1 à l'erreur numérique près = erreur 
if np.trapz(D1_law, x1) < 0.95 or np.trapz(D0_law, x0) < 0.95:
    print("ERREUR : le calcul des lois de porbabilités est différent de 1 à l'erreur près")
    print("Intégrale de D0 =", np.trapz(D0_law, x0))
    print("Intégrale de D1 =", np.trapz(D1_law, x1))
    print() 
    sys.exit()
    
n = 5
ROC_0, ROC_1, seuil, a = ROC(D0_law, D1_law, x0, x1, n)

"""Histogramme"""
plt.figure(1)
plt.hist(D0, label="Pas de bascule")
plt.hist(D1, label="Bascule")
plt.xlabel("D-statistique")
plt.ylabel("P(D)")
plt.title("Histogramme distribution observé D-statistique")
plt.legend()

"""Loi de proba estimé"""
plt.figure(2)
plt.plot(x0, D0_law, label="Null-Model")
plt.plot(x1, D1_law, label="Shift-Model")

#plt.plot([seuil, seuil],([0 for i in range(n)],[0.002 for i in range(n)]),"--", color="black", linewidth=1)
#plt.plot([], [], "--", color="black", label="Seuil ROC")

plt.xlabel("D-statistique", fontsize=20)
plt.ylabel("P(D)", fontsize=20)
plt.title("Densité de probabilité D-statistique", fontsize=30)
plt.grid()
plt.legend()

"""Courbe ROC"""
plt.figure(3)
plt.plot(ROC_0, ROC_1, "*-")
plt.xlabel("Taux de faux positif", fontsize=20)
plt.ylabel("Taux de vrai positif", fontsize=20)
plt.title("Courbe ROC", fontsize=30)
plt.grid()

ROC_0 = np.sort([0.0]+ROC_0+[1.0])  # Trie dans l'ordre les valeurs de ROC 
ROC_1 = np.sort([0.0]+ROC_1+[1.0])  # Trie dans l'ordre les valeurs de ROC 

# Permet de discriminer le cas où les courbes ne se chevauchent pas 
if a == 0:
    print("AUC =",1)
    print("Le test fais 100% de vrai positif")

# Calcul de l'AUC
else:
    AUC = np.trapz(ROC_1, ROC_0)
    print("Le score AUC de la courbe ROC est :", round(AUC,3))
    print("Le test distingue donc à juste titre une bascule", round(AUC*100,3),"% du temps")
    print("Autrement dit, le système se trompe", round((1-AUC)*100,4),"% du temps")
    print()


# Procédure pour calculer le recouvrement d'une courbe sur l'autre
grid_min = min(np.min(D0), np.min(D1))
grid_max = max(np.max(D0), np.max(D1))
c = (grid_max-grid_min)*0.1
grid = np.linspace(grid_min-c, grid_max+c, 1000)

D0_over = KDE(D0, grid, h0)
D1_over = KDE(D1, grid, h1)

overlap = np.trapz(np.minimum(D0_over, D1_over), grid)
if overlap > 0.8:
    print("AVERTISSEMENT : Les distributions se chevauchent fortement, à hauteur de",round(overlap*100,3), "%")
    print("L'interprétation graphique et les métriques sont faussées, ne pas en tenir compte, le test est faux dans tous les cas")









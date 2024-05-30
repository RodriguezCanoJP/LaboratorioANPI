import numpy as np
import matplotlib.pyplot as plt
import time

def svdCompact(A):
    m, n = A.shape
    if m > n:
        M1 = A.T@A
        D, V = np.linalg.eig(M1)
        const = n*np.max(D)*2.22e-16
        y = (D>const)
        rA = np.sum(y)
        D = y*D
        idx = D.argsort()[::-1]   
        D = D[idx]
        V = V[idx]
        Vr = V[:,:rA]
        Sr = np.diag(D[:rA])
        Ur = (1/D[:rA])*(A@Vr)
    else:
        M1 = A@A.T
        D, V = np.linalg.eig(M1)
        const = n*np.max(D)*2.22e-16
        y = (D>const)
        rA = np.sum(y)
        D = y*D
        idx = D.argsort()[::-1]   
        D = D[idx]
        U = V[idx]
        Ur = U[:,:rA]
        Sr = np.diag(D[:rA])
        Vr = (1/D[:rA]).T@A@Ur
        
    return Vr, Sr, Ur

def grafica_tiempos():
    dimensiones = range(5,13)
    tiempos_svd = []
    tiempos_svd_compacta = []
    for i in dimensiones:
        A = np.random.rand(i,i-1)
        start = time.time()
        np.linalg.svd(A)
        finish = time.time()
        svdCompact(A)
        finish2 = time.time()
        tiempos_svd.append(finish - start)
        tiempos_svd_compacta.append(finish2 - finish)
    
    plt.plot(dimensiones, tiempos_svd, 'r-',label="Numpy")
    plt.plot(dimensiones, tiempos_svd_compacta, 'b-', label="SVD Compacta")
    plt.title("Tiempos de ejecución de métodos de SVD")
    plt.xlabel("Dimensiones")
    plt.ylabel("Tiempo(s)")
    plt.legend()
    plt.grid()
    plt.show()

grafica_tiempos()
import numpy as np
import matplotlib.pyplot as plt
import PIL
import time
import os
import glob

def inicializa_imagenes():
    """
    inicializa la matriz del conjuto de rostros de las imagenes de 112x92 pixeles (10304 pixeles en total) que
    corresponden a un vector columna por imagen.
    Return: matriz de rostros de entrenamiento y matriz de rostos de comparación
    """
    X_train = np.empty((10304, 360))
    X_compare = np.empty((10304, 40))
    
    training_folder = os.getcwd() + "/training"
    compare_folder = os.getcwd() + "/compare"
    
    idx=0
    
    pattern = os.path.join(compare_folder, "p*.jpg")
    files = glob.glob(pattern)
    
    for file in files:
        image = np.asarray(PIL.Image.open(file)).reshape(-1,1)
        X_compare[:,idx:idx+1] = image
        idx+=1
    
    idx=0
    
    for i in range(1, 41):
        subdirectory = os.path.join(training_folder, f"s{i}")
        pattern = os.path.join(subdirectory, "*.jpg")
        files = glob.glob(pattern)

        for file in files:
            image = np.asarray(PIL.Image.open(file)).reshape(-1,1)
            X_train[:,idx:idx+1] = image
            idx+=1
    return X_train, X_compare

def show_face(face_image):
    """
    face_image: vector de valores de pixeles de la imagen de rostro
    grafica los pixeles de la imagen en una escala de grises
    """
    image = face_image.reshape(112,92)
    plt.imshow(image, cmap="gray")
    plt.axis("off")

def best_coordinate(face_sample, m=m, x=x):
    """
    @params
    face_sample: matriz de rostro con los valores del rostro promedio restado
    m = tamaño de columnas de matriz del conjunto de rostros de entrenamiento (la cantidad de diferentes rostros)
    x = matriz de vectores de coordenadas
    """
    err = np.empty(m)
    for i in range(m):
        err[i]=np.linalg.norm(face_sample - x[:,i].reshape(-1,1))
    return np.where(err == err.min())[0]

def resultados(n, train_mat, compare_mat, singular_matrix, r):
    """
    @params
    n: valor entre 0 y 39, rostro de comparacion
    train_mat: matriz de conjunto de rostros de entrenamiento
    compare_mat: matriz de conjunto de rostros de comparacion
    singular_matrix: matrix U resultado de SVD
    r: rango de A
    
    Grafica el rostro a comparar y el rosto que se identifica
    """
    x_sample = singular_matrix[:,:r].T @ compare_mat[:,n].reshape(-1,1)
    plt.clf()
    plt.subplot(1,2,1)
    show_face(compare_mat[:,n])
    plt.title("Rostro Nuevo")
    plt.subplot(1,2,2)
    plt.title("Rostro Identificado")
    show_face(train_mat[:,best_coordinate(x_sample)])
    plt.show()


X_train, X_compare = inicializa_imagenes()

#A todos los pixeles se le resta el valor promedio correspondiente de su fila
A_compare = X_compare - X_compare.mean(axis=1).reshape(-1,1)
A_train = X_train - X_train.mean(axis=1).reshape(-1,1)

#Decomposición de valores singulares
U, S, V = np.linalg.svd(A_train)

# Rango de A
m = len(S)
r = len((S > m * 2.2e-16 * S[0]))

#Vectores de coordenadas x
x = U[:,:r].T@A_train

resultados(29, A_train, A_compare, U, r)
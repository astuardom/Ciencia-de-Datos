# 1. Importar la librería NumPy
import numpy as np

# 2. Crear un vector de 10 elementos con valores del 1 al 10 utilizando arange()
vector = np.arange(1, 11)
print("Vector del 1 al 10:")
print(vector)

# 3. Generar una matriz de 3x3 con valores aleatorios entre 0 y 1 usando random.rand()
matriz_aleatoria = np.random.rand(3, 3)
print("\nMatriz 3x3 con valores aleatorios entre 0 y 1:")
print(matriz_aleatoria)

# 4. Crear una matriz identidad de tamaño 4x4 utilizando eye()
matriz_identidad = np.eye(4)
print("\nMatriz identidad 4x4:")
print(matriz_identidad)

# 5. Redimensionar el vector creado en el punto 2 en una matriz de 2x5 usando reshape()
matriz_2x5 = vector.reshape(2, 5)
print("\nVector redimensionado a matriz 2x5:")
print(matriz_2x5)

# 6. Seleccionar los elementos mayores a 5 del vector original
mayores_a_5 = vector[vector > 5]
print("\nElementos mayores a 5 del vector:")
print(mayores_a_5)

# 7. Realizar una operación matemática entre arreglos (suma de dos arreglos de tamaño 5)
a = np.arange(5)        
b = np.arange(5, 10)    
suma = a + b
print("\nSuma de arreglos:")
print("a =", a)
print("b =", b)
print("a + b =", suma)

# 8. Aplicar una función matemática (raíz cuadrada del vector original)
raiz_cuadrada = np.sqrt(vector)
print("\nRaíz cuadrada de cada elemento del vector original:")
print(raiz_cuadrada)

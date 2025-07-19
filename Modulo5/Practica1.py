# la musica de fondo mejora la memoria a corto plazo

# problema: estudiantes suelen poner música mientras estudian y afirman recordar mejor

# Hipótesis nula (H0): escuchar música instrumental no cambia las puntuaciones de memoria

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import os

# Estilo de gráfico
plt.style.use('ggplot')

# 1. Crear el conjunto de datos simulado
n = 15  # número de estudiantes

# Simular puntuaciones de memoria con y sin música
con_musica = np.random.normal(loc=75, scale=10, size=n).round(2)
sin_musica = np.random.normal(loc=70, scale=10, size=n).round(2)

# 2. Crear un DataFrame
df = pd.DataFrame({
    'participante': [f'M{i+1}' for i in range(n)] * 2,
    'grupo': ['sin_musica'] * n + ['con_musica'] * n,
    'puntacion': np.concatenate([sin_musica, con_musica])
})

# 3. Mostrar las primeras filas
print(df.head())

# 4. Guardar CSV en subcarpeta 'entrada'
os.makedirs('entrada', exist_ok=True)  # crea la carpeta si no existe
df.to_csv('entrada/datos_musica.csv', index=False)
print("✅ Archivo guardado correctamente en 'entrada/datos_musica.csv'")

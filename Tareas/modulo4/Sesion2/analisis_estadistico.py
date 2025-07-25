import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os

# 1. Crear el DataFrame con los datos ficticios

datos = {
    'ID': [1,2,3,4,5,6,7,8,9,10],
    'Nombre': ['Ana','Juan','Luis','Marta','Pedro','Sofía','Carlos','Elena','Miguel','Paula'],
    'Edad': [25,30,28,22,35,27,29,24,31,26],
    'Ingresos': [2500,3200,2800,2200,4100,2900,3100,2700,3300,2600],
    'Género': ['Femenino','Masculino','Masculino','Femenino','Masculino','Femenino','Masculino','Femenino','Masculino','Femenino'],
    'Ciudad': ['Lima','Bogotá','Lima','Quito','Santiago','Lima','Buenos Aires','Quito','Santiago','Buenos Aires']
}
df = pd.DataFrame(datos)

# 1. Definir Variables
variables = {
    'ID': 'Cuantitativa discreta (identificador)',
    'Nombre': 'Categórica nominal',
    'Edad': 'Cuantitativa discreta',
    'Ingresos': 'Cuantitativa continua',
    'Género': 'Categórica nominal',
    'Ciudad': 'Categórica nominal'
}
print('1. Tipos de variables:')
for var, tipo in variables.items():
    print(f'- {var}: {tipo}')
print('\n')

# 2. Tablas de Frecuencia
print('2. Tablas de frecuencia:')
# Variable categórica: Género
print('\nTabla de frecuencia para Género:')
print(df['Género'].value_counts())
# Variable cuantitativa discreta: Edad
print('\nTabla de frecuencia para Edad:')
print(df['Edad'].value_counts().sort_index())
print('\n')

# 3. Medidas de Tendencia Central para Ingresos
print('3. Medidas de tendencia central para Ingresos:')
media = df['Ingresos'].mean()
mediana = df['Ingresos'].median()
moda = df['Ingresos'].mode()[0]
print(f'- Media: {media}')
print(f'- Mediana: {mediana}')
print(f'- Moda: {moda}')
print('\n')

# 4. Medidas de Dispersión para Ingresos
print('4. Medidas de dispersión para Ingresos:')
rango = df['Ingresos'].max() - df['Ingresos'].min()
varianza = df['Ingresos'].var(ddof=0)
desv_std = df['Ingresos'].std(ddof=0)
print(f'- Rango: {rango}')
print(f'- Varianza: {varianza}')
print(f'- Desviación estándar: {desv_std}')
print('\n')

# 5. Visualización de Datos
print('5. Visualización de datos:')
# Crear carpeta de salida si no existe
output_dir = 'Tareas/modulo4/Sesion2/salida'
os.makedirs(output_dir, exist_ok=True)
# Histograma de Ingresos
plt.figure(figsize=(8,4))
plt.hist(df['Ingresos'], bins=5, color='skyblue', edgecolor='black')
plt.title('Histograma de Ingresos')
plt.xlabel('Ingresos')
plt.ylabel('Frecuencia')
plt.grid(axis='y', alpha=0.75)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'histograma_ingresos.png'))
plt.show()

# Boxplot de Edad
plt.figure(figsize=(6,4))
plt.boxplot(df['Edad'], vert=False, patch_artist=True, boxprops=dict(facecolor='lightgreen'))
plt.title('Boxplot de Edad')
plt.xlabel('Edad')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'boxplot_edad.png'))
plt.show()

print(f'¡Análisis completado! Las gráficas se han guardado en {output_dir}/.') 
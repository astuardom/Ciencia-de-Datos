import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar el dataset
df = pd.read_csv("entrada/empleados_150.csv")

#paso 2 impeccion inicial
print("primera fila del dataframe:")
print(df.head(10))
print(df.describe())
print("informacion del dataframe:")
print(df.info())

# Paso 3: identificar problemas
print("Valores nulos en cada columna:")
print(df.isnull().sum())
print("Valores duplicados:", df.duplicated().sum())
print("Tipos de datos:")
print(df.dtypes)

# Paso 4: limpieza de datos
df = df.drop_duplicates()  # Eliminar duplicados 

df['Edad'] = pd.to_numeric(df['Edad'], errors='coerce')  # Convertir a numérico, forzando errores a NaN
df['Edad'].fillna(df['Edad'].median())  # Rellenar NaN con la media  
df['Salario'].fillna(df['Salario'].median())  # Rellenar NaN con la media

# Paso 5: estadísticas descriptivas
print("\n medidas de tendencia central (Edad):")
print("media:", df['Edad'].mean())
print("mediana:", df['Edad'].median())
print("moda:", df['Edad'].mode())

print("\n medidas de dispersión (salario):")
print("Rango:", df['Salario'].max())
print("mediana:", df['Salario'].median())
print("Desviación estándar:", df['Salario'].std())



# Paso 6: visualización de datos

#5.1 Histograma de Edad
sns.histplot(df['Edad'], kde=True)
plt.title('Histograma de Edad')
plt.xlabel('Edad')
plt.ylabel('Frecuencia')
plt.show()

#5.2 Boxplto de Salario
sns.boxplot(x=df['Salario'])
plt.title('Boxplot de Salario')
plt.xlabel('Salario')
plt.show()

#5.3 Carrelacion entre Edad y Salario
# Scatterplot de Edad vs Salario
sns.scatterplot(x='Edad', y='Salario', data=df)
plt.title('Scatterplot de Edad vs Salario') 
plt.xlabel('Edad')
plt.ylabel('Salario')   
plt.show()

#5.4 pairplot(analisis multivariado visual)
sns.pairplot(df[['Edad', 'Salario']])
plt.suptitle('Pairplot de Edad y Salario')
plt.show()

#5.5 Regresión lineal entre Edad y Salario
sns.regplot(x='Edad', y='Salario', data=df)
plt.title('Regresión Lineal: Edad vs Salario')
plt.xlabel('Edad')
plt.ylabel('Salario')
plt.show()

#5.6 Boxplot agrupando Edad por tramos
df['Edad_Grupo'] = pd.cut(df['Edad'], bins=[20, 30, 40, 50, 60], labels=['20-30', '30-40', '40-50', '50-60'])
sns.boxplot(x='Edad_Grupo', y='Salario', data=df)
plt.title('Salario por Grupo de Edad')
plt.xlabel('Grupo de Edad')
plt.ylabel('Salario')
plt.show()

#5.7 Curva de salario promedio por edad
df.groupby('Edad')['Salario'].mean().plot(kind='line', marker='o')
plt.title('Salario Promedio por Edad')
plt.xlabel('Edad')
plt.ylabel('Salario Promedio')
plt.grid(True)
plt.show()

#5.8 Densidad conjunta (Jointplot) entre Edad y Salario
sns.jointplot(x='Edad', y='Salario', data=df, kind='kde', fill=True, cmap='Blues')
plt.suptitle('Densidad conjunta: Edad vs Salario', y=1.02)
plt.show()


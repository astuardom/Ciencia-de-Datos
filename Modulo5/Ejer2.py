
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import os

plt.style.use('ggplot')
sns.set(style='whitegrid')

np.random.seed(123)
n = 100
memoria_ejercicio = np.random.normal(loc=78, scale=8, size=n).round(1)
memoria_reposo = np.random.normal(loc=70, scale=8, size=n).round(1)

df = pd.DataFrame({
    "Empleado": [f"E{i+1}" for i in range(n)] * 2,
    "Condición": ["Reposo"] * n + ["Ejercicio"] * n,
    "PuntajeMemoria": np.concatenate([memoria_reposo, memoria_ejercicio])
})

os.makedirs('entrada', exist_ok=True)
df.to_csv('entrada/memoria_ejercicio.csv', index=False)

print("\n📋 EJERCICIO 2 – Puntaje de memoria")
print(df.head())

# Boxplot corregido
plt.figure(figsize=(8, 5))
sns.boxplot(x='Condición', y='PuntajeMemoria', data=df, hue='Condición', palette='Set2', legend=False)
plt.title("Puntaje de memoria por condición")
plt.ylabel("Puntaje (0–100)")
plt.xlabel("Condición")
plt.show()

# Histograma corregido
plt.figure(figsize=(8, 5))
sns.histplot(data=df, x='PuntajeMemoria', hue='Condición', kde=True, bins=12, palette='Set1')
plt.title("Distribución de puntajes de memoria")
plt.xlabel("Puntaje de memoria")
plt.show()

# Barplot corregido
plt.figure(figsize=(7, 5))
sns.barplot(x='Condición', y='PuntajeMemoria', data=df, errorbar='sd', hue='Condición', palette='pastel', legend=False)
plt.title("Promedio de memoria con desviación estándar")
plt.ylabel("Puntaje (0–100)")
plt.xlabel("Condición")
plt.show()

# Prueba t de Student
grupo_ejercicio = df[df['Condición'] == 'Ejercicio']['PuntajeMemoria']
grupo_reposo = df[df['Condición'] == 'Reposo']['PuntajeMemoria']

t_stat, p_val = stats.ttest_ind(grupo_ejercicio, grupo_reposo, equal_var=False)

print(f"\nEstadístico t: {t_stat:.3f}")
print(f"Valor p: {p_val:.4f}")

alpha = 0.05
if p_val < alpha:
    print("❗ Se rechaza la hipótesis nula: el ejercicio tiene un efecto significativo en la memoria.")
else:
    print("✅ No se rechaza la hipótesis nula: no hay evidencia suficiente de efecto.")

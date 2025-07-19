
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import os

plt.style.use('ggplot')
sns.set(style='whitegrid')

np.random.seed(456)
n = 100
tiempo_musica = np.random.normal(loc=290, scale=25, size=n).round(1)
tiempo_silencio = np.random.normal(loc=310, scale=25, size=n).round(1)

df = pd.DataFrame({
    "Usuario": [f"U{i+1}" for i in range(n)] * 2,
    "Grupo": ["Silencio"] * n + ["Música"] * n,
    "TiempoTest": np.concatenate([tiempo_silencio, tiempo_musica])
})

os.makedirs('entrada', exist_ok=True)
df.to_csv('entrada/test_concentracion_musica.csv', index=False)

print("\n📋 EJERCICIO 3 – Tiempo de resolución de test")
print(df.head())

# Boxplot corregido
plt.figure(figsize=(8, 5))
sns.boxplot(x='Grupo', y='TiempoTest', data=df, hue='Grupo', palette='Set2', legend=False)
plt.title("Tiempo para resolver test lógico")
plt.ylabel("Tiempo (segundos)")
plt.xlabel("Grupo")
plt.show()

# Histograma corregido
plt.figure(figsize=(8, 5))
sns.histplot(data=df, x='TiempoTest', hue='Grupo', kde=True, bins=12, palette='Set1')
plt.title("Distribución de tiempos por grupo")
plt.xlabel("Tiempo (segundos)")
plt.show()

# Barplot corregido
plt.figure(figsize=(7, 5))
sns.barplot(x='Grupo', y='TiempoTest', data=df, errorbar='sd', hue='Grupo', palette='pastel', legend=False)
plt.title("Tiempo promedio con desviación estándar")
plt.ylabel("Tiempo (segundos)")
plt.xlabel("Grupo")
plt.show()

# Prueba t de Student
grupo_musica = df[df['Grupo'] == 'Música']['TiempoTest']
grupo_silencio = df[df['Grupo'] == 'Silencio']['TiempoTest']

t_stat, p_val = stats.ttest_ind(grupo_musica, grupo_silencio, equal_var=False)

print(f"\nEstadístico t: {t_stat:.3f}")
print(f"Valor p: {p_val:.4f}")

alpha = 0.05
if p_val < alpha:
    print("❗ Se rechaza la hipótesis nula: la música influye significativamente en el tiempo de resolución.")
else:
    print("✅ No se rechaza la hipótesis nula: no hay evidencia suficiente de efecto.")

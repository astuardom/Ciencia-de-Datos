
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import os

plt.style.use('ggplot')
sns.set(style='whitegrid')

# Simulación reproducible
np.random.seed(42)
n = 100

velocidad_cafeina = np.random.normal(loc=75, scale=8, size=n).round(1)
velocidad_sin = np.random.normal(loc=68, scale=8, size=n).round(1)

df = pd.DataFrame({
    "Participante": [f"P{i+1}" for i in range(n)] * 2,
    "Grupo": ["SinCafeína"] * n + ["Cafeína"] * n,
    "Velocidad": np.concatenate([velocidad_sin, velocidad_cafeina])
})

# Guardar CSV
os.makedirs('entrada', exist_ok=True)
df.to_csv('entrada/velocidad_cafeina.csv', index=False)

# Mostrar primeras filas en terminal
print("\n📋 EJERCICIO 1 – Velocidad de escritura")
print(df.head())

# Boxplot corregido
plt.figure(figsize=(8, 5))
sns.boxplot(x='Grupo', y='Velocidad', data=df, hue='Grupo', palette='Set2', legend=False)
plt.title("Velocidad de escritura por grupo")
plt.ylabel("Palabras por minuto")
plt.xlabel("Grupo")
plt.show()

# Histograma corregido
plt.figure(figsize=(8, 5))
sns.histplot(data=df, x='Velocidad', hue='Grupo', kde=True, bins=12, palette='Set1')
plt.title("Distribución de velocidad de escritura")
plt.xlabel("Velocidad (ppm)")
plt.show()

# Barplot corregido
plt.figure(figsize=(7, 5))
sns.barplot(x='Grupo', y='Velocidad', data=df, errorbar='sd', hue='Grupo', palette='pastel', legend=False)
plt.title("Velocidad promedio con desviación estándar")
plt.ylabel("Palabras por minuto")
plt.xlabel("Grupo")
plt.show()

# Prueba t de Student
grupo_cafeina = df[df['Grupo'] == 'Cafeína']['Velocidad']
grupo_sin = df[df['Grupo'] == 'SinCafeína']['Velocidad']

t_stat, p_val = stats.ttest_ind(grupo_cafeina, grupo_sin, equal_var=False)

print(f"\nEstadístico t: {t_stat:.3f}")
print(f"Valor p: {p_val:.4f}")

alpha = 0.05
if p_val < alpha:
    print("❗ Se rechaza la hipótesis nula: la cafeína mejora significativamente la velocidad.")
else:
    print("✅ No se rechaza la hipótesis nula: no hay evidencia suficiente de efecto.")

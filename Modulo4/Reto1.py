import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os

# Crear carpeta de salida para los gráficos
os.makedirs("salida", exist_ok=True)

# Paso 1: Cargar y limpiar datos
df = pd.read_csv("entrada/clima.csv")
df['fecha'] = pd.to_datetime(df['fecha'], errors='coerce')
df['precipitación'] = df['precipitación'].fillna(0)
df = df[(df['temperatura'] >= -50) & (df['temperatura'] <= 60)]
df = df.dropna(subset=['fecha'])

# Paso 2: Gráficos y análisis

# Histograma de temperatura
plt.figure(figsize=(8, 5))
sns.histplot(df['temperatura'], kde=True, bins=30, color='skyblue')
plt.axvline(df['temperatura'].mean(), color='red', linestyle='--', label=f"Media: {df['temperatura'].mean():.2f}°C")
plt.axvline(df['temperatura'].median(), color='green', linestyle='--', label=f"Mediana: {df['temperatura'].median():.2f}°C")
plt.title("Histograma Distribución de Temperatura")
plt.xlabel("Temperatura (°C)")
plt.ylabel("Frecuencia")
plt.legend()
plt.tight_layout()
plt.savefig("salida/histograma_temperatura.png")
plt.show()
plt.close()

# Boxplot de temperatura por ciudad
plt.figure(figsize=(10, 6))
sns.boxplot(x='ciudad', y='temperatura', hue='ciudad', data=df, palette='muted', legend=False)
plt.title("Boxplot de Temperatura por Ciudad")
plt.xlabel("Ciudad")
plt.ylabel("Temperatura (°C)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("salida/boxplot_temperatura_ciudad.png")
plt.show()
plt.close()


# Detección de outliers (IQR)
def detectar_outliers(col):
    Q1, Q3 = df[col].quantile([0.25, 0.75])
    IQR = Q3 - Q1
    return df[(df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)]

outliers_temp = detectar_outliers("temperatura")
outliers_humedad = detectar_outliers("humedad")

# Scatterplot temperatura vs humedad
plt.figure(figsize=(8, 6))
sns.scatterplot(x='temperatura', y='humedad', hue='ciudad', data=df)
plt.title("Scatterplot Temperatura vs Humedad")
plt.xlabel("Temperatura (°C)")
plt.ylabel("Humedad (%)")
plt.legend(title="Ciudad", loc='upper right')
plt.tight_layout()
plt.savefig("salida/scatter_temp_vs_humedad.png")
plt.show()
plt.close()

# Mapa de calor de correlación
plt.figure(figsize=(6, 5))
sns.heatmap(df[['temperatura', 'humedad', 'precipitación']].corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlación entre Variables Climáticas")
plt.tight_layout()
plt.savefig("salida/heatmap_correlacion.png")
plt.show()
plt.close()

# Barplot de temperatura promedio por ciudad (sin errorbar)
plt.figure(figsize=(8, 5))
sns.barplot(x='ciudad', y='temperatura', data=df, estimator=np.mean, errorbar=None, hue='ciudad', palette='muted', legend=False)
plt.title("Barplot Temperatura Promedio por Ciudad")
plt.xlabel("Ciudad")
plt.ylabel("Temperatura Promedio (°C)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("salida/barplot_temp_promedio.png")
plt.show()
plt.close()


# Línea de temperatura en el tiempo por ciudad
plt.figure(figsize=(10, 6))
sns.lineplot(x='fecha', y='temperatura', hue='ciudad', data=df)
plt.title("Evolución de Temperatura en el Tiempo")
plt.xlabel("Fecha")
plt.ylabel("Temperatura (°C)")
plt.tight_layout()
plt.savefig("salida/lineplot_temperatura_tiempo.png")
plt.show()
plt.close()

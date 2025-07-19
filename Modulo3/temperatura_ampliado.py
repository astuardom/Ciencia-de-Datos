import pandas as pd
import numpy as np

#cargar el archivo CSV
df = pd.read_csv('lectura_archivos/entrada/reto1_temperatura_ampliado.csv')

# temperatura maxima - media
df["Temperatura_Maxima"] = pd.to_numeric(df["Temperatura_Maxima"], errors="coerce")
media_temp = df["Temperatura_Maxima"].mean()
df["Temperatura_Maxima"] = df["Temperatura_Maxima"].fillna(media_temp)
print(f"\n Media usada para rellenar nulos en Temperatura_Maxima: {media_temp:.2f}°C")

#temperarura categoría
bins = [-np.inf, 15, 25, np.inf]
labels = ["Frío", "Templado", "Caluroso"]
df["Categoria_Temp"] = pd.cut(df["Temperatura_Maxima"], bins=bins, labels=labels)

print("\n Distribución de temperaturas por categoría:")
print(df["Categoria_Temp"].value_counts())

# analizar por ciudad
promedios = df.groupby("Ciudad")["Temperatura_Maxima"].mean().sort_values()
print("\n Promedio de temperatura por ciudad:")
print(promedios)

#temperatura y humedad
df["Humedad_Relativa"] = pd.to_numeric(df["Humedad_Relativa"], errors="coerce")
correlacion = df["Temperatura_Maxima"].corr(df["Humedad_Relativa"])
print(f"\n Correlación entre Temperatura_Maxima y Humedad_Relativa: {correlacion:.2f}")

# guardar archivo procesado
df.to_csv("reto1_temperatura_procesado.csv", index=False)
print("\n Archivo procesado guardado como 'reto1_temperatura_procesado.csv'")

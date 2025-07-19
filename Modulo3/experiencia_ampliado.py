import pandas as pd
import numpy as np

# Cargar el archivo CSV
df = pd.read_csv('lectura_archivos/entrada/reto3_experiencia_ampliado.csv')

#Convertir a numérico por seguridad 
df["Experiencia"] = pd.to_numeric(df["Experiencia"], errors="coerce")
df["Edad"] = pd.to_numeric(df["Edad"], errors="coerce")

# Rellenar nulos en Experiencia con la mediana por Área
nulos = df["Experiencia"].isnull().sum()
df["Experiencia"] = df.groupby("Area")["Experiencia"].transform(
    lambda x: x.fillna(x.median())
)
print(f"\n Valores nulos en Experiencia detectados y rellenados con la mediana por Área: {nulos} nulos tratados.")

# Detección y tratamiento de outliers en Experiencia
q1 = df["Experiencia"].quantile(0.25)
q3 = df["Experiencia"].quantile(0.75)
iqr = q3 - q1
limite_inferior = q1 - 1.5 * iqr
limite_superior = q3 + 1.5 * iqr

outliers = df[(df["Experiencia"] < limite_inferior) | (df["Experiencia"] > limite_superior)]
print(f"\n Outliers encontrados en Experiencia: {len(outliers)} registros")
print(outliers[["ID", "Experiencia"]])


# Clasificación en grupos de experiencia
bins = [-1, 2, 5, 10, np.inf]
labels = ["Principiante", "Intermedio", "Avanzado", "Experto"]
df["Grupo_Experiencia"] = pd.cut(df["Experiencia"], bins=bins, labels=labels)

print("\n Distribución por Grupo_Experiencia:")
print(df["Grupo_Experiencia"].value_counts())

# Análisis entre Edad y Experiencia
correlacion = df["Edad"].corr(df["Experiencia"])
print(f"\n Correlación entre Edad y Experiencia: {correlacion:.2f}")
if correlacion > 0.7:
    interpretacion = "Alta relación positiva"
elif correlacion > 0.3:
    interpretacion = "Relación moderada positiva"
elif correlacion > 0:
    interpretacion = "Relación débil positiva"
elif correlacion < -0.7:
    interpretacion = "Alta relación negativa"
elif correlacion < -0.3:
    interpretacion = "Relación moderada negativa"
elif correlacion < 0:
    interpretacion = "Relación débil negativa"
else:
    interpretacion = "Sin relación"

print(f"Interpretación: {interpretacion}")

# Guardar archivo procesado
df.to_csv("reto3_experiencia_procesado.csv", index=False)
print("\n Archivo guardado como 'reto3_experiencia_procesado.csv'")
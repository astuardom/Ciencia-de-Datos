import pandas as pd
import numpy as np

# Cargar el archivo CSV
df = pd.read_csv('lectura_archivos/entrada/reto2_calificaciones_ampliado.csv')

# eliminar duplicados
df = df.drop_duplicates()
df["Calificacion"] = pd.to_numeric(df["Calificacion"], errors="coerce")
nulos = df["Calificacion"].isnull().sum()
print(f"\n Duplicados eliminados. Valores nulos en Calificación: {nulos}")

# importar nulos
df["Calificacion"] = df.groupby("Materia")["Calificacion"].transform(
    lambda x: x.fillna(x.mean())
)
print("\n Nulos imputados con la media por Materia.")

# cuartiles
q1 = df["Calificacion"].quantile(0.25)
q2 = df["Calificacion"].quantile(0.50)
q3 = df["Calificacion"].quantile(0.75)
iqr = q3 - q1
limite_inferior = q1 - 1.5 * iqr
limite_superior = q3 + 1.5 * iqr

outliers = df[(df["Calificacion"] < limite_inferior) | (df["Calificacion"] > limite_superior)]
valores_no_outliers = df[(df["Calificacion"] >= limite_inferior) & (df["Calificacion"] <= limite_superior)]

print("\n[3] Análisis de Calificaciones:")
print(f"Q1: {q1:.2f}")
print(f"Q2 (mediana): {q2:.2f}")
print(f"Q3: {q3:.2f}")
print(f"IQR: {iqr:.2f}")
print(f"Limite inferior: {limite_inferior:.2f}")
print(f"Limite superior: {limite_superior:.2f}")
print(f"\nOutliers encontrados ({len(outliers)} registros):")
print(outliers[["ID", "Calificacion"]])

# Análisis por género y materia 
print("\n Promedio de calificaciones por Género y Materia:")
resumen = df.groupby(["Genero", "Materia"])["Calificacion"].mean().round(2)
print(resumen)


# Guardar archivo procesado
df.to_csv("reto2_calificaciones_procesado.csv", index=False)
print("\n Archivo guardado como 'reto2_calificaciones_procesado.csv'")
import pandas as pd
import numpy as np

# Cargar el archivo CSV
df = pd.read_csv('lectura_archivos/entrada/ejemplo_datos_outliers.csv')

# Inspeccionar los datos
print(df.head())
print(df.info())

# Limpiar errores de datos no numéricos
df["Edad"] = pd.to_numeric(df["Edad"], errors='coerce')
df["Salario"] = pd.to_numeric(df["Salario"], errors='coerce')
df["Horas_Trabajo_Semanal"] = pd.to_numeric(df["Horas_Trabajo_Semanal"], errors='coerce')

# Guardar datos limpios
df.to_csv("lectura_archivos/salida/datos_limpios.csv", index=False, na_rep="NaN")

# Eliminar duplicados
df_sin_duplicados = df.drop_duplicates().copy()
df_sin_duplicados.to_csv("lectura_archivos/salida/datos_sin_duplicados.csv", index=False, na_rep="NaN")

# Mostrar diferencias
antes = len(df)
despues = len(df_sin_duplicados)
eliminados = antes - despues
print("Filas antes:", antes)
print("Filas después:", despues)  
print("Filas eliminadas:", eliminados)

df_sin_duplicados["Edad"] = df_sin_duplicados["Edad"].fillna(df["Edad"].mean())
df_sin_duplicados["Salario"] = df_sin_duplicados["Salario"].fillna(df["Salario"].mean())
df_sin_duplicados["Horas_Trabajo_Semanal"] = df_sin_duplicados["Horas_Trabajo_Semanal"].fillna(df["Horas_Trabajo_Semanal"].mean())

df_sin_duplicados.to_csv("lectura_archivos/salida/datos_rellenados.csv", index=False)

df["Horas_Trabajo_Semanal"] =pd.cut(df["Horas_Trabajo_Semanal"], bins=[0,30,40,50,np.inf], labels=["Bajo", "Medio", "Alto", "Muy Alto"])
df.to_csv("lectura_archivos/salida/datos_categorizados.csv", index=False)


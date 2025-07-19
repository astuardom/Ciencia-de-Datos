import pandas as pd
import numpy as np

# 1. LIMPIEZA Y TRANSFORMACIÓN DE DATOS
df = pd.read_csv("migracion.csv")

# Tratar valores perdidos
df["Cantidad_Migrantes"] = df["Cantidad_Migrantes"].fillna(df["Cantidad_Migrantes"].median())
df["Razon_Migracion"] = df["Razon_Migracion"].fillna("Desconocida")

# Filtrar outliers en PIB_Origen usando IQR
Q1 = df["PIB_Origen"].quantile(0.25)
Q3 = df["PIB_Origen"].quantile(0.75)
IQR = Q3 - Q1
limite_superior = Q3 + 1.5 * IQR
df = df[df["PIB_Origen"] <= limite_superior]

# Reemplazar valores de Razon_Migracion
mapa_razones = {
    "Económica": "Trabajo",
    "Conflicto": "Guerra",
    "Educación": "Estudios",
    "Cambio Climático": "Clima",
    "Desconocida": "Otra"
}
df["Razon_Migracion"] = df["Razon_Migracion"].replace(mapa_razones)

# 2. ANÁLISIS EXPLORATORIO
print("\n🔹 Primeras filas:\n", df.head())
print("\n🔹 Info:\n"); df.info()
print("\n🔹 Describe:\n", df.describe())
print("\n🔹 Media migrantes:", df["Cantidad_Migrantes"].mean())
print("🔹 Mediana migrantes:", df["Cantidad_Migrantes"].median())
print("🔹 PIB Origen promedio:", df["PIB_Origen"].mean())
print("🔹 PIB Destino promedio:", df["PIB_Destino"].mean())
print("🔹 Conteo razones migración:\n", df["Razon_Migracion"].value_counts())

# 3. AGRUPAMIENTO Y SUMARIZACIÓN
print("\n🔸 Total migrantes por razón:\n", df.groupby("Razon_Migracion")["Cantidad_Migrantes"].sum())
print("\n🔸 Promedio IDH Origen por razón:\n", df.groupby("Razon_Migracion")["IDH_Origen"].mean())
print("\n🔸 Ordenado por cantidad de migrantes:\n", df.sort_values(by="Cantidad_Migrantes", ascending=False).head())

# 4. FILTROS Y NUEVAS COLUMNAS
df_conflicto = df[df["Razon_Migracion"] == "Guerra"]
print("\n🔹 Migraciones por conflicto:\n", df_conflicto)

df_idh_alto = df[df["IDH_Destino"] > 0.90]
print("\n🔹 Migraciones con IDH destino > 0.90:\n", df_idh_alto)

df["Diferencia_IDH"] = df["IDH_Destino"] - df["IDH_Origen"]

# 5. EXPORTACIÓN
df.to_csv("C:/Users/admin/Documents/Ciencias de Datos/Tareas/Migracion_Limpio.csv", index=False)
print("\n✅ Archivo exportado como Migracion_Limpio.csv")

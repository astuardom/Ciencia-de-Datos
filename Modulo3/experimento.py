import pandas as pd

# 1. Leer el archivo CSV original
df = pd.read_csv("experimentos_ciencias.csv")

# Copia para no modificar el original
df_original = df.copy()

# 2. Filtrar resultados por experimento (ej: "Fotosíntesis")
filtro_fotosintesis = df[df["Experimento"] == "Fotosíntesis"].copy()

# 3. Convertir unidades / normalizar resultados
# Normaliza Fotosíntesis (% O2) multiplicando por 100
df["Resultado_Normalizado"] = df.apply(
    lambda row: row["Resultado"] * 100 if row["Experimento"] == "Fotosíntesis" and pd.notnull(row["Resultado"]) else row["Resultado"],
    axis=1
)

# 4. Calcular promedios por experimento
promedios = df.groupby("Experimento", as_index=False)["Resultado"].mean()

# 5. Limpiar datos nulos o unidades mal escritas (limpieza selectiva)
df["Unidad"] = df["Unidad"].fillna("Desconocido")
df["Resultado"] = df["Resultado"].fillna(0)
df["Observaciones"] = df["Observaciones"].fillna("Sin observaciones")

# 6. Agrupar y contar experimentos por grado
conteo_experimentos = df.groupby("Grado")["Experimento"].value_counts().unstack(fill_value=0)

# 7. Detectar anomalías por experimento usando IQR
def detectar_outliers(grupo):
    grupo = grupo.copy()
    q1 = grupo["Resultado"].quantile(0.25)
    q3 = grupo["Resultado"].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return grupo[(grupo["Resultado"] < lower) | (grupo["Resultado"] > upper)]

# Elimina warning de pandas al usar groupby + apply
anomalías = df.groupby("Experimento", group_keys=False).apply(detectar_outliers).reset_index(drop=True)

# 8. Exportar reporte limpio a nuevo archivo CSV
df.to_csv("experimentos_ciencias_reporte.csv", index=False)

# 9. Imprimir resultados clave
print("Filtrado por experimento (Fotosíntesis):")
print(filtro_fotosintesis)

print("\n Promedios por experimento:")
print(promedios)

print("\n Conteo de experimentos por grado:")
print(conteo_experimentos)

print("\n Anomalías detectadas:")
print(anomalías)

print("\n Reporte exportado como: 'experimentos_ciencias_reporte.csv'")

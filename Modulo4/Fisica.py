# Fisica.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

output_dir = "salidas"
os.makedirs(output_dir, exist_ok=True)

def cargar_datos():
    try:
        df_colisiones = pd.read_csv("entrada/colisiones_principal.csv")
        df_detectores = pd.read_csv("entrada/detector_secundario.csv")
        print("📥 Datos cargados correctamente.")
        return df_colisiones, df_detectores
    except FileNotFoundError as e:
        print(f"❌ Error al cargar archivos: {e}")
        return None, None

def explorar_datos(df):
    print("🔍 Exploración inicial:")
    print(df.head())
    print(df.info())
    print(df.describe())
    print(df.isnull().sum())

def calcular_frecuencias(df):
    return df["Tipo_Particula"].value_counts(), df["Estabilidad"].value_counts()

def calcular_estadisticas(df):
    return {
        "energia": {
            "media": df["Energia"].mean(),
            "mediana": df["Energia"].median(),
            "moda": df["Energia"].mode().iloc[0],
            "std": df["Energia"].std(ddof=1),
            "percentiles": df["Energia"].quantile([0.25, 0.5, 0.75]).to_dict(),
            "rango": df["Energia"].max() - df["Energia"].min()
        },
        "momento": {
            "media": df["Momento"].mean(),
            "mediana": df["Momento"].median(),
            "moda": df["Momento"].mode().iloc[0],
            "std": df["Momento"].std(ddof=1),
            "percentiles": df["Momento"].quantile([0.25, 0.5, 0.75]).to_dict(),
            "rango": df["Momento"].max() - df["Momento"].min()
        }
    }

def detectar_outliers(df):
    Q1 = df["Energia"].quantile(0.25)
    Q3 = df["Energia"].quantile(0.75)
    IQR = Q3 - Q1
    lim_inf = Q1 - 1.5 * IQR
    lim_sup = Q3 + 1.5 * IQR
    outliers_iqr = df[(df["Energia"] < lim_inf) | (df["Energia"] > lim_sup)]
    outliers_extremos = df[df["Energia"] > 20]
    return pd.concat([outliers_iqr, outliers_extremos]).drop_duplicates()

def exportar_graficos(df, df_limpio, df_final):
    plt.figure(figsize=(8, 5))
    sns.boxplot(x=df["Energia"])
    plt.title("Boxplot de Energía")
    plt.savefig(f"{output_dir}/boxplot_energia.png")
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.histplot(df["Energia"], kde=True)
    plt.title("Histograma de Energía (Original)")
    plt.xlabel("Energía (TeV)")
    plt.savefig(f"{output_dir}/histograma_energia_original.png")
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.histplot(df_limpio["Energia"], kde=True)
    plt.title("Histograma de Energía (Limpio)")
    plt.xlabel("Energía (TeV)")
    plt.savefig(f"{output_dir}/histograma_energia_limpio.png")
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df_final, x="Tipo_Particula", y="Momento")
    plt.xticks(rotation=45)
    plt.title("Boxplot del Momento por Tipo de Partícula")
    plt.ylabel("Momento (GeV/c)")
    plt.savefig(f"{output_dir}/boxplot_momento_particula.png")
    plt.close()

    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df_final, x="Energia", y="Momento", hue="Estabilidad")
    plt.title("Energía vs. Momento por Estabilidad")
    plt.xlabel("Energía (TeV)")
    plt.ylabel("Momento (GeV/c)")
    plt.savefig(f"{output_dir}/dispersión_energia_momento.png")
    plt.close()

def exportar_datos(frecuencia_particula, frecuencia_estabilidad, agrupado, df_final, estadisticas):
    frecuencia_particula.to_csv(f"{output_dir}/frecuencia_particula.csv")
    frecuencia_estabilidad.to_csv(f"{output_dir}/frecuencia_estabilidad.csv")
    agrupado.to_csv(f"{output_dir}/agrupacion_resumen.csv", index=False)
    df_final.to_csv(f"{output_dir}/df_fisica_completo.csv", index=False)

    with open(f"{output_dir}/estadisticas.txt", "w") as f:
        for var, stats in estadisticas.items():
            f.write(f"Estadísticas para {var}:\n")
            for k, v in stats.items():
                f.write(f"  {k}: {v}\n")
            f.write("\n")

# ==== FLUJO PRINCIPAL ====
df_colisiones, df_detectores = cargar_datos()
if df_colisiones is not None and df_detectores is not None:
    explorar_datos(df_colisiones)

    print("📊 Calculando estadísticas...")
    frecuencia_particula, frecuencia_estabilidad = calcular_frecuencias(df_colisiones)
    estadisticas = calcular_estadisticas(df_colisiones)

    print("🚨 Detectando outliers...")
    total_outliers = detectar_outliers(df_colisiones)

    print("🧹 Limpiando datos...")
    df_colisiones_limpio = df_colisiones[~df_colisiones.index.isin(total_outliers.index)]

    print("🔗 Fusionando datos...")
    df_fisica_completo = df_colisiones_limpio.merge(
        df_detectores, on="ID", how="left", suffixes=("", "_detector")
    )

    print("📈 Agrupando datos...")
    agrupado = df_fisica_completo.groupby(["Tipo_Particula", "Estabilidad"]).agg(
        promedio_energia=("Energia", "mean"),
        promedio_momento=("Momento", "mean"),
        conteo=("Energia", "count")
    ).reset_index()

    print("🖼️ Exportando visualizaciones...")
    exportar_graficos(df_colisiones, df_colisiones_limpio, df_fisica_completo)

    print("💾 Exportando archivos de salida...")
    exportar_datos(frecuencia_particula, frecuencia_estabilidad, agrupado, df_fisica_completo, estadisticas)

    print("✅ Análisis finalizado exitosamente.")
else:
    print("🚫 El análisis fue cancelado por error en la carga de datos.")

from matplotlib.backends.backend_pdf import PdfPages

# Exportar a PDF
with PdfPages('salidas/reporte_colisiones.pdf') as pdf:

    # Página 1: Portada
    fig = plt.figure(figsize=(11, 8.5))
    plt.suptitle("Reporte de Análisis de Datos de Colisiones", fontsize=18, y=0.95)
    plt.text(0.1, 0.6, f"Registros originales: {len(df_colisiones)}", fontsize=12)
    plt.text(0.1, 0.5, f"Registros luego de limpieza: {len(df_colisiones_limpio)}", fontsize=12)
    plt.axis('off')
    pdf.savefig(fig)
    plt.close()

    # Página 2: Tabla de frecuencia por tipo de partícula
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis('off')
    tabla_freq = frecuencia_particula.reset_index()
    tabla_freq.columns = ['Tipo de Partícula', 'Frecuencia']
    table = ax.table(cellText=tabla_freq.values,
                     colLabels=tabla_freq.columns,
                     loc='center',
                     cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.3, 1.5)
    ax.set_title("Frecuencia por Tipo de Partícula", fontsize=14, pad=20)
    pdf.savefig()
    plt.close()

    # Página 3: Tabla de estadísticas descriptivas
    resumen_estadisticas = pd.DataFrame({
        "Media": [estadisticas["energia"]["media"], estadisticas["momento"]["media"]],
        "Mediana": [estadisticas["energia"]["mediana"], estadisticas["momento"]["mediana"]],
        "Moda": [estadisticas["energia"]["moda"], estadisticas["momento"]["moda"]],
        "Desv. Estándar": [estadisticas["energia"]["std"], estadisticas["momento"]["std"]],
        "Rango": [estadisticas["energia"]["rango"], estadisticas["momento"]["rango"]],
    }, index=["Energía", "Momento"])

    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis('off')
    table = ax.table(cellText=resumen_estadisticas.round(2).values,
                     rowLabels=resumen_estadisticas.index,
                     colLabels=resumen_estadisticas.columns,
                     loc='center',
                     cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.4, 1.6)
    ax.set_title("Estadísticas Descriptivas de Energía y Momento", fontsize=14, pad=20)
    pdf.savefig()
    plt.close()

    # Página 4: Histogramas de Energía
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    sns.histplot(df_colisiones["Energia"], kde=True, ax=ax1, color='skyblue')
    ax1.set_title("Histograma Energía (Original)")
    sns.histplot(df_colisiones_limpio["Energia"], kde=True, ax=ax2, color='lightgreen')
    ax2.set_title("Histograma Energía (Limpio)")
    fig.suptitle("Comparación de Histogramas de Energía", fontsize=14)
    pdf.savefig()
    plt.close()

    # Página 5: Boxplot Energía (original vs limpio)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    sns.boxplot(x=df_colisiones["Energia"], ax=ax1, color='skyblue')
    ax1.set_title("Boxplot Energía (Original)")
    sns.boxplot(x=df_colisiones_limpio["Energia"], ax=ax2, color='lightgreen')
    ax2.set_title("Boxplot Energía (Limpio)")
    fig.suptitle("Comparación Boxplots de Energía", fontsize=14)
    pdf.savefig()
    plt.close()

    # Página 6: Boxplot Momento por tipo de partícula
    fig = plt.figure(figsize=(10, 6))
    sns.boxplot(data=df_fisica_completo, x="Tipo_Particula", y="Momento")
    plt.xticks(rotation=45)
    plt.title("Boxplot del Momento por Tipo de Partícula")
    plt.ylabel("Momento (GeV/c)")
    pdf.savefig()
    plt.close()

    # Página 7: Dispersión Energía vs Momento
    fig = plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df_fisica_completo, x="Energia", y="Momento", hue="Estabilidad")
    plt.title("Dispersión: Energía vs Momento por Estabilidad")
    plt.xlabel("Energía (TeV)")
    plt.ylabel("Momento (GeV/c)")
    pdf.savefig()
    plt.close()

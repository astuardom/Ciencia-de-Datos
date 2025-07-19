import pandas as pd
import numpy as np     
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import os

# Crear carpeta de salida
os.makedirs('salida', exist_ok=True)

# Cargar datos
df = pd.read_csv('entrada/dataset_con_outliers.csv')

# Frecuencias categóricas
freq_color = df['Color'].value_counts()
freq_satisfaction = df['Satisfaction'].value_counts(sort=False)

# Emparejar filas para tabla
max_len = max(len(freq_color), len(freq_satisfaction))
freq_color = freq_color.reindex(range(max_len)).fillna('')
freq_satisfaction = freq_satisfaction.reindex(range(max_len)).fillna('')
tabla_frecuencias = pd.DataFrame({'Color': freq_color.values, 'Satisfaction': freq_satisfaction.values})

# Estadísticas
def calcular_estadisticas(dataframe):
    return pd.DataFrame({
        'Mean': dataframe.mean().round(2),
        'Median': dataframe.median().round(2),
        'Moda': dataframe.mode().iloc[0],
        'Variance': dataframe.var(ddof=1).round(2),
        'Std_dev': dataframe.std(ddof=1).round(2),
        'iQR': (dataframe.quantile(0.75) - dataframe.quantile(0.25)).round(2),
    })

# Función outliers
def detectar_outliers_iqr(series):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    inf = Q1 - 1.5 * IQR
    sup = Q3 + 1.5 * IQR
    outliers = (series < inf) | (series > sup)
    return outliers, inf, sup

# Estadísticas
estadisticas_originales = calcular_estadisticas(df[['Children', 'Temperature']])
out_children, li_c, ls_c = detectar_outliers_iqr(df['Children'])
out_temp, li_t, ls_t = detectar_outliers_iqr(df['Temperature'])

df_clean = df[df['Children'].between(li_c, ls_c) & df['Temperature'].between(li_t, ls_t)]
estadisticas_limpias = calcular_estadisticas(df_clean[['Children', 'Temperature']])

# Reorganizar resumen para buena visualización
resumen = pd.concat([estadisticas_originales.T, estadisticas_limpias.T], axis=1)
resumen.columns = ['Original', 'Limpio']

# Exportar a PDF
with PdfPages('salida/analisis_reporte_mejorado.pdf') as pdf:

    # Página 1: Portada
    fig = plt.figure(figsize=(11, 8.5))
    plt.suptitle("Reporte de Análisis de Datos", fontsize=16, y=0.95)
    plt.text(0.1, 0.6, f"Registros originales: {len(df)}", fontsize=12)
    plt.text(0.1, 0.5, f"Registros limpios: {len(df_clean)}", fontsize=12)
    plt.axis('off')
    pdf.savefig(fig)
    plt.close()

    # Página 2: Tabla de frecuencias
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis('off')
    table = ax.table(cellText=tabla_frecuencias.values,
                     colLabels=tabla_frecuencias.columns,
                     loc='center',
                     cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.3, 1.5)
    ax.set_title("Frecuencia de Variables Categóricas", fontsize=14, pad=20)
    pdf.savefig()
    plt.close()

    # Página 3: Tabla de estadísticas descriptivas horizontal
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis('off')
    table = ax.table(cellText=resumen.values.round(2),
                     rowLabels=resumen.index,
                     colLabels=resumen.columns,
                     loc='center',
                     cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.5, 1.5)
    ax.set_title("Estadísticas Descriptivas (Original vs Limpio)", fontsize=14, pad=20)
    pdf.savefig()
    plt.close()

    # Página 4: Histogramas Children
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.hist(df['Children'], bins=range(df['Children'].min(), df['Children'].max()+1), edgecolor='black')
    ax1.set_title("Histograma Children (Original)")
    ax2.hist(df_clean['Children'], bins=range(df_clean['Children'].min(), df_clean['Children'].max()+1), edgecolor='black', color='green')
    ax2.set_title("Histograma Children (Limpio)")
    fig.suptitle("Histograma Comparativo - Children", fontsize=14)
    pdf.savefig()
    plt.close()

    # Página 5: Boxplots Children
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    sns.boxplot(x=df['Children'], ax=ax1, color='skyblue')
    ax1.set_title("Boxplot Children (Original)")
    sns.boxplot(x=df_clean['Children'], ax=ax2, color='lightgreen')
    ax2.set_title("Boxplot Children (Limpio)")
    fig.suptitle("Boxplot Comparativo - Children", fontsize=14)
    pdf.savefig()
    plt.close()

    # Página 6: Histogramas Temperature
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.hist(df['Temperature'], bins=20, edgecolor='black')
    ax1.set_title("Histograma Temperature (Original)")
    ax2.hist(df_clean['Temperature'], bins=20, edgecolor='black', color='orange')
    ax2.set_title("Histograma Temperature (Limpio)")
    fig.suptitle("Histograma Comparativo - Temperature", fontsize=14)
    pdf.savefig()
    plt.close()

    # Página 7: Boxplots Temperature
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    sns.boxplot(y=df['Temperature'], ax=ax1, color='lightblue')
    ax1.set_title("Boxplot Temperature (Original)")
    sns.boxplot(y=df_clean['Temperature'], ax=ax2, color='lightgreen')
    ax2.set_title("Boxplot Temperature (Limpio)")
    fig.suptitle("Boxplot Comparativo - Temperature", fontsize=14)
    pdf.savefig()
    plt.close()

# Reto 4 – Segmentación Operativa de Pozos Petroleros
# Empresa: Saudi Aramco | Departamento: Operaciones Subterráneas
# Área Responsable: Ciencia de Datos y Optimización de Producción

# Este programa tiene como objetivo analizar y visualizar la producción de petróleo
# y gas natural en pozos ubicados en diferentes campos de Arabia Saudita,
# utilizando técnicas avanzadas de visualización, segmentación y regresión múltiple.

# El conjunto de datos incluye variables como:
# - Producción de petróleo (oil_bbl)
# - Producción de gas (gas_mscf)
# - Corte de agua (water_cut_pct)
# - Presión del yacimiento (reservoir_pressure_psi)
# - Temperatura del yacimiento (reservoir_temperature_c)
# - Profundidad del pozo (well_depth_m)

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from jinja2 import Template
from datetime import datetime
import warnings
import os

warnings.filterwarnings("ignore")

# Función para generar y guardar gráficos

def guardar_grafico(figura, nombre):
    figura.savefig(nombre, dpi=300, bbox_inches='tight')
    plt.close(figura)

# Crear carpetas necesarias
os.makedirs("graficos", exist_ok=True)
os.makedirs("entrada", exist_ok=True)

# Validación inicial
df = pd.read_csv("entrada/petroleum_saudi_500.csv")
df['date'] = pd.to_datetime(df['date'])
print("\n✔️ Datos cargados correctamente")
print("\n🔹 Información general del DataFrame:")
print(df.info())
print("\n🔹 Estadísticas descriptivas:")
print(df.describe())

if df.isnull().sum().any():
    print("\n⚠️ Valores nulos encontrados:")
    print(df.isnull().sum())
else:
    print("\n✅ No hay valores nulos en el dataset")

# Clasificación de condiciones de operación
df['profundidad_cat'] = pd.cut(df['well_depth_m'], [0, 3100, 3500, np.inf], labels=['Poco profundo', 'Intermedio', 'Profundo'])
df['corte_agua_cat'] = pd.cut(df['water_cut_pct'], [-1, 25, 40, 100], labels=['Bajo', 'Medio', 'Alto'])
df['presion_cat'] = pd.qcut(df['reservoir_pressure_psi'], q=3, labels=['Baja', 'Media', 'Alta'])

print("\n🔹 Distribución de pozos por categoría de profundidad:")
print(df['profundidad_cat'].value_counts())

print("\n🔹 Promedio de producción por categoría de corte de agua:")
print(df.groupby('corte_agua_cat')[['oil_bbl', 'gas_mscf']].mean())

print("\n🔹 Media de presión y temperatura por campo petrolero:")
print(df.groupby('field')[['reservoir_pressure_psi', 'reservoir_temperature_c']].mean())

# Análisis de frecuencia
cross1 = pd.crosstab(df['field'], df['corte_agua_cat'])
cross2 = pd.crosstab(df['profundidad_cat'], df['presion_cat'])

fig1 = plt.figure(figsize=(8, 5))
sns.heatmap(cross1, annot=True, cmap='Blues', annot_kws={'size': 9})
plt.title("Relación entre Campo y Corte de Agua")
plt.xlabel("Corte de Agua")
plt.ylabel("Campo Petrolero")
guardar_grafico(fig1, "graficos/grafico_cross1.png")

fig2 = plt.figure(figsize=(6, 4))
sns.heatmap(cross2, annot=True, cmap='OrRd', annot_kws={'size': 9})
plt.title("Relación entre Profundidad del Pozo y Presión")
plt.xlabel("Presión del Yacimiento")
plt.ylabel("Categoría de Profundidad")
guardar_grafico(fig2, "graficos/grafico_cross2.png")

# Comparación de grupos
grouped1 = df.groupby('corte_agua_cat')[['oil_bbl', 'gas_mscf']].mean()
grouped2 = df.groupby('field')[['reservoir_pressure_psi', 'reservoir_temperature_c']].mean()

fig3 = plt.figure(figsize=(6, 4))
sns.boxplot(x='presion_cat', y='oil_bbl', data=df)
plt.title("Distribución de Producción de Petróleo según Presión")
plt.xlabel("Categoría de Presión")
plt.ylabel("Producción de Petróleo (oil_bbl)")
guardar_grafico(fig3, "graficos/grafico_boxplot_oil.png")

# Histograma + KDE
fig6 = plt.figure(figsize=(8, 5))
sns.histplot(df['oil_bbl'], kde=True, color='green')
plt.title('Distribución de Producción de Petróleo (oil_bbl)')
plt.xlabel('Producción de Petróleo (barriles)')
plt.ylabel('Frecuencia')
guardar_grafico(fig6, 'graficos/histograma_oil.png')

# Matriz de Dispersión (Pairplot)
sns.pairplot(df[['oil_bbl', 'gas_mscf', 'water_cut_pct', 'reservoir_pressure_psi']])
plt.suptitle("Matriz de Dispersión entre Variables Clave", y=1.02)
plt.savefig("graficos/grafico_pairplot.png", dpi=300, bbox_inches='tight')
plt.close()

# Correlación
corr = df[['oil_bbl', 'gas_mscf', 'water_cut_pct', 'reservoir_pressure_psi',
           'reservoir_temperature_c', 'well_depth_m']].corr()
fig4 = plt.figure(figsize=(8, 5))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Matriz de Correlación entre Variables Operativas")
guardar_grafico(fig4, "graficos/grafico_correlacion.png")

# Clustering
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(df[['oil_bbl', 'gas_mscf', 'reservoir_pressure_psi', 'water_cut_pct']])
kmeans = KMeans(n_clusters=3, random_state=42)
df['cluster'] = kmeans.fit_predict(data_scaled)

fig5 = plt.figure(figsize=(6, 4))
sns.scatterplot(x='oil_bbl', y='gas_mscf', hue='cluster', data=df, palette='viridis')
plt.title("Clustering de Pozos según Producción de Petróleo y Gas")
plt.xlabel("Producción de Petróleo (oil_bbl)")
plt.ylabel("Producción de Gas (gas_mscf)")
plt.legend(title="Cluster")
guardar_grafico(fig5, "graficos/grafico_clustering.png")

print("\n🔹 Promedio de producción por clúster:")
print(df.groupby('cluster')[['oil_bbl', 'gas_mscf']].mean())

# Regresión lineal múltiple
X = df[['reservoir_pressure_psi', 'water_cut_pct', 'well_depth_m']]
y = df['oil_bbl']
model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)
r2 = round(r2_score(y, y_pred), 4)
mse = round(mean_squared_error(y, y_pred), 2)
mae = round(mean_absolute_error(y, y_pred), 2)

# Informe HTML
html_template = """
<!DOCTYPE html>
<html lang='es'>
<head><meta charset='UTF-8'><title>Informe Web – Segmentación Pozos</title></head>
<body style='font-family: Arial; margin: 40px;'>
<h1>Informe Web: Segmentación de Pozos Petroleros</h1>
<h2>Saudi Aramco – Ciencia de Datos</h2>
<p><strong>Fecha de generación:</strong> {{ fecha }}</p>
<h3 id='objetivo'>1. Objetivo del Análisis</h3>
<p>Identificar patrones operativos y agrupar pozos similares mediante análisis estadístico, segmentación y modelos predictivos.</p>
<h3 id='corte'>2. Producción promedio por categoría de corte de agua</h3>
{{ tabla1 }}
<h3>3. Promedio de presión y temperatura por campo</h3>
{{ tabla2 }}
<h3>4. Visualizaciones Interpretadas</h3>
<ul>
  <li><strong>Campo vs Corte de Agua:</strong><br><img src='graficos/grafico_cross1.png' width='500'></li>
  <li><strong>Profundidad vs Presión:</strong><br><img src='graficos/grafico_cross2.png' width='500'></li>
  <li><strong>Producción vs Presión (Boxplot):</strong><br><img src='graficos/grafico_boxplot_oil.png' width='500'></li>
  <li><strong>Distribución de Producción de Petróleo:</strong><br><img src='graficos/histograma_oil.png' width='500'></li>
  <li><strong>Matriz de Dispersión:</strong><br><img src='graficos/grafico_pairplot.png' width='500'></li>
  <li><strong>Matriz de Correlación:</strong><br><img src='graficos/grafico_correlacion.png' width='500'></li>
  <li><strong>Segmentación con KMeans:</strong><br><img src='graficos/grafico_clustering.png' width='500'></li>
</ul>
<h3>5. Correlación entre Variables</h3>
<pre>{{ correlacion }}</pre>
<h3>6. Evaluación del Modelo</h3>
<ul>
  <li><strong>R²:</strong> {{ r2 }}</li>
  <li><strong>MSE:</strong> {{ mse }}</li>
  <li><strong>MAE:</strong> {{ mae }}</li>
</ul>
<h3>7. Conclusiones</h3>
<ul>
  <li>Los pozos con mayor corte de agua tienden a producir menos.</li>
  <li>Existen tres clústeres operativos bien diferenciados.</li>
  <li>El modelo lineal tiene bajo poder explicativo, se sugiere probar modelos más complejos.</li>
</ul>
</body>
</html>
"""

render = Template(html_template).render(
    fecha=datetime.today().strftime("%Y-%m-%d"),
    tabla1=grouped1.to_html(classes='table', border=0),
    tabla2=grouped2.to_html(classes='table', border=0),
    correlacion=corr.to_string(),
    r2=r2,
    mse=mse,
    mae=mae
)

with open("informe_segmentacion_pozos.html", "w", encoding="utf-8") as f:
    f.write(render)

print("\n✅ Informe generado como 'informe_segmentacion_pozos.html'")

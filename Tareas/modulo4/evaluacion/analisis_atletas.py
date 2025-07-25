import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mode, pearsonr
from sklearn.linear_model import LinearRegression
import os

# 1. Análisis Exploratorio de Datos
entrada_path = 'entrada/olimpicos.csv'
df = pd.read_csv(entrada_path)

salida_dir = 'salidas'
os.makedirs(salida_dir, exist_ok=True)

# Guardar resumen de resultados
resumen_path = os.path.join(salida_dir, 'resumen_resultados.txt')
resumen = []

print('Primeras 5 filas:')
print(df.head())
resumen.append('Primeras 5 filas:\n' + df.head().to_string() + '\n')

print('\nInfo del DataFrame:')
df.info()
with open(resumen_path, 'w', encoding='utf-8') as f:
    f.write('Info del DataFrame:\n')
    df.info(buf=f)
    f.write('\n')

print('\nEstadísticas descriptivas:')
describe_str = df.describe().to_string()
print(describe_str)
resumen.append('Estadísticas descriptivas:\n' + describe_str + '\n')

# Histograma de entrenamientos semanales
plt.figure(figsize=(7,4))
plt.hist(df['Entrenamientos_Semanales'], bins=6, color='#1976D2', edgecolor='black', alpha=0.8)
plt.title('Distribución de entrenamientos semanales', fontsize=14, fontweight='bold')
plt.xlabel('Entrenamientos semanales', fontsize=12)
plt.ylabel('Frecuencia', fontsize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(salida_dir, 'histograma_entrenamientos.png'))
plt.close()

# 2. Estadística Descriptiva
print('\nTipos de variable:')
tipos = {
    'Atleta': 'Categórica nominal',
    'Edad': 'Cuantitativa discreta',
    'Altura_cm': 'Cuantitativa continua',
    'Peso_kg': 'Cuantitativa continua',
    'Deporte': 'Categórica nominal',
    'Entrenamientos_Semanales': 'Cuantitativa discreta',
    'Medallas_Totales': 'Cuantitativa discreta',
    'Pais': 'Categórica nominal'
}
for col, tipo in tipos.items():
    print(f'- {col}: {tipo}')
resumen.append('Tipos de variable:\n' + '\n'.join([f'- {col}: {tipo}' for col, tipo in tipos.items()]) + '\n')

media_medallas = df['Medallas_Totales'].mean()
mediana_medallas = df['Medallas_Totales'].median()
moda_medallas = mode(df['Medallas_Totales'], keepdims=True).mode[0]
print(f'\nMedia de medallas: {media_medallas}')
print(f'Mediana de medallas: {mediana_medallas}')
print(f'Moda de medallas: {moda_medallas}')
resumen.append(f'Media de medallas: {media_medallas}\nMediana de medallas: {mediana_medallas}\nModa de medallas: {moda_medallas}\n')

std_altura = df['Altura_cm'].std()
print(f'Desviación estándar de la altura: {std_altura:.2f} cm')
resumen.append(f'Desviación estándar de la altura: {std_altura:.2f} cm\n')

# Boxplot de peso para detectar outliers y mostrar valores
plt.figure(figsize=(7,4))
box = sns.boxplot(x=df['Peso_kg'], color='#FFA000', fliersize=8, linewidth=2, boxprops=dict(alpha=0.7))
plt.title('Boxplot del peso de los atletas', fontsize=14, fontweight='bold')
plt.xlabel('Peso (kg)', fontsize=12)
plt.xticks(fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(salida_dir, 'boxplot_peso.png'))
plt.close()

# Mostrar valores atípicos
q1 = df['Peso_kg'].quantile(0.25)
q3 = df['Peso_kg'].quantile(0.75)
iqr = q3 - q1
outliers = df[(df['Peso_kg'] < q1 - 1.5*iqr) | (df['Peso_kg'] > q3 + 1.5*iqr)]['Peso_kg']
if not outliers.empty:
    print('Valores atípicos en peso:', outliers.values)
    resumen.append('Valores atípicos en peso: ' + ', '.join(map(str, outliers.values)) + '\n')
else:
    print('No se detectaron valores atípicos en peso.')
    resumen.append('No se detectaron valores atípicos en peso.\n')

# 3. Análisis de Correlación
corr_pearson, pval = pearsonr(df['Entrenamientos_Semanales'], df['Medallas_Totales'])
print(f'\nCorrelación de Pearson entre entrenamientos semanales y medallas totales: {corr_pearson:.2f} (p={pval:.4f})')
resumen.append(f'Correlación de Pearson entre entrenamientos semanales y medallas totales: {corr_pearson:.2f} (p={pval:.4f})\n')

plt.figure(figsize=(7,5))
sns.scatterplot(x='Peso_kg', y='Medallas_Totales', data=df, s=120, color='#C62828', edgecolor='k', alpha=0.85)
plt.title('Relación entre peso y medallas totales', fontsize=14, fontweight='bold')
plt.xlabel('Peso (kg)', fontsize=12)
plt.ylabel('Medallas totales', fontsize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(salida_dir, 'scatter_peso_medallas.png'))
plt.close()

interpretacion_corr = ''
if abs(corr_pearson) > 0.7:
    interpretacion_corr = 'Existe una correlación fuerte.'
elif abs(corr_pearson) > 0.4:
    interpretacion_corr = 'Existe una correlación moderada.'
else:
    interpretacion_corr = 'La correlación es débil o nula.'
print('Interpretación:', interpretacion_corr)
resumen.append('Interpretación correlación: ' + interpretacion_corr + '\n')

# 4. Regresión Lineal
X = df[['Entrenamientos_Semanales']].values
y = df['Medallas_Totales'].values
modelo = LinearRegression()
modelo.fit(X, y)
intercepto = modelo.intercept_
pendiente = modelo.coef_[0]
R2 = modelo.score(X, y)
print(f'\nRegresión lineal: Medallas_Totales ~ Entrenamientos_Semanales')
print(f'Intercepto: {intercepto:.2f}, Pendiente: {pendiente:.2f}, R²: {R2:.2f}')
resumen.append(f'Regresión lineal: Medallas_Totales ~ Entrenamientos_Semanales\nIntercepto: {intercepto:.2f}, Pendiente: {pendiente:.2f}, R²: {R2:.2f}\n')

interpretacion_reg = f"Por cada entrenamiento semanal adicional, se espera que el número de medallas aumente en {pendiente:.2f}. El R² de {R2:.2f} indica que el modelo explica el {R2*100:.1f}% de la variabilidad en las medallas."
print('Interpretación regresión:', interpretacion_reg)
resumen.append('Interpretación regresión: ' + interpretacion_reg + '\n')

plt.figure(figsize=(7,5))
sns.regplot(x='Entrenamientos_Semanales', y='Medallas_Totales', data=df, color='#1976D2', line_kws={"color": "#C62828", "lw":2})
plt.title('Regresión lineal: Medallas vs Entrenamientos semanales', fontsize=14, fontweight='bold')
plt.xlabel('Entrenamientos semanales', fontsize=12)
plt.ylabel('Medallas totales', fontsize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(salida_dir, 'regresion_medallas_entrenamientos.png'))
plt.close()

# 5. Visualización de Datos
# Heatmap de correlación
plt.figure(figsize=(8,6))
matriz_corr = df.select_dtypes(include=[np.number]).corr()
sns.heatmap(matriz_corr, annot=True, cmap='coolwarm', linewidths=2, linecolor='black', fmt='.2f', square=True, cbar_kws={"label": "Correlación"})
plt.title('Heatmap de correlación entre variables numéricas', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(salida_dir, 'heatmap_correlacion.png'))
plt.close()

# Boxplot de medallas por disciplina
plt.figure(figsize=(8,5))
sns.boxplot(x='Deporte', y='Medallas_Totales', data=df, palette='Set2', linewidth=2, fliersize=8, boxprops=dict(alpha=0.8))
plt.title('Distribución de medallas por disciplina deportiva', fontsize=15, fontweight='bold')
plt.xlabel('Disciplina deportiva', fontsize=12)
plt.ylabel('Medallas totales', fontsize=12)
plt.xticks(fontsize=11)
plt.yticks(fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(salida_dir, 'boxplot_medallas_disciplina.png'))
plt.close()

# Guardar resumen de resultados
with open(resumen_path, 'a', encoding='utf-8') as f:
    for linea in resumen:
        f.write(linea + '\n') 
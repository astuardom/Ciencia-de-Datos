# 1. Instalación e importación de librerías
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 2. Creación del conjunto de datos
np.random.seed(42)
datos = {
    'Edad': np.random.randint(18, 60, 20),
    'Ingresos': np.random.randint(2000, 8000, 20),
    'Años de Educación': np.random.randint(8, 20, 20),
    'Horas de Sueño': np.random.uniform(5, 9, 20)
}
df = pd.DataFrame(datos)

print('Conjunto de datos:')
print(df.head())
print('\n')

# 3. Cálculo de la matriz de correlación
matriz_corr = df.corr()
print('Matriz de correlación:')
print(matriz_corr)
print('\n')

# Crear carpeta de salida si no existe
output_dir = 'salidas'
os.makedirs(output_dir, exist_ok=True)

# Exportar el DataFrame y la matriz de correlación a CSV
# (opcional, útil para entregar o revisar)
df.to_csv(os.path.join(output_dir, 'datos_heatmap.csv'), index=False)
matriz_corr.to_csv(os.path.join(output_dir, 'matriz_correlacion.csv'))

# 4. Generación del heatmap mejorado
dibujo = plt.figure(figsize=(9,7))
sns.heatmap(matriz_corr, annot=True, cmap='coolwarm', linewidths=2, linecolor='black', fmt='.2f',
            square=True, cbar_kws={"shrink": .8, "label": "Correlación"})
plt.title('Heatmap de Correlación entre Variables', fontsize=16, fontweight='bold')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12, rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'heatmap_correlacion.png'))
plt.show()

# 5. Interpretación
print('Interpretación:')
print('El heatmap muestra la fuerza y dirección de la correlación entre las variables numéricas. Los valores cercanos a 1 indican correlación positiva fuerte, cercanos a -1 indican correlación negativa fuerte, y cercanos a 0 indican poca o ninguna correlación. En este ejemplo, los datos son simulados, por lo que las correlaciones pueden ser débiles o aleatorias.') 
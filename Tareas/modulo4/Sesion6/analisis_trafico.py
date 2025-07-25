import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# 2. Cargar los datos desde un archivo CSV
entrada_path = 'entrada/datos_trafico.csv'
df = pd.read_csv(entrada_path)

salida_dir = 'salidas'
os.makedirs(salida_dir, exist_ok=True)

# 3. Gráfico de líneas: cantidad de vehículos por día
plt.figure(figsize=(10,5))
plt.plot(df['Día'], df['Vehículos'], marker='o', color='#1976D2', linestyle='-', linewidth=2)
plt.title('Evolución diaria del tráfico vehicular en la avenida principal', fontsize=15, fontweight='bold')
plt.xlabel('Día del mes', fontsize=12)
plt.ylabel('Cantidad de vehículos', fontsize=12)
plt.xticks(df['Día'], rotation=45)
plt.yticks(fontsize=10)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(salida_dir, 'grafico_lineas_vehiculos.png'))
plt.show()

# 4. Histograma: distribución de velocidades
plt.figure(figsize=(8,5))
plt.hist(df['Velocidad_Promedio'], bins=8, color='#FFA000', edgecolor='black', alpha=0.85)
plt.title('Distribución de la velocidad promedio de los vehículos', fontsize=15, fontweight='bold')
plt.xlabel('Velocidad promedio (km/h)', fontsize=12)
plt.ylabel('Frecuencia de días', fontsize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(salida_dir, 'histograma_velocidades.png'))
plt.show()

# 5. Gráfico de dispersión: vehículos vs accidentes
plt.figure(figsize=(8,6))
plt.scatter(df['Vehículos'], df['Accidentes'], c='#C62828', edgecolor='k', s=90, alpha=0.8)
plt.title('Relación entre cantidad de vehículos y número de accidentes diarios', fontsize=15, fontweight='bold')
plt.xlabel('Cantidad de vehículos', fontsize=12)
plt.ylabel('Número de accidentes', fontsize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.grid(alpha=0.2)
plt.tight_layout()
plt.savefig(os.path.join(salida_dir, 'grafico_accidentes.png'))
plt.show() 
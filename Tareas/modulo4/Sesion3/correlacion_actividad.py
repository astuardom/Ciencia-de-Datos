import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import os

# 1. Creación de Datos Simulados
np.random.seed(42)
# Horas de ejercicio por semana (10 a 15 personas)
horas_ejercicio = np.random.randint(0, 8, 15)
# Presión arterial sistólica (relacionada inversamente con el ejercicio, más ruido)
presion_arterial = 130 - horas_ejercicio * 4 + np.random.normal(0, 3, 15)

# 2. Construcción de una Tabla de Contingencia
# Grupo de edad y tipo de dieta
grupo_edad = np.random.choice(['Joven', 'Adulto', 'Mayor'], 15)
tipo_dieta = np.random.choice(['Saludable', 'Regular', 'No saludable'], 15)

df = pd.DataFrame({
    'Horas de ejercicio': horas_ejercicio,
    'Presión arterial': presion_arterial,
    'Grupo de edad': grupo_edad,
    'Tipo de dieta': tipo_dieta
})

print('Datos simulados:')
print(df)
print('\n')

# Tabla de contingencia
print('Tabla de contingencia (Grupo de edad vs Tipo de dieta):')
tabla_contingencia = pd.crosstab(df['Grupo de edad'], df['Tipo de dieta'])
print(tabla_contingencia)
print('\n')

# 3. Visualización con Scatterplot mejorada
# Crear carpeta de salida si no existe
output_dir = 'salidas'
os.makedirs(output_dir, exist_ok=True)

# Asignar un color a cada grupo de edad
colores = {'Joven': 'green', 'Adulto': 'blue', 'Mayor': 'orange'}
colores_puntos = df['Grupo de edad'].map(colores)

plt.figure(figsize=(8,5))
for grupo in df['Grupo de edad'].unique():
    subset = df[df['Grupo de edad'] == grupo]
    plt.scatter(subset['Horas de ejercicio'], subset['Presión arterial'],
                color=colores[grupo], label=grupo, edgecolor='k', s=80)
# Línea de tendencia
m, b = np.polyfit(df['Horas de ejercicio'], df['Presión arterial'], 1)
plt.plot(df['Horas de ejercicio'], m*df['Horas de ejercicio'] + b, color='red', linestyle='--', label='Tendencia')
plt.title('Relación entre horas de ejercicio y presión arterial')
plt.xlabel('Horas de ejercicio por semana')
plt.ylabel('Presión arterial sistólica')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'scatter_ejercicio_presion.png'))
plt.show()

# 4. Coeficiente de Correlación de Pearson
coef, p_valor = pearsonr(df['Horas de ejercicio'], df['Presión arterial'])
print(f'Coeficiente de correlación de Pearson: {coef:.2f}')
print(f'Valor p: {p_valor:.4f}')
if abs(coef) > 0.7:
    interpretacion = 'fuerte'
elif abs(coef) > 0.4:
    interpretacion = 'moderada'
else:
    interpretacion = 'débil'
print(f'Interpretación: Existe una correlación {interpretacion} entre las variables.')
print('\n')

# 5. Reflexión sobre Correlación vs. Causalidad (mejorada)
print('Reflexión:')
print('La correlación indica que existe una relación estadística entre dos variables, pero NO implica causalidad.')
print('Por ejemplo, aunque observamos que a mayor cantidad de horas de ejercicio la presión arterial tiende a ser menor, esto no significa necesariamente que el ejercicio sea la única causa de la disminución de la presión arterial. Pueden existir otros factores involucrados, como la dieta, genética, estrés, etc.')
print('\nEjemplo de correlación sin causalidad:')
print('- El número de personas que se ahogan en piscinas y la cantidad de películas en las que aparece cierto actor pueden estar correlacionados en un periodo, pero claramente no hay relación causal.')
print('\nEjemplo de causalidad sin correlación fuerte:')
print('- Fumar causa cáncer de pulmón, pero en una muestra pequeña o sesgada, la correlación puede no ser evidente.')
print('\nEn resumen:')
print('- Correlación significa que dos variables se mueven juntas, pero no necesariamente una causa la otra.')
print('- Causalidad significa que un cambio en una variable provoca un cambio en la otra.')
print('¡Siempre analiza el contexto antes de concluir causalidad!') 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os

# 1. Creación de Datos Simulados
np.random.seed(42)
# Temperatura ambiente (°C)
temperatura = np.random.uniform(15, 35, 30)
# Consumo de energía (kWh), relación lineal + ruido
desviacion = np.random.normal(0, 5, 30)
consumo_energia = 2.5 * temperatura + 10 + desviacion

# Guardar datos en DataFrame
X = temperatura.reshape(-1, 1)
y = consumo_energia

df = pd.DataFrame({'Temperatura (°C)': temperatura, 'Consumo de energía (kWh)': consumo_energia})

# Crear carpeta de salida
output_dir = 'salidas'
os.makedirs(output_dir, exist_ok=True)

df.to_csv(os.path.join(output_dir, 'datos_simulados.csv'), index=False)

print('Datos simulados:')
print(df.head())
print('\n')

# 2. Implementación del Modelo de Regresión Lineal
modelo = LinearRegression()
modelo.fit(X, y)
intercepto = modelo.intercept_
pendiente = modelo.coef_[0]
print('Coeficientes de la regresión:')
print(f'- Intercepto: {intercepto:.2f}')
print(f'- Pendiente: {pendiente:.2f}')
print('\n')

# 3. Predicción de Valores con el Modelo
predicciones = modelo.predict(X)
print('Primeros 5 valores predichos:')
print(predicciones[:5])
print('\n')

# Guardar predicciones en el DataFrame y archivo

df['Predicción (kWh)'] = predicciones

df.to_csv(os.path.join(output_dir, 'datos_con_predicciones.csv'), index=False)

# Graficar datos y regresión
plt.figure(figsize=(8,5))
plt.scatter(temperatura, consumo_energia, color='blue', label='Datos reales', edgecolor='k')
plt.plot(temperatura, predicciones, color='red', label='Regresión lineal')
plt.title('Regresión lineal: Temperatura vs Consumo de energía')
plt.xlabel('Temperatura (°C)')
plt.ylabel('Consumo de energía (kWh)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'regresion_lineal.png'))
plt.show()

# 4. Evaluación del Modelo
mse = mean_squared_error(y, predicciones)
mae = mean_absolute_error(y, predicciones)
print('Métricas de error:')
print(f'- Error Cuadrático Medio (MSE): {mse:.2f}')
print(f'- Error Absoluto Medio (MAE): {mae:.2f}')

# Interpretación
print('\nInterpretación:')
print('El modelo de regresión lineal simple ajusta una recta a los datos simulados. Un MSE y MAE bajos indican buen ajuste, pero siempre hay que considerar la variabilidad de los datos y el contexto del problema.') 
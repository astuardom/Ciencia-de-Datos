# 1. Importar librerías
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 2. Cargar el conjunto de datos
df = pd.read_csv("entrada/forest_growth_data.csv")
print("Primeras filas del dataset:")
print(df.head(5))

# 3. Visualizar los datos
# Relación entre tree_age y tree_height
plt.figure(figsize=(8, 5))
plt.scatter(df['tree_age'], df["tree_height"], alpha=0.7, color='green')
plt.title("Edad del Árbol vs Altura")
plt.xlabel("Edad del Árbol (años)")
plt.ylabel("Altura del Árbol (m)")
plt.grid(True)
plt.show()

# 4. Ajustar el modelo de regresión lineal simple
X = df[['tree_age']]  # Matriz de características (n, 1)
y = df['tree_height']  # Vector objetivo (n,)
model = LinearRegression()
model.fit(X, y)

# 5. Ver el resumen del modelo
print("\nResumen del Modelo")
print(f"Intercepto (β0): {model.intercept_:.3f}")
print(f"Pendiente (β1): {model.coef_[0]:.3f}")

# 6. Interpretación del modelo
print(f"Interpretación: por cada año adicional de edad, la altura promedio del árbol aumenta {model.coef_[0]:.2f} metros.")

# 7. Visualizar la línea de regresión
plt.figure(figsize=(8, 5))
plt.scatter(df['tree_age'], df["tree_height"], alpha=0.7, color='blue', label="Datos reales")
plt.plot(df['tree_age'], model.predict(X), color='red', label="Línea de Regresión")
plt.title("Regresión Lineal Simple")
plt.xlabel("Edad del Árbol (años)")
plt.ylabel("Altura del Árbol (m)")
plt.legend()
plt.grid(True)
plt.show()

# 8. Evaluar el rendimiento del modelo
y_pred = model.predict(X)
mse = mean_squared_error(y, y_pred)
mae = mean_absolute_error(y, y_pred)
r2 = r2_score(y, y_pred)

print("\nEvaluación del Modelo")
print(f"Error Cuadrático Medio (MSE): {mse:.2f}")
print(f"Error Absoluto Medio (MAE): {mae:.2f}")
print(f"Coeficiente de Determinación (R²): {r2:.3f}")

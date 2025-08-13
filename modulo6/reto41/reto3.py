# ==============================================
# Reto 3: Predicción de Rendimiento de Cultivos Agrícolas
# ==============================================

# 1️⃣ Importar librerías necesarias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Configuración visual
sns.set(style="whitegrid")
plt.rcParams["figure.dpi"] = 100

# 2️⃣ Cargar el dataset
# Asegúrate de poner la ruta correcta de tu archivo CSV descargado
df = pd.read_csv("Crop_recommendation.csv")

print("🔹 Vista inicial de los datos:")
print(df.head())
print("\n📏 Dimensiones:", df.shape)

# 3️⃣ Preparar los datos
# Para este reto, simularemos que 'label' (cultivo) influye en el rendimiento.
# Creamos una columna ficticia de rendimiento basada en N, P, K, temperatura y humedad.
np.random.seed(42)
df["yield_ton_per_ha"] = (
    0.05 * df["N"] +
    0.04 * df["P"] +
    0.03 * df["K"] +
    0.2 * df["temperature"] +
    0.15 * df["humidity"] +
    np.random.normal(0, 2, len(df))
)

# Variables independientes
X = df[["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]]

# Variable dependiente
y = df["yield_ton_per_ha"]

# Manejar valores faltantes (si hubiera)
X = X.fillna(X.mean())

# 4️⃣ Dividir datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5️⃣ Inicializar y entrenar el modelo
modelo = LinearRegression()
modelo.fit(X_train, y_train)

# 6️⃣ Realizar predicciones
y_pred = modelo.predict(X_test)

# 7️⃣ Calcular métricas de desempeño
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n📊 Métricas del modelo:")
print(f"MAE:  {mae:.2f}")
print(f"MSE:  {mse:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R²:   {r2:.4f}")

# 8️⃣ Visualización: Rendimiento real vs predicho
plt.figure(figsize=(7, 5))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.7)
plt.xlabel("Rendimiento Real (ton/ha)")
plt.ylabel("Rendimiento Predicho (ton/ha)")
plt.title("Comparación de Rendimiento Real vs Predicho")
plt.plot([y.min(), y.max()], [y.min(), y.max()], '--', color='red')
plt.show()

# 9️⃣ Visualización de importancia de variables (coeficientes del modelo)
coeficientes = pd.DataFrame({
    "Variable": X.columns,
    "Coeficiente": modelo.coef_
}).sort_values(by="Coeficiente", ascending=False)

plt.figure(figsize=(8, 5))
sns.barplot(x="Coeficiente", y="Variable", data=coeficientes, palette="viridis")
plt.title("Importancia de las Variables en la Predicción")
plt.show()

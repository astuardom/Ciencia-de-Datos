import pandas as pd
import matplotlib.pyplot as plt

# 1. Cargar datos
url = "https://raw.githubusercontent.com/vega/vega-datasets/main/data/seattle-weather.csv"
df = pd.read_csv(url)
df["date"] = pd.to_datetime(df["date"])
df["month"] = df["date"].dt.month

# 2. Cálculos por mes
monthly_data = df.groupby("month").agg({
    "weather": lambda x: (x == "sun").mean(),  # proporción de días soleados
    "precipitation": "mean",
    "temp_max": "mean"
}).rename(columns={"weather": "sun_days", "precipitation": "avg_precip", "temp_max": "avg_temp_max"})

# Mostrar tabla resumen
print("📊 Estadísticas mensuales:")
print(monthly_data)

# 3. Gráfico de líneas
plt.figure(figsize=(10, 6))
plt.plot(monthly_data.index, monthly_data["sun_days"], label="Días soleados (%)", marker='o')
plt.plot(monthly_data.index, monthly_data["avg_temp_max"], label="Temp. Máx Promedio (°C)", marker='o')
plt.plot(monthly_data.index, monthly_data["avg_precip"], label="Precipitación Promedio (mm)", marker='o')

plt.title("Estacionalidad del clima en Seattle (2012)")
plt.xlabel("Mes")
plt.ylabel("Valores")
plt.xticks(ticks=range(1,13), labels=["Ene", "Feb", "Mar", "Abr", "May", "Jun", 
                                      "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"])
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()

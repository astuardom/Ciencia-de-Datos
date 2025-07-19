import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Cargar y preparar el dataset
url = "https://raw.githubusercontent.com/vega/vega-datasets/main/data/seattle-weather.csv"
df = pd.read_csv(url)
df["date"] = pd.to_datetime(df["date"])
df["month"] = df["date"].dt.month

# 2. Tomar muestras
np.random.seed(42)  # Para reproducibilidad

# Muestra aleatoria simple (60 días al azar)
sample_random = df.sample(n=60)

# Muestra sesgada (solo días calurosos: temp_max ≥ 25)
sample_biased = df[df["temp_max"] >= 25].sample(n=60, random_state=42)

# 3. Calcular estadísticas
def calc_stats(sample, label):
    mean_precip = sample["precipitation"].mean()
    var_precip = sample["precipitation"].var()
    pct_dry_days = (sample["precipitation"] == 0).mean() * 100

    print(f"📊 Estadísticas para {label}:")
    print(f"- Media de precipitación: {mean_precip:.3f}")
    print(f"- Varianza de precipitación: {var_precip:.3f}")
    print(f"- % de días sin lluvia: {pct_dry_days:.1f}%\n")

calc_stats(sample_random, "Muestra Aleatoria")
calc_stats(sample_biased, "Muestra Sesgada (temp_max ≥ 25)")

# 4. Boxplot comparativo
combined = pd.concat([
    sample_random.assign(muestra="Aleatoria"),
    sample_biased.assign(muestra="Sesgada")
])

plt.figure(figsize=(8, 6))
sns.boxplot(
    data=combined,
    x="muestra",
    y="precipitation",
    hue="muestra",           # Asigna hue explícitamente
    palette="Set2",
    dodge=False              # Para que no se duplique la caja
)
plt.title("Comparación de Precipitación entre Muestras")
plt.ylabel("Precipitación (mm)")
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()





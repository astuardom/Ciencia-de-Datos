import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Cargar y preparar el dataset
url = "https://raw.githubusercontent.com/vega/vega-datasets/main/data/seattle-weather.csv"
df = pd.read_csv(url)
df["date"] = pd.to_datetime(df["date"])
df["month"] = df["date"].dt.month
df["year"] = df["date"].dt.year
df["day"] = df["date"].dt.day

# 2. Definición de eventos
df["A"] = df["precipitation"] >= 1           # Evento A: lluvia significativa
df["B"] = df["weather"] == "rain"            # Evento B: día etiquetado como 'rain'
df["C"] = df["temp_max"] >= 25               # Evento C: día caluroso (≥ 25°C)

# 3. Cálculo de probabilidades
P_A = df["A"].mean()
P_B = df["B"].mean()
P_C = df["C"].mean()

P_A_given_B = df[df["B"]]["A"].mean()
P_A_given_C = df[df["C"]]["A"].mean()

print("📊 Probabilidades simples:")
print(f"P(A) - Precipitación ≥ 1: {P_A:.3f}")
print(f"P(B) - Clima 'rain': {P_B:.3f}")
print(f"P(C) - Temperatura máxima ≥ 25°C: {P_C:.3f}")

print("\n📌 Probabilidades condicionales:")
print(f"P(A | B) - Lluvia si el clima es 'rain': {P_A_given_B:.3f}")
print(f"P(A | C) - Lluvia si temp_max ≥ 25: {P_A_given_C:.3f}")

# 4. Gráfico de barras – Frecuencia mensual de días con precipitación ≥ 1
monthly_precip_days = df[df["A"]].groupby("month").size()

plt.figure(figsize=(10, 5))
monthly_precip_days.plot(kind='bar', color='skyblue')
plt.title("Frecuencia mensual de días con precipitación ≥ 1")
plt.xlabel("Mes")
plt.ylabel("Número de días")
plt.xticks(ticks=range(0,12), labels=["Ene", "Feb", "Mar", "Abr", "May", "Jun", 
                                      "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"], rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# 5. Heatmap – P(A | B ∧ C) por mes
df["B_and_C"] = df["B"] & df["C"]

# Filtrar días donde B y C son verdaderos
cond_group = df[df["B_and_C"]].groupby("month")["A"].mean().reset_index(name='P(A|B∧C)')

# Crear heatmap
pivot = cond_group.pivot_table(index="month", values="P(A|B∧C)")

plt.figure(figsize=(6, 5))
sns.heatmap(pivot, annot=True, fmt=".2f", cmap="coolwarm", cbar_kws={"label": "P(A | B ∧ C)"})
plt.title("Probabilidad condicional P(A | B ∧ C) por mes")
plt.xlabel("Mes")
plt.ylabel("")
plt.yticks(ticks=[i + 0.5 for i in range(12)], labels=["Ene", "Feb", "Mar", "Abr", "May", "Jun", 
                                                       "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"], rotation=0)
plt.tight_layout()
plt.show()

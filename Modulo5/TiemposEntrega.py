import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm

# ----------------------------------------
# PARÁMETROS GENERALES
# ----------------------------------------
media_real = 48
desviacion_real = 12
n_muestras = 1000
escala_exponencial = media_real  # Media de la exponencial

# ----------------------------------------
# 1. LEY DE LOS GRANDES NÚMEROS
# ----------------------------------------
tamaños = [10, 50, 500]
medias_por_n = {}

for n in tamaños:
    muestras = [np.mean(np.random.exponential(scale=escala_exponencial, size=n)) for _ in range(n_muestras)]
    medias_por_n[n] = muestras

# Visualización de convergencia
plt.figure(figsize=(10, 6))
for n, medias in medias_por_n.items():
    plt.plot(medias, label=f"n = {n}", alpha=0.6)
plt.axhline(media_real, color='black', linestyle='--', label='Media real (48h)')
plt.title("Ley de los Grandes Números – Convergencia de la media muestral")
plt.xlabel("Índice de muestra")
plt.ylabel("Media de la muestra")
plt.legend()
plt.tight_layout()
plt.show()

# ----------------------------------------
# 2. TEOREMA DEL LÍMITE CENTRAL (n = 50)
# ----------------------------------------
n = 50
medias_n50 = [np.mean(np.random.exponential(scale=escala_exponencial, size=n)) for _ in range(n_muestras)]

# Histograma y curva normal
plt.figure(figsize=(8, 5))
sns.histplot(medias_n50, bins=30, kde=False, stat='density', color='skyblue', label='Distribución simulada')
x = np.linspace(min(medias_n50), max(medias_n50), 100)
plt.plot(x, norm.pdf(x, loc=media_real, scale=desviacion_real/np.sqrt(n)), label='Normal teórica', color='red')
plt.title("TLC: Distribución muestral de la media (n=50)")
plt.xlabel("Media muestral")
plt.ylabel("Densidad")
plt.legend()
plt.tight_layout()
plt.show()

# ----------------------------------------
# 3. PROBABILIDAD P(media_muestral > 52)
# ----------------------------------------
z_score = (52 - media_real) / (desviacion_real / np.sqrt(n))
prob_mayor_52 = 1 - norm.cdf(z_score)

# Visualización
x = np.linspace(40, 60, 300)
y = norm.pdf(x, loc=media_real, scale=desviacion_real / np.sqrt(n))
plt.figure(figsize=(8, 5))
plt.plot(x, y, label="Distribución normal de la media")
plt.fill_between(x, y, where=(x > 52), color='orange', alpha=0.5, label=f"P(>52h) ≈ {prob_mayor_52:.3f}")
plt.axvline(52, color='orange', linestyle='--')
plt.title("Probabilidad de que la media muestral sea mayor a 52h")
plt.xlabel("Media muestral")
plt.ylabel("Densidad")
plt.legend()
plt.tight_layout()
plt.show()

# ----------------------------------------
# 4. DISTRIBUCIÓN MUESTRAL DE PROPORCIONES (n = 100)
# ----------------------------------------
n_prop = 100
proporciones = []
for _ in range(n_muestras):
    muestra = np.random.exponential(scale=escala_exponencial, size=n_prop)
    prop = np.mean(muestra < 50)
    proporciones.append(prop)

proporcion_entre = np.mean((np.array(proporciones) >= 0.6) & (np.array(proporciones) <= 0.7))

# Histograma
plt.figure(figsize=(8, 5))
sns.histplot(proporciones, bins=30, kde=False, stat='density', color='mediumseagreen')
plt.axvline(0.6, color='black', linestyle='--')
plt.axvline(0.7, color='black', linestyle='--')
plt.title(f"Distribución muestral de proporciones (n=100)\nP(0.6 ≤ p ≤ 0.7) ≈ {proporcion_entre:.3f}")
plt.xlabel("Proporción de entregas < 50h")
plt.ylabel("Densidad")
plt.tight_layout()
plt.show()

# ----------------------------------------
# 5. RESULTADOS NUMÉRICOS RESUMIDOS
# ----------------------------------------
print("\n📊 RESUMEN ESTADÍSTICO:")
for n, stats in medias_por_n.items():
    print(f"- n = {n}: media ≈ {np.mean(stats):.3f}, varianza ≈ {np.var(stats):.3f}")

print(f"\n➡ Teorema del Límite Central (n=50): media ≈ {np.mean(medias_n50):.3f}, varianza ≈ {np.var(medias_n50):.3f}")
print(f"➡ Probabilidad de que media muestral > 52h: {prob_mayor_52:.4f}")
print(f"➡ Proporción media de entregas < 50h: {np.mean(proporciones):.3f}")
print(f"➡ P(0.6 ≤ p ≤ 0.7): {proporcion_entre:.4f}")

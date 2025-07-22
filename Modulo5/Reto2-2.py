# Proyecto 2: Forestal – Salud de los árboles
# Análisis de proporción de árboles sanos

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm, binom
from math import ceil, sqrt

# ===================================================
# 1. FUNCIONES UTILITARIAS
# ===================================================
def intervalo_confianza_proporcion(p_hat, n, confianza):
    """Calcula el intervalo de confianza para una proporción."""
    z = norm.ppf(1 - (1 - confianza) / 2)
    margen_error = z * np.sqrt(p_hat * (1 - p_hat) / n)
    return (p_hat - margen_error, p_hat + margen_error, margen_error, z)

def calcular_tamanio_muestral(z, p, E):
    """Calcula el tamaño muestral necesario para estimar una proporción."""
    n = (z ** 2) * p * (1 - p) / (E ** 2)
    return ceil(n)

def imprimir_separador(titulo):
    print('\n' + '=' * 70)
    print(titulo)
    print('=' * 70)

# ===================================================
# 2. DATOS INICIALES
# ===================================================
# Simulación reproducible de 1000 árboles: 88% sanos, 12% enfermos
n = 1000
np.random.seed(42)  # Semilla para reproducibilidad
sanos = int(n * 0.88)
enfermos = n - sanos
estado = [1]*sanos + [0]*enfermos
np.random.shuffle(estado)  # Aleatorizar

df = pd.DataFrame({'estado': estado})

# Mostrar el número de simulaciones
global_n = len(df)
print(f"Número de simulaciones (árboles evaluados): {global_n}")

# ===================================================
# 3. CÁLCULO DE PROPORCIÓN, HIPÓTESIS Y PRUEBA Z
# ===================================================
proporcion_sanos = df['estado'].mean()
valor_esperado = 0.85

# Hipótesis
print("\nHIPÓTESIS:")
print("   H₀: p ≤ 0.85 (El porcentaje de árboles sanos es 85% o menos)")
print("   H₁: p > 0.85 (El porcentaje de árboles sanos es mayor a 85%)")

# Prueba Z para proporciones
z = (proporcion_sanos - valor_esperado) / np.sqrt(valor_esperado * (1 - valor_esperado) / n)
from scipy.stats import norm as normdist
p_value = 1 - normdist.cdf(z)
print(f"\nEstadístico Z: {z:.3f}")
print(f"Valor p: {p_value:.4f}")

# ===================================================
# 4. INTERVALO DE CONFIANZA Y TAMAÑO MUESTRAL
# ===================================================
confianza = 0.90
ic_low, ic_upp, margen_error, z_90 = intervalo_confianza_proporcion(proporcion_sanos, n, confianza)
print(f"\nIntervalo de confianza del 90%: ({ic_low:.3f}, {ic_upp:.3f})")

# Tamaño muestral necesario para margen de ±3% con 95% de confianza
confianza_95 = 0.95
z_95 = norm.ppf(1 - (1 - confianza_95) / 2)
margen_error_3 = 0.03
n_necesario = calcular_tamanio_muestral(z_95, proporcion_sanos, margen_error_3)
print(f"Tamaño muestral necesario para margen ±3% y 95% de confianza: {n_necesario}")

# ===================================================
# 5. VISUALIZACIÓN
# ===================================================
imprimir_separador('VISUALIZACIÓN ESTADÍSTICA')

fig, axs = plt.subplots(1, 2, figsize=(13, 5))

# --- Gráfico de barras ---
conteo = df['estado'].value_counts().sort_index()
axs[0].bar(['Enfermo', 'Sano'], conteo, color=['red', 'green'])
axs[0].set_title('Estado de los árboles')
axs[0].set_ylabel('Cantidad')

# --- Curva binomial teórica ---
x = np.arange(0, n+1)
binom_teorica = binom.pmf(x, n, valor_esperado)
axs[1].plot(x, binom_teorica, label='Binomial teórica (p=0.85)')
axs[1].axvline(sum(estado), color='red', linestyle='--', label='Observado')
axs[1].set_title('Curva binomial teórica vs. observada')
axs[1].set_xlabel('Número de árboles sanos')
axs[1].set_ylabel('Probabilidad')
axs[1].legend()

plt.tight_layout()
plt.show()

# ===================================================
# 6. INTERPRETACIÓN FINAL
# ===================================================
imprimir_separador('INTERPRETACIÓN FINAL')

if p_value < 0.05:
    print("Se rechaza H₀: Hay evidencia de que más del 85% de los árboles están sanos.")
else:
    print("No se rechaza H₀: No hay evidencia suficiente para afirmar que más del 85% están sanos.")

print(f"\nProporción observada: {proporcion_sanos:.3f}")
print(f"Intervalo de confianza 90%: ({ic_low:.3f}, {ic_upp:.3f})")
print(f"Tamaño muestral recomendado para ±3% y 95% confianza: {n_necesario}")
print("\nLa diferencia observada es significativa si el valor p es menor a 0.05. Si no, podría deberse al azar o a tamaño muestral insuficiente.")

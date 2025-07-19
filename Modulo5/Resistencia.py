import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from math import ceil, sqrt

# ===================================================
# 1. FUNCIONES UTILITARIAS
# ===================================================

def intervalo_confianza_media(xbar, sigma, n, confianza):
    """Calcula el intervalo de confianza para la media con sigma conocida."""
    z = norm.ppf(1 - (1 - confianza) / 2)
    margen_error = z * sigma / sqrt(n)
    return (xbar - margen_error, xbar + margen_error, margen_error, z)

def intervalo_confianza_proporcion(p_hat, n, confianza):
    """Calcula el intervalo de confianza para una proporción."""
    z = norm.ppf(1 - (1 - confianza) / 2)
    margen_error = z * sqrt(p_hat * (1 - p_hat) / n)
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

media = 400               # Media muestral (MPa)
sigma = 10                # Desviación estándar poblacional (MPa)
n_resistencia = 70        # Tamaño muestral para resistencia
confianza_media = 0.99    # Nivel de confianza para media

p_defectuosos = 20 / 50   # Proporción muestral defectuosos
n_proporcion = 50         # Tamaño de muestra para proporción
confianza_proporcion = 0.95
E1 = 0.04                 # Margen de error deseado 1
E2 = 0.03                 # Margen de error deseado 2

# ===================================================
# 3. CÁLCULO DE INTERVALOS Y TAMAÑO MUESTRAL
# ===================================================

# --- Intervalo de confianza para la media ---
ic_media_inf, ic_media_sup, E_media, z_99 = intervalo_confianza_media(
    media, sigma, n_resistencia, confianza_media)

# --- Intervalo de confianza para la proporción ---
ic_prop_inf, ic_prop_sup, E_prop, z_95 = intervalo_confianza_proporcion(
    p_defectuosos, n_proporcion, confianza_proporcion)

# --- Tamaños muestrales requeridos ---
n_req_04 = calcular_tamanio_muestral(z_95, p_defectuosos, E1)
n_req_03 = calcular_tamanio_muestral(z_95, p_defectuosos, E2)

# ===================================================
# 4. RESULTADOS Y RECOMENDACIONES
# ===================================================

imprimir_separador('RESULTADOS NUMÉRICOS')

print(f"Intervalo de confianza del 99% para la media de resistencia:")
print(f"   [{ic_media_inf:.2f}, {ic_media_sup:.2f}] MPa")
print(f"Margen de error: ±{E_media:.2f} MPa\n")

print(f"Intervalo de confianza del 95% para la proporción de defectuosos (muestra de 50):")
print(f"   [{ic_prop_inf:.2f}, {ic_prop_sup:.2f}]")
print(f"Margen de error: ±{E_prop:.2f}\n")

print(f"Tamaño muestral requerido para estimar proporción con margen ±0.04 al 95%: {n_req_04} piezas")
print(f"Tamaño muestral requerido para margen ±0.03 al 95%: {n_req_03} piezas\n")

# Recomendaciones automáticas
imprimir_separador('RECOMENDACIONES AUTOMÁTICAS')
if n_proporcion < n_req_04:
    print(f"Advertencia: El tamaño de muestra actual para proporción (n={n_proporcion}) es menor al recomendado para E=0.04 ({n_req_04}).")
    print(f"→ Se recomienda aumentar el tamaño de muestra para una mayor precisión en la estimación de la proporción.")
else:
    print(f"El tamaño de muestra usado para proporción es adecuado para un margen de error de ±0.04.")

# ===================================================
# 5. VISUALIZACIONES MEJORADAS
# ===================================================

imprimir_separador('VISUALIZACIÓN ESTADÍSTICA')

fig, axs = plt.subplots(1, 2, figsize=(13, 5))

# --- Gráfico de resistencia ---
x = np.linspace(370, 430, 400)
y = norm.pdf(x, media, sigma)
axs[0].plot(x, y, label='Distribución Normal (Resistencia)', color='steelblue')
axs[0].axvline(ic_media_inf, color='red', linestyle='--', label='Límite IC 99%')
axs[0].axvline(ic_media_sup, color='red', linestyle='--')
axs[0].fill_between(x, y, where=(x >= ic_media_inf) & (x <= ic_media_sup), color='orange', alpha=0.3, label='Intervalo 99%')
axs[0].set_title('Distribución Resistencia (MPa)')
axs[0].set_xlabel('Resistencia (MPa)')
axs[0].set_ylabel('Densidad')
axs[0].legend()

# --- Gráfico de proporción de defectuosos ---
proporciones = np.linspace(0, 1, 300)
desv_p = sqrt(p_defectuosos * (1 - p_defectuosos) / n_proporcion)
dist_prop = norm.pdf(proporciones, p_defectuosos, desv_p)
axs[1].plot(proporciones, dist_prop, label='Distribución Proporción', color='seagreen')
axs[1].axvline(ic_prop_inf, color='purple', linestyle='--', label='Límite IC 95%')
axs[1].axvline(ic_prop_sup, color='purple', linestyle='--')
axs[1].fill_between(proporciones, dist_prop, where=(proporciones >= ic_prop_inf) & (proporciones <= ic_prop_sup), color='violet', alpha=0.2, label='Intervalo 95%')
axs[1].set_title('Proporción Piezas Defectuosas')
axs[1].set_xlabel('Proporción')
axs[1].set_ylabel('Densidad')
axs[1].legend()

plt.tight_layout()
plt.show()

# ===================================================
# 6. INTERPRETACIÓN FINAL
# ===================================================

imprimir_separador('INTERPRETACIÓN FINAL')

print(f"1. Con 99% de confianza, la resistencia promedio real de las piezas está entre {ic_media_inf:.2f} y {ic_media_sup:.2f} MPa.")
print(f"2. Para estimar la proporción real de defectuosos con un margen de ±0.04 y 95% de confianza, debes analizar al menos {n_req_04} piezas.")
print(f"3. Si deseas una estimación más precisa (±0.03), el tamaño mínimo de muestra debe ser {n_req_03} piezas.")
print(f"4. La visualización muestra claramente los intervalos de confianza y te ayuda a comunicar resultados técnicos en informes y presentaciones.")

# OPCIONAL: Si quieres guardar los resultados en un archivo txt o csv, puedo agregarlo fácilmente.

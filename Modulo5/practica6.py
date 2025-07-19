import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Definir los parámetros
n = 25  # tamaño de la muestra
mean = 32  # media de la muestra
mu_0 = 30  # media hipotética
std = 5  # desviación estándar
alpha = 0.05  # nivel de significancia
df = n - 1  # grados de libertad

# Calcular el estadístico de prueba
# Fórmula: t = (x̄ - μ₀) / (s/√n)

t_stat = (mean - mu_0) / (std / np.sqrt(n))
print(f"Estadístico de prueba: {t_stat:.2f}")

# Generar valores para distribución t
x = np.linspace(-4, 4, 100)  # rango de valores para la gráfica
y = stats.t.pdf(x, df)  # densidad de probabilidad

# Crear la gráfica
plt.figure(figsize=(10, 6))
plt.plot(x, y, 'b-', label="Distribución t(df=24)")

# Rellenar la región de rechazo
plt.fill_between(x, y, where=x>=t_stat, color="red", alpha=0.3, label="Región de rechazo (derecha)")
plt.fill_between(x, y, where=x<=-t_stat, color="red", alpha=0.3, label="Región de rechazo (izquierda)")

# Agregar la línea vertical del estadístico de prueba
plt.axvline(t_stat, color="green", linestyle="--", label=f"Estadístico de prueba: {t_stat:.2f}")

plt.title("Distribución t de Student")
plt.xlabel("Valores de t")
plt.ylabel("Densidad de probabilidad")
plt.legend()
plt.show()  


#imprimir decision
critical_value = stats.t.ppf(1-alpha/2,df)
print(f"Valor critico: {critical_value:.2f}")
if t_stat > critical_value or t_stat < -critical_value:
    print("Rechazar la hipotesis nula")
else:
    print("No se rechaza la hipotesis nula")






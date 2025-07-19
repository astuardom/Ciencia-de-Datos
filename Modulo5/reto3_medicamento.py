"""
Reto 3: Eficiencia de un Nuevo Medicamento
Análisis estadístico para probar la eficacia de un medicamento en la reducción de presión arterial
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd

def analizar_eficiencia_medicamento():
    """
    Analiza la eficiencia de un nuevo medicamento para reducir la presión arterial
    """
    
    print("=" * 70)
    print("RETO 3: EFICIENCIA DE UN NUEVO MEDICAMENTO")
    print("=" * 70)
    
    # ===================================================
    # DATOS DEL PROBLEMA
    # ===================================================
    
    # Parámetros de la muestra
    n = 40                    # Tamaño de la muestra
    x_barra = 118            # Media muestral (mmHg)
    s = 6                    # Desviación estándar muestral (mmHg)
    mu_0 = 120               # Media poblacional hipotética (mmHg)
    alpha = 0.05             # Nivel de significancia
    df = n - 1               # Grados de libertad
    
    print(f"\n📊 DATOS DEL ESTUDIO:")
    print(f"   • Tamaño de muestra: {n} pacientes")
    print(f"   • Presión arterial promedio observada: {x_barra} mmHg")
    print(f"   • Desviación estándar: {s} mmHg")
    print(f"   • Presión arterial objetivo del laboratorio: {mu_0} mmHg")
    print(f"   • Nivel de significancia: α = {alpha}")
    
    # ===================================================
    # PASO 1: ESTABLECER HIPÓTESIS
    # ===================================================
    
    print(f"\n🔬 PASO 1: HIPÓTESIS")
    print(f"   H₀: μ = {mu_0} mmHg (El medicamento NO reduce la presión a {mu_0} mmHg)")
    print(f"   H₁: μ < {mu_0} mmHg (El medicamento SÍ reduce la presión por debajo de {mu_0} mmHg)")
    print(f"   → Prueba UNILATERAL (cola izquierda)")
    
    # ===================================================
    # PASO 2: NIVEL DE SIGNIFICANCIA Y TIPO DE PRUEBA
    # ===================================================
    
    print(f"\n📈 PASO 2: NIVEL DE SIGNIFICANCIA")
    print(f"   • α = {alpha} ({alpha*100}%)")
    print(f"   • Tipo de prueba: UNILATERAL (cola izquierda)")
    print(f"   • Grados de libertad: df = {n} - 1 = {df}")
    
    # ===================================================
    # PASO 3: CALCULAR ESTADÍSTICA DE PRUEBA
    # ===================================================
    
    # Fórmula: t = (x̄ - μ₀) / (s/√n)
    t_stat = (x_barra - mu_0) / (s / np.sqrt(n))
    
    print(f"\n🧮 PASO 3: ESTADÍSTICA DE PRUEBA")
    print(f"   Fórmula: t = (x̄ - μ₀) / (s/√n)")
    print(f"   t = ({x_barra} - {mu_0}) / ({s}/√{n})")
    print(f"   t = {x_barra - mu_0} / {s / np.sqrt(n):.4f}")
    print(f"   t = {t_stat:.4f}")
    
    # ===================================================
    # PASO 4: VALOR CRÍTICO Y VALOR P
    # ===================================================
    
    # Para prueba unilateral (cola izquierda), usamos alpha directamente
    t_critical = stats.t.ppf(alpha, df)
    p_value = stats.t.cdf(t_stat, df)
    
    print(f"\n📊 PASO 4: VALOR CRÍTICO Y VALOR P")
    print(f"   • Valor crítico (t₍α,df₎): t₍{alpha},{df}₎ = {t_critical:.4f}")
    print(f"   • Valor p: P(T ≤ {t_stat:.4f}) = {p_value:.4f}")
    
    # ===================================================
    # PASO 5: DECISIÓN Y CONCLUSIÓN
    # ===================================================
    
    print(f"\n🎯 PASO 5: DECISIÓN Y CONCLUSIÓN")
    
    # Criterio 1: Comparar estadístico de prueba con valor crítico
    if t_stat < t_critical:
        decision_critico = "RECHAZAR H₀"
        conclusion_critico = "El medicamento SÍ reduce significativamente la presión arterial"
    else:
        decision_critico = "NO RECHAZAR H₀"
        conclusion_critico = "No hay evidencia suficiente para afirmar que el medicamento reduce la presión"
    
    # Criterio 2: Comparar valor p con nivel de significancia
    if p_value < alpha:
        decision_pvalue = "RECHAZAR H₀"
        conclusion_pvalue = "El valor p es menor que α, evidencia estadísticamente significativa"
    else:
        decision_pvalue = "NO RECHAZAR H₀"
        conclusion_pvalue = "El valor p es mayor que α, no hay evidencia estadísticamente significativa"
    
    print(f"   📋 Criterio del valor crítico:")
    print(f"      • Decisión: {decision_critico}")
    print(f"      • Justificación: t = {t_stat:.4f} {'<' if t_stat < t_critical else '≥'} {t_critical:.4f}")
    print(f"      • Conclusión: {conclusion_critico}")
    
    print(f"\n   📋 Criterio del valor p:")
    print(f"      • Decisión: {decision_pvalue}")
    print(f"      • Justificación: p = {p_value:.4f} {'<' if p_value < alpha else '≥'} {alpha}")
    print(f"      • Conclusión: {conclusion_pvalue}")
    
    # ===================================================
    # VISUALIZACIÓN
    # ===================================================
    
    print(f"\n📊 VISUALIZACIÓN DE LA PRUEBA")
    
    # Generar valores para la distribución t
    x = np.linspace(-4, 2, 200)
    y = stats.t.pdf(x, df)
    
    # Crear la gráfica
    plt.figure(figsize=(12, 8))
    
    # Distribución t
    plt.plot(x, y, 'b-', linewidth=2, label=f'Distribución t(df={df})')
    
    # Región de rechazo (cola izquierda)
    plt.fill_between(x, y, where=x <= t_critical, color='red', alpha=0.3, 
                     label=f'Región de rechazo (α = {alpha})')
    
    # Estadístico de prueba
    plt.axvline(t_stat, color='green', linestyle='--', linewidth=2, 
                label=f'Estadístico de prueba: t = {t_stat:.4f}')
    
    # Valor crítico
    plt.axvline(t_critical, color='red', linestyle=':', linewidth=2, 
                label=f'Valor crítico: t = {t_critical:.4f}')
    
    # Configuración de la gráfica
    plt.title('Prueba t Unilateral: Eficiencia del Medicamento\nH₀: μ = 120 vs H₁: μ < 120', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Valores de t', fontsize=12)
    plt.ylabel('Densidad de probabilidad', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Agregar anotaciones
    plt.annotate(f'p = {p_value:.4f}', xy=(t_stat, stats.t.pdf(t_stat, df)), 
                xytext=(t_stat-1, stats.t.pdf(t_stat, df)+0.02),
                arrowprops=dict(arrowstyle='->', color='green'),
                fontsize=10, color='green')
    
    plt.tight_layout()
    plt.show()
    
    # ===================================================
    # RESUMEN FINAL
    # ===================================================
    
    print(f"\n" + "=" * 70)
    print(f"📋 RESUMEN FINAL")
    print(f"=" * 70)
    
    print(f"   🎯 HIPÓTESIS:")
    print(f"      H₀: μ = {mu_0} mmHg")
    print(f"      H₁: μ < {mu_0} mmHg")
    
    print(f"\n   📊 RESULTADOS:")
    print(f"      • Estadístico de prueba: t = {t_stat:.4f}")
    print(f"      • Valor crítico: t₍{alpha},{df}₎ = {t_critical:.4f}")
    print(f"      • Valor p: {p_value:.4f}")
    
    print(f"\n   ✅ DECISIÓN:")
    if p_value < alpha:
        print(f"      RECHAZAR H₀ al nivel de significancia α = {alpha}")
        print(f"      → El medicamento SÍ reduce significativamente la presión arterial")
        print(f"      → La afirmación del laboratorio es CORRECTA")
    else:
        print(f"      NO RECHAZAR H₀ al nivel de significancia α = {alpha}")
        print(f"      → No hay evidencia suficiente para afirmar que el medicamento reduce la presión")
        print(f"      → La afirmación del laboratorio NO puede ser confirmada")
    
    print(f"\n   💡 INTERPRETACIÓN:")
    print(f"      Con {alpha*100}% de confianza, {'existe' if p_value < alpha else 'no existe'} evidencia")
    print(f"      estadística suficiente para concluir que el medicamento reduce la presión arterial")
    print(f"      por debajo de {mu_0} mmHg.")

if __name__ == "__main__":
    analizar_eficiencia_medicamento() 
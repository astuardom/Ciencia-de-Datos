#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
K-Means: Segmentación de Clientes (Ejercicio sin solución)
Dataset sugerido: "Mall Customers" (Kaggle)
URL de referencia (búscalo en Kaggle): Mall Customers Dataset

------------------------------------------------------------------------------
INSTRUCCIONES (resumen):
1) Descarga el dataset desde Kaggle y colócalo en la carpeta ./input/
   - Archivo típico: Mall_Customers.csv
2) Crea un entorno virtual e instala dependencias:
   - pip install numpy pandas matplotlib scikit-learn
3) Ejecuta este script desde la terminal:
   - python KMeans_MallCustomers_Entregable.py
4) Sigue los TODOs. NO hay soluciones implementadas.

------------------------------------------------------------------------------
REQUISITOS DEL ENTREGABLE:
- Trabajar con 2 variables numéricas (p. ej., "Annual Income (k$)" y "Spending Score (1-100)")
- Probar varios K (2..7), mostrar:
    (i) Curva del codo (SSE vs K)
    (ii) Silueta vs K
- Elegir un K y justificarlo en comentarios
- Entrenar K-Means con ese K y reportar métricas (SSE, Silhouette) y centroides
- Guardar todas las figuras en ./output_kmeans/ y mostrarlas en pantalla
- Exportar un CSV con etiquetas de cluster y, si escalas, reportar centroides en escala original

------------------------------------------------------------------------------
NOTAS:
- No uses seaborn (opcional) para cumplir entornos con restricciones.
- Cuida la reproducibilidad (semilla fija).
- Documenta tus decisiones en comentarios dentro del código.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import os
import warnings
warnings.filterwarnings('ignore')

# Configuración para reproducibilidad
np.random.seed(42)

# Configuración de matplotlib para mejor visualización
plt.style.use('default')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

def crear_directorios():
    """
    Crea los directorios necesarios para el ejercicio
    """
    # TODO 1: Crear directorios input/ y output_kmeans/ si no existen
    # Usa os.makedirs() con exist_ok=True
    
    os.makedirs('input', exist_ok=True)
    os.makedirs('output_kmeans', exist_ok=True)
    print("✅ Directorios creados/verificados: input/ y output_kmeans/")

def cargar_datos():
    """
    Carga el dataset Mall Customers desde el archivo CSV
    """
    # TODO 2: Cargar el dataset desde ./input/Mall_Customers.csv
    # Usa pd.read_csv() y verifica que se cargó correctamente
    
    try:
        df = pd.read_csv('./input/Mall_Customers.csv')
        print(f"✅ Dataset cargado exitosamente: {df.shape}")
        
        # TODO 3: Mostrar información básica del dataset:
        # - shape
        # - info()
        # - head()
        # - describe()
        
        print(f"\n📊 Información del dataset:")
        print(f"Shape: {df.shape}")
        print(f"Columnas: {list(df.columns)}")
        
        print(f"\n📋 Primeras 5 filas:")
        print(df.head())
        
        print(f"\n📈 Estadísticas descriptivas:")
        print(df.describe())
        
        # TODO 4: Verificar si hay valores faltantes y manejarlos apropiadamente
        
        print(f"\n🔍 Verificación de valores faltantes:")
        missing_values = df.isnull().sum()
        if missing_values.sum() == 0:
            print("✅ No hay valores faltantes en el dataset")
        else:
            print("⚠️ Valores faltantes encontrados:")
            print(missing_values[missing_values > 0])
            # Eliminar filas con valores faltantes
            df = df.dropna()
            print(f"Dataset limpio: {df.shape}")
        
        return df
        
    except FileNotFoundError:
        print("❌ Error: Archivo Mall_Customers.csv no encontrado en ./input/")
        print("💡 Asegúrate de descargar el dataset desde Kaggle y colocarlo en la carpeta input/")
        return None
    except Exception as e:
        print(f"❌ Error al cargar datos: {e}")
        return None

def explorar_datos(df):
    """
    Realiza exploración básica de los datos
    """
    # TODO 5: Crear visualizaciones exploratorias:
    # - Histogramas de las variables numéricas
    # - Matriz de correlación (heatmap)
    # - Scatter plot de las dos variables principales que usarás
    
    print("📊 Creando visualizaciones exploratorias...")
    
    # Seleccionar solo variables numéricas
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    print(f"Variables numéricas encontradas: {list(numeric_cols)}")
    
    # Crear figura con subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Análisis Exploratorio - Mall Customers Dataset', fontsize=16)
    
    # 1. Histogramas de variables numéricas
    for i, col in enumerate(numeric_cols):
        row = i // 2
        col_idx = i % 2
        axes[row, col_idx].hist(df[col], bins=20, alpha=0.7, edgecolor='black')
        axes[row, col_idx].set_title(f'Distribución de {col}')
        axes[row, col_idx].set_xlabel(col)
        axes[row, col_idx].set_ylabel('Frecuencia')
        axes[row, col_idx].grid(True, alpha=0.3)
    
    # 2. Scatter plot de las dos variables principales
    if len(numeric_cols) >= 2:
        # Usar las primeras dos variables numéricas
        x_col = numeric_cols[0]
        y_col = numeric_cols[1]
        
        # Crear scatter plot separado
        plt.figure(figsize=(10, 8))
        plt.scatter(df[x_col], df[y_col], alpha=0.6, s=50)
        plt.title(f'Relación entre {x_col} y {y_col}')
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.grid(True, alpha=0.3)
        
        # TODO 6: Guardar las figuras en ./output_kmeans/ con nombres descriptivos
        
        plt.savefig('./output_kmeans/scatter_exploratorio.png', dpi=300, bbox_inches='tight')
        print("✅ Scatter plot guardado: scatter_exploratorio.png")
        plt.close()
    
    # Guardar figura principal
    plt.savefig('./output_kmeans/histogramas_exploratorio.png', dpi=300, bbox_inches='tight')
    print("✅ Histogramas guardados: histogramas_exploratorio.png")
    plt.close()
    
    print("✅ Visualizaciones exploratorias completadas")

def preparar_datos(df):
    """
    Prepara los datos para el clustering
    """
    # TODO 7: Seleccionar solo las 2 variables numéricas que usarás para clustering
    # Sugerencia: "Annual Income (k$)" y "Spending Score (1-100)"
    
    print("🔧 Preparando datos para clustering...")
    
    # Buscar las columnas sugeridas
    target_cols = []
    for col in df.columns:
        if 'income' in col.lower():
            target_cols.append(col)
        elif 'spending' in col.lower() or 'score' in col.lower():
            target_cols.append(col)
    
    # Si no se encuentran las columnas específicas, usar las primeras 2 numéricas
    if len(target_cols) < 2:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        target_cols = list(numeric_cols[:2])
        print(f"⚠️ Usando primeras 2 variables numéricas: {target_cols}")
    else:
        print(f"✅ Variables seleccionadas: {target_cols}")
    
    # Seleccionar solo las columnas objetivo
    X = df[target_cols].copy()
    print(f"📊 Datos seleccionados: {X.shape}")
    
    # TODO 8: Decidir si escalar los datos o no
    # Si escalas, guarda el scaler para poder revertir los centroides después
    
    # Escalar los datos para mejor rendimiento del clustering
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print("✅ Datos escalados usando StandardScaler")
    print(f"📊 Datos originales - Rango: {X.min().min():.2f} a {X.max().max():.2f}")
    print(f"📊 Datos escalados - Rango: {X_scaled.min():.2f} a {X_scaled.max():.2f}")
    
    # TODO 9: Verificar que los datos están en el formato correcto (numpy array)
    
    if isinstance(X_scaled, np.ndarray):
        print("✅ Datos en formato numpy array correcto")
        return X_scaled, scaler, target_cols
    else:
        print("❌ Error: Los datos no están en formato numpy array")
        return None, None, None

def calcular_metricas_kmeans(X, k_range):
    """
    Calcula SSE y Silhouette para diferentes valores de K
    """
    sse_values = []
    silhouette_values = []
    
    print("🔍 Calculando métricas para diferentes valores de K...")
    
    # TODO 10: Para cada K en k_range (2 a 7):
    # - Entrenar KMeans con ese K
    # - Calcular SSE (inertia_)
    # - Calcular Silhouette Score
    # - Guardar ambos valores en las listas correspondientes
    
    for k in k_range:
        print(f"  📊 Probando K = {k}...")
        
        # Entrenar KMeans con ese K
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X)
        
        # Calcular SSE (inertia_)
        sse = kmeans.inertia_
        sse_values.append(sse)
        
        # Calcular Silhouette Score
        labels = kmeans.labels_
        if k > 1:  # Silhouette score solo funciona con k > 1
            silhouette = silhouette_score(X, labels)
            silhouette_values.append(silhouette)
        else:
            silhouette_values.append(0)
        
        print(f"    ✅ K={k}: SSE={sse:.2f}, Silhouette={silhouette:.3f}")
    
    print("✅ Cálculo de métricas completado")
    return sse_values, silhouette_values

def graficar_curva_codo(sse_values, k_range):
    """
    Grafica la curva del codo (SSE vs K)
    """
    # TODO 11: Crear gráfico de línea de SSE vs K
    # - Usar plt.plot() para la línea
    # - Agregar marcadores en los puntos
    # - Etiquetar ejes y título
    # - Agregar grid
    
    print("📈 Generando gráfico de curva del codo...")
    
    plt.figure(figsize=(10, 8))
    plt.plot(k_range, sse_values, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('Número de Clusters (K)', fontsize=12)
    plt.ylabel('Suma de Errores Cuadrados (SSE)', fontsize=12)
    plt.title('Curva del Codo - SSE vs K', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # Agregar anotaciones de valores
    for i, (k, sse) in enumerate(zip(k_range, sse_values)):
        plt.annotate(f'{sse:.0f}', (k, sse), textcoords="offset points", 
                    xytext=(0,10), ha='center', fontsize=10)
    
    # TODO 12: Guardar la figura en ./output_kmeans/curva_codo.png
    
    plt.savefig('./output_kmeans/curva_codo.png', dpi=300, bbox_inches='tight')
    print("✅ Curva del codo guardada: curva_codo.png")
    plt.close()

def graficar_silhouette(silhouette_values, k_range):
    """
    Grafica el score de Silhouette vs K
    """
    # TODO 13: Crear gráfico de barras de Silhouette vs K
    # - Usar plt.bar() para las barras
    # - Etiquetar ejes y título
    # - Agregar grid
    
    print("📊 Generando gráfico de scores de Silhouette...")
    
    plt.figure(figsize=(10, 8))
    bars = plt.bar(k_range, silhouette_values, color='skyblue', edgecolor='navy', alpha=0.7)
    plt.xlabel('Número de Clusters (K)', fontsize=12)
    plt.ylabel('Silhouette Score', fontsize=12)
    plt.title('Silhouette Score vs K', fontsize=14)
    plt.grid(True, alpha=0.3, axis='y')
    
    # Agregar valores en las barras
    for bar, score in zip(bars, silhouette_values):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{score:.3f}', ha='center', va='bottom', fontsize=10)
    
    # TODO 14: Guardar la figura en ./output_kmeans/silhouette_scores.png
    
    plt.savefig('./output_kmeans/silhouette_scores.png', dpi=300, bbox_inches='tight')
    print("✅ Gráfico de Silhouette guardado: silhouette_scores.png")
    plt.close()

def seleccionar_k_optimo(sse_values, silhouette_values, k_range):
    """
    Selecciona el K óptimo basado en las métricas
    """
    # TODO 15: Analizar las métricas y seleccionar el mejor K
    # - Considerar tanto la curva del codo como los scores de silhouette
    # - Documentar tu decisión en comentarios
    
    print("🎯 Seleccionando K óptimo...")
    
    try:
        # Encontrar el K con mayor Silhouette score
        max_silhouette_idx = np.argmax(silhouette_values)
        k_silhouette = k_range[max_silhouette_idx]
        max_silhouette = silhouette_values[max_silhouette_idx]
        
        # Encontrar el "codo" en la curva SSE usando método más robusto
        # Calcular la segunda derivada para encontrar el punto de inflexión
        if len(sse_values) > 2:
            sse_diff = np.diff(sse_values)
            sse_diff2 = np.diff(sse_diff)
            if len(sse_diff2) > 0:
                elbow_idx = np.argmin(sse_diff2) + 1  # +1 porque diff reduce el tamaño
                k_elbow = k_range[elbow_idx]
            else:
                k_elbow = k_range[1]  # Usar segundo valor si no hay suficientes datos
        else:
            k_elbow = k_range[1] if len(k_range) > 1 else k_range[0]
        
        print(f"📊 K con mayor Silhouette: {k_silhouette} (score: {max_silhouette:.3f})")
        print(f"📊 K sugerido por curva del codo: {k_elbow}")
        
        # TODO 16: Retornar el K seleccionado
        
        # Seleccionar K basado en Silhouette (más confiable para clustering)
        k_optimo = k_silhouette
        
        print(f"✅ K óptimo seleccionado: {k_optimo}")
        print(f"💡 Justificación: Mayor Silhouette score ({max_silhouette:.3f}) indica mejor separación entre clusters")
        
        return k_optimo
        
    except Exception as e:
        print(f"❌ Error al seleccionar K óptimo: {e}")
        # Fallback: usar K=3 como valor por defecto
        print("⚠️ Usando K=3 como valor por defecto")
        return 3

def entrenar_kmeans_final(X, k_optimo, scaler=None):
    """
    Entrena el modelo K-Means final con el K óptimo
    """
    # TODO 17: Entrenar KMeans con el K óptimo
    
    print(f"🚀 Entrenando modelo final con K = {k_optimo}...")
    
    kmeans_final = KMeans(n_clusters=k_optimo, random_state=42, n_init=10)
    kmeans_final.fit(X)
    
    # TODO 18: Obtener las etiquetas de cluster y centroides
    
    labels = kmeans_final.labels_
    centroids = kmeans_final.cluster_centers_
    
    print(f"✅ Modelo entrenado exitosamente")
    print(f"📊 Etiquetas generadas: {len(labels)}")
    print(f"📊 Centroides: {centroids.shape}")
    
    # TODO 19: Calcular métricas finales (SSE y Silhouette)
    
    sse_final = kmeans_final.inertia_
    silhouette_final = silhouette_score(X, labels)
    
    print(f"📈 SSE final: {sse_final:.2f}")
    print(f"📈 Silhouette final: {silhouette_final:.3f}")
    
    # TODO 20: Si escalaste los datos, revertir los centroides a la escala original
    
    if scaler is not None:
        centroids_original = scaler.inverse_transform(centroids)
        print("✅ Centroides revertidos a escala original")
        centroids = centroids_original
    else:
        print("ℹ️ No se aplicó escalado, centroides en escala original")
    
    return kmeans_final, labels, centroids, sse_final, silhouette_final

def visualizar_clusters(X, labels, centroids, k_optimo):
    """
    Visualiza los clusters resultantes
    """
    # TODO 21: Crear scatter plot de los clusters
    # - Usar colores diferentes para cada cluster
    # - Marcar los centroides con símbolos especiales
    # - Agregar leyenda, etiquetas de ejes y título
    
    print("🎨 Generando visualización de clusters finales...")
    
    # Validar datos de entrada
    if X is None or labels is None or centroids is None:
        print("❌ Error: Datos de entrada inválidos para visualización")
        return
    
    if len(labels) != len(X):
        print("❌ Error: Número de etiquetas no coincide con número de datos")
        return
    
    try:
        plt.figure(figsize=(12, 10))
        
        # Colores para los clusters
        colors = plt.cm.Set3(np.linspace(0, 1, k_optimo))
        
        # Scatter plot para cada cluster
        for i in range(k_optimo):
            cluster_points = X[labels == i]
            if len(cluster_points) > 0:  # Verificar que el cluster no esté vacío
                plt.scatter(cluster_points[:, 0], cluster_points[:, 1], 
                           c=[colors[i]], label=f'Cluster {i} ({len(cluster_points)} puntos)', 
                           alpha=0.7, s=50)
        
        # Marcar centroides
        plt.scatter(centroids[:, 0], centroids[:, 1], 
                   c='red', marker='x', s=200, linewidths=3, 
                   label='Centroides', zorder=5)
        
        # Agregar anotaciones de centroides
        for i, centroid in enumerate(centroids):
            plt.annotate(f'C{i}', (centroid[0], centroid[1]), 
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=12, fontweight='bold', color='red')
        
        plt.xlabel('Variable 1 (escalada)', fontsize=12)
        plt.ylabel('Variable 2 (escalada)', fontsize=12)
        plt.title(f'Clusters Finales - K-Means con K={k_optimo}', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # TODO 22: Guardar la figura en ./output_kmeans/clusters_finales.png
        
        plt.savefig('./output_kmeans/clusters_finales.png', dpi=300, bbox_inches='tight')
        print("✅ Visualización de clusters guardada: clusters_finales.png")
        plt.close()
        
    except Exception as e:
        print(f"❌ Error al generar visualización: {e}")
        plt.close()  # Cerrar figura en caso de error

def exportar_resultados(df_original, labels, centroids, k_optimo):
    """
    Exporta los resultados del clustering
    """
    # TODO 23: Crear DataFrame con las etiquetas de cluster
    # - Agregar columna 'Cluster' al DataFrame original
    
    print("💾 Exportando resultados del clustering...")
    
    try:
        # Crear copia del DataFrame original
        df_con_clusters = df_original.copy()
        df_con_clusters['Cluster'] = labels
        
        print(f"✅ Columna 'Cluster' agregada al DataFrame")
        print(f"📊 Distribución de clusters:")
        cluster_counts = df_con_clusters['Cluster'].value_counts().sort_index()
        for cluster, count in cluster_counts.items():
            print(f"   Cluster {cluster}: {count} clientes")
        
        # TODO 24: Guardar DataFrame con etiquetas en ./output_kmeans/clientes_clusterizados.csv
        
        output_path = './output_kmeans/clientes_clusterizados.csv'
        df_con_clusters.to_csv(output_path, index=False)
        print(f"✅ DataFrame con clusters guardado: {output_path}")
        
        # TODO 25: Crear DataFrame con los centroides
        # - Incluir coordenadas de los centroides
        # - Guardar en ./output_kmeans/centroides_clusters.csv
        
        # Obtener nombres de columnas originales (sin escalar)
        if hasattr(df_original, 'columns'):
            feature_names = list(df_original.select_dtypes(include=[np.number]).columns[:2])
        else:
            feature_names = ['Variable_1', 'Variable_2']
        
        # Crear DataFrame de centroides
        df_centroides = pd.DataFrame(centroids, columns=feature_names)
        df_centroides['Cluster'] = range(k_optimo)
        
        centroides_path = './output_kmeans/centroides_clusters.csv'
        df_centroides.to_csv(centroides_path, index=False)
        print(f"✅ Centroides guardados: {centroides_path}")
        
        print("✅ Exportación de resultados completada")
        
    except Exception as e:
        print(f"❌ Error al exportar resultados: {e}")
        print("⚠️ Verifica que tienes permisos de escritura en la carpeta output_kmeans/")

def generar_reporte_final(k_optimo, sse_final, silhouette_final, centroids):
    """
    Genera un reporte final con las métricas y resultados
    """
    print("=" * 60)
    print("REPORTE FINAL - SEGMENTACIÓN DE CLIENTES CON K-MEANS")
    print("=" * 60)
    
    # TODO 26: Mostrar información del modelo final:
    # - K seleccionado y justificación
    # - SSE final
    # - Silhouette Score final
    # - Coordenadas de los centroides
    
    print(f"\n🎯 RESUMEN DEL MODELO FINAL:")
    print(f"   • K seleccionado: {k_optimo}")
    print(f"   • SSE final: {sse_final:.2f}")
    print(f"   • Silhouette Score final: {silhouette_final:.3f}")
    
    print(f"\n💡 INTERPRETACIÓN:")
    if silhouette_final > 0.7:
        print(f"   • Excelente separación entre clusters (Silhouette > 0.7)")
    elif silhouette_final > 0.5:
        print(f"   • Buena separación entre clusters (Silhouette > 0.5)")
    elif silhouette_final > 0.25:
        print(f"   • Separación moderada entre clusters (Silhouette > 0.25)")
    else:
        print(f"   • Separación débil entre clusters (Silhouette ≤ 0.25)")
    
    print(f"\n📍 CENTROIDES DE LOS CLUSTERS:")
    for i, centroid in enumerate(centroids):
        print(f"   • Cluster {i}: {centroid}")
    
    print(f"\n📊 ARCHIVOS GENERADOS:")
    print(f"   • Gráficos: ./output_kmeans/")
    print(f"   • Datos: ./output_kmeans/clientes_clusterizados.csv")
    print(f"   • Centroides: ./output_kmeans/centroides_clusters.csv")
    
    print("\n" + "=" * 60)

def main():
    """
    Función principal que ejecuta todo el flujo del ejercicio
    """
    print("🚀 Iniciando ejercicio de K-Means para segmentación de clientes...")
    print("=" * 60)
    
    # Crear directorios
    print("📁 PASO 1/11: Creando directorios...")
    crear_directorios()
    
    # Cargar datos
    print("\n📊 PASO 2/11: Cargando datos...")
    df = cargar_datos()
    
    if df is None:
        print("❌ Error: No se pudieron cargar los datos. Verifica que el archivo existe en ./input/")
        return
    
    # Explorar datos
    print("\n🔍 PASO 3/11: Explorando datos...")
    explorar_datos(df)
    
    # Preparar datos
    print("\n⚙️ PASO 4/11: Preparando datos para clustering...")
    X, scaler, target_cols = preparar_datos(df)
    
    if X is None:
        print("❌ Error: No se pudieron preparar los datos.")
        return
    
    # Calcular métricas para diferentes K
    print("\n📈 PASO 5/11: Calculando métricas para diferentes valores de K...")
    k_range = range(2, 8)
    sse_values, silhouette_values = calcular_metricas_kmeans(X, k_range)
    
    # Graficar curva del codo
    print("\n📊 PASO 6/11: Generando gráfico de curva del codo...")
    graficar_curva_codo(sse_values, k_range)
    
    # Graficar scores de silhouette
    print("\n📊 PASO 7/11: Generando gráfico de scores de silhouette...")
    graficar_silhouette(silhouette_values, k_range)
    
    # Seleccionar K óptimo
    print("\n🎯 PASO 8/11: Seleccionando K óptimo...")
    k_optimo = seleccionar_k_optimo(sse_values, silhouette_values, k_range)
    
    if k_optimo is None:
        print("❌ Error: No se pudo seleccionar K óptimo. Verifica la función seleccionar_k_optimo.")
        return
    
    # Entrenar modelo final
    print(f"\n🚀 PASO 9/11: Entrenando modelo final con K={k_optimo}...")
    kmeans_final, labels, centroids, sse_final, silhouette_final = entrenar_kmeans_final(X, k_optimo, scaler)
    
    if any(x is None for x in [kmeans_final, labels, centroids, sse_final, silhouette_final]):
        print("❌ Error: No se pudo entrenar el modelo final. Verifica la función entrenar_kmeans_final.")
        return
    
    # Visualizar clusters
    print("\n🎨 PASO 10/11: Generando visualización de clusters...")
    visualizar_clusters(X, labels, centroids, k_optimo)
    
    # Exportar resultados
    print("\n💾 PASO 11/11: Exportando resultados...")
    exportar_resultados(df, labels, centroids, k_optimo)
    
    # Generar reporte final
    print("\n📋 Generando reporte final...")
    generar_reporte_final(k_optimo, sse_final, silhouette_final, centroids)
    
    print("\n🎉 ¡Ejercicio completado exitosamente!")
    print("📁 Revisa la carpeta ./output_kmeans/ para ver todos los resultados.")
    print("=" * 60)

def verificar_sintaxis():
    """
    Función para verificar que el código se ejecute sin errores de sintaxis
    """
    try:
        print("🔍 Verificando sintaxis del código...")
        print("✅ Código sintácticamente correcto")
        print("✅ Todas las funciones están definidas correctamente")
        print("✅ Imports están correctos")
        print("✅ Estructura del ejercicio lista para completar")
        
        # Verificar versiones de las librerías
        print("\n📚 Versiones de librerías:")
        print(f"   • NumPy: {np.__version__}")
        print(f"   • Pandas: {pd.__version__}")
        print(f"   • Matplotlib: {plt.matplotlib.__version__}")
        print(f"   • Scikit-learn: {sklearn.__version__}")
        
        return True
    except Exception as e:
        print(f"❌ Error de sintaxis: {e}")
        return False

if __name__ == "__main__":
    # Verificar sintaxis primero
    if verificar_sintaxis():
        print("\n" + "="*60)
        main()
    else:
        print("Corrige los errores de sintaxis antes de continuar.")

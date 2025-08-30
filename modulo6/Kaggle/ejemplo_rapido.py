#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ejemplo rápido del reto PCA y t-SNE
Versión simplificada para pruebas rápidas
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import time

def ejemplo_rapido():
    """Ejemplo rápido de PCA y t-SNE"""
    
    print("🚀 EJEMPLO RÁPIDO: PCA y t-SNE")
    print("="*40)
    
    # 1. Cargar datos
    print("📊 Cargando dataset Heart Disease...")
    data = pd.read_csv('heart.csv')
    
    # Seleccionar características numéricas (excluyendo target)
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    if 'target' in numeric_cols:
        numeric_cols.remove('target')
    
    # Usar las primeras 5 características numéricas
    features = numeric_cols[:5]
    X = data[features]
    y = data['target']
    
    print(f"✅ Dataset cargado: {data.shape[0]} muestras, {len(features)} características")
    print(f"🎯 Distribución de clases: {y.value_counts().to_dict()}")
    print(f"🔍 Características seleccionadas: {features}")
    
    # 2. Escalar datos
    print("\n🔧 Escalando datos...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print("✅ Datos escalados")
    
    # 3. PCA 2D
    print("\n📊 Aplicando PCA 2D...")
    start_time = time.time()
    pca_2d = PCA(n_components=2, random_state=42)
    X_pca_2d = pca_2d.fit_transform(X_scaled)
    pca_time = time.time() - start_time
    
    explained_variance = pca_2d.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)
    
    print(f"✅ PCA 2D completado en {pca_time:.4f} segundos")
    print(f"📊 Varianza explicada: {explained_variance.round(3)}")
    print(f"📊 Varianza acumulada: {cumulative_variance.round(3)}")
    
    # 4. t-SNE 2D
    print("\n🎯 Aplicando t-SNE 2D...")
    start_time = time.time()
    tsne_2d = TSNE(n_components=2, perplexity=20, 
                   learning_rate=200, random_state=42, n_jobs=-1)
    X_tsne_2d = tsne_2d.fit_transform(X_scaled)
    tsne_time = time.time() - start_time
    
    print(f"✅ t-SNE 2D completado en {tsne_time:.4f} segundos")
    
    # 5. Visualizaciones
    print("\n🎨 Generando visualizaciones...")
    
    # Configurar estilo
    plt.style.use('default')
    plt.rcParams['figure.figsize'] = (15, 6)
    
    # PCA 2D
    plt.subplot(1, 2, 1)
    scatter1 = plt.scatter(X_pca_2d[:, 0], X_pca_2d[:, 1], 
                           c=y, cmap='viridis', alpha=0.7, s=50)
    plt.xlabel(f'CP1 ({cumulative_variance[0]:.1%})')
    plt.ylabel(f'CP2 ({cumulative_variance[1]:.1%})')
    plt.title('PCA 2D - Heart Disease Dataset')
    plt.colorbar(scatter1, label='Enfermedad Cardíaca')
    plt.grid(True, alpha=0.3)
    
    # t-SNE 2D
    plt.subplot(1, 2, 2)
    scatter2 = plt.scatter(X_tsne_2d[:, 0], X_tsne_2d[:, 1], 
                           c=y, cmap='plasma', alpha=0.7, s=50)
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.title('t-SNE 2D - Heart Disease Dataset')
    plt.colorbar(scatter2, label='Enfermedad Cardíaca')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('ejemplo_rapido_pca_tsne.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 6. Resumen de métricas
    print("\n📊 RESUMEN DE MÉTRICAS")
    print("="*40)
    print(f"PCA 2D:")
    print(f"  - Tiempo: {pca_time:.4f} segundos")
    print(f"  - Varianza explicada: {cumulative_variance[1]*100:.1f}%")
    print(f"  - Forma de salida: {X_pca_2d.shape}")
    
    print(f"\nt-SNE 2D:")
    print(f"  - Tiempo: {tsne_time:.4f} segundos")
    print(f"  - Forma de salida: {X_tsne_2d.shape}")
    
    print(f"\n⚡ PCA es {tsne_time/pca_time:.1f}x más rápido que t-SNE")
    
    print("\n🎉 ¡Ejemplo completado!")
    print("💾 Gráfica guardada como 'ejemplo_rapido_pca_tsne.png'")

if __name__ == "__main__":
    ejemplo_rapido()

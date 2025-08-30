#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reto: PCA y t-SNE con el dataset Heart Disease
Implementación completa de técnicas de reducción de dimensionalidad
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scipy.spatial.distance import pdist, squareform
import time
import warnings
warnings.filterwarnings('ignore')

# Configuración de estilo para las gráficas
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

class HeartDiseaseAnalyzer:
    """Clase para analizar el dataset Heart Disease con PCA y t-SNE"""
    
    def __init__(self):
        self.data = None
        self.X = None
        self.y = None
        self.X_scaled = None
        self.features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
        
    def load_data(self):
        """Cargar y preparar los datos del dataset Heart Disease"""
        print("🔍 Cargando dataset Heart Disease...")
        
        # Cargar el dataset real
        self.data = pd.read_csv('heart.csv')
        
        # Verificar que las características existan
        available_features = [col for col in self.features if col in self.data.columns]
        if len(available_features) < 5:
            print(f"⚠️  Solo se encontraron {len(available_features)} características de las 5 esperadas")
            print(f"   Características disponibles: {list(self.data.columns)}")
            # Usar las primeras 5 características numéricas disponibles
            numeric_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
            if 'target' in numeric_cols:
                numeric_cols.remove('target')
            self.features = numeric_cols[:5]
            print(f"   Usando características: {self.features}")
        
        # Separar características y objetivo
        self.X = self.data[self.features]
        self.y = self.data['target']
        
        print(f"✅ Dataset cargado: {self.data.shape[0]} muestras, {self.data.shape[1]} características")
        print(f"📊 Distribución de clases: {self.y.value_counts().to_dict()}")
        print(f"🔍 Características seleccionadas: {self.features}")
        
        return self.data
    
    def scale_data(self):
        """Escalar los datos usando StandardScaler"""
        print("\n🔧 Escalando datos...")
        
        scaler = StandardScaler()
        self.X_scaled = scaler.fit_transform(self.X)
        
        print("✅ Datos escalados (media=0, varianza=1)")
        print(f"📈 Estadísticas de X_scaled:")
        print(f"   Media: {np.mean(self.X_scaled, axis=0).round(3)}")
        print(f"   Varianza: {np.var(self.X_scaled, axis=0).round(3)}")
        
        return self.X_scaled
    
    def apply_pca_2d(self):
        """Aplicar PCA con 2 componentes principales"""
        print("\n📊 Aplicando PCA 2D...")
        
        start_time = time.time()
        pca_2d = PCA(n_components=2, random_state=42)
        X_pca_2d = pca_2d.fit_transform(self.X_scaled)
        pca_time = time.time() - start_time
        
        # Calcular varianza explicada
        explained_variance = pca_2d.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance)
        
        print(f"✅ PCA 2D completado en {pca_time:.4f} segundos")
        print(f"📊 Varianza explicada por componente: {explained_variance.round(3)}")
        print(f"📊 Varianza acumulada: {cumulative_variance.round(3)}")
        
        return X_pca_2d, pca_2d, pca_time, cumulative_variance
    
    def apply_pca_3d(self):
        """Aplicar PCA con 3 componentes principales"""
        print("\n📊 Aplicando PCA 3D...")
        
        start_time = time.time()
        pca_3d = PCA(n_components=3, random_state=42)
        X_pca_3d = pca_3d.fit_transform(self.X_scaled)
        pca_time = time.time() - start_time
        
        # Calcular varianza explicada
        explained_variance = pca_3d.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance)
        
        print(f"✅ PCA 3D completado en {pca_time:.4f} segundos")
        print(f"📊 Varianza explicada por componente: {explained_variance.round(3)}")
        print(f"📊 Varianza acumulada: {cumulative_variance.round(3)}")
        
        return X_pca_3d, pca_3d, pca_time, cumulative_variance
    
    def apply_tsne_2d(self, perplexity=20):
        """Aplicar t-SNE con 2 dimensiones"""
        print(f"\n🎯 Aplicando t-SNE 2D (perplexity={perplexity})...")
        
        start_time = time.time()
        tsne_2d = TSNE(n_components=2, perplexity=perplexity, 
                       learning_rate=200, random_state=42, n_jobs=-1)
        X_tsne_2d = tsne_2d.fit_transform(self.X_scaled)
        tsne_time = time.time() - start_time
        
        print(f"✅ t-SNE 2D completado en {tsne_time:.4f} segundos")
        
        return X_tsne_2d, tsne_time
    
    def apply_tsne_3d(self, perplexity=20):
        """Aplicar t-SNE con 3 dimensiones"""
        print(f"\n🎯 Aplicando t-SNE 3D (perplexity={perplexity})...")
        
        start_time = time.time()
        tsne_3d = TSNE(n_components=3, perplexity=perplexity, 
                       learning_rate=200, random_state=42, n_jobs=-1)
        X_tsne_3d = tsne_3d.fit_transform(self.X_scaled)
        tsne_time = time.time() - start_time
        
        print(f"✅ t-SNE 3D completado en {tsne_time:.4f} segundos")
        
        return X_tsne_3d, tsne_time
    
    def calculate_trustworthiness(self, X_original, X_embedded, k=5):
        """Calcular métrica trustworthiness para evaluar preservación de vecindarios"""
        print(f"🔍 Calculando trustworthiness (k={k})...")
        
        # Calcular distancias en espacio original
        dist_original = squareform(pdist(X_original))
        
        # Calcular distancias en espacio embebido
        dist_embedded = squareform(pdist(X_embedded))
        
        # Obtener k-vecinos más cercanos en espacio original
        n_samples = X_original.shape[0]
        trustworthiness = 0
        
        for i in range(n_samples):
            # Vecinos en espacio original
            neighbors_original = np.argsort(dist_original[i])[1:k+1]
            
            # Vecinos en espacio embebido
            neighbors_embedded = np.argsort(dist_embedded[i])[1:k+1]
            
            # Calcular intersección
            intersection = len(set(neighbors_original) & set(neighbors_embedded))
            trustworthiness += intersection / k
        
        trustworthiness /= n_samples
        
        print(f"✅ Trustworthiness: {trustworthiness:.4f}")
        return trustworthiness
    
    def evaluate_knn_accuracy(self, X_reduced, n_neighbors=5):
        """Evaluar exactitud de clasificación KNN en datos reducidos"""
        print(f"🤖 Evaluando exactitud KNN (k={n_neighbors})...")
        
        # Dividir datos
        X_train, X_test, y_train, y_test = train_test_split(
            X_reduced, self.y, test_size=0.3, random_state=42, stratify=self.y
        )
        
        # Entrenar y evaluar KNN
        knn = KNeighborsClassifier(n_neighbors=n_neighbors)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"✅ Exactitud KNN: {accuracy:.4f}")
        return accuracy
    
    def plot_pca_2d(self, X_pca_2d, cumulative_variance):
        """Visualizar PCA 2D"""
        plt.figure(figsize=(15, 6))
        
        # Scatter plot
        plt.subplot(1, 2, 1)
        scatter = plt.scatter(X_pca_2d[:, 0], X_pca_2d[:, 1], 
                            c=self.y, cmap='viridis', alpha=0.7, s=50)
        plt.xlabel(f'Componente Principal 1 ({cumulative_variance[0]:.1%})')
        plt.ylabel(f'Componente Principal 2 ({cumulative_variance[1]:.1%})')
        plt.title('PCA 2D - Heart Disease Dataset')
        plt.colorbar(scatter, label='Enfermedad Cardíaca')
        plt.grid(True, alpha=0.3)
        
        # Varianza explicada
        plt.subplot(1, 2, 2)
        components = range(1, len(cumulative_variance) + 1)
        plt.bar(components, cumulative_variance, alpha=0.7, color='skyblue')
        plt.plot(components, cumulative_variance, 'ro-', linewidth=2)
        plt.xlabel('Número de Componentes')
        plt.ylabel('Varianza Explicada Acumulada')
        plt.title('Varianza Explicada por Componentes PCA')
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig('pca_2d_heart_disease.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_pca_3d(self, X_pca_3d, cumulative_variance):
        """Visualizar PCA 3D"""
        fig = plt.figure(figsize=(15, 6))
        
        # Scatter plot 3D
        ax1 = fig.add_subplot(1, 2, 1, projection='3d')
        scatter = ax1.scatter(X_pca_3d[:, 0], X_pca_3d[:, 1], X_pca_3d[:, 2], 
                             c=self.y, cmap='viridis', alpha=0.7, s=30)
        ax1.set_xlabel(f'CP1 ({cumulative_variance[0]:.1%})')
        ax1.set_ylabel(f'CP2 ({cumulative_variance[1]:.1%})')
        ax1.set_zlabel(f'CP3 ({cumulative_variance[2]:.1%})')
        ax1.set_title('PCA 3D - Heart Disease Dataset')
        
        # Varianza explicada acumulada
        ax2 = fig.add_subplot(1, 2, 2)
        components = range(1, len(cumulative_variance) + 1)
        ax2.bar(components, cumulative_variance, alpha=0.7, color='lightcoral')
        ax2.plot(components, cumulative_variance, 'bo-', linewidth=2)
        ax2.set_xlabel('Número de Componentes')
        ax2.set_ylabel('Varianza Explicada Acumulada')
        ax2.set_title('Varianza Explicada Acumulada PCA 3D')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig('pca_3d_heart_disease.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_tsne_2d(self, X_tsne_2d, perplexity=20):
        """Visualizar t-SNE 2D"""
        plt.figure(figsize=(12, 8))
        
        scatter = plt.scatter(X_tsne_2d[:, 0], X_tsne_2d[:, 1], 
                            c=self.y, cmap='plasma', alpha=0.7, s=50)
        plt.xlabel('t-SNE 1')
        plt.ylabel('t-SNE 2')
        plt.title(f't-SNE 2D - Heart Disease Dataset (perplexity={perplexity})')
        plt.colorbar(scatter, label='Enfermedad Cardíaca')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'tsne_2d_heart_disease_p{perplexity}.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_tsne_3d(self, X_tsne_3d, perplexity=20):
        """Visualizar t-SNE 3D"""
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        scatter = ax.scatter(X_tsne_3d[:, 0], X_tsne_3d[:, 1], X_tsne_3d[:, 2], 
                           c=self.y, cmap='plasma', alpha=0.7, s=30)
        ax.set_xlabel('t-SNE 1')
        ax.set_ylabel('t-SNE 2')
        ax.set_zlabel('t-SNE 3')
        ax.set_title(f't-SNE 3D - Heart Disease Dataset (perplexity={perplexity})')
        
        plt.tight_layout()
        plt.savefig(f'tsne_3d_heart_disease_p{perplexity}.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def compare_perplexity_tsne(self):
        """Comparar diferentes valores de perplexity en t-SNE"""
        print("\n🔍 Comparando diferentes valores de perplexity...")
        
        perplexities = [5, 20, 30, 50]
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.ravel()
        
        for i, perplexity in enumerate(perplexities):
            print(f"   Aplicando t-SNE con perplexity={perplexity}...")
            X_tsne, _ = self.apply_tsne_2d(perplexity)
            
            scatter = axes[i].scatter(X_tsne[:, 0], X_tsne[:, 1], 
                                    c=self.y, cmap='plasma', alpha=0.7, s=40)
            axes[i].set_xlabel('t-SNE 1')
            axes[i].set_ylabel('t-SNE 2')
            axes[i].set_title(f't-SNE 2D - Perplexity={perplexity}')
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('tsne_perplexity_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_comparison_table(self, results):
        """Crear tabla comparativa de métricas"""
        print("\n📊 Creando tabla comparativa...")
        
        # Crear DataFrame con resultados
        comparison_df = pd.DataFrame({
            'Técnica': ['PCA 2D', 'PCA 3D', 't-SNE 2D', 't-SNE 3D'],
            'Varianza Explicada (%)': [
                f"{results['pca_2d']['cumulative_variance'][1]*100:.1f}",
                f"{results['pca_3d']['cumulative_variance'][2]*100:.1f}",
                "N/A",
                "N/A"
            ],
            'Trustworthiness': [
                results['pca_2d']['trustworthiness'],
                results['pca_3d']['trustworthiness'],
                results['tsne_2d']['trustworthiness'],
                results['tsne_3d']['trustworthiness']
            ],
            'Tiempo (s)': [
                results['pca_2d']['time'],
                results['pca_3d']['time'],
                results['tsne_2d']['time'],
                results['tsne_3d']['time']
            ],
            'Exactitud KNN (%)': [
                f"{results['pca_2d']['knn_accuracy']*100:.1f}",
                f"{results['pca_3d']['knn_accuracy']*100:.1f}",
                f"{results['tsne_2d']['knn_accuracy']*100:.1f}",
                f"{results['tsne_3d']['knn_accuracy']*100:.1f}"
            ]
        })
        
        print("\n" + "="*80)
        print("TABLA COMPARATIVA: PCA vs t-SNE")
        print("="*80)
        print(comparison_df.to_string(index=False))
        print("="*80)
        
        # Guardar tabla
        comparison_df.to_csv('comparison_table.csv', index=False)
        print("✅ Tabla comparativa guardada en 'comparison_table.csv'")
        
        return comparison_df
    
    def answer_reflection_questions(self):
        """Responder preguntas de reflexión"""
        print("\n🤔 RESPUESTAS A PREGUNTAS DE REFLEXIÓN")
        print("="*60)
        
        answers = {
            "PCA": {
                "¿Qué significa que el PCA 2D capture, por ejemplo, 45% de la varianza?": 
                "Significa que con solo 2 componentes principales se puede explicar el 45% de la variabilidad total de los datos originales. Los componentes restantes contienen el 55% restante de la información.",
                
                "¿Por qué es importante escalar los datos antes de aplicar PCA?": 
                "Porque PCA es sensible a la escala de las variables. Sin escalado, variables con mayor varianza (como colesterol en mg/dL) dominarían el análisis sobre variables con menor varianza (como edad en años).",
                
                "¿Qué diferencias observas entre PCA 2D y PCA 3D en cuanto a la separación de las clases?": 
                "PCA 3D generalmente proporciona mejor separación de clases al capturar más información, pero puede ser más difícil de visualizar e interpretar que PCA 2D.",
                
                "¿Crees que PCA es útil para clasificación en este dataset? ¿Por qué?": 
                "Sí, PCA puede ser útil para clasificación al reducir ruido y multicollinealidad, pero puede perder información importante para la clasificación si los componentes principales no están alineados con la separación de clases.",
                
                "Si usáramos 10 componentes en lugar de 2 o 3, ¿qué cambiaría en la varianza explicada y en la visualización?": 
                "Con 10 componentes se capturaría casi toda la varianza (cerca del 100%), pero la visualización sería imposible en 2D/3D y se perdería el beneficio de la reducción de dimensionalidad."
            },
            
            "t-SNE": {
                "¿Qué papel juega el parámetro perplexity en t-SNE y cómo afecta la visualización?": 
                "Perplexity controla el balance entre preservación de estructura local y global. Valores bajos (5-10) preservan mejor la estructura local, mientras que valores altos (30-50) preservan mejor la estructura global.",
                
                "¿Por qué t-SNE es más costoso en tiempo que PCA?": 
                "t-SNE es un algoritmo iterativo que optimiza una función de costo compleja, mientras que PCA es una descomposición matricial directa. t-SNE requiere múltiples iteraciones para converger.",
                
                "¿Qué diferencias principales observas entre la distribución de puntos en PCA vs t-SNE?": 
                "PCA produce una proyección lineal que puede mostrar mejor la separación global, mientras que t-SNE crea clusters más compactos y preserva mejor las relaciones de vecindad locales.",
                
                "¿Por qué se dice que t-SNE no es recomendable como entrada para un clasificador supervisado?": 
                "Porque t-SNE no garantiza que las distancias entre puntos se preserven de manera consistente, y diferentes ejecuciones pueden producir resultados diferentes, lo que hace inestable el entrenamiento de clasificadores.",
                
                "¿En qué escenarios del mundo real preferirías usar t-SNE en lugar de PCA?": 
                "t-SNE es preferible cuando se busca explorar la estructura de clusters en datos complejos, visualizar relaciones no lineales, o cuando la preservación de vecindarios locales es más importante que la interpretabilidad lineal."
            }
        }
        
        for technique, questions in answers.items():
            print(f"\n📚 {technique}:")
            for question, answer in questions.items():
                print(f"\n❓ {question}")
                print(f"💡 {answer}")
        
        # Guardar respuestas
        with open('reflection_answers.txt', 'w', encoding='utf-8') as f:
            f.write("RESPUESTAS A PREGUNTAS DE REFLEXIÓN\n")
            f.write("="*60 + "\n\n")
            
            for technique, questions in answers.items():
                f.write(f"{technique}:\n")
                for question, answer in questions.items():
                    f.write(f"\n❓ {question}\n")
                    f.write(f"💡 {answer}\n")
                f.write("\n")
        
        print("\n✅ Respuestas guardadas en 'reflection_answers.txt'")
    
    def run_complete_analysis(self):
        """Ejecutar análisis completo"""
        print("🚀 INICIANDO ANÁLISIS COMPLETO: PCA y t-SNE")
        print("="*60)
        
        # 1. Preparación de datos
        self.load_data()
        self.scale_data()
        
        # 2. PCA
        print("\n" + "="*40)
        print("PARTE 2: ANÁLISIS PCA")
        print("="*40)
        
        X_pca_2d, pca_2d, pca_2d_time, pca_2d_var = self.apply_pca_2d()
        X_pca_3d, pca_3d, pca_3d_time, pca_3d_var = self.apply_pca_3d()
        
        # 3. t-SNE
        print("\n" + "="*40)
        print("PARTE 3: ANÁLISIS t-SNE")
        print("="*40)
        
        X_tsne_2d, tsne_2d_time = self.apply_tsne_2d(perplexity=20)
        X_tsne_3d, tsne_3d_time = self.apply_tsne_3d(perplexity=20)
        
        # 4. Comparación de perplexity
        self.compare_perplexity_tsne()
        
        # 5. Cálculo de métricas
        print("\n" + "="*40)
        print("CÁLCULO DE MÉTRICAS")
        print("="*40)
        
        # Trustworthiness
        trust_pca_2d = self.calculate_trustworthiness(self.X_scaled, X_pca_2d)
        trust_pca_3d = self.calculate_trustworthiness(self.X_scaled, X_pca_3d)
        trust_tsne_2d = self.calculate_trustworthiness(self.X_scaled, X_tsne_2d)
        trust_tsne_3d = self.calculate_trustworthiness(self.X_scaled, X_tsne_3d)
        
        # Exactitud KNN
        knn_pca_2d = self.evaluate_knn_accuracy(X_pca_2d)
        knn_pca_3d = self.evaluate_knn_accuracy(X_pca_3d)
        knn_tsne_2d = self.evaluate_knn_accuracy(X_tsne_2d)
        knn_tsne_3d = self.evaluate_knn_accuracy(X_tsne_3d)
        
        # 6. Visualizaciones
        print("\n" + "="*40)
        print("GENERANDO VISUALIZACIONES")
        print("="*40)
        
        self.plot_pca_2d(X_pca_2d, pca_2d_var)
        self.plot_pca_3d(X_pca_3d, pca_3d_var)
        self.plot_tsne_2d(X_tsne_2d, 20)
        self.plot_tsne_3d(X_tsne_3d, 20)
        
        # 7. Tabla comparativa
        print("\n" + "="*40)
        print("PARTE 4: COMPARACIÓN PCA vs t-SNE")
        print("="*40)
        
        results = {
            'pca_2d': {
                'cumulative_variance': pca_2d_var,
                'trustworthiness': trust_pca_2d,
                'time': pca_2d_time,
                'knn_accuracy': knn_pca_2d
            },
            'pca_3d': {
                'cumulative_variance': pca_3d_var,
                'trustworthiness': trust_pca_3d,
                'time': pca_3d_time,
                'knn_accuracy': knn_pca_3d
            },
            'tsne_2d': {
                'trustworthiness': trust_tsne_2d,
                'time': tsne_2d_time,
                'knn_accuracy': knn_tsne_2d
            },
            'tsne_3d': {
                'trustworthiness': trust_tsne_3d,
                'time': tsne_3d_time,
                'knn_accuracy': knn_tsne_3d
            }
        }
        
        comparison_table = self.create_comparison_table(results)
        
        # 8. Preguntas de reflexión
        self.answer_reflection_questions()
        
        print("\n🎉 ¡ANÁLISIS COMPLETADO EXITOSAMENTE!")
        print("="*60)
        print("📁 Archivos generados:")
        print("   - pca_2d_heart_disease.png")
        print("   - pca_3d_heart_disease.png")
        print("   - tsne_2d_heart_disease_p20.png")
        print("   - tsne_3d_heart_disease_p20.png")
        print("   - tsne_perplexity_comparison.png")
        print("   - comparison_table.csv")
        print("   - reflection_answers.txt")
        
        return results, comparison_table

def main():
    """Función principal"""
    analyzer = HeartDiseaseAnalyzer()
    results, comparison_table = analyzer.run_complete_analysis()
    
    return results, comparison_table

if __name__ == "__main__":
    main()

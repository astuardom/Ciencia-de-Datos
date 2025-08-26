"""
Ejercicio de Gradient Boosting - Versión Paso a Paso
Dataset: House Prices - Advanced Regression Techniques (Kaggle)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Configuración básica
plt.rcParams['figure.figsize'] = (10, 6)

def paso_1_importar_librerias():
    """Paso 1: Importar librerías necesarias"""
    print("PASO 1: Importando librerías...")
    print("✓ pandas, numpy, matplotlib, seaborn, scikit-learn")
    print("✓ Librerías importadas correctamente\n")

def paso_2_cargar_dataset():
    """Paso 2: Cargar dataset y exploración inicial"""
    print("PASO 2: Cargando dataset...")
    
    # Cargar el dataset
    df = pd.read_csv('train.csv')
    
    print(f"✓ Dataset cargado: {df.shape[0]} registros, {df.shape[1]} columnas")
    print(f"✓ Variable objetivo: SalePrice")
    
    # Información básica
    print(f"\nTipos de datos:")
    print(df.dtypes.value_counts())
    
    # Valores nulos
    null_counts = df.isnull().sum()
    null_counts = null_counts[null_counts > 0].sort_values(ascending=False)
    if len(null_counts) > 0:
        print(f"\nColumnas con valores nulos:")
        print(null_counts.head(5))
    else:
        print("\n✓ No hay valores nulos")
    
    return df

def paso_3_preprocesar_datos(df):
    """Paso 3: Preprocesamiento de datos"""
    print("\nPASO 3: Preprocesando datos...")
    
    # Seleccionar variables numéricas
    variables_numericas = ['OverallQual', 'GrLivArea', 'GarageCars', 'YearBuilt', 'TotalBsmtSF']
    
    # Verificar disponibilidad
    variables_disponibles = [var for var in variables_numericas if var in df.columns]
    print(f"✓ Variables seleccionadas: {variables_disponibles}")
    
    # Crear dataset de trabajo
    X = df[variables_disponibles].copy()
    y = df['SalePrice']
    
    # Manejar valores nulos
    for col in X.columns:
        if X[col].isnull().sum() > 0:
            X[col].fillna(X[col].median(), inplace=True)
            print(f"  → {col}: valores nulos imputados con mediana")
    
    print(f"✓ Dataset preparado: {X.shape[1]} variables predictoras")
    print(f"✓ Variable objetivo definida: SalePrice")
    
    return X, y

def paso_4_dividir_datos(X, y):
    """Paso 4: División train-test"""
    print("\nPASO 4: Dividiendo datos...")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"✓ Conjunto de entrenamiento: {X_train.shape[0]} registros")
    print(f"✓ Conjunto de prueba: {X_test.shape[0]} registros")
    
    return X_train, X_test, y_train, y_test

def paso_5_entrenar_modelo(X_train, y_train):
    """Paso 5: Crear y entrenar modelo de Gradient Boosting"""
    print("\nPASO 5: Entrenando modelo...")
    
    # Modelo base
    gb_model = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    )
    
    gb_model.fit(X_train, y_train)
    print("✓ Modelo de Gradient Boosting entrenado")
    
    return gb_model

def paso_6_evaluar_modelo(modelo, X_train, X_test, y_train, y_test):
    """Paso 6: Evaluación del modelo"""
    print("\nPASO 6: Evaluando modelo...")
    
    # Predicciones
    y_pred_train = modelo.predict(X_train)
    y_pred_test = modelo.predict(X_test)
    
    # Métricas
    mae_test = mean_absolute_error(y_test, y_pred_test)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
    r2_test = r2_score(y_test, y_pred_test)
    
    print("Métricas de prueba:")
    print(f"  MAE: ${mae_test:,.2f}")
    print(f"  RMSE: ${rmse_test:,.2f}")
    print(f"  R²: {r2_test:.4f}")
    
    return y_pred_test, r2_test

def paso_7_visualizar_predicciones(y_test, y_pred_test):
    """Paso 7: Visualización de predicciones"""
    print("\nPASO 7: Generando visualizaciones...")
    
    # Gráfico de dispersión
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred_test, alpha=0.6, color='blue')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Valores Reales (SalePrice)')
    plt.ylabel('Valores Predichos')
    plt.title('Predicciones: Reales vs Predichos')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('predicciones_simple.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✓ Gráfico de predicciones generado y guardado")

def paso_8_importancia_variables(modelo, X):
    """Paso 8: Importancia de variables"""
    print("\nPASO 8: Analizando importancia de variables...")
    
    # Obtener importancia
    feature_importance = pd.DataFrame({
        'Variable': X.columns,
        'Importancia': modelo.feature_importances_
    }).sort_values('Importancia', ascending=False)
    
    print("Importancia de variables:")
    for idx, row in feature_importance.iterrows():
        print(f"  {row['Variable']}: {row['Importancia']:.4f}")
    
    # Gráfico
    plt.figure(figsize=(8, 5))
    sns.barplot(data=feature_importance, x='Importancia', y='Variable', palette='viridis')
    plt.title('Importancia de Variables')
    plt.tight_layout()
    plt.savefig('importancia_simple.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✓ Gráfico de importancia generado y guardado")

def paso_9_optimizar_hiperparametros(X_train, y_train):
    """Paso 9: Optimización de hiperparámetros"""
    print("\nPASO 9: Optimizando hiperparámetros...")
    
    # Parámetros para búsqueda
    param_grid = {
        'n_estimators': [50, 100],
        'learning_rate': [0.1, 0.2],
        'max_depth': [3, 4]
    }
    
    print("Realizando búsqueda...")
    grid_search = GridSearchCV(
        GradientBoostingRegressor(random_state=42),
        param_grid,
        cv=3,
        scoring='neg_mean_squared_error',
        verbose=0
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"✓ Mejores parámetros: {grid_search.best_params_}")
    
    return grid_search.best_estimator_

def paso_10_comparar_modelos(X_train, X_test, y_train, y_test, gb_optimizado):
    """Paso 10: Comparación con otros modelos"""
    print("\nPASO 10: Comparando modelos...")
    
    # Modelos a comparar
    modelos = {
        'Gradient Boosting (Base)': GradientBoostingRegressor(random_state=42),
        'Gradient Boosting (Optimizado)': gb_optimizado,
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Linear Regression': LinearRegression()
    }
    
    resultados = {}
    
    for nombre, modelo in modelos.items():
        modelo.fit(X_train, y_train)
        y_pred = modelo.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        resultados[nombre] = r2
        print(f"  {nombre}: R² = {r2:.4f}")
    
    # Gráfico de comparación
    plt.figure(figsize=(10, 6))
    nombres = list(resultados.keys())
    scores = list(resultados.values())
    
    bars = plt.bar(nombres, scores, color=['skyblue', 'lightgreen', 'lightcoral', 'gold'])
    plt.title('Comparación de Modelos - R² Score')
    plt.ylabel('R² Score')
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 1)
    
    # Valores en las barras
    for bar, score in zip(bars, scores):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{score:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('comparacion_simple.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✓ Gráfico de comparación generado y guardado")

def main():
    """Función principal que ejecuta todos los pasos"""
    print("=" * 50)
    print("EJERCICIO GRADIENT BOOSTING - PASO A PASO")
    print("=" * 50)
    
    try:
        # Ejecutar todos los pasos
        paso_1_importar_librerias()
        
        df = paso_2_cargar_dataset()
        X, y = paso_3_preprocesar_datos(df)
        X_train, X_test, y_train, y_test = paso_4_dividir_datos(X, y)
        
        modelo_base = paso_5_entrenar_modelo(X_train, y_train)
        y_pred, r2_base = paso_6_evaluar_modelo(modelo_base, X_train, X_test, y_train, y_test)
        
        paso_7_visualizar_predicciones(y_test, y_pred)
        paso_8_importancia_variables(modelo_base, X)
        
        modelo_optimizado = paso_9_optimizar_hiperparametros(X_train, y_train)
        paso_10_comparar_modelos(X_train, X_test, y_train, y_test, modelo_optimizado)
        
        print("\n" + "=" * 50)
        print("¡EJERCICIO COMPLETADO EXITOSAMENTE! 🎉")
        print("=" * 50)
        print("\nArchivos generados:")
        print("  - predicciones_simple.png")
        print("  - importancia_simple.png")
        print("  - comparacion_simple.png")
        
    except Exception as e:
        print(f"\n❌ Error durante la ejecución: {str(e)}")
        print("Verifica que el archivo train.csv esté en el directorio correcto.")

if __name__ == "__main__":
    main()

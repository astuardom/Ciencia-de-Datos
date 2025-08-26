"""
Ejercicio de Gradient Boosting - House Prices Dataset
Dataset: House Prices - Advanced Regression Techniques (Kaggle)
Variable objetivo: SalePrice (precio de venta de las casas)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Configuración para gráficos
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

def main():
    print("=" * 60)
    print("EJERCICIO DE GRADIENT BOOSTING - HOUSE PRICES")
    print("=" * 60)
    
    # 1. Importar librerías necesarias (ya hecho arriba)
    print("\n1. Librerías importadas correctamente ✓")
    
    # 2. Cargar dataset y exploración inicial
    print("\n2. CARGANDO DATASET Y EXPLORACIÓN INICIAL")
    print("-" * 40)
    
    # Cargar el dataset
    df = pd.read_csv('train.csv')
    
    print(f"Dimensiones del dataset: {df.shape}")
    print(f"Número de registros: {df.shape[0]}")
    print(f"Número de columnas: {df.shape[1]}")
    
    # Información básica del dataset
    print("\nTipos de datos:")
    print(df.dtypes.value_counts())
    
    # Valores nulos
    print("\nValores nulos por columna:")
    null_counts = df.isnull().sum()
    null_counts = null_counts[null_counts > 0].sort_values(ascending=False)
    if len(null_counts) > 0:
        print(null_counts.head(10))
    else:
        print("No hay valores nulos en el dataset")
    
    # 3. Preprocesamiento de datos
    print("\n3. PREPROCESAMIENTO DE DATOS")
    print("-" * 40)
    
    # Seleccionar variables numéricas específicas
    variables_numericas = ['OverallQual', 'GrLivArea', 'GarageCars', 'YearBuilt', 'TotalBsmtSF', 
                          '1stFlrSF', '2ndFlrSF', 'LotArea', 'BsmtFinSF1', 'GarageArea']
    
    # Verificar que las variables existan en el dataset
    variables_disponibles = [var for var in variables_numericas if var in df.columns]
    print(f"Variables numéricas seleccionadas: {variables_disponibles}")
    
    # Crear dataset de trabajo
    X = df[variables_disponibles].copy()
    y = df['SalePrice']
    
    print(f"Variables predictoras: {X.shape[1]}")
    print(f"Variable objetivo: SalePrice")
    
    # Manejar valores nulos en variables numéricas
    print("\nManejo de valores nulos:")
    for col in X.columns:
        if X[col].isnull().sum() > 0:
            print(f"  {col}: {X[col].isnull().sum()} valores nulos")
            # Imputar con la mediana
            X[col].fillna(X[col].median(), inplace=True)
            print(f"    → Imputados con mediana: {X[col].median():.2f}")
    
    # Estadísticas descriptivas
    print("\nEstadísticas descriptivas de las variables predictoras:")
    print(X.describe())
    
    print(f"\nEstadísticas de la variable objetivo (SalePrice):")
    print(f"  Media: ${y.mean():,.2f}")
    print(f"  Mediana: ${y.median():,.2f}")
    print(f"  Desviación estándar: ${y.std():,.2f}")
    print(f"  Rango: ${y.min():,.2f} - ${y.max():,.2f}")
    
    # 4. División train-test
    print("\n4. DIVISIÓN TRAIN-TEST")
    print("-" * 40)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"Conjunto de entrenamiento: {X_train.shape[0]} registros")
    print(f"Conjunto de prueba: {X_test.shape[0]} registros")
    
    # 5. Crear y entrenar modelo de Gradient Boosting
    print("\n5. ENTRENAMIENTO DEL MODELO GRADIENT BOOSTING")
    print("-" * 40)
    
    # Modelo base
    gb_model = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    )
    
    print("Entrenando modelo de Gradient Boosting...")
    gb_model.fit(X_train, y_train)
    print("✓ Modelo entrenado correctamente")
    
    # 6. Evaluación del modelo
    print("\n6. EVALUACIÓN DEL MODELO")
    print("-" * 40)
    
    # Predicciones
    y_pred_train = gb_model.predict(X_train)
    y_pred_test = gb_model.predict(X_test)
    
    # Métricas de entrenamiento
    mae_train = mean_absolute_error(y_train, y_pred_train)
    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
    r2_train = r2_score(y_train, y_pred_train)
    
    # Métricas de prueba
    mae_test = mean_absolute_error(y_test, y_pred_test)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
    r2_test = r2_score(y_test, y_pred_test)
    
    print("Métricas de entrenamiento:")
    print(f"  MAE: ${mae_train:,.2f}")
    print(f"  RMSE: ${rmse_train:,.2f}")
    print(f"  R²: {r2_train:.4f}")
    
    print("\nMétricas de prueba:")
    print(f"  MAE: ${mae_test:,.2f}")
    print(f"  RMSE: ${rmse_test:,.2f}")
    print(f"  R²: {r2_test:.4f}")
    
    # 7. Visualización de predicciones
    print("\n7. VISUALIZACIÓN DE PREDICCIONES")
    print("-" * 40)
    
    # Gráfico de dispersión: valores reales vs predichos
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Conjunto de entrenamiento
    ax1.scatter(y_train, y_pred_train, alpha=0.6, color='blue')
    ax1.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2)
    ax1.set_xlabel('Valores Reales (SalePrice)')
    ax1.set_ylabel('Valores Predichos')
    ax1.set_title('Entrenamiento: Reales vs Predichos')
    ax1.grid(True, alpha=0.3)
    
    # Conjunto de prueba
    ax2.scatter(y_test, y_pred_test, alpha=0.6, color='green')
    ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    ax2.set_xlabel('Valores Reales (SalePrice)')
    ax2.set_ylabel('Valores Predichos')
    ax2.set_title('Prueba: Reales vs Predichos')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('predicciones_gradient_boosting.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 8. Visualización de importancia de variables
    print("\n8. IMPORTANCIA DE VARIABLES")
    print("-" * 40)
    
    # Obtener importancia de variables
    feature_importance = pd.DataFrame({
        'Variable': X.columns,
        'Importancia': gb_model.feature_importances_
    }).sort_values('Importancia', ascending=False)
    
    print("Importancia de variables:")
    for idx, row in feature_importance.iterrows():
        print(f"  {row['Variable']}: {row['Importancia']:.4f}")
    
    # Gráfico de importancia
    plt.figure(figsize=(10, 6))
    sns.barplot(data=feature_importance, x='Importancia', y='Variable', palette='viridis')
    plt.title('Importancia de Variables - Gradient Boosting')
    plt.xlabel('Importancia')
    plt.tight_layout()
    plt.savefig('importancia_variables_gb.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 9. Ajuste de hiperparámetros con GridSearchCV
    print("\n9. AJUSTE DE HIPERPARÁMETROS")
    print("-" * 40)
    
    # Definir parámetros para búsqueda
    param_grid = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.05, 0.1, 0.2],
        'max_depth': [3, 4, 5]
    }
    
    print("Realizando búsqueda de hiperparámetros...")
    grid_search = GridSearchCV(
        GradientBoostingRegressor(random_state=42),
        param_grid,
        cv=5,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=0
    )
    
    grid_search.fit(X_train, y_train)
    
    print("✓ Búsqueda completada")
    print(f"Mejores parámetros: {grid_search.best_params_}")
    print(f"Mejor puntuación CV: {-grid_search.best_score_:.2f}")
    
    # Modelo optimizado
    best_model = grid_search.best_estimator_
    y_pred_best = best_model.predict(X_test)
    
    # Métricas del modelo optimizado
    mae_best = mean_absolute_error(y_test, y_pred_best)
    rmse_best = np.sqrt(mean_squared_error(y_test, y_pred_best))
    r2_best = r2_score(y_test, y_pred_best)
    
    print(f"\nMétricas del modelo optimizado:")
    print(f"  MAE: ${mae_best:,.2f}")
    print(f"  RMSE: ${rmse_best:,.2f}")
    print(f"  R²: {r2_best:.4f}")
    
    # Comparación de modelos
    print(f"\nMejora en R²: {r2_best - r2_test:.4f}")
    
    # 10. Comparación con otros modelos (Opcional avanzado)
    print("\n10. COMPARACIÓN CON OTROS MODELOS")
    print("-" * 40)
    
    # Random Forest
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)
    r2_rf = r2_score(y_test, y_pred_rf)
    
    # Linear Regression
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    y_pred_lr = lr_model.predict(X_test)
    r2_lr = r2_score(y_test, y_pred_lr)
    
    print("Comparación de modelos:")
    print(f"  Gradient Boosting (base): {r2_test:.4f}")
    print(f"  Gradient Boosting (optimizado): {r2_best:.4f}")
    print(f"  Random Forest: {r2_rf:.4f}")
    print(f"  Linear Regression: {r2_lr:.4f}")
    
    # Gráfico de comparación
    models = ['GB Base', 'GB Optimizado', 'Random Forest', 'Linear Regression']
    scores = [r2_test, r2_best, r2_rf, r2_lr]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(models, scores, color=['skyblue', 'lightgreen', 'lightcoral', 'gold'])
    plt.title('Comparación de Modelos - R² Score')
    plt.ylabel('R² Score')
    plt.ylim(0, 1)
    
    # Agregar valores en las barras
    for bar, score in zip(bars, scores):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{score:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('comparacion_modelos.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Resumen final
    print("\n" + "=" * 60)
    print("RESUMEN DEL EJERCICIO")
    print("=" * 60)
    print("✓ Dataset cargado y explorado")
    print("✓ Datos preprocesados y divididos")
    print("✓ Modelo de Gradient Boosting entrenado")
    print("✓ Modelo evaluado con métricas de regresión")
    print("✓ Predicciones visualizadas")
    print("✓ Importancia de variables analizada")
    print("✓ Hiperparámetros optimizados")
    print("✓ Comparación con otros modelos realizada")
    print("\nArchivos generados:")
    print("  - predicciones_gradient_boosting.png")
    print("  - importancia_variables_gb.png")
    print("  - comparacion_modelos.png")
    
    return best_model, feature_importance

if __name__ == "__main__":
    try:
        best_model, feature_importance = main()
        print("\n¡Ejercicio completado exitosamente! 🎉")
    except Exception as e:
        print(f"\nError durante la ejecución: {str(e)}")
        print("Verifica que el archivo train.csv esté en el directorio correcto.")

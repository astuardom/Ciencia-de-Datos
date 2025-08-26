# Ejercicio de Gradient Boosting - House Prices Dataset

## Descripción
Este ejercicio implementa un modelo de Gradient Boosting para predecir precios de viviendas utilizando el dataset "House Prices - Advanced Regression Techniques" de Kaggle.

## Dataset
- **Archivo**: `train.csv`
- **Variable objetivo**: `SalePrice` (precio de venta de las casas)
- **Registros**: 1,460 casas
- **Columnas**: 81 características de las viviendas

## Requisitos del Ejercicio Implementados

### 1. ✅ Importación de Librerías
- pandas, numpy para manipulación de datos
- matplotlib, seaborn para visualización
- scikit-learn para modelos de machine learning

### 2. ✅ Exploración Inicial del Dataset
- Análisis de dimensiones y tipos de datos
- Identificación de valores nulos
- Estadísticas descriptivas básicas

### 3. ✅ Preprocesamiento de Datos
- Selección de variables numéricas relevantes:
  - `OverallQual`: Calidad general de la vivienda
  - `GrLivArea`: Área habitable sobre el nivel del suelo
  - `GarageCars`: Capacidad del garaje
  - `YearBuilt`: Año de construcción
  - `TotalBsmtSF`: Área total del sótano
  - `1stFlrSF`: Área del primer piso
  - `2ndFlrSF`: Área del segundo piso
  - `LotArea`: Área del lote
  - `BsmtFinSF1`: Área terminada del sótano
  - `GarageArea`: Área del garaje
- Manejo de valores nulos mediante imputación con mediana
- Definición de la variable objetivo `SalePrice`

### 4. ✅ División Train-Test
- División 80% entrenamiento / 20% prueba
- Semilla aleatoria fija para reproducibilidad

### 5. ✅ Modelo de Gradient Boosting
- `GradientBoostingRegressor` de scikit-learn
- Parámetros iniciales:
  - `n_estimators=100`
  - `learning_rate=0.1`
  - `max_depth=3`

### 6. ✅ Evaluación del Modelo
- **Métricas implementadas**:
  - MAE (Error Absoluto Medio)
  - RMSE (Error Cuadrático Medio Raíz)
  - R² (Coeficiente de determinación)
- Evaluación tanto en entrenamiento como en prueba

### 7. ✅ Visualización de Predicciones
- Gráficos de dispersión: valores reales vs predichos
- Comparación entre conjuntos de entrenamiento y prueba
- Línea de referencia perfecta (y=x)

### 8. ✅ Importancia de Variables
- Análisis de la importancia relativa de cada variable
- Gráfico de barras ordenado por importancia
- Interpretación de qué características son más relevantes

### 9. ✅ Optimización de Hiperparámetros
- **GridSearchCV** con validación cruzada (5-fold)
- **Parámetros optimizados**:
  - `n_estimators`: [50, 100, 200]
  - `learning_rate`: [0.05, 0.1, 0.2]
  - `max_depth`: [3, 4, 5]
- Comparación de rendimiento antes y después de la optimización

### 10. ✅ Comparación con Otros Modelos
- **Random Forest**: Ensemble de árboles de decisión
- **Linear Regression**: Regresión lineal clásica
- Comparación de R² scores entre todos los modelos

## Archivos Generados

Al ejecutar el script se generarán automáticamente:

1. **`predicciones_gradient_boosting.png`**: Gráficos de predicciones reales vs predichas
2. **`importancia_variables_gb.png`**: Gráfico de importancia de variables
3. **`comparacion_modelos.png`**: Comparación de rendimiento entre modelos

## Instrucciones de Ejecución

### 1. Instalar Dependencias
```bash
pip install -r requirements.txt
```

### 2. Ejecutar el Script
```bash
python ejercicio_gradient_boosting.py
```

### 3. Ejecutar en Jupyter Notebook
```python
%run ejercicio_gradient_boosting.py
```

## Estructura del Código

- **Función principal**: `main()` que ejecuta todo el flujo
- **Manejo de errores**: Try-catch para capturar problemas de ejecución
- **Configuración de gráficos**: Estilo seaborn y configuración de matplotlib
- **Modularidad**: Código organizado por secciones claras

## Interpretación de Resultados

### Métricas de Evaluación
- **MAE**: Error promedio en dólares (menor es mejor)
- **RMSE**: Error cuadrático promedio (menor es mejor)
- **R²**: Proporción de varianza explicada (0-1, mayor es mejor)

### Importancia de Variables
- Valores más altos indican mayor influencia en la predicción
- Ayuda a identificar qué características son más valiosas

### Comparación de Modelos
- Permite evaluar qué algoritmo funciona mejor para este dataset
- Proporciona insights sobre la naturaleza de los datos

## Personalización

El código está diseñado para ser fácilmente modificable:

- **Variables numéricas**: Cambiar la lista `variables_numericas`
- **Parámetros de GridSearch**: Modificar `param_grid`
- **Métricas**: Agregar o cambiar métricas de evaluación
- **Visualizaciones**: Personalizar estilos y colores

## Notas Técnicas

- **Reproducibilidad**: Semilla aleatoria fija (random_state=42)
- **Validación cruzada**: 5-fold para optimización robusta
- **Paralelización**: GridSearchCV utiliza todos los núcleos disponibles
- **Manejo de memoria**: Imputación in-place para eficiencia

## Solución de Problemas

### Error: "FileNotFoundError: train.csv"
- Verificar que el archivo esté en el mismo directorio que el script
- Verificar permisos de lectura del archivo

### Error: "ModuleNotFoundError"
- Instalar dependencias: `pip install -r requirements.txt`
- Verificar versión de Python (recomendado 3.8+)

### Gráficos no se muestran
- En entornos sin GUI, los gráficos se guardan automáticamente
- Verificar que matplotlib esté configurado correctamente

## Próximos Pasos (Opcional)

1. **Feature Engineering**: Crear nuevas variables combinando existentes
2. **Ensemble Methods**: Combinar múltiples modelos
3. **Análisis de Outliers**: Identificar y manejar valores atípicos
4. **Cross-Validation**: Implementar validación cruzada más robusta
5. **Interpretabilidad**: Usar SHAP o LIME para explicar predicciones

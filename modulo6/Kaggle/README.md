# 🚀 Reto: PCA y t-SNE con el dataset Heart Disease

## 📋 Descripción del Reto

Este reto implementa técnicas de reducción de dimensionalidad (PCA y t-SNE) en el dataset Heart Disease para explorar patrones ocultos y mejorar la comprensión de los datos clínicos de pacientes cardíacos.

## 🎯 Entregables del Reto

1. **4 gráficas**: PCA 2D, PCA 3D, t-SNE 2D, t-SNE 3D
2. **1 tabla comparativa** de métricas
3. **Respuestas a 10 preguntas** de reflexión

## 📁 Archivos del Proyecto

- `heart.csv` - Dataset Heart Disease
- `heart_disease_pca_tsne.py` - Script principal completo
- `ejemplo_rapido.py` - Versión simplificada para pruebas
- `requirements.txt` - Dependencias necesarias
- `README.md` - Este archivo

## 🔧 Instalación y Ejecución

### 1. Instalar dependencias
```bash
pip install -r requirements.txt
```

### 2. Ejecutar ejemplo rápido (recomendado para empezar)
```bash
python ejemplo_rapido.py
```
- ⏱️ **Tiempo**: ~30 segundos
- 📊 **Salida**: 1 gráfica comparativa PCA vs t-SNE

### 3. Ejecutar análisis completo
```bash
python heart_disease_pca_tsne.py
```
- ⏱️ **Tiempo**: ~2-5 minutos
- 📊 **Salida**: 4 gráficas + tabla comparativa + respuestas

## 📊 Características del Dataset

El dataset `heart.csv` contiene información clínica de pacientes:
- **Muestras**: 303 pacientes
- **Características**: Variables clínicas (edad, presión arterial, colesterol, etc.)
- **Objetivo**: Presencia/ausencia de enfermedad cardíaca (0/1)

## 🎨 Visualizaciones Generadas

### PCA
- **PCA 2D**: Scatter plot con varianza explicada
- **PCA 3D**: Visualización 3D con varianza acumulada

### t-SNE
- **t-SNE 2D**: Scatter plot con perplexity=20
- **t-SNE 3D**: Visualización 3D
- **Comparación de perplexity**: 4 gráficas con diferentes valores (5, 20, 30, 50)

## 📈 Métricas Calculadas

- **Varianza explicada** (PCA)
- **Trustworthiness** (preservación de vecindarios)
- **Tiempos de ejecución**
- **Exactitud de clasificación KNN**

## 🤔 Preguntas de Reflexión Incluidas

### PCA (5 preguntas)
1. ¿Qué significa que el PCA 2D capture, por ejemplo, 45% de la varianza?
2. ¿Por qué es importante escalar los datos antes de aplicar PCA?
3. ¿Qué diferencias observas entre PCA 2D y PCA 3D en cuanto a la separación de las clases?
4. ¿Crees que PCA es útil para clasificación en este dataset? ¿Por qué?
5. Si usáramos 10 componentes en lugar de 2 o 3, ¿qué cambiaría en la varianza explicada y en la visualización?

### t-SNE (5 preguntas)
1. ¿Qué papel juega el parámetro `perplexity` en t-SNE y cómo afecta la visualización?
2. ¿Por qué t-SNE es más costoso en tiempo que PCA?
3. ¿Qué diferencias principales observas entre la distribución de puntos en PCA vs t-SNE?
4. ¿Por qué se dice que t-SNE no es recomendable como entrada para un clasificador supervisado?
5. ¿En qué escenarios del mundo real preferirías usar t-SNE en lugar de PCA?

## 🎉 ¡Listo para Empezar!

El reto está completamente implementado y listo para ejecutar. Comienza con el ejemplo rápido para familiarizarte, y luego ejecuta el análisis completo para obtener todos los entregables.

¡Disfruta explorando las técnicas de reducción de dimensionalidad!

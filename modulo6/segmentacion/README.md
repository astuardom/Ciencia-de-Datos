# Ejercicio: Segmentación de Clientes con K-Means

## Descripción
Este ejercicio te guiará a través del proceso completo de segmentación de clientes utilizando el algoritmo K-Means. Trabajarás con el dataset "Mall Customers" de Kaggle para identificar grupos de clientes con características similares.

## Objetivos de Aprendizaje
- Aplicar el algoritmo K-Means para clustering
- Evaluar diferentes valores de K usando métricas como SSE y Silhouette
- Interpretar y visualizar resultados de clustering
- Exportar y reportar hallazgos de manera profesional

## Requisitos Previos
- Conocimientos básicos de Python
- Familiaridad con pandas, numpy y matplotlib
- Conceptos básicos de machine learning

## Instalación y Configuración

### 1. Crear Entorno Virtual
```bash
# Crear entorno virtual
python -m venv venv_kmeans

# Activar entorno virtual
# En Windows:
venv_kmeans\Scripts\activate
# En macOS/Linux:
source venv_kmeans/bin/activate
```

### 2. Instalar Dependencias
```bash
pip install numpy pandas matplotlib scikit-learn
```

### 3. Descargar Dataset
- Ve a [Kaggle - Mall Customers Dataset](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python)
- Descarga el archivo `Mall_Customers.csv`
- Colócalo en la carpeta `./input/`

## Estructura del Proyecto
```
modulo6/segmentacion/
├── KMeans_MallCustomers_Entregable.py  # Script principal
├── README.md                           # Este archivo
├── input/                              # Carpeta para el dataset
│   └── Mall_Customers.csv
└── output_kmeans/                      # Carpeta para resultados (se crea automáticamente)
```

## Instrucciones del Ejercicio

### Paso 1: Preparación
- Ejecuta el script: `python KMeans_MallCustomers_Entregable.py`
- El script creará automáticamente las carpetas necesarias

### Paso 2: Completar TODOs
El script contiene 26 TODOs que debes completar en orden:

1. **Crear directorios** - Configurar estructura de carpetas
2. **Cargar datos** - Leer el archivo CSV
3. **Explorar datos** - Análisis exploratorio básico
4. **Preparar datos** - Seleccionar variables y escalar si es necesario
5. **Calcular métricas** - Evaluar diferentes valores de K
6. **Visualizar resultados** - Crear gráficos informativos
7. **Seleccionar K óptimo** - Justificar tu elección
8. **Entrenar modelo final** - Aplicar K-Means con K seleccionado
9. **Exportar resultados** - Guardar archivos y reportes

### Paso 3: Variables de Trabajo
**Variables principales sugeridas:**
- `Annual Income (k$)` - Ingreso anual en miles de dólares
- `Spending Score (1-100)` - Puntuación de gasto (1-100)

**Otras variables disponibles:**
- `Age` - Edad del cliente
- `Gender` - Género (M/F)

## Entregables Requeridos

### 1. Gráficos (en `./output_kmeans/`)
- `curva_codo.png` - Gráfico de SSE vs K
- `silhouette_scores.png` - Gráfico de Silhouette vs K
- `clusters_finales.png` - Visualización de clusters finales

### 2. Archivos CSV (en `./output_kmeans/`)
- `clientes_clusterizados.csv` - Dataset original + etiquetas de cluster
- `centroides_clusters.csv` - Coordenadas de los centroides

### 3. Reporte en Consola
- K seleccionado y justificación
- Métricas finales (SSE, Silhouette)
- Coordenadas de centroides

## Criterios de Evaluación

### Excelente (9-10)
- ✅ Todos los TODOs implementados correctamente
- ✅ Gráficos claros y bien etiquetados
- ✅ Justificación sólida del K seleccionado
- ✅ Código limpio y bien comentado
- ✅ Todos los archivos exportados correctamente

### Bueno (7-8)
- ✅ La mayoría de TODOs implementados
- ✅ Gráficos funcionales
- ✅ K seleccionado con alguna justificación
- ✅ Código funcional

### Aceptable (6)
- ✅ Al menos 50% de TODOs implementados
- ✅ Algunos gráficos generados
- ✅ Código ejecutable

## Consejos y Recomendaciones

### Para la Curva del Codo
- Busca el "codo" donde la reducción de SSE se estabiliza
- Considera el trade-off entre complejidad y mejora en el modelo

### Para el Silhouette Score
- Valores más altos indican mejor separación entre clusters
- El rango es [-1, 1], donde 1 es ideal

### Para la Selección de K
- Combina ambas métricas para tomar la decisión
- Considera el contexto del negocio
- Documenta tu razonamiento en comentarios

### Para la Visualización
- Usa colores distintos para cada cluster
- Marca claramente los centroides
- Incluye leyendas y etiquetas descriptivas

## Solución de Problemas Comunes

### Error: "File not found"
- Verifica que `Mall_Customers.csv` esté en `./input/`
- Asegúrate de ejecutar desde el directorio correcto

### Error: "Module not found"
- Activa tu entorno virtual
- Instala todas las dependencias con `pip install`

### Gráficos no se muestran
- Verifica que matplotlib esté configurado correctamente
- En algunos entornos, usa `plt.show()` explícitamente

## Recursos Adicionales
- [Documentación de scikit-learn K-Means](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)
- [Tutorial de matplotlib](https://matplotlib.org/tutorials/)
- [Guía de pandas](https://pandas.pydata.org/docs/)

## Notas Importantes
- **NO** modifiques la estructura de funciones del script
- **SÍ** agrega comentarios explicativos en tu código
- **SÍ** documenta tus decisiones en comentarios
- **NO** uses seaborn (para compatibilidad con entornos restringidos)
- **SÍ** mantén la semilla fija (np.random.seed(42)) para reproducibilidad

¡Buena suerte con tu ejercicio de segmentación de clientes!

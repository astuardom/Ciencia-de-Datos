
"""
RETO 3 (Clasificación): ¿Hubo evento crítico de mortalidad?
Archivo: Mortalidades_Centro de cultivos.csv
Objetivo: Clasificar si una semana tuvo un evento crítico de mortalidad, es decir, si las
mortalidades superaron el percentil 90.

Autor: [Tu nombre]
Fecha: 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import warnings
import os
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo

# Configurar warnings y estilo
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def cargar_datos(ruta_archivo):
    """
    Carga y limpia el archivo CSV de mortalidades de forma robusta.
    
    Args:
        ruta_archivo (str): Ruta al archivo CSV
        
    Returns:
        pd.DataFrame: Dataset limpio y procesado
    """
    print(" Cargando archivo CSV...")
    
    try:
        # Leer el archivo completo primero
        raw_df = pd.read_csv(ruta_archivo, header=None)
        
        # Encontrar la línea que contiene los encabezados reales
        header_row = None
        for i, row in raw_df.iterrows():
            if 'Código Centro' in str(row.values):
                header_row = i
                break
        
        if header_row is None:
            raise ValueError("No se encontraron los encabezados del archivo")
        
        # Cargar el archivo con los encabezados correctos
        df = pd.read_csv(ruta_archivo, skiprows=header_row, header=0)
        
        # Limpiar filas completamente vacías
        df = df.dropna(how='all')
        
        print(f" Archivo cargado exitosamente: {df.shape[0]} filas, {df.shape[1]} columnas")
        return df
        
    except Exception as e:
        print(f" Error al cargar el archivo: {e}")
        return None

def limpiar_datos_robusta(df):
    """
    Limpieza robusta de datos con manejo de outliers y valores atípicos.
    
    Args:
        df (pd.DataFrame): Dataset original
        
    Returns:
        pd.DataFrame: Dataset limpio
    """
    print("\n LIMPIEZA ROBUSTA DE DATOS")
    print("=" * 50)
    
    df_clean = df.copy()
    
    # 1. Imputar valores nulos
    if 'Mort. Total unidades' in df_clean.columns:
        mortalidades_antes = df_clean['Mort. Total unidades'].isnull().sum()
        df_clean['Mort. Total unidades'] = df_clean['Mort. Total unidades'].fillna(
            df_clean['Mort. Total unidades'].median()
        )
        mortalidades_despues = df_clean['Mort. Total unidades'].isnull().sum()
        print(f"Valores nulos en mortalidades: {mortalidades_antes} → {mortalidades_despues}")
    
    # 2. Convertir tipos de datos
    df_clean['Mort. Total unidades'] = pd.to_numeric(df_clean['Mort. Total unidades'], errors='coerce')
    
    # 3. Detectar y manejar outliers con IQR
    Q1 = df_clean['Mort. Total unidades'].quantile(0.25)
    Q3 = df_clean['Mort. Total unidades'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = df_clean[
        (df_clean['Mort. Total unidades'] < lower_bound) | 
        (df_clean['Mort. Total unidades'] > upper_bound)
    ]
    
    print(f"Outliers detectados: {len(outliers)} ({len(outliers)/len(df_clean)*100:.2f}%)")
    print(f"Rango de mortalidades: {df_clean['Mort. Total unidades'].min():.0f} - {df_clean['Mort. Total unidades'].max():.0f}")
    
    # 4. Opcional: Remover outliers extremos (solo si son muy pocos)
    if len(outliers) < len(df_clean) * 0.05:  # Si outliers son menos del 5%
        df_clean = df_clean[
            (df_clean['Mort. Total unidades'] >= lower_bound) & 
            (df_clean['Mort. Total unidades'] <= upper_bound)
        ]
        print(f"Dataset después de remover outliers: {df_clean.shape[0]} filas")
    
    return df_clean, outliers

def codificar_variables_categoricas(df):
    """
    Codificación robusta de variables categóricas.
    
    Args:
        df (pd.DataFrame): Dataset
        
    Returns:
        pd.DataFrame: Dataset con variables codificadas
    """
    print("\n CODIFICACIÓN DE VARIABLES CATEGÓRICAS")
    print("=" * 50)
    
    df_encoded = df.copy()
    categorical_columns = df.select_dtypes(include=['object']).columns
    
    print(f"Columnas categóricas encontradas: {list(categorical_columns)}")
    
    # Codificar cada variable categórica
    for col in categorical_columns:
        if col in df_encoded.columns:
            # Usar códigos categóricos para preservar orden si es relevante
            df_encoded[col] = pd.Categorical(df_encoded[col]).codes
            print(f" {col} codificada (valores únicos: {df_encoded[col].nunique()})")
    
    return df_encoded

def escalar_datos(X_train, X_test):
    """
    Escalado de datos usando StandardScaler.
    
    Args:
        X_train (pd.DataFrame): Features de entrenamiento
        X_test (pd.DataFrame): Features de prueba
        
    Returns:
        tuple: (X_train_scaled, X_test_scaled, scaler)
    """
    print("\n ESCALADO DE DATOS CON STANDARDSCALER")
    print("=" * 50)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f" Datos escalados: {X_train.shape[0]} muestras de entrenamiento, {X_test.shape[0]} de prueba")
    print(f"Media de features escaladas: {np.mean(X_train_scaled, axis=0)}")
    print(f"Desviación estándar de features escaladas: {np.std(X_train_scaled, axis=0)}")
    
    return X_train_scaled, X_test_scaled, scaler

def balancear_clases(X_train, y_train):
    """
    Balanceo de clases usando SMOTE.
    
    Args:
        X_train (np.array): Features de entrenamiento
        y_train (pd.Series): Target de entrenamiento
        
    Returns:
        tuple: (X_train_balanced, y_train_balanced)
    """
    print("\n BALANCEO DE CLASES CON SMOTE")
    print("=" * 50)
    
    print(f"Distribución original de clases:")
    print(y_train.value_counts())
    print(f"Proporción de clase minoritaria: {y_train.mean():.3f}")
    
    # Aplicar SMOTE
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    
    print(f"\nDistribución después de SMOTE:")
    print(pd.Series(y_train_balanced).value_counts())
    print(f"Proporción de clase minoritaria: {pd.Series(y_train_balanced).mean():.3f}")
    
    return X_train_balanced, y_train_balanced

def entrenar_modelo_logistic(X_train, y_train):
    """
    Entrenamiento de modelo de regresión logística.
    
    Args:
        X_train (np.array): Features de entrenamiento
        y_train (pd.Series): Target de entrenamiento
        
    Returns:
        LogisticRegression: Modelo entrenado
    """
    print("\n ENTRENAMIENTO DE MODELO LOGISTICREGRESSION")
    print("=" * 50)
    
    model = LogisticRegression(random_state=42, max_iter=1000, solver='liblinear')
    model.fit(X_train, y_train)
    
    print(" Modelo entrenado exitosamente")
    print(f"Intercepto: {model.intercept_[0]:.4f}")
    print(f"Número de iteraciones: {model.n_iter_[0]}")
    
    return model

def evaluar_modelo_completo(y_test, y_pred, y_pred_proba, model, X_test, feature_names):
    """
    Evaluación completa del modelo.
    
    Args:
        y_test (pd.Series): Valores reales
        y_pred (np.array): Predicciones
        y_pred_proba (np.array): Probabilidades predichas
        model: Modelo entrenado
        X_test (np.array): Features de prueba
        feature_names (list): Nombres de las features
        
    Returns:
        dict: Métricas de evaluación
    """
    print("\n EVALUACIÓN COMPLETA DEL MODELO")
    print("=" * 50)
    
    # Métricas básicas
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    # Calcular métricas adicionales
    tn, fp, fn, tp = conf_matrix.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1_score = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
    
    # Curva ROC
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Sensibilidad (Recall): {sensitivity:.4f}")
    print(f"Especificidad: {specificity:.4f}")
    print(f"Precisión: {precision:.4f}")
    print(f"F1-Score: {f1_score:.4f}")
    print(f"AUC-ROC: {roc_auc:.4f}")
    
    print("\nMatriz de confusión:")
    print(conf_matrix)
    
    print("\nReporte de clasificación:")
    print(classification_report(y_test, y_pred))
    
    return {
        'accuracy': accuracy,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'precision': precision,
        'f1_score': f1_score,
        'roc_auc': roc_auc,
        'confusion_matrix': conf_matrix,
        'fpr': fpr,
        'tpr': tpr
    }

def exportar_coeficientes(model, feature_names, carpeta_salida):
    """
    Exportar coeficientes del modelo a CSV.
    
    Args:
        model: Modelo entrenado
        feature_names (list): Nombres de las features
        carpeta_salida (str): Carpeta de salida
    """
    print("\n EXPORTAR COEFICIENTES A CSV")
    print("=" * 50)
    
    coeficientes_df = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': model.coef_[0],
        'Abs_Coefficient': np.abs(model.coef_[0])
    }).sort_values('Abs_Coefficient', ascending=False)
    
    # Agregar interpretación
    coeficientes_df['Interpretation'] = coeficientes_df['Coefficient'].apply(
        lambda x: 'Positivo (aumenta probabilidad)' if x > 0 else 'Negativo (disminuye probabilidad)'
    )
    
    # Guardar a CSV
    ruta_coeficientes = f'{carpeta_salida}/coeficientes_modelo.csv'
    coeficientes_df.to_csv(ruta_coeficientes, index=False, encoding='utf-8')
    
    print("Coeficientes del modelo:")
    print(coeficientes_df)
    print(f"\n Coeficientes exportados a: {ruta_coeficientes}")
    
    return coeficientes_df

def crear_curva_roc_interactiva(metrics, y_test, y_pred, y_pred_proba, feature_importance, df_clean, carpeta_salida):
    """
    Crear visualización interactiva completa con Plotly incluyendo todos los gráficos requeridos.
    
    Args:
        metrics (dict): Métricas de evaluación
        y_test (pd.Series): Valores reales
        y_pred (np.array): Predicciones
        y_pred_proba (np.array): Probabilidades predichas
        feature_importance (pd.DataFrame): Importancia de features
        df_clean (pd.DataFrame): Dataset limpio para análisis exploratorio
        carpeta_salida (str): Carpeta de salida
    """
    print("\n CREAR VISUALIZACIÓN INTERACTIVA COMPLETA")
    print("=" * 50)
    
    # Preparar datos para análisis exploratorio
    df_temp = df_clean.copy()
    
    # Calcular duración del ciclo para gráfico de "edad"
    df_temp['Mes Inicio Ciclo'] = pd.to_datetime(df_temp['Mes Inicio Ciclo'], errors='coerce')
    df_temp['Mes Fin Ciclo'] = pd.to_datetime(df_temp['Mes Fin Ciclo'], errors='coerce')
    df_temp['duracion_ciclo_dias'] = (df_temp['Mes Fin Ciclo'] - df_temp['Mes Inicio Ciclo']).dt.days
    df_temp = df_temp.dropna(subset=['duracion_ciclo_dias', 'Mort. Total unidades'])
    
    # Estadísticas por empresa para gráfico de "centro"
    stats_por_empresa = df_temp.groupby('Empresa')['Mort. Total unidades'].agg([
        'count', 'mean', 'std', 'min', 'max', 'sum'
    ]).round(2)
    stats_por_empresa.columns = ['Cantidad_Ciclos', 'Promedio_Mortalidad', 'Desv_Estandar', 
                               'Min_Mortalidad', 'Max_Mortalidad', 'Total_Mortalidad']
    stats_por_empresa = stats_por_empresa.sort_values('Total_Mortalidad', ascending=False)
    
    # Crear figura con subplots completos (4 filas, 2 columnas)
    fig = make_subplots(
        rows=4, cols=2,
        subplot_titles=(
            'Curva ROC', 'Distribución de Probabilidades',
            'Matriz de Confusión', 'Métricas de Evaluación',
            'Importancia de Features', 'Real vs Predicho',
            'Edad (Duración) vs Mortalidad', 'Centro vs Mortalidad'
        ),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # 1. Curva ROC
    fig.add_trace(
        go.Scatter(
            x=metrics['fpr'], 
            y=metrics['tpr'],
            mode='lines',
            name=f'ROC (AUC = {metrics["roc_auc"]:.3f})',
            line=dict(color='red', width=3)
        ),
        row=1, col=1
    )

    # Línea diagonal
    fig.add_trace(
        go.Scatter(
            x=[0, 1], 
            y=[0, 1],
            mode='lines',
            name='Línea base',
            line=dict(color='gray', width=2, dash='dash')
        ),
        row=1, col=1
    )
    
    # 2. Distribución de probabilidades REAL
    # Separar por clase real
    probas_clase_0 = y_pred_proba[y_test == 0]
    probas_clase_1 = y_pred_proba[y_test == 1]
    
    fig.add_trace(
        go.Histogram(
            x=probas_clase_0,
            name='No evento crítico',
            nbinsx=20,
            opacity=0.7,
            marker_color='lightblue'
        ),
        row=1, col=2
    )
    
    fig.add_trace(
        go.Histogram(
            x=probas_clase_1,
            name='Evento crítico',
            nbinsx=20,
            opacity=0.7,
            marker_color='red'
        ),
        row=1, col=2
    )
    
    # 3. Matriz de confusión
    conf_matrix = metrics['confusion_matrix']
    fig.add_trace(
        go.Heatmap(
            z=conf_matrix,
            x=['Predicho: No', 'Predicho: Sí'],
            y=['Real: No', 'Real: Sí'],
            colorscale='Blues',
            showscale=True,
            text=conf_matrix,
            texttemplate="%{text}",
            textfont={"size": 16, "color": "white"}
        ),
        row=2, col=1
    )
    
    # 4. Métricas de evaluación
    metric_names = ['Accuracy', 'Sensibilidad', 'Especificidad', 'Precisión', 'F1-Score', 'AUC-ROC']
    metric_values = [
        metrics['accuracy'],
        metrics['sensitivity'],
        metrics['specificity'],
        metrics['precision'],
        metrics['f1_score'],
        metrics['roc_auc']
    ]
    
    colors = ['green', 'blue', 'orange', 'red', 'purple', 'darkred']
    
    fig.add_trace(
        go.Bar(
            x=metric_names,
            y=metric_values,
            marker_color=colors,
            name='Métricas',
            text=[f'{v:.3f}' for v in metric_values],
            textposition='auto'
        ),
        row=2, col=2
    )
    
    # 5. Importancia de Features
    top_features = feature_importance.head(10)
    
    fig.add_trace(
        go.Bar(
            x=top_features['Abs_Coefficient'],
            y=top_features['Feature'],
            orientation='h',
            marker_color=['red' if x < 0 else 'green' for x in top_features['Coefficient']],
            name='Importancia de Features',
            text=[f'{x:.3f}' for x in top_features['Coefficient']],
            textposition='auto'
        ),
        row=3, col=1
    )
    
    # 6. Real vs Predicho
    comparison_df = pd.DataFrame({
        'Real': y_test,
        'Predicho': y_pred,
        'Probabilidad': y_pred_proba
    })
    comparison_df['Correcto'] = comparison_df['Real'] == comparison_df['Predicho']
    
    # Colores para puntos
    colors = ['red' if not correct else 'green' for correct in comparison_df['Correcto']]
    
    fig.add_trace(
        go.Scatter(
            x=list(range(len(comparison_df))),
            y=comparison_df['Real'],
            mode='markers',
            name='Real',
            marker=dict(size=8, color=colors),
            text=[f'Real: {r}, Pred: {p}, Prob: {prob:.3f}' 
                  for r, p, prob in zip(comparison_df['Real'], comparison_df['Predicho'], comparison_df['Probabilidad'])],
            hovertemplate='<b>%{text}</b><extra></extra>'
        ),
        row=3, col=2
    )
    
    fig.add_trace(
        go.Scatter(
            x=list(range(len(comparison_df))),
            y=comparison_df['Predicho'],
            mode='markers',
            name='Predicho',
            marker=dict(size=6, color='blue', symbol='x'),
            text=[f'Real: {r}, Pred: {p}, Prob: {prob:.3f}' 
                  for r, p, prob in zip(comparison_df['Real'], comparison_df['Predicho'], comparison_df['Probabilidad'])],
            hovertemplate='<b>%{text}</b><extra></extra>'
        ),
        row=3, col=2
    )
    
    # Actualizar layout
    fig.update_layout(
        title_text="Análisis Completo de Clasificación de Eventos Críticos - Visualización Interactiva",
        height=1600,
        width=1400,
        showlegend=True,
        template="plotly_white"
    )
    
    # 7. Gráfico de Edad (Duración) vs Mortalidad
    fig.add_trace(
        go.Scatter(
            x=df_temp['duracion_ciclo_dias'],
            y=df_temp['Mort. Total unidades'],
            mode='markers',
            name='Duración vs Mortalidad',
            marker=dict(size=6, color='red', opacity=0.6),
            text=[f'Duración: {dur:.0f} días<br>Mortalidad: {mort:,.0f}' 
                  for dur, mort in zip(df_temp['duracion_ciclo_dias'], df_temp['Mort. Total unidades'])],
            hovertemplate='<b>%{text}</b><extra></extra>'
        ),
        row=4, col=1
    )
    
    # Agregar línea de tendencia para duración vs mortalidad
    z = np.polyfit(df_temp['duracion_ciclo_dias'], df_temp['Mort. Total unidades'], 1)
    p = np.poly1d(z)
    fig.add_trace(
        go.Scatter(
            x=df_temp['duracion_ciclo_dias'],
            y=p(df_temp['duracion_ciclo_dias']),
            mode='lines',
            name='Tendencia',
            line=dict(color='blue', width=2, dash='dash'),
            showlegend=False
        ),
        row=4, col=1
    )
    
    # 8. Gráfico de Centro vs Mortalidad (Top 10 empresas)
    top_10_empresas = stats_por_empresa.head(10)
    
    fig.add_trace(
        go.Bar(
            x=top_10_empresas.index,
            y=top_10_empresas['Total_Mortalidad'],
            name='Mortalidad Total por Empresa',
            marker_color='orange',
            text=[f'{val:,.0f}' for val in top_10_empresas['Total_Mortalidad']],
            textposition='auto',
            hovertemplate='<b>%{x}</b><br>Mortalidad Total: %{y:,.0f}<extra></extra>'
        ),
        row=4, col=2
    )
    
    # Actualizar ejes
    # Fila 1
    fig.update_xaxes(title_text="False Positive Rate", row=1, col=1)
    fig.update_yaxes(title_text="True Positive Rate", row=1, col=1)
    fig.update_xaxes(title_text="Probabilidad Predicha", row=1, col=2)
    fig.update_yaxes(title_text="Frecuencia", row=1, col=2)
    
    # Fila 2
    fig.update_xaxes(title_text="", row=2, col=1)
    fig.update_yaxes(title_text="", row=2, col=1)
    fig.update_xaxes(title_text="Métricas", row=2, col=2)
    fig.update_yaxes(title_text="Valor", row=2, col=2)
    
    # Fila 3
    fig.update_xaxes(title_text="Coeficiente Absoluto", row=3, col=1)
    fig.update_yaxes(title_text="Features", row=3, col=1)
    fig.update_xaxes(title_text="Muestra", row=3, col=2)
    fig.update_yaxes(title_text="Evento Crítico (0=No, 1=Sí)", row=3, col=2)
    
    # Fila 4
    fig.update_xaxes(title_text="Duración del Ciclo (días)", row=4, col=1)
    fig.update_yaxes(title_text="Mortalidad Total (unidades)", row=4, col=1)
    fig.update_xaxes(title_text="Empresa", row=4, col=2)
    fig.update_yaxes(title_text="Mortalidad Total (unidades)", row=4, col=2)
    
    # Guardar como HTML
    ruta_html = f'{carpeta_salida}/evaluacion_interactiva.html'
    fig.write_html(ruta_html)
    
    print(f"Visualización interactiva completa guardada en: {ruta_html}")
    print("Gráficos incluidos:")
    print("   - Curva ROC con AUC")
    print("   - Distribución de probabilidades por clase")
    print("   - Matriz de confusión")
    print("   - Métricas de evaluación")
    print("   - Importancia de features")
    print("   - Comparación real vs predicho")
    print("   - Edad (Duración del ciclo) vs Mortalidad")
    print("   - Centro (Empresa) vs Mortalidad")
    
    return fig

def crear_grafico_edad_vs_mortalidad(df, carpeta_salida):
    """
    Crear gráfico de Edad (duración del ciclo) vs Mortalidad.
    
    Args:
        df (pd.DataFrame): Dataset con datos limpios
        carpeta_salida (str): Carpeta de salida
    """
    print("\n GRÁFICO: EDAD (DURACIÓN CICLO) VS MORTALIDAD")
    print("=" * 50)
    
    try:
        # Crear variable de "edad" basada en la duración del ciclo
        df_temp = df.copy()
        
        # Convertir fechas
        df_temp['Mes Inicio Ciclo'] = pd.to_datetime(df_temp['Mes Inicio Ciclo'], errors='coerce')
        df_temp['Mes Fin Ciclo'] = pd.to_datetime(df_temp['Mes Fin Ciclo'], errors='coerce')
        
        # Calcular duración del ciclo en días (proxy de "edad")
        df_temp['duracion_ciclo_dias'] = (df_temp['Mes Fin Ciclo'] - df_temp['Mes Inicio Ciclo']).dt.days
        
        # Filtrar datos válidos
        df_temp = df_temp.dropna(subset=['duracion_ciclo_dias', 'Mort. Total unidades'])
        
        print(f"Datos válidos para análisis: {len(df_temp)} registros")
        print(f"Duración promedio del ciclo: {df_temp['duracion_ciclo_dias'].mean():.1f} días")
        print(f"Rango de duración: {df_temp['duracion_ciclo_dias'].min():.0f} - {df_temp['duracion_ciclo_dias'].max():.0f} días")
        
        # Crear figura con subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Análisis de Edad (Duración del Ciclo) vs Mortalidad', fontsize=16, fontweight='bold')
        
        # 1. Scatter plot: Duración vs Mortalidad
        axes[0, 0].scatter(df_temp['duracion_ciclo_dias'], df_temp['Mort. Total unidades'], 
                           alpha=0.6, color='red', s=50)
        axes[0, 0].set_xlabel('Duración del Ciclo (días)')
        axes[0, 0].set_ylabel('Mortalidad Total (unidades)')
        axes[0, 0].set_title('Duración del Ciclo vs Mortalidad')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Agregar línea de tendencia
        z = np.polyfit(df_temp['duracion_ciclo_dias'], df_temp['Mort. Total unidades'], 1)
        p = np.poly1d(z)
        axes[0, 0].plot(df_temp['duracion_ciclo_dias'], p(df_temp['duracion_ciclo_dias']), 
                        "r--", alpha=0.8, linewidth=2)
        
        # 2. Boxplot por categorías de duración
        df_temp['categoria_duracion'] = pd.cut(df_temp['duracion_ciclo_dias'], 
                                              bins=5, labels=['Muy Corto', 'Corto', 'Medio', 'Largo', 'Muy Largo'])
        
        sns.boxplot(data=df_temp, x='categoria_duracion', y='Mort. Total unidades', ax=axes[0, 1])
        axes[0, 1].set_xlabel('Categoría de Duración del Ciclo')
        axes[0, 1].set_ylabel('Mortalidad Total (unidades)')
        axes[0, 1].set_title('Mortalidad por Categoría de Duración')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. Histograma de duración del ciclo
        axes[1, 0].hist(df_temp['duracion_ciclo_dias'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        axes[1, 0].set_xlabel('Duración del Ciclo (días)')
        axes[1, 0].set_ylabel('Frecuencia')
        axes[1, 0].set_title('Distribución de Duración del Ciclo')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Correlación
        correlacion = df_temp['duracion_ciclo_dias'].corr(df_temp['Mort. Total unidades'])
        axes[1, 1].text(0.5, 0.5, f'Correlación: {correlacion:.3f}', 
                        ha='center', va='center', transform=axes[1, 1].transAxes,
                        fontsize=14, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
        axes[1, 1].set_title('Coeficiente de Correlación')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        # Guardar gráfico
        ruta_grafico = f'{carpeta_salida}/grafico_edad_vs_mortalidad.png'
        plt.savefig(ruta_grafico, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f" Gráfico guardado en: {ruta_grafico}")
        print(f" Correlación entre duración y mortalidad: {correlacion:.3f}")
        
        return df_temp
        
    except Exception as e:
        print(f" Error al crear gráfico de edad vs mortalidad: {e}")
        return None

def crear_grafico_centro_vs_mortalidad(df, carpeta_salida):
    """
    Crear gráfico de Centro vs Mortalidad.
    
    Args:
        df (pd.DataFrame): Dataset con datos limpios
        carpeta_salida (str): Carpeta de salida
    """
    print("\n GRÁFICO: CENTRO VS MORTALIDAD")
    print("=" * 50)
    
    try:
        # Preparar datos
        df_temp = df.copy()
        
        # Agrupar por empresa (centro) y calcular estadísticas
        stats_por_empresa = df_temp.groupby('Empresa')['Mort. Total unidades'].agg([
            'count', 'mean', 'std', 'min', 'max', 'sum'
        ]).round(2)
        
        stats_por_empresa.columns = ['Cantidad_Ciclos', 'Promedio_Mortalidad', 'Desv_Estandar', 
                                   'Min_Mortalidad', 'Max_Mortalidad', 'Total_Mortalidad']
        stats_por_empresa = stats_por_empresa.sort_values('Total_Mortalidad', ascending=False)
        
        print(f"Empresas analizadas: {len(stats_por_empresa)}")
        print(f"Total de ciclos: {stats_por_empresa['Cantidad_Ciclos'].sum()}")
        
        # Crear figura con subplots
        fig, axes = plt.subplots(2, 2, figsize=(18, 12))
        fig.suptitle('Análisis de Centros (Empresas) vs Mortalidad', fontsize=16, fontweight='bold')
        
        # 1. Top 10 empresas por mortalidad total
        top_10_empresas = stats_por_empresa.head(10)
        
        bars = axes[0, 0].bar(range(len(top_10_empresas)), top_10_empresas['Total_Mortalidad'], 
                              color='red', alpha=0.7)
        axes[0, 0].set_xlabel('Empresa')
        axes[0, 0].set_ylabel('Mortalidad Total (unidades)')
        axes[0, 0].set_title('Top 10 Empresas por Mortalidad Total')
        axes[0, 0].set_xticks(range(len(top_10_empresas)))
        axes[0, 0].set_xticklabels([emp[:20] + '...' if len(emp) > 20 else emp 
                                    for emp in top_10_empresas.index], rotation=45, ha='right')
        
        # Agregar valores en las barras
        for i, bar in enumerate(bars):
            height = bar.get_height()
            axes[0, 0].text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:,.0f}', ha='center', va='bottom', fontsize=8)
        
        # 2. Promedio de mortalidad por empresa
        top_10_promedio = stats_por_empresa.nlargest(10, 'Promedio_Mortalidad')
        
        bars2 = axes[0, 1].bar(range(len(top_10_promedio)), top_10_promedio['Promedio_Mortalidad'], 
                               color='orange', alpha=0.7)
        axes[0, 1].set_xlabel('Empresa')
        axes[0, 1].set_ylabel('Mortalidad Promedio (unidades)')
        axes[0, 1].set_title('Top 10 Empresas por Mortalidad Promedio')
        axes[0, 1].set_xticks(range(len(top_10_promedio)))
        axes[0, 1].set_xticklabels([emp[:20] + '...' if len(emp) > 20 else emp 
                                   for emp in top_10_promedio.index], rotation=45, ha='right')
        
        # Agregar valores en las barras
        for i, bar in enumerate(bars2):
            height = bar.get_height()
            axes[0, 1].text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:,.0f}', ha='center', va='bottom', fontsize=8)
        
        # 3. Boxplot de mortalidad por empresa (top 8)
        top_8_empresas = stats_por_empresa.head(8).index
        df_top8 = df_temp[df_temp['Empresa'].isin(top_8_empresas)]
        
        sns.boxplot(data=df_top8, x='Empresa', y='Mort. Total unidades', ax=axes[1, 0])
        axes[1, 0].set_xlabel('Empresa')
        axes[1, 0].set_ylabel('Mortalidad Total (unidades)')
        axes[1, 0].set_title('Distribución de Mortalidad por Empresa (Top 8)')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 4. Scatter plot: Cantidad de ciclos vs Mortalidad promedio
        axes[1, 1].scatter(stats_por_empresa['Cantidad_Ciclos'], 
                           stats_por_empresa['Promedio_Mortalidad'], 
                           alpha=0.6, s=50, color='green')
        axes[1, 1].set_xlabel('Cantidad de Ciclos')
        axes[1, 1].set_ylabel('Mortalidad Promedio (unidades)')
        axes[1, 1].set_title('Cantidad de Ciclos vs Mortalidad Promedio')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Agregar línea de tendencia
        z = np.polyfit(stats_por_empresa['Cantidad_Ciclos'], stats_por_empresa['Promedio_Mortalidad'], 1)
        p = np.poly1d(z)
        axes[1, 1].plot(stats_por_empresa['Cantidad_Ciclos'], 
                        p(stats_por_empresa['Cantidad_Ciclos']), 
                        "g--", alpha=0.8, linewidth=2)
        
        plt.tight_layout()
        
        # Guardar gráfico
        ruta_grafico = f'{carpeta_salida}/grafico_centro_vs_mortalidad.png'
        plt.savefig(ruta_grafico, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Guardar estadísticas a CSV
        ruta_stats = f'{carpeta_salida}/estadisticas_por_empresa.csv'
        stats_por_empresa.to_csv(ruta_stats, encoding='utf-8')
        
        print(f" Gráfico guardado en: {ruta_grafico}")
        print(f" Estadísticas por empresa guardadas en: {ruta_stats}")
        print(f" Empresa con mayor mortalidad total: {stats_por_empresa.index[0]}")
        print(f" Empresa con mayor mortalidad promedio: {stats_por_empresa.nlargest(1, 'Promedio_Mortalidad').index[0]}")
        
        return stats_por_empresa
        
    except Exception as e:
        print(f" Error al crear gráfico de centro vs mortalidad: {e}")
        return None

def main():
    """
    Función principal que ejecuta el análisis completo con técnicas avanzadas.
    """
    print("=" * 60)
    print("RETO 3: CLASIFICACIÓN AVANZADA - EVENTOS CRÍTICOS DE MORTALIDAD")
    print("=" * 60)
    print(f"Fecha de ejecución: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # Configurar carpeta de salida
    carpeta_salida = "salida"
    if not os.path.exists(carpeta_salida):
        os.makedirs(carpeta_salida)
    
    # 1. Cargar datos
    print("\n1.  CARGA Y LIMPIEZA ROBUSTA DE DATOS")
    df = cargar_datos('entrada/Mortalidades_Centro de cultivos.csv')
    if df is None:
        return
    
    # 2. Limpieza robusta
    df_clean, outliers = limpiar_datos_robusta(df)
    
    # 3. Codificación de variables categóricas
    print("\n2.  CODIFICACIÓN DE VARIABLES CATEGÓRICAS")
    df_encoded = codificar_variables_categoricas(df_clean)
    
    # 4. Crear variable objetivo
    print("\n3.  CREAR VARIABLE OBJETIVO")
    Q90 = df_encoded['Mort. Total unidades'].quantile(0.90)
    df_encoded['evento_critico'] = (df_encoded['Mort. Total unidades'] > Q90).astype(int)
    
    print(f"Percentil 90 de mortalidades: {Q90:.2f}")
    print(f"Eventos críticos: {df_encoded['evento_critico'].sum()} ({df_encoded['evento_critico'].mean()*100:.2f}%)")
    
    # 5. Preparar datos para ML
    print("\n4.  PREPARAR DATOS PARA MACHINE LEARNING")
    exclude_cols = ['evento_critico', 'Mort. Total unidades']
    feature_cols = [col for col in df_encoded.columns if col not in exclude_cols]
    
    X = df_encoded[feature_cols]
    y = df_encoded['evento_critico']
    
    print(f"Features (X): {X.shape}")
    print(f"Target (y): {y.shape}")
    print(f"Columnas de features: {list(X.columns)}")
    
    # 6. Separar datos
    print("\n5.  SEPARAR DATOS DE ENTRENAMIENTO Y PRUEBA")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"Entrenamiento: {X_train.shape[0]} muestras")
    print(f"Prueba: {X_test.shape[0]} muestras")
    print(f"Proporción de eventos críticos en entrenamiento: {y_train.mean():.3f}")
    print(f"Proporción de eventos críticos en prueba: {y_test.mean():.3f}")
    
    # 7. Escalar datos
    print("\n6.  ESCALADO DE DATOS CON STANDARDSCALER")
    X_train_scaled, X_test_scaled, scaler = escalar_datos(X_train, X_test)
    
    # 8. Balancear clases
    print("\n7.  BALANCEO DE CLASES CON SMOTE")
    X_train_balanced, y_train_balanced = balancear_clases(X_train_scaled, y_train)
    
    # 9. Entrenar modelo
    print("\n8.  ENTRENAMIENTO DE MODELO LOGISTICREGRESSION")
    model = entrenar_modelo_logistic(X_train_balanced, y_train_balanced)
    
    # 10. Realizar predicciones
    print("\n9.  REALIZAR PREDICCIONES")
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    print(f"Predicciones realizadas: {len(y_pred)}")
    print(f"Predicciones positivas: {y_pred.sum()}")
    
    # 11. Evaluación completa
    print("\n10.  EVALUACIÓN COMPLETA")
    metrics = evaluar_modelo_completo(y_test, y_pred, y_pred_proba, model, X_test_scaled, X.columns)
    
    # 12. Exportar coeficientes
    print("\n11.  EXPORTAR COEFICIENTES")
    coeficientes_df = exportar_coeficientes(model, X.columns, carpeta_salida)
    
    # 13. Crear gráficos de análisis exploratorio
    print("\n12.  CREAR GRÁFICOS DE ANÁLISIS EXPLORATORIO")
    crear_grafico_edad_vs_mortalidad(df_clean, carpeta_salida)
    crear_grafico_centro_vs_mortalidad(df_clean, carpeta_salida)
    
    # 14. Crear visualizaciones interactivas
    print("\n13.  CREAR VISUALIZACIONES INTERACTIVAS")
    fig = crear_curva_roc_interactiva(metrics, y_test, y_pred, y_pred_proba, coeficientes_df, df_clean, carpeta_salida)
    
    # RESUMEN FINAL
    print("\n" + "=" * 60)
    print("RESUMEN DEL ANÁLISIS AVANZADO - TODAS LAS TÉCNICAS IMPLEMENTADAS")
    print("=" * 60)
    print(f" Dataset: {df_clean.shape[0]} filas, {df_clean.shape[1]} columnas")
    print(f" Eventos críticos: {df_encoded['evento_critico'].sum()} ({df_encoded['evento_critico'].mean()*100:.2f}%)")
    print(f" Accuracy del modelo: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f" Outliers detectados: {len(outliers)}")
    print(f" Archivos generados en: {carpeta_salida}/")
    print("=" * 60)
    
    print("\n ¡ANÁLISIS AVANZADO COMPLETADO EXITOSAMENTE!")
    print(" Todas las técnicas avanzadas han sido implementadas:")
    print("   - Carga y limpieza robusta de datos")
    print("   - Detección de outliers")
    print("   - Codificación de variables categóricas")
    print("   - Escalado de datos con StandardScaler")
    print("   - Balanceo de clases con SMOTE")
    print("   - Entrenamiento de modelo con LogisticRegression")
    print("   - Evaluación completa con múltiples métricas")
    print("   - Exportación de coeficientes a CSV")
    print("   - Visualización interactiva completa con todos los gráficos (.html)")
    print("   - Gráficos de análisis exploratorio integrados")

if __name__ == "__main__":
    main() 
# -*- coding: utf-8 -*-
"""
Reto 1 – Clasificación de Especies de Pingüinos (con exportación de resultados)
-------------------------------------------------------------------------------
Dataset: https://www.kaggle.com/datasets/parulpandey/palmer-archipelago-antarctica-penguin-data
Archivo esperado: penguins_size.csv  (colócalo en el mismo directorio de este script)

Qué hace este script:
1) Carga y prepara datos (nulos, tipos, codificación categórica).
2) Separa en train/test con semilla reproducible.
3) Entrena 4 modelos: Regresión Logística, Árbol, Random Forest y SVM (kernel lineal).
4) Evalúa con Accuracy, F1 macro, reportes y matrices de confusión.
5) Genera y GUARDA:
   - CSV con métricas por modelo
   - CSV con reportes de clasificación por modelo
   - CSV con matrices de confusión por modelo
   - CSV con predicciones por modelo (y_test vs y_pred)
   - PNG de matrices de confusión (todas en grilla)
   - PNG de comparativas (Accuracy y F1-Macro)
   - README.txt con un resumen del experimento

Requisitos:
- pandas, numpy, scikit-learn, matplotlib, seaborn
Instalación (opcional):
pip install pandas numpy scikit-learn matplotlib seaborn
"""

import os
import sys
import json
import time
from datetime import datetime
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# ---------------------------------------------------------------------
# 0) Configuración general y utilidades de guardado
# ---------------------------------------------------------------------
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
sns.set(style="whitegrid")
plt.rcParams["figure.dpi"] = 110

CSV_NAME = "penguins_size.csv"  # nombre del archivo descargado de Kaggle

STAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
BASE_OUT = os.path.join("salidas")
RUN_DIR = os.path.join(BASE_OUT, f"ejecucion_{STAMP}")
FIG_DIR = os.path.join(RUN_DIR, "figs")
TAB_DIR = os.path.join(RUN_DIR, "tablas")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TAB_DIR, exist_ok=True)

def save_fig(path):
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()

def save_df(df: pd.DataFrame, path: str):
    df.to_csv(path, index=False, encoding="utf-8-sig")

def write_text(path: str, text: str):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)

# ---------------------------------------------------------------------
# 1) Carga de datos
# ---------------------------------------------------------------------
if not os.path.exists(CSV_NAME):
    sys.exit(
        f"ERROR: No se encontró '{CSV_NAME}'. "
        "Descarga el CSV desde Kaggle y colócalo en este mismo directorio."
    )

df = pd.read_csv(CSV_NAME)
df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

cols_requeridas = {
    "species", "island", "culmen_length_mm", "culmen_depth_mm",
    "flipper_length_mm", "body_mass_g", "sex"
}
faltantes = cols_requeridas - set(df.columns)
if faltantes:
    sys.exit(
        "ERROR: Faltan columnas en el CSV para este script.\n"
        f"Faltantes: {faltantes}\n"
        "Verifica que descargaste 'penguins_size.csv' correcto."
    )

# ---------------------------------------------------------------------
# 2) Limpieza y preparación
# ---------------------------------------------------------------------
df["sex"] = df["sex"].replace({".": np.nan, "": np.nan})
num_cols = ["culmen_length_mm", "culmen_depth_mm", "flipper_length_mm", "body_mass_g"]
for c in num_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")

df = df.dropna(subset=["species"])
df = df.dropna(subset=num_cols, how="any")
df = df[df["species"].isin(["Adelie", "Chinstrap", "Gentoo"])]

cat_cols = ["island", "sex"]
X = df[cat_cols + num_cols].copy()
y = df["species"].copy()

# ---------------------------------------------------------------------
# 3) Split Train/Test
# ---------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=RANDOM_STATE, stratify=y
)

# ---------------------------------------------------------------------
# 4) Preprocesamiento
# ---------------------------------------------------------------------
preprocess_scaled = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ("num", StandardScaler(), num_cols),
    ],
    remainder="drop",
    verbose_feature_names_out=False,
)

preprocess_tree = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ("num", "passthrough", num_cols),
    ],
    remainder="drop",
    verbose_feature_names_out=False,
)

# ---------------------------------------------------------------------
# 5) Modelos (Pipelines)
# ---------------------------------------------------------------------
models = {
    "LogisticRegression": Pipeline(steps=[
        ("prep", preprocess_scaled),
        ("clf", LogisticRegression(max_iter=2000, multi_class="auto", random_state=RANDOM_STATE))
    ]),
    "DecisionTree": Pipeline(steps=[
        ("prep", preprocess_tree),
        ("clf", DecisionTreeClassifier(random_state=RANDOM_STATE))
    ]),
    "RandomForest": Pipeline(steps=[
        ("prep", preprocess_tree),
        ("clf", RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1))
    ]),
    "SVM_linear": Pipeline(steps=[
        ("prep", preprocess_scaled),
        ("clf", SVC(kernel="linear", probability=True, random_state=RANDOM_STATE))
    ]),
}

# ---------------------------------------------------------------------
# 6) Entrenamiento y evaluación
# ---------------------------------------------------------------------
def evaluar_modelo(y_true, y_pred) -> Tuple[float, float, dict, np.ndarray]:
    acc = accuracy_score(y_true, y_pred)
    f1m = f1_score(y_true, y_pred, average="macro")
    report_dict = classification_report(y_true, y_pred, digits=4, output_dict=True)
    cm = confusion_matrix(y_true, y_pred, labels=sorted(y_true.unique()))
    return acc, f1m, report_dict, cm

resultados: Dict[str, Dict] = {}
t0 = time.time()

for nombre, pipe in models.items():
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    acc, f1m, report_dict, cm = evaluar_modelo(y_test, preds)

    resultados[nombre] = {
        "modelo": pipe,
        "accuracy": acc,
        "f1_macro": f1m,
        "reporte_dict": report_dict,
        "cm": cm,
        "preds": preds
    }

elapsed = time.time() - t0

# ---------------------------------------------------------------------
# 7) Exportar TABLAS (CSV)
# ---------------------------------------------------------------------
# 7.1 Métricas por modelo
metricas_df = pd.DataFrame([
    {"modelo": m, "accuracy": res["accuracy"], "f1_macro": res["f1_macro"]}
    for m, res in resultados.items()
]).sort_values(by="accuracy", ascending=False)
save_df(metricas_df, os.path.join(TAB_DIR, "metricas_por_modelo.csv"))

# 7.2 Reportes de clasificación por modelo
for m, res in resultados.items():
    rep_dict = res["reporte_dict"]
    # Convertir a DataFrame (clases + promedios)
    rep_df = pd.DataFrame(rep_dict).T.reset_index().rename(columns={"index": "clase_o_promedio"})
    save_df(rep_df, os.path.join(TAB_DIR, f"reporte_clasificacion_{m}.csv"))

# 7.3 Matrices de confusión por modelo
labels_sorted = sorted(y_test.unique())
for m, res in resultados.items():
    cm = res["cm"]
    cm_df = pd.DataFrame(cm, index=[f"real_{l}" for l in labels_sorted],
                            columns=[f"pred_{l}" for l in labels_sorted])
    save_df(cm_df.reset_index(), os.path.join(TAB_DIR, f"matriz_confusion_{m}.csv"))

# 7.4 Predicciones (y_test vs y_pred) por modelo
y_test_df = pd.DataFrame({"y_real": y_test.values})
for m, res in resultados.items():
    pred_df = y_test_df.copy()
    pred_df["y_pred"] = res["preds"]
    save_df(pred_df, os.path.join(TAB_DIR, f"predicciones_{m}.csv"))

# ---------------------------------------------------------------------
# 8) Exportar GRÁFICAS (PNG)
# ---------------------------------------------------------------------
# 8.1 Matrices de confusión en grilla
def plot_confusion_matrices_grid(resultados: Dict[str, Dict], labels, save_path: str):
    n = len(resultados)
    cols = 2
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(10, 4 * rows))
    axes = np.array(axes).reshape(rows, cols)

    # Limpia ejes por defecto
    for ax in axes.flat:
        ax.axis("off")

    i = 0
    for nombre, res in resultados.items():
        r = i // cols
        c = i % cols
        ax = axes[r, c]
        cm = res["cm"]
        sns.heatmap(cm, annot=True, fmt="d", cbar=False, ax=ax,
                    xticklabels=labels, yticklabels=labels)
        ax.set_title(f"Matriz de confusión – {nombre}")
        ax.set_xlabel("Predicción")
        ax.set_ylabel("Real")
        ax.axis("on")
        i += 1

    save_fig(save_path)

plot_confusion_matrices_grid(
    resultados, labels_sorted,
    os.path.join(FIG_DIR, "matrices_confusion_grid.png")
)

# 8.2 Comparativas de Accuracy y F1
def plot_bar_metric(metric_values, metric_name: str, save_path: str):
    modelos = list(metric_values.keys())
    valores = [metric_values[m] for m in modelos]
    plt.figure(figsize=(8, 4))
    sns.barplot(x=modelos, y=valores, errorbar=None)
    plt.title(f"Comparativa de {metric_name} por modelo")
    plt.ylabel(metric_name)
    plt.ylim(0, 1)
    for i, v in enumerate(valores):
        plt.text(i, min(v + 0.02, 0.99), f"{v:.3f}", ha="center", va="bottom")
    save_fig(save_path)

plot_bar_metric(
    {m: res["accuracy"] for m, res in resultados.items()},
    "Accuracy",
    os.path.join(FIG_DIR, "comparativa_accuracy.png")
)

plot_bar_metric(
    {m: res["f1_macro"] for m, res in resultados.items()},
    "F1-Macro",
    os.path.join(FIG_DIR, "comparativa_f1_macro.png")
)

# ---------------------------------------------------------------------
# 9) Resumen del experimento (README)
# ---------------------------------------------------------------------
summary = {
    "timestamp": STAMP,
    "dataset_csv": CSV_NAME,
    "total_registros": int(len(df)),
    "train_size": int(len(X_train)),
    "test_size": int(len(X_test)),
    "target_clases": sorted(df["species"].unique().tolist()),
    "features": cat_cols + num_cols,
    "modelos": list(models.keys()),
    "mejor_modelo_por_accuracy": metricas_df.iloc[0]["modelo"],
    "accuracy_mejor_modelo": float(metricas_df.iloc[0]["accuracy"]),
    "f1macro_mejor_modelo": float(metricas_df.iloc[0]["f1_macro"]),
    "tiempo_total_seg": round(elapsed, 3)
}
write_text(os.path.join(RUN_DIR, "README.txt"),
           "Resumen de la ejecución\n"
           "========================\n\n" +
           json.dumps(summary, indent=2, ensure_ascii=False))

# (Extra) también guardamos el JSON crudo por si quieres leerlo desde otro script
with open(os.path.join(RUN_DIR, "resumen.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

# ---------------------------------------------------------------------
# 10) Mensaje final
# ---------------------------------------------------------------------
print("\n================= RESULTADOS =================")
print(metricas_df.to_string(index=False))
print(f"\n✅ Archivos guardados en: {RUN_DIR}")
print("   - tablas/*.csv (métricas, reportes, matrices, predicciones)")
print("   - figs/*.png (matrices de confusión y comparativas)")
print("   - README.txt y resumen.json")

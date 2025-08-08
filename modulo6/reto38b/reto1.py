# reto1.py
# -*- coding: utf-8 -*-

import os
import base64
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import (
    LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler,
    RobustScaler, PowerTransformer, Normalizer
)
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# =========================
# 0) RUTAS (input/output)
# =========================
INPUT_PATH = os.path.join("input", "Cosechas_Cosechas_2023b.csv")
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.rcParams["figure.dpi"] = 160  # nitidez para PNG

# ==================================
# 1) Lectura robusta del CSV
# (soporta cabecera extra en fila 1)
# ==================================
def read_csv_robust(path):
    try:
        df_try = pd.read_csv(path, header=1)
        if "Toneladas Cosechadas" in df_try.columns or any("tonel" in c.lower() for c in df_try.columns):
            return df_try
    except Exception:
        pass
    return pd.read_csv(path, header=0)

if not os.path.exists(INPUT_PATH):
    raise SystemExit(f"No se encontró el archivo en {INPUT_PATH}. Verifica la ruta.")

df = read_csv_robust(INPUT_PATH)
# elimina columnas índice basura
df = df.drop(columns=[c for c in df.columns if str(c).startswith("Unnamed")], errors="ignore")

# ==================================
# 2) Exploración (head/info/describe/nulos)
# ==================================
print("Primeras filas (head):")
print(df.head())
print("\nInfo:")
print(df.info())
print("\nDescribe:")
print(df.describe(include="all"))
print("\nNulos por columna:")
print(df.isnull().sum())

# ==================================
# 3) Mapeo flexible de columnas
# ==================================
def find_col(candidates, cols):
    for c in candidates:
        if c in cols:
            return c
    return None

cols = df.columns.tolist()
col_ton     = find_col(["Toneladas Cosechadas", "Toneladas", "toneladas", "Toneladas_Cosechadas"], cols)
col_empresa = find_col(["Empresa", "empresa"], cols)
col_especie = find_col(["Especie", "especie"], cols)
col_zona    = find_col(["Zona", "zona", "Región", "Region"], cols)
col_periodo = find_col(["Periodo Información", "Periodo", "Periodo_Informacion"], cols)
col_inicio  = find_col(["Mes Inicio Ciclo", "Inicio Ciclo", "Fecha Inicio"], cols)
col_fin     = find_col(["Mes Fin Ciclo", "Fin Ciclo", "Fecha Fin"], cols)
col_centro  = find_col(["Código Centro", "Codigo Centro", "Código", "Codigo"], cols)

if col_ton is None:
    raise SystemExit(f"No se encontró la columna objetivo de toneladas. Columnas: {cols}")

# ==================================
# 4) Limpieza y conversión de tipos
# ==================================
# Normaliza strings de categóricas (espacios duros, trims)
for c in [col_empresa, col_especie, col_zona, col_periodo, col_centro]:
    if c is not None:
        df[c] = (df[c].astype(str)
                        .str.replace("\u00a0", " ", regex=False)  # NBSP
                        .str.strip())

# Toneladas → numérico (maneja miles y coma decimal)
df[col_ton] = (
    df[col_ton].astype(str)
      .str.replace(".", "", regex=False)  # quita miles
      .str.replace(",", ".", regex=False) # coma → punto
)
df[col_ton] = pd.to_numeric(df[col_ton], errors="coerce")

# Fechas (si existen)
for c in [col_inicio, col_fin]:
    if c is not None:
        df[c] = pd.to_datetime(df[c], errors="coerce")

# Ingeniería: duración del ciclo en días
if col_inicio is not None and col_fin is not None:
    df["ciclo_dias"] = (df[col_fin] - df[col_inicio]).dt.days
else:
    df["ciclo_dias"] = np.nan

# ==================================
# 5) Tratamiento de faltantes (pre-codificación)
# - Numéricos → mediana
# - Categóricos → modo o "Desconocido"
# ==================================
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = [c for c in [col_empresa, col_especie, col_zona, col_periodo, col_centro] if c is not None]

# imputación numérica
for c in num_cols:
    if df[c].isna().any():
        df[c].fillna(df[c].median(), inplace=True)

# imputación categórica
for c in cat_cols:
    if df[c].isna().any():
        moda = df[c].mode(dropna=True)
        fill_val = "Desconocido" if moda.empty else moda.iloc[0]
        df[c].fillna(fill_val, inplace=True)

# asegurar y sin NaN
if df[col_ton].isna().any():
    df[col_ton] = df[col_ton].fillna(df[col_ton].median())

print("\nNulos tras imputación base:")
print(df.isnull().sum())

# ==================================
# 6) Codificación categórica
# - LabelEncoder: Empresa, Especie, Código Centro
# - OneHotEncoder: Zona, Periodo
# ==================================
df_enc = df.copy()

for c in [col_empresa, col_especie, col_centro]:
    if c is not None:
        le = LabelEncoder()
        df_enc[c + "_LE"] = le.fit_transform(df_enc[c].astype(str))

ohe_df = pd.DataFrame(index=df_enc.index)
onehot_cols = [c for c in [col_zona, col_periodo] if c is not None]
if onehot_cols:
    ct = ColumnTransformer(
        transformers=[("ohe", OneHotEncoder(handle_unknown="ignore"), onehot_cols)],
        remainder="drop"
    )
    ohe_matrix = ct.fit_transform(df_enc)
    ohe_feature_names = ct.named_transformers_["ohe"].get_feature_names_out(onehot_cols)
    ohe_df = pd.DataFrame(ohe_matrix.toarray(), columns=ohe_feature_names, index=df_enc.index)
    df_enc = pd.concat([df_enc, ohe_df], axis=1)

# ==================================
# 7) Escalamiento/transformaciones de la Y (target)
# ==================================
y_raw = df_enc[col_ton].values.reshape(-1, 1)

std_scaler    = StandardScaler()
minmax_scaler = MinMaxScaler()
robust_scaler = RobustScaler()

df_enc["Ton_scaled_standard"] = std_scaler.fit_transform(y_raw)
df_enc["Ton_scaled_minmax"]   = minmax_scaler.fit_transform(y_raw)
df_enc["Ton_scaled_robust"]   = robust_scaler.fit_transform(y_raw)

df_enc["Ton_log1p"]     = np.log1p(df_enc[col_ton].clip(lower=0))
df_enc["Ton_sqrt"]      = np.sqrt(df_enc[col_ton].clip(lower=0))
pt = PowerTransformer(method="yeo-johnson", standardize=False)  # Yeo-Johnson (apto para 0/negativos)
df_enc["Ton_yeojohnson"] = pt.fit_transform(y_raw).ravel()

normalizer = Normalizer(norm="l2")
df_enc["Ton_norm_l2"] = normalizer.fit_transform(y_raw).ravel()

# ==================================
# 8) Ensamble de features (X) y target (y)
# ==================================
feature_cols = []

for c in [col_empresa, col_especie, col_centro]:
    if c is not None and (c + "_LE") in df_enc.columns:
        feature_cols.append(c + "_LE")

if not ohe_df.empty:
    feature_cols.extend(ohe_df.columns.tolist())

# num engineered
if "ciclo_dias" in df_enc.columns:
    feature_cols.append("ciclo_dias")

# fallback si quedara vacío
if not feature_cols:
    df_enc["_row_idx"] = np.arange(len(df_enc))
    feature_cols = ["_row_idx"]

X = df_enc[feature_cols].copy()
y = df_enc[col_ton].copy()

# Dump de features y target para depurar
pd.concat([X, y.rename(col_ton)], axis=1).to_csv(os.path.join(OUTPUT_DIR, "Xy_modelado.csv"), index=False)

# ==================================
# 9) Split + Imputación post-split (evita leakage)
# ==================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

imputer = SimpleImputer(strategy="median")
imputer.fit(X_train)
X_train = pd.DataFrame(imputer.transform(X_train), columns=X.columns, index=X_train.index)
X_test  = pd.DataFrame(imputer.transform(X_test),  columns=X.columns, index=X_test.index)

if y_train.isna().any():
    y_train = y_train.fillna(y_train.median())
if y_test.isna().any():
    y_test = y_test.fillna(y_train.median())

# ==================================
# 10) Modelado (3 regresores)
# ==================================
models = {
    "LinearRegression": LinearRegression(),
    "RandomForestRegressor": RandomForestRegressor(n_estimators=300, random_state=42),
    "GradientBoostingRegressor": GradientBoostingRegressor(random_state=42),
}

results = []
fitted = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    fitted[name] = model
    y_pred = model.predict(X_test)
    results.append({
        "Modelo": name,
        "MSE": mean_squared_error(y_test, y_pred),
        "R2":  r2_score(y_test, y_pred),
        "MAE": mean_absolute_error(y_test, y_pred),
    })

results_df = pd.DataFrame(results).sort_values("MSE", ascending=True).reset_index(drop=True)
best_model_name = results_df.loc[0, "Modelo"]
best_mse = results_df.loc[0, "MSE"]
best_r2 = results_df.loc[0, "R2"]
best_mae = results_df.loc[0, "MAE"]

print("\nResultados de modelos:")
print(results_df)

# Guarda métricas
metrics_csv_path = os.path.join(OUTPUT_DIR, "metricas_modelos.csv")
results_df.to_csv(metrics_csv_path, index=False)

# Guardar mejor modelo (si joblib disponible)
best_est = fitted[best_model_name]
model_path = None
try:
    import joblib
    model_path = os.path.join(OUTPUT_DIR, f"mejor_modelo_{best_model_name}.joblib")
    joblib.dump(best_est, model_path)
except Exception as e:
    print("Aviso: no se pudo guardar el modelo (instala joblib si falta).", e)

# Importancia de features (si aplica)
feature_importance = None
if hasattr(best_est, "feature_importances_"):
    feature_importance = pd.DataFrame({
        "feature": X.columns,
        "importance": best_est.feature_importances_
    }).sort_values("importance", ascending=False)
    fi_html = feature_importance.head(20).to_html(index=False, float_format="%.6f",
                                                  classes="table table-striped table-sm")
else:
    fi_html = "<p class='text-secondary'>No disponible para este modelo.</p>"

# ==================================
# 11) Visualización con seaborn.kdeplot (filtrando varianza cero)
# ==================================
sns.set(style="whitegrid")
dist_cols = ["Ton_scaled_standard", "Ton_scaled_minmax", "Ton_scaled_robust",
             "Ton_log1p", "Ton_sqrt", "Ton_yeojohnson", "Ton_norm_l2"]
available = [c for c in dist_cols if c in df_enc.columns]

# Muestreo opcional para evitar cuelgues en KDE con datasets gigantes
max_kde = 50000
df_kde = df_enc.sample(max_kde, random_state=42) if len(df_enc) > max_kde else df_enc

fig, ax = plt.subplots(figsize=(10, 6))
for c in available:
    s = df_kde[c].dropna()
    if s.nunique() > 1:  # evita warning de varianza cero
        sns.kdeplot(s, label=c, ax=ax, linewidth=2.0, warn_singular=False)
ax.set_title("Distribuciones KDE de Toneladas transformadas/escaladas")
ax.set_xlabel("Valor")
ax.set_ylabel("Densidad")
ax.legend(loc="best")
plt.tight_layout()
plot_path = os.path.join(OUTPUT_DIR, "distribuciones_toneladas.png")
plt.savefig(plot_path, bbox_inches="tight")
plt.close(fig)

with open(plot_path, "rb") as f:
    plot_b64 = base64.b64encode(f.read()).decode("utf-8")

# ==================================
# 12) Tabla resumen (media/std) por técnica
# ==================================
summary_rows = []
for c in available:
    s = df_enc[c].dropna()
    if len(s) > 0:
        summary_rows.append({"Variable": c, "Media": float(s.mean()), "DesvEst": float(s.std())})
summary_df = pd.DataFrame(summary_rows)
summary_csv_path = os.path.join(OUTPUT_DIR, "resumen_transformaciones.csv")
summary_df.to_csv(summary_csv_path, index=False)

# ==================================
# 13) Dashboard HTML interactivo (Bootstrap + Tabs + Búsqueda + Plotly)
#     - Mantiene el PNG de seaborn por requisito
#     - Agrega un gráfico Plotly (JS via CDN) con histogramas densidad
# ==================================
results_table_html = results_df.to_html(index=False, float_format="%.4f", classes="table table-striped table-sm")
summary_table_html = summary_df.to_html(index=False, float_format="%.6f", classes="table table-striped table-sm")

def interpretation_text():
    return (
        f"<p><b>Mejor modelo:</b> {best_model_name} "
        f"(MSE={best_mse:.4f}, R²={best_r2:.4f}, MAE={best_mae:.4f}).</p>"
        "<p><b>Transformaciones:</b> <i>log1p</i> y <i>Yeo-Johnson</i> tienden a reducir asimetrías; "
        "<i>sqrt</i> suaviza valores altos. <b>Escaladores:</b> <i>Standard</i> centra/escala, "
        "<i>MinMax</i> lleva a [0,1] y <i>Robust</i> resiste outliers.</p>"
        "<p><b>Codificación:</b> LabelEncoder en Empresa/Especie/Centro y One-Hot en Zona/Período permite a los modelos "
        "capturar diferencias por categoría; los árboles suelen rendir bien con interacciones no lineales.</p>"
    )

# Datos para Plotly: arrays por cada columna disponible (downsample opcional)
plotly_series = {}
for c in available:
    s = df_enc[c].dropna()
    if s.nunique() > 1 and len(s) > 0:
        if len(s) > 30000:
            s = s.sample(30000, random_state=42)
        plotly_series[c] = s.tolist()

plotly_json = json.dumps(plotly_series)  # se inyecta al JS

dashboard_html = f"""
<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="utf-8" />
<title>Dashboard – Reto 1: Toneladas Cosechadas</title>
<meta name="viewport" content="width=device-width, initial-scale=1" />
<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
<script src="https://cdn.plot.ly/plotly-2.30.0.min.js"></script>
<style>
  :root {{
    --bg: #0b0f1a;
    --grad1: #0b0f1a;
    --grad2: #0d1b2a;
    --fg: #f8fafc;
    --muted: #c7d1db;
    --card: rgba(255,255,255,0.06);
    --border: rgba(255,255,255,0.14);
    --accent: #00e5ff;   /* cian vibrante */
    --accent2: #ff4ecd;  /* magenta neón */
    --accent3: #ffd166;  /* amarillo cálido */
    --shadow: 0 10px 30px rgba(0,0,0,.35), inset 0 0 0 1px rgba(255,255,255,.04);
  }}
  body {{
    margin: 0; padding: 32px;
    color: var(--fg);
    background: radial-gradient(1200px 800px at 10% 10%, #0a1325 0%, transparent 50%),
                radial-gradient(1000px 700px at 90% 20%, #1b1140 0%, transparent 55%),
                linear-gradient(160deg, var(--grad1), var(--grad2));
    font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, Arial;
  }}
  h1 {{ letter-spacing: .2px; margin-bottom: 8px; }}
  h4 {{ margin-bottom: 12px; }}
  .badge-custom {{
    background: linear-gradient(135deg, var(--accent), var(--accent2));
    color: #0b0f1a; border: none; padding: 6px 10px; border-radius: 999px; font-weight: 700;
    box-shadow: 0 6px 18px rgba(0, 229, 255, .25);
  }}
  .card {{
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 16px;
    box-shadow: var(--shadow);
    backdrop-filter: blur(6px);
  }}
  .nav-pills .nav-link {{
    color: var(--fg);
    border: 1px solid var(--border);
    background: rgba(255,255,255,0.04);
    transition: transform .15s ease, background .2s ease, box-shadow .2s ease;
  }}
  .nav-pills .nav-link:hover {{
    transform: translateY(-1px);
    box-shadow: 0 8px 18px rgba(0,0,0,.25);
  }}
  .nav-pills .nav-link.active {{
    color: #0b0f1a;
    background: linear-gradient(135deg, var(--accent), var(--accent2));
    border-color: transparent;
    box-shadow: 0 10px 22px rgba(255, 78, 205, .25);
  }}
  .table {{ color: #ffffff; font-size: 14px; }}
  .table thead th {{
    color: #ffffff;
    background: linear-gradient(180deg, rgba(255,255,255,.08), rgba(255,255,255,.02));
    border-bottom: 1px solid var(--border);
  }}
  .table tbody tr:hover {{ background: rgba(255,255,255,.05); }}
  .table tbody td {{ color: #ffffff; }}
  .form-control {{
    background: rgba(255,255,255,.06);
    border-color: var(--border);
    color: var(--fg);
    box-shadow: inset 0 0 0 1px rgba(255,255,255,.03);
  }}
  .form-control:focus {{
    border-color: var(--accent);
    box-shadow: 0 0 0 3px rgba(0,229,255,.2);
  }}
  img {{
    max-width: 100%;
    border-radius: 12px;
    border: 1px solid var(--border);
    box-shadow: 0 12px 28px rgba(0,0,0,.35);
  }}
  details summary {{
    cursor: pointer;
    color: var(--accent3);
  }}
  details[open] summary {{ color: var(--accent); }}
  .subtitle {{ color: #ffffff; margin-bottom: 16px; }}
  .text-secondary {{ color: #ffffff !important; }}
  .card p {{ color: #ffffff !important; }}
  .card h4 {{ color: #ffffff !important; }}
  .card span {{ color: #ffffff !important; }}
  .card b {{ color: #ffffff !important; }}
  .card i {{ color: #ffffff !important; }}
  .card ul {{ color: #ffffff !important; }}
  .card li {{ color: #ffffff !important; }}
  .card details {{ color: #ffffff !important; }}
  .card summary {{ color: #ffffff !important; }}
  .tab-pane {{ color: #ffffff !important; }}
  .tab-pane p {{ color: #ffffff !important; }}
  .tab-pane h4 {{ color: #ffffff !important; }}
  .tab-pane span {{ color: #ffffff !important; }}
  .tab-pane b {{ color: #ffffff !important; }}
  .tab-pane i {{ color: #ffffff !important; }}
  .tab-pane ul {{ color: #ffffff !important; }}
  .tab-pane li {{ color: #ffffff !important; }}
  .card * {{ color: #ffffff !important; }}
  .tab-pane * {{ color: #ffffff !important; }}
  .container-fluid * {{ color: #ffffff !important; }}
  .container-fluid p {{ color: #ffffff !important; }}
  .container-fluid h1 {{ color: #ffffff !important; }}
  .container-fluid h4 {{ color: #ffffff !important; }}
  .container-fluid span {{ color: #ffffff !important; }}
  .container-fluid b {{ color: #ffffff !important; }}
  .container-fluid i {{ color: #ffffff !important; }}
  .container-fluid ul {{ color: #ffffff !important; }}
  .container-fluid li {{ color: #ffffff !important; }}
  #table-modelos * {{ color: #ffffff !important; }}
  #table-resumen * {{ color: #ffffff !important; }}
  #table-modelos table {{ color: #ffffff !important; }}
  #table-resumen table {{ color: #ffffff !important; }}
  #table-modelos th {{ color: #ffffff !important; }}
  #table-modelos td {{ color: #ffffff !important; }}
  #table-resumen th {{ color: #ffffff !important; }}
  #table-resumen td {{ color: #ffffff !important; }}
</style>

</head>
<body class="p-4">
  <div class="container-fluid">
    <h1 class="mb-2">Reto 1 – Toneladas Cosechadas <span class="badge badge-custom">Dashboard</span></h1>
    <p class="text-secondary">Preprocesamiento, transformaciones de la variable objetivo y comparación de modelos.</p>

    <ul class="nav nav-pills mb-3" id="pills-tab" role="tablist">
      <li class="nav-item" role="presentation">
        <button class="nav-link active" id="pills-modelos-tab" data-bs-toggle="pill" data-bs-target="#pills-modelos" type="button" role="tab">Modelos</button>
      </li>
      <li class="nav-item" role="presentation">
        <button class="nav-link" id="pills-plotly-tab" data-bs-toggle="pill" data-bs-target="#pills-plotly" type="button" role="tab">Distribuciones (Interactivo)</button>
      </li>
      <li class="nav-item" role="presentation">
        <button class="nav-link" id="pills-kde-tab" data-bs-toggle="pill" data-bs-target="#pills-kde" type="button" role="tab">Distribuciones (PNG)</button>
      </li>
      <li class="nav-item" role="presentation">
        <button class="nav-link" id="pills-resumen-tab" data-bs-toggle="pill" data-bs-target="#pills-resumen" type="button" role="tab">Resumen</button>
      </li>
      <li class="nav-item" role="presentation">
        <button class="nav-link" id="pills-interp-tab" data-bs-toggle="pill" data-bs-target="#pills-interp" type="button" role="tab">Interpretación</button>
      </li>
      <li class="nav-item" role="presentation">
        <button class="nav-link" id="pills-fi-tab" data-bs-toggle="pill" data-bs-target="#pills-fi" type="button" role="tab">Importancias</button>
      </li>
      <li class="nav-item" role="presentation">
        <button class="nav-link" id="pills-meta-tab" data-bs-toggle="pill" data-bs-target="#pills-meta" type="button" role="tab">Metadata</button>
      </li>
    </ul>

    <div class="tab-content" id="pills-tabContent">
      <div class="tab-pane fade show active" id="pills-modelos" role="tabpanel">
        <div class="card p-3 mb-3">
          <h4>Comparación de Modelos</h4>
          <input id="search-modelos" class="form-control form-control-sm mb-2" placeholder="Filtrar filas..." />
          <div id="table-modelos">{results_table_html}</div>
          <details class="mt-2"><summary>Cómo leer las métricas</summary>
            <span class="text-secondary">Se elige el modelo con <b>menor MSE</b>. R²: proporción de varianza explicada. MAE: error medio absoluto.</span>
          </details>
        </div>
      </div>

      <div class="tab-pane fade" id="pills-plotly" role="tabpanel">
        <div class="card p-3 mb-3">
          <h4>Distribuciones Interactivas</h4>
          <div id="plotlyDiv" style="height:560px;"></div>
        </div>
      </div>

      <div class="tab-pane fade" id="pills-kde" role="tabpanel">
        <div class="card p-3 mb-3">
          <h4>Distribuciones KDE (seaborn)</h4>
          <img src="data:image/png;base64,{plot_b64}" alt="Distribuciones KDE" />
        </div>
      </div>

      <div class="tab-pane fade" id="pills-resumen" role="tabpanel">
        <div class="card p-3 mb-3">
          <h4>Resumen de Transformaciones</h4>
          <input id="search-resumen" class="form-control form-control-sm mb-2" placeholder="Filtrar filas..." />
          <div id="table-resumen">{summary_table_html}</div>
        </div>
      </div>

      <div class="tab-pane fade" id="pills-interp" role="tabpanel">
        <div class="card p-3 mb-3">
          <h4>Interpretación</h4>
          {interpretation_text()}
        </div>
      </div>

      <div class="tab-pane fade" id="pills-fi" role="tabpanel">
        <div class="card p-3 mb-3">
          <h4>Importancia de Features (Top 20)</h4>
          {fi_html}
          <p class="text-secondary">Si el mejor modelo no es de árboles, esta sección puede no mostrar valores.</p>
        </div>
      </div>

      <div class="tab-pane fade" id="pills-meta" role="tabpanel">
        <div class="card p-3 mb-3">
          <h4>Metadata</h4>
          <ul class="mb-2">
            <li>Archivo procesado: <b>{INPUT_PATH}</b></li>
            <li>Filas: <b>{len(df_enc):,}</b> – Columnas: <b>{len(df_enc.columns):,}</b></li>
            <li>Mejor modelo: <b>{best_model_name}</b></li>
          </ul>
          <details><summary>Columnas de features usadas</summary>
            <span class="text-secondary">{', '.join(feature_cols)}</span>
          </details>
        </div>
      </div>
    </div>
  </div>

<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/js/bootstrap.bundle.min.js"></script>
<script>
  // Búsqueda simple en tablas (client-side)
  function filterTable(inputId, tableContainerId) {{
    const q = document.getElementById(inputId).value.toLowerCase();
    const container = document.getElementById(tableContainerId);
    const table = container.querySelector('table');
    if (!table) return;
    const rows = table.querySelectorAll('tbody tr');
    rows.forEach(tr => {{
      const text = tr.innerText.toLowerCase();
      tr.style.display = text.includes(q) ? '' : 'none';
    }});
  }}
  document.getElementById('search-modelos').addEventListener('input', () => filterTable('search-modelos', 'table-modelos'));
  document.getElementById('search-resumen').addEventListener('input', () => filterTable('search-resumen', 'table-resumen'));

  // Plotly: genera un histograma densidad por cada serie
  const series = {plotly_json};
  const traces = [];
  for (const [name, arr] of Object.entries(series)) {{
    traces.push({{
      type: 'histogram',
      x: arr,
      histnorm: 'probability density',
      name: name,
      opacity: 0.55
    }});
  }}
  const layout = {{
    paper_bgcolor: '#0b0f1a',
    plot_bgcolor: 'rgba(255,255,255,0.02)',
    font: {{ color: '#f8fafc' }},
    barmode: 'overlay',
    margin: {{ t: 30, r: 10, l: 40, b: 40 }},
    xaxis: {{ gridcolor: 'rgba(255,255,255,0.10)', zerolinecolor: 'rgba(255,255,255,0.10)' }},
    yaxis: {{ gridcolor: 'rgba(255,255,255,0.10)', title: 'Densidad', zerolinecolor: 'rgba(255,255,255,0.10)' }},
    legend: {{
      bgcolor: 'rgba(11,15,26,0.6)',
      bordercolor: 'rgba(255,255,255,0.12)',
      borderwidth: 1
    }}
  }};
  Plotly.newPlot('plotlyDiv', traces, layout, {{displayModeBar: true, responsive: true}});
</script>
</body>
</html>
"""

dash_path = os.path.join(OUTPUT_DIR, "dashboard_reto1.html")
with open(dash_path, "w", encoding="utf-8") as f:
    f.write(dashboard_html)

# ==================================
# 14) Guardados finales
# ==================================
processed_path = os.path.join(OUTPUT_DIR, "dataset_procesado.csv")
df_enc.to_csv(processed_path, index=False)

# Log final de archivos
print("\n=== ARCHIVOS GENERADOS ===")
print(f"- Dataset procesado: {processed_path}")
print(f"- Tabla resumen: {summary_csv_path}")
print(f"- Métricas modelos: {metrics_csv_path}")
print(f"- Gráfico (seaborn PNG): {plot_path}")
print(f"- Dashboard HTML (interactivo): {dash_path}")
if model_path:
    print(f"- Mejor modelo guardado: {model_path}")
print("\nListo ✅")

import pandas as pd
import numpy as np

# Paso 1: Carga de los datos
df_salud = pd.read_csv('lectura_archivos/entrada/encuesta_salud.csv')
df_region = pd.read_csv('lectura_archivos/entrada/region_referencia.csv')

# Paso 2: Inspección del DataFrame
print("\n=== Información General ===")
print(df_salud.info())
print(df_salud.head())

# Paso 3: Conversión a numérico
cols_numericas = ['Peso', 'Altura', 'Presion_Arterial']
df_salud[cols_numericas] = df_salud[cols_numericas].apply(pd.to_numeric, errors='coerce')

# Paso 4: Eliminación de duplicados
df_salud = df_salud.drop_duplicates()

# Paso 5: Cálculo del IMC y categorización
df_salud['IMC'] = df_salud['Peso'] / (df_salud['Altura'] ** 2)

# Discretización del IMC
def categorizar_imc(imc):
    if pd.isna(imc):
        return np.nan
    elif imc < 18.5:
        return 'Bajo'
    elif 18.5 <= imc < 25:
        return 'Normal'
    elif 25 <= imc < 30:
        return 'Sobrepeso'
    else:
        return 'Obesidad'

df_salud['Categoria_IMC'] = df_salud['IMC'].apply(categorizar_imc)

# Paso 6: Rellenar ingresos nulos con la media por región
df_salud['Ingresos'] = pd.to_numeric(df_salud['Ingresos'], errors='coerce')
df_salud['Ingresos'] = df_salud.groupby('Region')['Ingresos'].transform(lambda x: x.fillna(x.mean()))

# Paso 7: Unión con archivo de regiones
df_completo = pd.merge(df_salud, df_region, on='Region', how='left')

# Paso 8: Agrupar por Zona y calcular promedios de IMC y presión
df_resumen = df_completo.groupby('Zona')[['IMC', 'Presion_Arterial']].mean()
print("\n=== Promedios por Zona ===")
print(df_resumen)

# Paso 9: Crear índice jerárquico con Zona y Sexo
df_completo.set_index(['Zona', 'Sexo'], inplace=True)

# Paso 10: Comparativa usando .unstack()
print("\n=== Comparativa de IMC por Zona y Sexo ===")
print(df_completo['IMC'].groupby(level=[0,1]).mean().unstack())

print("\n=== Comparativa de Presión Arterial por Zona y Sexo ===")
print(df_completo['Presion_Arterial'].groupby(level=[0,1]).mean().unstack())

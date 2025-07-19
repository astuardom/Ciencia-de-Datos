import pandas as pd
import numpy as np

# Cargar los archivos CSV
df_ventas = pd.read_csv('lectura_archivos/entrada/ventas_diarias.csv')
df_empleados = pd.read_csv('lectura_archivos/entrada/empleados_sucursal.csv')    
df_productos = pd.read_csv('lectura_archivos/entrada/productos_categoria.csv')

# Vista previa de los dataframes originales
print("\n ==== Data Frame Originales ====")
print(df_ventas.head(9))

# Paso 1: Indexación jerárquica (MultiIndex)
df_multiindex = df_ventas.set_index(['Sucursal', 'Fecha', 'Categoria'])
print("\n ==== Data Frame con MultiIndex ====")
print(df_multiindex.head(10))  # CORREGIDO: nombre correcto de variable

# Todas las ventas en sucursal 'Sur'
#print("\n ==== Ventas en Sucursal Sur ====")
#print(df_multiindex.loc['Sur'])

# Ventas en 'Sur' el 2024-06-01
#print("\n ==== Ventas en Sur el 2024-06-01 ====")
#print(df_multiindex.loc[('Sur', '2024-06-01')])

# Ventas de categoría 'Electronica' en Sur el 2024-06-01
#print("\n ==== Ventas de Electrónica en Sur el 2024-06-01 ====")
#print(df_multiindex.xs(('Sur', '2024-06-01', 'Electronica')))

#resumen diario de ventas por sucursal electronica
#ventas_elect=df_multiindex.xs('Electronica', level='Categoria')
#print("\n ==== Resumen Diario de Ventas por Sucursal Electrónica ====")



# Paso 2: Agrupamiento y agregación
df_grupoped = df_ventas.groupby(['Sucursal', 'Categoria'])['Ventas'].sum().reset_index()
print("\n ==== Data Frame Agrupación y Agregación (Ventas Totales) ====")
print(df_grupoped.head(10))

df_grupoped2 = df_ventas.groupby(['Sucursal', 'Categoria'])[['Ventas', 'Unidades']].agg(['sum', 'mean']).reset_index()
print("\n ==== Data Frame Agrupación y Agregación (Sum y Mean) ====")
print(df_grupoped2.head(10))


# Paso 3: Transformación de estructura
df_pivotable = df_ventas.groupby(['Fecha', 'Categoria'])['Ventas'].sum().reset_index()
print("\n ==== Data Frame Pivot Table ====")
print(df_pivotable.head(10))

# Transformación en tabla pivote
df_pivote = df_pivotable.pivot(index='Fecha', columns='Categoria', values='Ventas')
print("\n ==== Data Frame Pivote ====")
print(df_pivote.head(10))

# Paso 4: despivote con melt
#revertimos el pivote enterior para volver
df_melted = df_pivote.reset_index().melt(id_vars='Fecha', var_name='Categoria', value_name='Ventas')
print("\n ==== Data Frame Melted ====")
print(df_melted.head(10))

# Paso 5: concatenación de DataFrames
df_extra = df_ventas.copy()
df_extra['Fecha'] = pd.to_datetime(df_extra['Fecha']) + pd.Timedelta(days=10)
df_extra['Fecha'] = df_extra['Fecha'].dt.strftime('%Y-%m-%d')
df_concatenado = pd.concat([df_ventas, df_extra], axis=0)
print("\n ==== Data Frame Concatenado ====")
print(df_concatenado.head(10))
df_concatenado.to_csv('lectura_archivos/salida/ventas_concatenadas.csv')

# Paso 6: combinación usando merge

# INNER JOIN
df_merged_inner = pd.merge(df_ventas, df_productos, on='Categoria', how='inner')
print("\n ==== Data Frame Merge (Ventas y Productos) INNER JOIN ====")
print(df_merged_inner.head(10))
df_merged_inner.to_csv('lectura_archivos/salida/ventas_productos_inner.csv', index=False)

# LEFT JOIN
df_merged_left = pd.merge(df_ventas, df_empleados, on='Sucursal', how='left')
print("\n ==== Data Frame Merge (Ventas y Empleados) LEFT JOIN ====")
print(df_merged_left.head(10))
df_merged_left.to_csv('lectura_archivos/salida/ventas_empleados_left.csv', index=False)

# RIGHT JOIN
df_merged_right = pd.merge(df_productos, df_ventas, on='Categoria', how='right')
print("\n ==== Data Frame Merge (Productos y Ventas) RIGHT JOIN ====")
print(df_merged_right.head(10))
df_merged_right.to_csv('lectura_archivos/salida/productos_ventas_right.csv', index=False)







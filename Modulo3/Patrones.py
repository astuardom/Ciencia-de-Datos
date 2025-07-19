import pandas as pd

# Datos corregidos y bien formateados
data = {
    'Cliente': ['C1', 'C2', 'C1', 'C3'],
    'Producto': ['Portátil', 'Teléfono', 'Tableta', 'Ratón'],
    'Categoria': ['Electrónica', 'Electrónica', 'Electrónica', 'Accesorios'],
    'Monto': [999.99, 499.99, 299.99, 19.99],
    'Fecha': ['2025-10-01', '2025-09-15', '2025-11-01', '2025-08-01']
}

# Crear el DataFrame
df = pd.DataFrame(data)

# Convertir a datetime
df['Fecha'] = pd.to_datetime(df['Fecha'])

# Definir último trimestre: septiembre a diciembre
inicio_trimestre = pd.to_datetime('2025-09-01')
fin_trimestre = pd.to_datetime('2025-12-31')

# Filtrar por fechas
filtro = (df['Fecha'] >= inicio_trimestre) & (df['Fecha'] <= fin_trimestre)
df_trim = df[filtro]

# Calcular monto total por cliente
monto_total = df_trim.groupby('Cliente')['Monto'].sum().sort_values(ascending=False)

# Estadísticas
minimo = df_trim['Monto'].min()
maximo = df_trim['Monto'].max()
media = df_trim['Monto'].mean()

# Categorías únicas
categorias = df_trim['Categoria'].nunique()

# Mostrar salida esperada
print("Clientes Principales por Monto Total:")
print(monto_total)

print(f"\nEstadísticas de Compras:\nMín: {minimo:.2f}, Máx: {maximo:.2f}, Media: {media:.2f}")
print(f"\nCategorías Únicas: {categorias}")

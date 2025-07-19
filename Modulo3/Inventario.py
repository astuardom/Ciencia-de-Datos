import pandas as pd

# Datos corregidos y limpios
data = {
    'Tienda': ['T1', 'T1', 'T2', 'T2'],
    'Producto': ['Portátil', 'Ratón', 'Teléfono', 'Tableta'],
    'Precio': [999.99, 19.99, 499.99, 299.99],
    'Cantidad': [5, 50, 8, 3]
}

# Crear DataFrame
df = pd.DataFrame(data)

# Calcular artículos con bajo inventario (<10 unidades)
bajo_inventario = df[df['Cantidad'] < 10][['Tienda', 'Producto', 'Cantidad']]

# Calcular valor total del inventario por tienda
df['Valor_Inventario'] = df['Precio'] * df['Cantidad']
valor_por_tienda = df.groupby('Tienda')['Valor_Inventario'].sum()

# Contar tipos de productos únicos
tipos_productos = df['Producto'].nunique()

# Mostrar resultados
print("Artículos con Bajo Inventario:")
print(bajo_inventario)

print("\nValor del Inventario por Tienda:")
print(valor_por_tienda)

print(f"\nTipos de Productos Únicos: {tipos_productos}")

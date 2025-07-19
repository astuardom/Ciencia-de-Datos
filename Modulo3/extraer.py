import pandas as pd

# 1. Leer la hoja específica 'Ventas2023' del archivo Excel
df = pd.read_excel('lectura_archivos/entrada/reporte_ventas.xlsx', sheet_name='Ventas2023')

# 2. Seleccionar solo las columnas Cliente y Monto
df_filtrado = df[['Cliente', 'Monto']]

# 3. Establecer la columna Cliente como índice
df_filtrado.set_index('Cliente', inplace=True)

# 4. Exportar a un archivo CSV
df_filtrado.to_csv('ventas_2023.csv', sep=',')  # Usa coma como separador

# 5. Imprimir resultado
print("✅ Datos extraídos de la hoja 'Ventas2023':")
print(df_filtrado)

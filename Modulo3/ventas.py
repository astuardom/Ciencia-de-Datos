import pandas as pd

# 1. Leer el archivo ventas.csv
df = pd.read_csv('lectura_archivos/entrada/retos/ventas_mensuales.csv', sep=',')

# 2. Filtrar las filas donde la cantidad es mayor a 10
df_filtrado = df[df['Cantidad'] > 10]

# 3. Exportar los datos filtrados a un nuevo archivo CSV sin índice y con delimitador ;
df_filtrado.to_csv('ventas_filtradas.csv', sep=';', index=False)

print("Archivo 'ventas_filtradas.csv' creado exitosamente con las ventas filtradas.")
print(df_filtrado)

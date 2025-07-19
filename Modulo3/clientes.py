import pandas as pd

# 1. Leer ambos archivos CSV con separador coma
df1 = pd.read_csv('lectura_archivos/entrada/retos/clientes1.csv', sep=',')
df2 = pd.read_csv('lectura_archivos/entrada/retos/clientes2.csv', sep=',')

# 2. Combinar ambos DataFrames
df_combinado = pd.concat([df1, df2], ignore_index=True)

# 3. Manejar valores nulos: reemplazar por 0 o 'Desconocido'
df_combinado['Edad'] = df_combinado['Edad'].fillna(0)
df_combinado['Ciudad'] = df_combinado['Ciudad'].fillna('Desconocido')

# 4. Exportar resultado a nuevo archivo CSV
df_combinado.to_csv('clientes_combinados.csv', sep=',', index=False)

# 5. Imprimir el resultado en pantalla
print("✅ Clientes combinados:")
print(df_combinado)

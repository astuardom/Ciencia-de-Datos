import pandas as pd

# 1. Leer el archivo Excel original
df = pd.read_excel('lectura_archivos/entrada/retos/empleados.xlsx')

# 2. Reemplazar valores nulos por 'Desconocido'
df_limpio = df.fillna('Desconocido')

# 3. Exportar a un nuevo archivo Excel limpio
df_limpio.to_excel('empleados_limpios.xlsx', index=False)

# 4. Imprimir el resultado en pantalla
print("✅ Datos limpios de empleados:")
print(df_limpio)

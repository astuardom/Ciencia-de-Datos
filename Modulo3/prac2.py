import pandas as pd
df = pd.read_excel('lectura_archivos/entrada/reporte.xlsx', sheet_name='Ventas2023')
print(df.head())    
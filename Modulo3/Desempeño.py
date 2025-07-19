import pandas as pd

# Crear el diccionario de datos
data = {
    'Nombre': ['Alicia', 'Bruno', 'Carlos', 'David'],
    'Departamento': ['Ventas', 'Marketing', 'TI', 'Ventas'],
    'Puntaje': [85, 90, 75, 88],
    'Calificacion': ['A', 'A', 'B', 'A']
}

# Crear el DataFrame con índice personalizado
df = pd.DataFrame(data, index=['E001', 'E002', 'E003', 'E004'])

# Filtrar empleados con puntaje > 80 y en Ventas o Marketing
filtro = (df['Puntaje'] > 80) & (df['Departamento'].isin(['Ventas', 'Marketing']))
altos_desempenos = df[filtro][['Nombre', 'Departamento', 'Puntaje']]

# Calcular promedio por departamento
promedios = df.groupby('Departamento')['Puntaje'].mean()

# Calificaciones únicas
calificaciones_unicas = df['Calificacion'].unique()

# Mostrar resultados
print("Altos Desempeños en Ventas/Marketing:\n", altos_desempenos)
print("\nPuntaje Promedio por Departamento:\n", promedios)
print("\nCalificaciones Únicas:", calificaciones_unicas)

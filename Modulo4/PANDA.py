# 1. Importar la librería Pandas y crear el DataFrame
import pandas as pd

data = {
    'Jugador': ['Lionel Messi', 'Cristiano Ronaldo', 'Kevin De Bruyne', 'Kylian Mbappé', 'Luka Modric'],
    'Posición': ['Delantero', 'Delantero', 'Mediocampista', 'Delantero', 'Mediocampista'],
    'Edad': [35, 38, 31, 24, 37],
    'Goles': [20, 18, 8, 25, 3],
    'Asistencias': [10, 5, 15, 12, 8]
}

df = pd.DataFrame(data)
print("1. DataFrame creado:")
print(df)

# 2. Mostrar los nombres de todos los jugadores
print("\n2. Nombres de los jugadores:")
print(df['Jugador'])

# 3. Filtrar jugadores con más de 10 goles
print("\n3. Jugadores con más de 10 goles:")
print(df[df['Goles'] > 10][['Jugador', 'Goles']])

# 4. Agregar columna 'Puntos' (Goles * 4) + (Asistencias * 2)
df['Puntos'] = (df['Goles'] * 4) + (df['Asistencias'] * 2)
print("\n4. DataFrame con columna 'Puntos':")
print(df)

# 5. Promedio de goles
promedio_goles = df['Goles'].mean()
print("\n5. Promedio de goles:")
print(promedio_goles)

# 6. Máximo y mínimo de asistencias
max_asist = df['Asistencias'].max()
min_asist = df['Asistencias'].min()
print("\n6. Máximo de asistencias:", max_asist)
print("   Mínimo de asistencias:", min_asist)

# 7. Contar jugadores por posición
conteo_posiciones = df['Posición'].value_counts()
print("\n7. Cantidad de jugadores por posición:")
print(conteo_posiciones)

# 8. Ordenar por goles en orden descendente
ordenado_goles = df.sort_values(by='Goles', ascending=False)
print("\n8. DataFrame ordenado por goles (descendente):")
print(ordenado_goles)

# 9. Estadísticas generales del DataFrame
print("\n9. Estadísticas generales:")
print(df.describe())

# 10. Contar cuántos jugadores hay en cada posición con value_counts()
print("\n10. Conteo por posición con value_counts():")
print(df['Posición'].value_counts())

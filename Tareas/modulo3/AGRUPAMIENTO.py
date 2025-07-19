import pandas as pd

# 1. Leer el archivo CSV original
df = pd.read_csv("estudiantes_calificaciones.csv")

# 1. Crear DataFrame con Indexación Jerárquica
df_hierarchical = df.set_index(["Estudiante", "Materia"])
print("1. DataFrame con Indexación Jerárquica:\n", df_hierarchical)

# 2. Acceder a datos con Indexación Jerárquica
calificacion_maria_historia = df_hierarchical.loc[("María", "Historia")]
print("\n2. Calificación de María en Historia:", calificacion_maria_historia["Calificación"])

# 3. Agrupar y Agregar Datos con groupby
grouped = df.groupby("Materia")["Calificación"]
promedio = grouped.mean()
maximo = grouped.max()
print("\n3. Promedio por Materia:\n", promedio)
print("\n3. Máxima Calificación por Materia:\n", maximo)

# 4. Pivoteo de DataFrame
pivot_df = df.pivot(index="Estudiante", columns="Materia", values="Calificación")
print("\n4. DataFrame Pivoteado:\n", pivot_df)

# 5. Despivoteo con melt
melted_df = pivot_df.reset_index().melt(id_vars="Estudiante", var_name="Materia", value_name="Calificación")
print("\n5. DataFrame Despivoteado:\n", melted_df)

# 6. Concatenación y Merge de DataFrames
df1 = pd.DataFrame({
    "ID_Estudiante": [1, 2],
    "Estudiante": ["Juan", "María"],
    "Carrera": ["Ingeniería", "Historia"]
})

df2 = pd.DataFrame({
    "ID_Estudiante": [1, 1, 2, 2],
    "Materia": ["Matemáticas", "Historia", "Matemáticas", "Historia"],
    "Calificación": [6.5, 5.8, 4.2, 6.0]
})

# Concatenación por filas
concat_df = pd.concat([df1, df2], axis=0, ignore_index=True)
print("\n6. Concatenación de df1 y df2:\n", concat_df)

# Merge por ID_Estudiante
merged_df = pd.merge(df1, df2, on="ID_Estudiante")
print("\n6. Merge por ID_Estudiante:\n", merged_df)

# Guardar resultados (opcional)
df_hierarchical.to_csv("1_indexacion_jerarquica.csv")
pivot_df.to_csv("4_pivoteo.csv")
melted_df.to_csv("5_despivoteo.csv", index=False)
df1.to_csv("6_df1.csv", index=False)
df2.to_csv("6_df2.csv", index=False)
concat_df.to_csv("6_concatenado.csv", index=False)
merged_df.to_csv("6_mergeado.csv", index=False)

import pandas as pd
import numpy as np
import os

def analizar_calificaciones():
    """
    Analiza calificaciones de estudiantes con limpieza de datos y análisis estadístico
    """
    try:
        # Verificar si el archivo existe
        archivo = 'lectura_archivos/entrada/reto2_calificaciones_ampliado.csv'
        
        if not os.path.exists(archivo):
            print(f"Error: No se encontró el archivo '{archivo}'")
            return None
            
        # Cargar el archivo CSV
        df = pd.read_csv(archivo)
        print(f"Archivo cargado exitosamente. Registros: {len(df)}")
        
        return df
        
    except FileNotFoundError:
        print("Error: No se pudo encontrar el archivo de entrada")
        return None
    except Exception as e:
        print(f"Error inesperado: {e}")
        return None

# Cargar datos
df = analizar_calificaciones()
if df is None:
    exit()

# Eliminar duplicados
df = df.drop_duplicates()
df["Calificacion"] = pd.to_numeric(df["Calificacion"], errors="coerce")
nulos = df["Calificacion"].isnull().sum()
print(f"\nDuplicados eliminados. Valores nulos en Calificación: {nulos}")

# Imputar nulos
df["Calificacion"] = df.groupby("Materia")["Calificacion"].transform(
    lambda x: x.fillna(x.mean())
)
print("\nNulos imputados con la media por Materia.")

# Calcular cuartiles y detectar outliers
q1 = df["Calificacion"].quantile(0.25)
q2 = df["Calificacion"].quantile(0.50)
q3 = df["Calificacion"].quantile(0.75)
iqr = q3 - q1
limite_inferior = q1 - 1.5 * iqr
limite_superior = q3 + 1.5 * iqr

outliers = df[(df["Calificacion"] < limite_inferior) | (df["Calificacion"] > limite_superior)]
valores_no_outliers = df[(df["Calificacion"] >= limite_inferior) & (df["Calificacion"] <= limite_superior)]

print("\n[3] Análisis de Calificaciones:")
print(f"Q1: {q1:.2f}")
print(f"Q2 (mediana): {q2:.2f}")
print(f"Q3: {q3:.2f}")
print(f"IQR: {iqr:.2f}")
print(f"Límite inferior: {limite_inferior:.2f}")
print(f"Límite superior: {limite_superior:.2f}")
print(f"\nOutliers encontrados ({len(outliers)} registros):")
print(outliers[["ID", "Calificacion"]])

# Análisis por género y materia 
print("\nPromedio de calificaciones por Género y Materia:")
resumen = df.groupby(["Genero", "Materia"])["Calificacion"].mean().round(2)
print(resumen)

# Guardar archivo procesado
try:
    df.to_csv("reto2_calificaciones_procesado.csv", index=False)
    print("\nArchivo guardado como 'reto2_calificaciones_procesado.csv'")
except Exception as e:
    print(f"\nError al guardar archivo: {e}")

if __name__ == "__main__":
    print("Análisis de calificaciones completado exitosamente.")
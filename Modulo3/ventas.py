import pandas as pd
import os

def procesar_ventas():
    """
    Procesa archivo de ventas mensuales y filtra registros con cantidad > 10
    """
    try:
        # Verificar si el archivo existe
        archivo_entrada = 'lectura_archivos/entrada/ventas_mensuales.csv'
        
        if not os.path.exists(archivo_entrada):
            print(f"Error: No se encontró el archivo '{archivo_entrada}'")
            print("Verificando archivos disponibles en el directorio...")
            
            # Listar archivos disponibles
            if os.path.exists('lectura_archivos/entrada/'):
                archivos = os.listdir('lectura_archivos/entrada/')
                print("Archivos disponibles:")
                for archivo in archivos:
                    if archivo.endswith('.csv'):
                        print(f"  - {archivo}")
            return
        
        # 1. Leer el archivo ventas.csv
        df = pd.read_csv(archivo_entrada, sep=',')
        print(f"Archivo cargado exitosamente. Registros: {len(df)}")
        
        # 2. Filtrar las filas donde la cantidad es mayor a 10
        df_filtrado = df[df['Cantidad'] > 10]
        print(f"Registros filtrados (Cantidad > 10): {len(df_filtrado)}")
        
        # 3. Exportar los datos filtrados a un nuevo archivo CSV sin índice y con delimitador ;
        archivo_salida = 'ventas_filtradas.csv'
        df_filtrado.to_csv(archivo_salida, sep=';', index=False)
        
        print(f"Archivo '{archivo_salida}' creado exitosamente con las ventas filtradas.")
        print("\nPrimeras 5 filas del archivo filtrado:")
        print(df_filtrado.head())
        
    except FileNotFoundError:
        print("Error: No se pudo encontrar el archivo de entrada")
    except pd.errors.EmptyDataError:
        print("Error: El archivo CSV está vacío")
    except Exception as e:
        print(f"Error inesperado: {e}")

if __name__ == "__main__":
    procesar_ventas()

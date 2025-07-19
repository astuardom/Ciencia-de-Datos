import pandas as pd

# Cargar el archivo CSV con un separador diferente
#df = pd.read_csv('lectura_archivos/entrada/estudiantes.csv', sep=';')
#print(df)

#eliminar una columna
#df = pd.read_csv('lectura_archivos/entrada/ventas.csv', sep=';')   
#df = df.dropna() 
#print (df)

#escrivir csv 
data = {
    "productos": ["peras", "durazno"],
    "precio": [0.5, 0.3]
}
df = pd.DataFrame(data)
df.to_csv('lectura_archivos/entrada/productos.csv', index=False, sep=';')    
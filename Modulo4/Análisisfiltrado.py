import pandas as pd
import random

# Datos base (como en la imagen)
productos_base = [
    ("Laptop", 800),
    ("Teclado", 30),
    ("Mouse", 15),
    ("Monitor", 200),
    ("Impresora", 150),
    ("Cargador", 25)
]

# Crear 100 registros aleatorios
data = []
for i in range(101, 201):  # IDs de 101 a 200
    producto, precio = random.choice(productos_base)
    cantidad = random.randint(1, 3)
    total = cantidad * precio
    data.append((i, producto, cantidad, precio, total))

# Crear el DataFrame original
df = pd.DataFrame(data, columns=["ID_Venta", "Producto", "Cantidad", "Precio", "Total"])

# Guardar el DataFrame como archivo CSV
df.to_csv("ventas_100.csv", index=False)  # 1. Guardar archivo CSV

# 1. Cargar el archivo CSV
df_loaded = pd.read_csv("ventas_100.csv")

# 2. Mostrar las primeras 5 filas
print("Primeras 5 filas:")
print(df_loaded.head())

# 3. Extraer columnas Producto y Precio
producto_precio = df_loaded[["Producto", "Precio"]]
print("\nProducto y Precio:")
print(producto_precio)

# 4. Filtrar productos con precio mayor a 50
filtrado = df_loaded[df_loaded["Precio"] > 50]
print("\nProductos con precio > 50:")
print(filtrado)

# 5. Guardar el DataFrame filtrado en un nuevo archivo CSV
filtrado.to_csv("ventas_filtradas.csv", index=False)

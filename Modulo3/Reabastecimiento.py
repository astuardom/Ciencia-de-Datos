import numpy as np

# Matriz de inventario: filas = productos, columnas = tiendas
inventario = np.array([
    [45, 8, 67, 92],    # Medicamentos
    [73, 55, 4, 88],    # Alimentos
    [19, 82, 63, 3],    # Accesorios
    [91, 47, 9, 76],    # Vacunas
    [34, 95, 81, 2]     # Antiparasitarios
])

# Ventas semanales promedio por producto
ventas_promedio = np.array([15, 25, 10, 22, 8])

# Nombres de los productos
productos = np.array(["Medicamentos", "Alimentos", "Accesorios", "Vacunas", "Antiparasitarios"])

print("Productos con existencias < 20:")
prioridad = []

# Recorrer la matriz con índices
for i in range(inventario.shape[0]):        # Filas = productos
    for j in range(inventario.shape[1]):    # Columnas = tiendas
        stock = inventario[i, j]
        if stock < 20:
            print(f"- {productos[i]}: Tienda {j+1} ({stock} unidades)")
            if ventas_promedio[i] > 20:
                prioridad.append((productos[i], j+1, stock, ventas_promedio[i]))

# Mostrar productos con prioridad de reabastecimiento
if prioridad:
    print("\nPrioridad de reabastecimiento (ventas > 20):")
    for producto, tienda, stock, ventas in prioridad:
        print(f"- {producto} (Tienda {tienda}: {stock} unidades, ventas: {ventas})")
else:
    print("\nNo hay productos con alta demanda y bajo stock.")

import statistics as stats

def balancear_reaccion(coeficientes):
    if len(coeficientes) < 4:
        print("Debe haber al menos 4 coeficientes (2 reactivos y 2 productos).")
        return

    mitad = len(coeficientes) // 2
    reactivos = coeficientes[:mitad]
    productos = coeficientes[mitad:]

    suma_reactivos = sum(reactivos)
    suma_productos = sum(productos)

    if abs(suma_reactivos - suma_productos) < 0.01:
        print(f"Reacción balanceada. Media: {stats.mean(coeficientes):.2f}")
    else:
        print(f"Reacción no balanceada. Media: {stats.mean(coeficientes):.2f}")


if __name__ == "__main__":
    coeficientes = [2, 1, 1, 2]  # Ejemplo: 2H2 + O2 -> 2H2O
    balancear_reaccion(coeficientes)

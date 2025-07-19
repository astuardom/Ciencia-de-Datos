import math as m

def calcular_descuento(precio, porcentaje):
    def validar_entrada(precio, porcentaje):
        return precio > 0 and 0 <= porcentaje <= 100

    if not validar_entrada(precio, porcentaje):
        print("Error: precio debe ser positivo y el porcentaje entre 0 y 100.")
        return None

    descuento = (1 - (porcentaje / 100)) * precio
    precio_final = m.floor(descuento)
    return precio_final

if __name__ == "__main__":
    precio = 1500
    porcentaje = 20
    resultado = calcular_descuento(precio, porcentaje)
    print(f"Precio final: {resultado}")
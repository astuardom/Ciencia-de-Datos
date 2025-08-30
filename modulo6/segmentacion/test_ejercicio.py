#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de prueba para verificar que el ejercicio de K-Means funcione correctamente
"""

import sys
import os

def test_imports():
    """Prueba que todos los imports funcionen correctamente"""
    try:
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import silhouette_score
        print("✅ Todos los imports funcionan correctamente")
        return True
    except ImportError as e:
        print(f"❌ Error de import: {e}")
        return False

def test_dataset():
    """Prueba que el dataset esté disponible"""
    try:
        dataset_path = "./input/Mall_Customers.csv"
        if os.path.exists(dataset_path):
            print(f"✅ Dataset encontrado en: {dataset_path}")
            return True
        else:
            print(f"❌ Dataset no encontrado en: {dataset_path}")
            return False
    except Exception as e:
        print(f"❌ Error al verificar dataset: {e}")
        return False

def test_syntax():
    """Prueba que el código principal tenga sintaxis correcta"""
    try:
        # Intentar compilar el archivo principal
        with open("KMeans_MallCustomers_Entregable.py", "r", encoding="utf-8") as f:
            code = f.read()
        
        # Compilar el código
        compile(code, "KMeans_MallCustomers_Entregable.py", "exec")
        print("✅ Sintaxis del código principal correcta")
        return True
    except SyntaxError as e:
        print(f"❌ Error de sintaxis: {e}")
        return False
    except Exception as e:
        print(f"❌ Error al verificar sintaxis: {e}")
        return False

def main():
    """Función principal de pruebas"""
    print("=" * 60)
    print("PRUEBAS DEL EJERCICIO K-MEANS")
    print("=" * 60)
    
    tests = [
        ("Imports", test_imports),
        ("Dataset", test_dataset),
        ("Sintaxis", test_syntax)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🔍 Probando: {test_name}")
        if test_func():
            passed += 1
        else:
            print(f"⚠️  {test_name} falló")
    
    print("\n" + "=" * 60)
    print(f"RESULTADO: {passed}/{total} pruebas pasaron")
    
    if passed == total:
        print("🎉 ¡Todo está listo para comenzar el ejercicio!")
        print("\nPróximos pasos:")
        print("1. Ejecuta: python KMeans_MallCustomers_Entregable.py")
        print("2. Comienza a completar los TODOs")
        print("3. ¡Buena suerte con tu segmentación de clientes!")
    else:
        print("⚠️  Corrige los problemas antes de continuar")
    
    print("=" * 60)

if __name__ == "__main__":
    main()

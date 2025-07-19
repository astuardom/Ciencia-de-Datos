import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

# Leer datos originales
url = 'https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv'
tips = pd.read_csv(url)

# Guardar CSV original
tips.to_csv('entrada/tips.csv', index=False)

# Crear versión determinística
tips_det = tips.copy()
tips_det['tip'] = (tips_det['total_bill'] * 0.15).round(2)
tips_det['Experiment'] = 'Deterministico'

# Crear versión aleatoria (igual que original)
tips_ran = tips.copy()
tips_ran['Experiment'] = 'Aleatorio'

# Unir ambos DataFrames
df = pd.concat([tips_det, tips_ran], ignore_index=True)
df.to_csv('entrada/tips_experiment.csv', index=False)

# Probabilidad básica
A = df['tip'] > 5                     # evento A: propinas mayores a 5
B = df['day'].str.lower() == 'sun'   # evento B: día domingo

# P(A), P(B), P(A ∩ B)
P_A = A.mean()            # Probabilidad de A
P_B = B.mean()            # Probabilidad de B
P_A_and_B = (A & B).mean()  # Intersección A ∩ B

print(f"P(A)  = {P_A:.4f}")
print(f"P(B)  = {P_B:.4f}")
print(f"P(A ∩ B) = {P_A_and_B:.4f}")

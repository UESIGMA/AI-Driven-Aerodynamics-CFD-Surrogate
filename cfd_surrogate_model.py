import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import warnings
warnings.filterwarnings('ignore')

# 1. Comprehensive Training Data
veriler = [
    [2, 40, 0,  2.25, 0.228, 0.0089], [2, 40, 3,  2.25, 0.543, 0.0105],
    [2, 40, 6,  2.25, 0.837, 0.0138], [2, 40, 9,  2.25, 1.150, 0.0187],
    [2, 40, 15, 2.25, 1.010, 0.0303], [0, 0,  5, 1.5, 0.548, 0.0150],
    [1, 40, 5, 1.5, 0.655, 0.0162], [2, 40, 5, 1.5, 0.767, 0.0185],
    [4, 40, 5, 1.5, 0.972, 0.0210], [6, 40, 5, 1.5, 1.150, 0.0245],
    [2, 40, 5, 5.0,  0.716, 0.0544], [2, 40, 5, 10.0, 0.814, 0.0501],
    [2, 40, 5, 20.0, 0.830, 0.0490], [2, 40, 5, 30.0, 0.853, 0.0450]
]

# DataFrame setup with English column names
df = pd.DataFrame(veriler, columns=['Camber_%', 'Position_%', 'AoA_Degree', 'Velocity_ms', 'CL', 'CD'])
X = df[['Camber_%', 'Position_%', 'AoA_Degree', 'Velocity_ms']]
y = df[['CL', 'CD']] 

# 2. Multi-Output Neural Network Initialization
model_ysa = make_pipeline(
    StandardScaler(),
    MLPRegressor(hidden_layer_sizes=(64, 64), activation='tanh', solver='lbfgs', max_iter=2000, random_state=42)
)
model_ysa.fit(X, y)

# 3. NEW SCENARIO TEST: NACA 3412 profile prediction
aoa_hassas = np.linspace(0, 15, 100)
yeni_veriler_ysa = pd.DataFrame({'Camber_%': 3, 'Position_%': 40, 'AoA_Degree': aoa_hassas, 'Velocity_ms': 15.0})

tahminler = model_ysa.predict(yeni_veriler_ysa)
tahmin_edilen_CL = tahminler[:, 0]
tahmin_edilen_CD = tahminler[:, 1]
verimlilik = tahmin_edilen_CL / tahmin_edilen_CD

# 4. Professional Dual-Axis Plot (English Labels)
fig, ax1 = plt.subplots(figsize=(10, 6))

color1 = 'tab:red'
ax1.set_xlabel('Angle of Attack (AoA - Degrees)', fontsize=12)
ax1.set_ylabel('Lift Coefficient (CL)', color=color1, fontsize=12, fontweight='bold')
ax1.plot(aoa_hassas, tahmin_edilen_CL, color=color1, linewidth=3, label='Predicted CL')
ax1.tick_params(axis='y', labelcolor=color1)
ax1.grid(True, linestyle='--', alpha=0.5)

ax2 = ax1.twinx()  
color2 = 'tab:blue'
ax2.set_ylabel('Drag Coefficient (CD)', color=color2, fontsize=12, fontweight='bold')
ax2.plot(aoa_hassas, tahmin_edilen_CD, color=color2, linewidth=3, linestyle='--', label='Predicted CD')
ax2.tick_params(axis='y', labelcolor=color2)

max_verim = max(verimlilik)
max_verim_aoa = aoa_hassas[np.argmax(verimlilik)]
max_verim_cl = tahmin_edilen_CL[np.argmax(verimlilik)]

ax1.annotate(f'Maximum Aerodynamic Efficiency\n(AoA: {max_verim_aoa:.1f}°, CL/CD: {max_verim:.1f})', 
             xy=(max_verim_aoa, max_verim_cl), 
             xytext=(max_verim_aoa-6, max_verim_cl-0.2),
             arrowprops=dict(facecolor='darkgreen', shrink=0.05, width=2, headwidth=8), 
             fontsize=11, color='darkgreen', fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="darkgreen", lw=2))

plt.title('AI-Driven CFD: NACA 3412 Aerodynamic Performance Prediction', fontsize=14, fontweight='bold')
fig.tight_layout()
plt.show()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import warnings
warnings.filterwarnings('ignore')

# 1. Eğitim Verilerimiz (Senin analiz raporundaki veriler)
veriler = [
    [2, 40, 0,  2.25, 0.228], [2, 40, 3,  2.25, 0.543], [2, 40, 6,  2.25, 0.837], 
    [2, 40, 9,  2.25, 1.150], [2, 40, 15, 2.25, 1.010],
    [0, 0,  5, 1.5, 0.548], [1, 40, 5, 1.5, 0.655], [2, 40, 5, 1.5, 0.767], 
    [4, 40, 5, 1.5, 0.972], [6, 40, 5, 1.5, 1.150],
    [2, 10, 5, 1.5, 0.860], [2, 20, 5, 1.5, 0.830], [2, 30, 5, 1.5, 0.800], 
    [2, 50, 5, 1.5, 0.730],
    [2, 40, 5, 5.0,  0.716], [2, 40, 5, 10.0, 0.814], [2, 40, 5, 20.0, 0.830], 
    [2, 40, 5, 30.0, 0.853]
]
df = pd.DataFrame(veriler, columns=['Kamburluk_%', 'Pozisyon_%', 'AoA_Derece', 'Hiz_ms', 'CL'])
X = df[['Kamburluk_%', 'Pozisyon_%', 'AoA_Derece', 'Hiz_ms']]
y = df['CL']

# 2. Yapay Sinir Ağı (YSA) Modeli Kurulumu
# Not: YSA'lar birim farklılıklarına duyarlıdır, bu yüzden StandardScaler ile veriyi ölçekliyoruz.
# 'lbfgs' çözücüsü ufak veri setlerinde pürüzsüz fiziksel eğriler bulmak için idealdir.
model_ysa = make_pipeline(
    StandardScaler(),
    MLPRegressor(hidden_layer_sizes=(32, 32), activation='tanh', solver='lbfgs', max_iter=2000, random_state=42)
)

model_ysa.fit(X, y)
print("✅ Yapay Sinir Ağı başarıyla eğitildi!\n")

# 3. Pürüzsüz bir eğri için 0'dan 15'e kadar 100 farklı hücum açısı (0, 0.15, 0.30... 15.0)
aoa_hassas = np.linspace(0, 15, 100)

yeni_veriler_ysa = pd.DataFrame({
    'Kamburluk_%': 3,
    'Pozisyon_%': 40,
    'AoA_Derece': aoa_hassas,
    'Hiz_ms': 15.0
})

# 100 farklı senaryonun CL değerini tek seferde tahmin et
tahmin_edilen_CL_ysa = model_ysa.predict(yeni_veriler_ysa)

# 4. Sonuçları Grafiğe Dökme
plt.figure(figsize=(10, 6))
plt.plot(aoa_hassas, tahmin_edilen_CL_ysa, linestyle='-', color='red', linewidth=2.5)

# Grafiğin görsel ayarları
plt.title('Yapay Sinir Ağı (YSA) ile NACA 3412 Pürüzsüz Aerodinamik Eğrisi', fontsize=14, fontweight='bold')
plt.xlabel('Hücum Açısı (AoA - Derece)', fontsize=12)
plt.ylabel('Kaldırma Katsayısı (CL)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)

# Maksimum Kaldırma (Stall Başlangıcı) noktasını bulup işaretle
max_cl_ysa = max(tahmin_edilen_CL_ysa)
max_cl_aoa_ysa = aoa_hassas[list(tahmin_edilen_CL_ysa).index(max_cl_ysa)]

plt.annotate(f'Stall Bölgesi Girişi\n(Açı: {max_cl_aoa_ysa:.1f}°, CL: {max_cl_ysa:.3f})', 
             xy=(max_cl_aoa_ysa, max_cl_ysa), 
             xytext=(max_cl_aoa_ysa-5, max_cl_ysa-0.1),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=7), 
             fontsize=11, color='darkred', fontweight='bold')

plt.show()

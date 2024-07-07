import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv('Rockpaper.csv')

# Definisikan variabel input dan output
score = ctrl.Antecedent(np.arange(data['Score'].min(), data['Score'].max() + 1, 1), 'score')
classification = ctrl.Consequent(np.arange(0, 11, 1), 'classification')

# Fuzzification untuk variabel score
score.automf(3)

# Membership functions untuk variabel output classification
classification['low'] = fuzz.trimf(classification.universe, [0, 0, 5])
classification['medium'] = fuzz.trimf(classification.universe, [0, 5, 10])
classification['high'] = fuzz.trimf(classification.universe, [5, 10, 10])

# Basis Aturan
# Asumsi aturan-aturan berdasarkan logika domain, dapat disesuaikan sesuai kebutuhan
rule1 = ctrl.Rule(score['poor'], classification['low'])
rule2 = ctrl.Rule(score['average'], classification['medium'])
rule3 = ctrl.Rule(score['good'], classification['high'])

# Sistem Inferensi
fis_ctrl = ctrl.ControlSystem([rule1, rule2, rule3])
fis = ctrl.ControlSystemSimulation(fis_ctrl)

# Penerapan FIS pada dataset
results = []
for index, row in data.iterrows():
    fis.input['score'] = row['Score']
    fis.compute()
    results.append(fis.output['classification'])

data['FIS_classification'] = results

# Visualisasi
plt.figure(figsize=(10, 6))
plt.scatter(data['Score'], data['FIS_classification'], c='blue', label='FIS Classification')
plt.xlabel('Score')
plt.ylabel('FIS Classification')
plt.title('FIS Classification vs Score')
plt.legend()
plt.show()

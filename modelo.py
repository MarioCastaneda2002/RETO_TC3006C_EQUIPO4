import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn

df = pd.read_csv(r'/Users/mariocastaneda/Downloads/archive/501.csv')
df1 = pd.read_csv(r'/Users/mariocastaneda/Downloads/archive/502.csv')
df2 = pd.read_csv(r'/Users/mariocastaneda/Downloads/archive/503.csv')
df3 = pd.read_csv(r'/Users/mariocastaneda/Downloads/archive/504.csv')
df4 = pd.read_csv(r'/Users/mariocastaneda/Downloads/archive/505.csv')
df5 = pd.read_csv(r'/Users/mariocastaneda/Downloads/archive/506.csv')
df6 = pd.read_csv(r'/Users/mariocastaneda/Downloads/archive/507.csv')
df7 = pd.read_csv(r'/Users/mariocastaneda/Downloads/archive/508.csv')
df8 = pd.read_csv(r'/Users/mariocastaneda/Downloads/archive/509.csv')
df9 = pd.read_csv(r'/Users/mariocastaneda/Downloads/archive/510.csv')
df10 = pd.read_csv(r'/Users/mariocastaneda/Downloads/archive/511.csv')
df11 = pd.read_csv(r'/Users/mariocastaneda/Downloads/archive/512.csv')
df12 = pd.read_csv(r'/Users/mariocastaneda/Downloads/archive/513.csv')
df13 = pd.read_csv(r'/Users/mariocastaneda/Downloads/archive/514.csv')
df14 = pd.read_csv(r'/Users/mariocastaneda/Downloads/archive/515.csv')

df = pd.concat([df, df1, df2, df3, df4, df5, df6, df7, df8, df9, df10, df11, df12, df13, df14])

# Submuestreo: # Conteo de clases
conteo_clases = df['label'].value_counts()

muestras_por_clase = 3726

submuestreo = pd.DataFrame(columns=df.columns)

# Itera sobre cada clase y realiza el submuestreo
for clase, cantidad in conteo_clases.items():
    if cantidad > muestras_por_clase:
        subconjunto = df[df['label'] == clase].sample(muestras_por_clase)
    else:
        subconjunto = df[df['label'] == clase]
    submuestreo = pd.concat([submuestreo, subconjunto])

x = submuestreo[['back_x','back_y','back_z','thigh_x','thigh_y','thigh_z']]
y = submuestreo['label']

x_1  =np.array(x)
y_1  =np.array(y)

from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier

mapeo = {1: 'Caminando', 3: 'Arrastrando_pies', 4: 'Subir_escaleras', 5: 'Bajar_escaleras', 6: 'Parado', 7: 'Sentado', 8: 'Acostado'}
mapear = np.vectorize(lambda x: mapeo.get(x, x))
y_nueva = mapear(y_1)

#Modelo Producción
print("----- Production model -----")
clf = AdaBoostClassifier(RandomForestClassifier(bootstrap = True, max_depth = 50, min_samples_leaf = 1, min_samples_split = 2, n_estimators = 500),n_estimators = 200, learning_rate = 0.65)
clf.fit(x_1, y_nueva)

def predecir(var1, var2, var3, var4, var5, var6):
    entrada = [[var1, var2, var3, var4, var5, var6]]
    resultado = clf.predict(entrada)
    return resultado[0]

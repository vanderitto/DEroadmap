import kagglehub
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Pobieranie danych
path = kagglehub.dataset_download("vipullrathod/fish-market")
print("Path to dataset files:", path)
print(os.listdir(path))

csv_path = os.path.join(path, "Fish.csv")
df = pd.read_csv(csv_path)

# Wyswietlanie szczegolow danych
print(df.head())
print(df.describe())

df['Weight'].hist()
plt.show()

df['Height'].hist()
plt.show()

# Wybieram rybe Bream
dfBream = df[df['Species'] == 'Bream']
dfBream_mW, dfBream_stdW = dfBream['Weight'].mean(), dfBream['Weight'].std()
dfBream_mH, dfBream_stdH = dfBream['Height'].mean(), dfBream['Height'].std()


newBreamWeight = np.random.normal(dfBream_mW, dfBream_stdW, 150)
newBreamHeight = np.random.normal(dfBream_mH, dfBream_stdH, 150)

newBreamHeight = np.clip(newBreamHeight, dfBream['Height'].min(), dfBream['Height'].max())
newBreamWeight = np.clip(newBreamWeight, dfBream['Weight'].min(), dfBream['Weight'].max())

# Porównanie na wykresie dla wagi Bream
plt.figure(figsize=(10, 5))
sns.histplot(dfBream['Weight'], bins=20, color='blue', label='Prawdziwe', kde=True, stat='density')
sns.histplot(newBreamWeight, bins=20, color='red', label='Syntetyczne', kde=True, stat='density')

plt.title("Porównanie rozkładu wagi - prawdziwe vs syntetyczne Bream")
plt.legend()
plt.show()




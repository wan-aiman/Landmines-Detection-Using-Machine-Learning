import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

path = "G:\My Drive\degree\intelligence-systems\Python\Mine_Dataset.xls"
df = pd.read_excel(path)

print(df)

samples = df.iloc[:, 0:2]

print(samples)

model = KMeans(n_clusters=5)  # 5 clusters, to compare Mine Type
model.fit(samples)

print(model)

labels = model.predict(samples)

print(labels)

xs = df.iloc[:, 0]  # V
ys = df.iloc[:, 1]  # H
plt.scatter(xs, ys, c=labels)

plt.show()
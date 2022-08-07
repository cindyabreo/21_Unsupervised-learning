import pandas as pd 
import sklearn
import seaborn as sns
from sklearn.cluster import KMeans
import seaborn as sns

#Step 1:
url1 = 'https://raw.githubusercontent.com/4GeeksAcademy/k-means-project-tutorial/main/housing.csv'
df = pd.read_csv(url1, header=0, sep=",")
df.head()
df.shape

#Step 2:
X = df.loc[:, ["MedInc", "Latitude", "Longitude"]]
X.head()

#Step 3:
kmeans = KMeans(n_clusters=6)
X["Cluster"] = kmeans.fit_predict(X)
X["Cluster"] = X["Cluster"].astype("category")
X.head()

sns.relplot(
    x="Longitude", y="Latitude", hue="Cluster", data=X, height=6,
);

kmeans = KMeans(n_clusters=3)
X["Cluster_3"] = kmeans.fit_predict(X)
X["Cluster_3"] = X["Cluster_3"].astype("category")

X.head()


sns.relplot(
    x="Longitude", y="Latitude", hue="Cluster_3", data=X, height=6,
);
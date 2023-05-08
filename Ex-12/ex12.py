import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

x=[4,5,10,4]
y=[24,22,21,21]

data=list(zip(x,y))

plt.scatter(x,y)
plt.show()

#n-init=10 controls the number of times the k-means algorithm will be run
kmeans=KMeans(n_clusters=2,n_init=10)   
kmeans.fit(data)
labels=kmeans.predict(data)

plt.scatter(x,y,c=labels)
plt.show()

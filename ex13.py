import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram,linkage
x=[4,5,10,4]
y=[24,22,21,21]
data=list(zip(x,y))
plt.scatter(x,y)
plt.show()
linkage_data=linkage(data,method='ward',metric='euclidean')
dendrogram(linkage_data)
plt.show()

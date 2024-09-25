import numpy as np
from PIL import Image as im
import matplotlib.pyplot as plt 
import sklearn.datasets as datasets

centers = [[0,0], [2,2], [-3,2], [2,-4]]
X, _ = datasets.make_blobs(n_samples=300, centers = centers, cluster_std =1, random_state=0)

class KMeans():
    def __init__(self, data, k):
        self.data = data
        self.k = k 
        self.assignment = [-1 for _ in range(len(data))]
        self.snaps =[]

    def snap(self, centers):
        TEMPFILE = "temp.png"

        fig, ax = plt.subplots()
        ax.scatter(X[:,0],X[:, 1], c=self.assignment)
        ax.scatter(centers[:,0], centers[:,1], c='r')
        fig.savefig(TEMPFILE)
        plt.close()
        self.snaps.append(im.fromarray(np.asarray(im.open(TEMPFILE))))

    def isunassigned():
        return self.assignment[i] == -1

    def initialize(self, i):
        return self.data[np.random.choice(len(self.data)-1, size = 3, replace=False)]# k points at random from our datasets
    
    def make_clusters(centers): #assign points to nearest center
        for i in range(len(self.assignment)):
            # iterate over every data point and every center and ask which center are you closest to and then assign to that one
            for i in range(len(self.assignment)): 
                for j in range(self.k):
                    if self.isunassigned(i): # if this point has not been assigned give it to first cluster
                        self.assignment[i] = j 
                        dist = self.dist(centers[j], self.data[i]) # compute distance
                    else:
                        new_dist = self.dist(centers[j], self.data[i]) # find new distance after assigning to new center
                        if new_dist < dist: 
                            self.assignment[i] = j # if we found a smaller value we reassign it to new cluster
                            dist = new.dist # so distance is now new distance

    def compute_centers(self):
        centers = [] # empty list
        for i in range(self.k): # i in range number of clusters that we want
            cluster = [] # cluster list 
            for j in range(len(self.assignment)): # for i in range of assignments
                if self.assignment[j] == i: # are you part of that cluster i?
                    cluster.append(self.data[j]) # if you are add to that cluster i
            centers.append(np.mean(np.array(cluster), axis = 0)) # append all clusters to centers
        return np.array(centers)
    
    def unassign(self):
        self.assignment = [-1 for _ in range(len(self.data))]

    def are_diff(self, centers, new_centers):
        for i in range(self.k):
            for j in range(self.k):
                if self.dist(centers[i], new_centers[i])== 0:
                    return True
                else:
                    return False
                
    def dist(self, x, y):
        return sum((x-y)**2) ** (1/2)

    def lloyds(self):
        centers = self.initalize() # initialization: number of centers
        self.make_clusters(centers) # assign points to centers
        new_centers = self.compute_centers() 
        while self.are_diff(centers, new_centers):
            self.unassign() # only if new clusters are different do we unassign
            centers = new_centers
            self.make_clusters(centers) # make clusters from centers
            new_centers = self.compute_centers() # compute centers again
        return 
    
    kmeans = KMEANS(X,4)
    kmeans.lloyds()
    images = kmeans.snaps

    images[0].save(
        'kmeans.gif',
        optimize=False, 
        save_all=True, 
        append_images=images[1:], 
        loop=0,
        duration=500
    )
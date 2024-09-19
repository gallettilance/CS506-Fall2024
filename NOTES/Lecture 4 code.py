import numpy as np
import 

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

    def lloyds(self):
        # initialization: number of centers
        centers = self.initalize()
        # assign points to centers
        self.make_clusters(centers)
        new_centers = self.compute_centers()
        self.unassign()
        while self.are_diff(centers, new_centers):
            centers = new_centers
            self.make_clusters(centers)
            new_centers = self.compute_centers()
            self.unassign()
        return 
    
    kmeans =KMEANS(X,4)
    kmeans.lloyds()
    images = kmeans.snaps

    images[0].save(
        'kmeans.gif'
        optimize=False, 
        save_all=True, 
        append_images=images[1:], 
        loop=0
        duration=500
    )
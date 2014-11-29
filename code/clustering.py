from __future__ import division
from linear_algebra import squared_distance, vector_mean, distance
import math, random
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

class KMeans:
    """performs k-means clustering"""

    def __init__(self, k):
        self.k = k          # number of clusters
        self.means = None   # means of clusters
        
    def classify(self, input):
        """return the index of the cluster closest to the input"""
        return min(range(self.k),
                   key=lambda i: squared_distance(input, self.means[i]))
                   
    def train(self, inputs):
    
        self.means = random.sample(inputs, self.k)
        assignments = None
        
        while True:
            # Find new assignments
            new_assignments = map(self.classify, inputs)

            # If no assignments have changed, we're done.
            if assignments == new_assignments:                
                return

            # Otherwise keep the new assignments,
            assignments = new_assignments    

            for i in range(self.k):
                i_points = [p for p, a in zip(inputs, assignments) if a == i]
                # avoid divide-by-zero if i_points is empty
                if i_points:                                
                    self.means[i] = vector_mean(i_points)    

def squared_clustering_errors(inputs, k):
    """finds the total squared error from k-means clustering the inputs"""
    clusterer = KMeans(k)
    clusterer.train(inputs)
    means = clusterer.means
    assignments = map(clusterer.classify, inputs)
    
    return sum(squared_distance(input,means[cluster])
               for input, cluster in zip(inputs, assignments))

def plot_squared_clustering_errors(plt):

    ks = range(1, len(inputs) + 1)
    errors = [squared_clustering_errors(inputs, k) for k in ks]

    plt.plot(ks, errors)
    plt.xticks(ks)
    plt.xlabel("k")
    plt.ylabel("total squared error")
    plt.show()

#
# using clustering to recolor an image
#

def recolor_image(input_file, k):

    img = mpimg.imread(path_to_png_file)
    pixels = [pixel for row in img for pixel in row]
    clusterer = KMeans(5)
    clusterer.train(pixels) # this might take a while    

    def recolor(pixel):
        cluster = clusterer.classify(pixel) # index of the closest cluster
        return clusterer.means[cluster]     # mean of the closest cluster

    new_img = [[recolor(pixel) for pixel in row]
               for row in img]

    plt.imshow(new_img)
    plt.axis('off')
    plt.show()

#
# hierarchical clustering
#

def cluster_distance(cluster1, cluster2, distance_agg=min):
    """finds the aggregate distance between elements of cluster1
    and elements of cluster2"""
    return distance_agg(distance(input_i, input_j)
                        for input_i in cluster1.members()
                        for input_j in cluster2.members())

class LeafCluster:
    """stores a single input
    it has 'infinite depth' so that we never try to split it"""

    def __init__(self, value):
        self.value = value
        self.depth = float('inf')
        
    def __repr__(self):
        return str(self.value)
        
    def members(self):
        """a LeafCluster has only one member"""
        return [self.value]
                
class MergedCluster:
    """a new cluster that's the result of 'merging' two clusters"""

    def __init__(self, branches, depth):
        self.branches = branches
        self.depth = depth

    def __repr__(self):
        """show as {(depth) child1, child2}"""
        return ("{(" + str(self.depth) + ") " +
                ", ".join(str(b) for b in self.branches) + " }")
        
    def members(self):
        """recursively get members by looking for members of branches"""
        return [member
                for cluster in self.branches
                for member in cluster.members()]


class BottomUpClusterer:

    def __init__(self, distance_agg=min):
        self.agg = distance_agg
        self.clusters = None
        
    def train(self, inputs):
        # start with each input its own cluster
        self.clusters = [LeafCluster(input) for input in inputs]

        while len(self.clusters) > 1:
                    
            # find the two closest clusters
            c1, c2 = min([(cluster1, cluster2)
                          for cluster1 in self.clusters
                          for cluster2 in self.clusters
                          if cluster1 != cluster2],
                         key=lambda (c1, c2): cluster_distance(c1, c2, 
                                                               self.agg))

            merged_cluster = MergedCluster([c1, c2], len(self.clusters))
                                            
            self.clusters = [c for c in self.clusters
                             if c not in [c1, c2]]
                              
            self.clusters.append(merged_cluster)
            
    def get_clusters(self, num_clusters):
        """extract num_clusters clusters from the hierachy"""
        
        clusters = self.clusters[:] # create a copy so we can modify it
        while len(clusters) < num_clusters:
            # choose the least deep cluster
            next_cluster = min(clusters, key=lambda c: c.depth)
            # remove it from the list
            clusters = [c for c in clusters if c != next_cluster]
            # and add its children
            clusters.extend(next_cluster.branches)

        return clusters





if __name__ == "__main__":

    inputs = [[-14,-5],[13,13],[20,23],[-19,-11],[-9,-16],[21,27],[-49,15],[26,13],[-46,5],[-34,-1],[11,15],[-49,0],[-22,-16],[19,28],[-12,-8],[-13,-19],[-41,8],[-11,-6],[-25,-9],[-18,-3]]

    random.seed(0) # so you get the same results as me
    clusterer = KMeans(3)
    clusterer.train(inputs)
    print "3-means:"
    print clusterer.means
    print

    random.seed(0)
    clusterer = KMeans(2)
    clusterer.train(inputs)
    print "2-means:"
    print clusterer.means
    print

    print "errors as a function of k"

    for k in range(1, len(inputs) + 1):
        print k, squared_clustering_errors(inputs, k)
    print


    print "bottom up hierarchical clustering"

    buc = BottomUpClusterer() # or BottomUpClusterer(max) if you like
    buc.train(inputs)
    print buc.clusters[0]

    print
    print "three clusters:"
    for cluster in buc.get_clusters(3):
        print cluster
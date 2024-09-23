# 9/16/2024
## GitHub Review 
- in order to collaborate on a repo without overwhelming with too many changes, you can fork the repo and create a pull request (what we are doing now)
- trying to contribute ot main repository (think of them as other branches):
  1. fork the repository and then clone it 
  2. add remotes -- origin points to your repository; upstream points to the repository you are trying to conribute to
  3. then work in your repo, add, commit --> makes a PR to the priginal repo to request that your changes can be reviewed and approved
- main branch should be stable - not where you should be doing development
## Linear Algebra
- hard to tell what exact factors contribute to a given result 
- need to know under which circumstances a hypothesis is made false (what data points makes it wrong)
- y = f(x) gives us an idea of how the data trends, but there will still be variation 
- confirmation bias -- just because you feel like you see relationship, doesn't necessarily mean that it is there(?)
- working with a finite set of points and asked to find this relationship
- positive example supports hypothesis; negative doesn't 


# 9/18/2024
## Clustering
Types of clustering:
- partitional - each object belongs to exactly one cluster 
- heirarchical - set if best cluster organized in a tree (like phylogenetic)
- density based - defined based on local density of points
- soft clustering - each point is assigned to every cluster with a certain probability

Paritional Clustering: 
- eg. blue and yellow dots in clusters: **variance** is smaller in the left figure (where is not a lot of overlap) and much larger when they is overlap/interspersing 
- 1/cardinality in cluster Ci * Sum (distance(between point and mean)^2)
  - $\frac{1}{|C_i|}*\sum{d(\mu, x_i)^2}$
  - gives you overall evaluation of clustering 
- Cost function: way to evaluate and compare solution (hope to find some algorithm that can find solutions that make the cost smaller)
  - sum from i to k of (sum of for every s in cluster i, the distance between the mean and the value squared)
  - $\sum_{i}^{k}\sum_{x \in C_i}{d(\mu, x)^2}$
- K-means -- when k = 1, then everything in one cluster; if k = n, then everything in its own cluster 
- algorithm -- take a partition and repeatedly adjust byt exchanging points between partitions depending on what it is closer to 
  - Lloyd's Algorithm: 
    - randomly pick k centers 
    - assign each point in the dataset to its closest center
    - compute the new centers as the means of each cluster
    - repeat 2 & 3 until convergence (when the centers no longer change)
- this algorithm is not always optimal: 
  - elongated clusters (dist between less than length of cluster)
  - randomly assigned centers between two and also two in one cluster 
  - outliers really throw off the clusters 

  # 9/23/2024
  
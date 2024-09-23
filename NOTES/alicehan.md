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
  ## Kmeans++
- Will Lloyd's always converge? 
  - there is only a finite number of partitions in the data set, so we will always be able to find the optimal even if by looking at all possible 
  - algorithm get stuck in a cycle/loop. Would only happen if you have 2 overlapping points and a randomized assignment of points to clusters, but our algorithm can spot that these are the same point and therefore is converged. 
  - so.. YES will always converge
- Does it always converge to the optimal solution? 
  - no, like example from last class, there can be clusters close together which might not count an organic cluster as a single cluster, but instead as two. 
  - if we were to pick the farthest away points as the centrs, they woudl just all be outliers as a full cluster
- Instead, **KMEANS++** allows you to combine the randomized centers while also using a probability proportional to the distance squared 
  1. start with a random center
  2. Let D(x) be the distance between x and the closest of the centers picked so far. Choose the next center with probability propoetional to D(x)^2
- The goal is to minimize the cost, but we don't want too many k points 
  - looking for point of diminishing return 
- How do we choose the right k?
  1. iterate through diff values of k (elbow method) 
  2. Use empirical / domain-specific knowledge
  3. Metric for evaulating the clustering 
- want to find way to evaulate the clusters
  - only evaluates if similar data points are in the same cluster, but not if dissimilar datapoints are in diff clusters
  - a = average within-cluster distance 
    - basically radius 
  - b = average intra-cluster distance 
    - distance between centers 
  - if (b-a) = 0, then a and b are overlapping or very close together 
    - we want to maximize b - a 
  - butttt this value doesnt really mean anything so we want it to be a ratio value between 0 and 1
    - (b - a) / max(a, b)
    - if value close to 1, that means that there is really good separation between the two clusters 
- Silhouette Scores: 
  - for each data point i: 
    - $a_i$ = mean distance from point i to every other point in its cluster
    - $b_i$ = smallest mean distance from point i to every point in another cluster 
    - essentially, distance to its neighbors compared to the distance to each of its neighbors 
    - $s_i = (b_i - a_i) / max(a_i, b_i)$ 
  - if each point's silhouette score is close to 1, then we know that this is a good partition 
  - we can then plot the silhouette scores and their average 

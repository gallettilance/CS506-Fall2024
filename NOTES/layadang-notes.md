# Lecture 5 (09/25)
Building a DENDOGRAM with info of which cluster is merging, which shows similarities among groups

## Hierarchical Clustering
### Agglomerative
- Start with every point in its own cluster
- At each step merge the two closest clusters
- Stop when every point is in the same cluster

### Divisive
- Start with every point in the same cluster
- At each step, split until every point is in its own cluster (K-means?)

## Agglomerative
What is a distance between two clusters? From last lecture, we have silhouette scores (distance between points (D), distance between clusters (d)), similarly, we have single-link distance and complete-link distance

### Single-link distance
To determine which cluster is closer, use the two closest points:

- D(C1, C2) = min(d(p1, p2)), where p1 and p2 are points in two different clusters

Handles clusters of different sizes, but noise between the two interferes with min distance

### Complete-link distance
Use the furthest possible distance between the two clusters as possible
- D(C1, C2) = max(d(p1, p2)), where p1 and p2 are points in two different clusters

Helps with noise, but not good with clusters of different sizes

### Average-link distance

Average of all distances between the points of all clusters

### Ward's distance
Most used method

Estimate penalty (variance of new cluster) if we were to merge too clusters, the difference between merged and unmerged is the score

## Density-based clustering

Cluster points tahta are densely packed together (allow for weird-shaped clusters)

**Density**: number of points over given area

Define a **min point** or minimum number of points within an area (**radius**), label as core (in the area), border (touching the border), or noise (outside the border)


**DFS/DBScan**: Start with a core point (has at least min number of other core points), add the neighbors, if neighbor is also core point, then continue


























# Lecture 4 (09/23)
## K-means
Lloyd's algorithm review
* randomly pick k centers
* assign each point to a center
* compute new centers
* repeat

## *Lloyd's algorithm always converges* --> True!

Proof by contradiction: iterating over limited partitions, there are no infinite loops

## *Lloyd's algorithm always picks the optimal solution* --> False!

The starting random centers affect the results, maybe pick points that are as far away as possible from each other. (Farthes First Traversal)

Problem: picking an outlier as a cluster, not random enough (K-means would not have had this happen)

Combine both methods? Randomness but weighted by distance (so further points have a higher chance of getting picked, but its not deterministic) <-- K-means++

### K-means++
Select points where each point has probability weighted by its distance from the last point (favor points that are further away)
Chance of picking any point == proportional to its distance squared

## Finding the correct k
The lowest cost is when k = # of data points

The point of diminishing returns (point in which adding another cluster decreases how much cost went down)

Sometimes, k needs prior knowledge / domain experts

Metric for evaluating a clustering output
* similar in same cluster, dissimilar in different clusters
* variance we expect?

## Silhouette scores
We need clusters to be FAR (inter-cluster differences) and COMPACT (within-cluster difference)

* (a) within cluster difference (compact)
* (b) inter cluster difference (far)

if (b-a) = 0, they're overlapping :( --> so (b-a) should be large

normalize it (s.t. (b-a) is a scale fromm 0 to 1):
* (b-a) / max(a, b)
* closer to 1 means one is so far comapared to the other (closer to perfect scenario)
* closer to 0 means there is an overlapping cluster (not ideal)

Calculate silhouette scores of every single point from each cluster
* look at all of them (especially outliers)
* look at ones that are near 0 and 1
* look at average silohette scores

# Lecture 3 (09/16)
## Cluster K-means
Clustering: grouping data points such that they are similar to each other and dissimilar from others

Clustering can be arbitrary!

Types of clustering
* partitional: each object belongs to exactly one cluster
(goal is to parition dataset into k clusters)
* hierarchial
* density-based
* soft clustering

Mathematical way to evaluate a parition: evaluate validity of a cluster by **sum of variance**
* if sum of smaller, it is a better cluster (points are closer together in the cluster)

## Cost Function
Method to evaluate and compare solutions, find some alg to min cost function

Evaluate partition: sum of all points in each cluster of squared distances from points to its center
* edge cases: k=1 all points are in one cluster, k=n each point is in its cluster

General algorithm idea: continue to adjust cluster and data points and assign data points to minimize cost  

**Lloyd's Algorithm**: 
(1) randomly pick k centers (assign each point in the dataset to its closest center)
(2) assign each point in the dataset 
(3) compute the new centers / true centers as the means of each cluster 
(4) repeat

# Problems
Elongated clusters with naturally high variance means it will be split up, despite beloning together

Can only make circular/bobular cluster shapes because of the mean-centering properties

Dependent on the "random" starting point / inital placement matters / algorithm is not optimal

Easily thrown off by outliers

# Lecture 2 (09/16)

## Additional Git information
**git fork**: one moment in time of a repo

**remote "upstream"**: main repo you forked from
* precedence over your changes

**remote "origin"**: your forked repo
* git push to your origin
* then git pull (request) your updates to upstream

**To sync:**
```
git pull upstream main

git push origin main
```
The main branch is supposed to be stable, not where you do development (needs to be in isolated area)
* always use a new branch!

## Data Science notes

**Confirmation bias**: subset of examples is not representative of the "truth", tendency to look for more positive examples rather than negative

_"all models are wrong but some are useful"_
* we have a finite set of points, but we're finding a continuous relationship 
* data changes over time (monitor that model over time, are we able to refresh and relearn?)
* what if tehre are infinite number of rules?

Hypothesis is a subset of the rule — do we need more positive/negative examples to learn about the rule?

Sometimes things are random, sometimes we just need more information (another feature is needed to find the relationship)

**Data science workflow**: Process data, explore data, extract features, create model

The data is everything, putting it in the right format is important 

Exploring data is important, visualize it, become an expert: if you don't understand it, how are models supposed to?

Extract features you think affects the outcome: transform the data into what you think is useful

At the very end is when you try to create the model.
* a rather small part of data science

## Worksheet questions
### _a) What property must a hypothesis have?_

Be a subset of the rule, or get closer to the rule

### _b) What examples would you have wanted to try?_

(8, 4, 2) does it need to be in ascending order?

### _c) Given the hypothesis (x, 2x, 3x), for each of the following, determine whether they are positive or negative examples:_

- (2, 4, 6)
- (6, 8 , 10)
- (1, 3, 5)

Positive, negative, negative

### _d) Describe steps of a Data Science Workflow_

Process the data

Explore the data: process again if necessary

Extract features: which ones seem relevant to the task? process and explore again if necessary

Create model: evaluate the model and see which steps to redo

### _e) Give a real world example for each of the following data types:_

- record: (name, age, balance)
- graph: supply + demand
- image: a big matrix of pixels
- text: list of words

### _f) Give a real world example of unsupervised learning_

Given a bunch of mass + size for various coins (unlabeled types), see the clusters and determine how many types of coins are there

What structures are there that are naturally occuring? Clusters?

### _g) Give a real world example of supervised learning_

Given a bunch of mass + size for various coins (labeled types: dimes, quarters, pennies, etc), see the clusters and predict coin type given mass and size


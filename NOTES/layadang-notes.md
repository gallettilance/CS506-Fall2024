# Lecture 13 (10/23)

## Midterm Launch

Two parts: 3-page write-up about findings and reproducible notebook of final Kaggle submission (70% of grade)
- The other 30% is Kaggle results

Key notes:
* Anything is fine except for deep learning! 
* Really, submitting twice is enough "if you're good at data science"
    * Limit is 5 submissions a day
* Plots are part of the 3-page limit, no appendix
* COMPETITION ENDS AT 4:30 PM MONDAY !
* Monetary prizes, you can get extra prizes if you beat the TAs' scores

Do:
* NOT cheat by finding the average rating of the movie (you are including the review in the average)
* NOT append data from outside, only use the training data
* Clean the data and check for NaNs
* Make lots of new features
* Keep track of all models


### Lecture today is making a Kaggle submission w/ starter code
#### (No attendance)


# Lecture 12 (10/21)
## Support Vector Machines (SVMs)
Find the widest decision boundary that separates the data set ("widest street")
- the two sides' equations are arbitrary (+1 / -1, etc.)
- equation of linear decision boundary (w^Tx+b=0)

Multiplying by a constant does not make the decision boundary change (c * w^Tx+ c * b=0)
- But the width of the street changes
- Inversely proportional to magnitude of w

The width of the street is inversely proportional to magnitude of w

How do we determine a good line? Move the line towards direction of misclassified point

None of the dataset points should be in the street

## Perceptron's Algorithm
1. Start with random lninne
2. Define: 
    - a total number of iterations
    - a learning rate (a)
    - an expanding rate (c<1)
3. Repeat for each iteration:
    - pick a point from dataset
    - if correct, do nothing, else:
        - Adjust w1 by adding (yi * a * x1), w2 by adding (yi * a * x2), and b by adding (yi * a)
    - expand or retract by width of c

 
# Lecture 11 (10/16)

**conditional probability**: changing the context of a probability

**Bayesian classifiers**: P(C | A1 and A2 and...) conditional probability based on class (C) and features (As)
- maximize this by estimating P(A1 and A2 and...| C)

## Model Evaluation
how do we know we have a good model?

**confusion and cost matrix**: award points for correct prediction, penalize for incorrect (scale by however much needed)
- determined with the help of experts/context, measureable somehow?

## Ensemble Methods

17 classifers are trained 
- assume independence
- all equal error rate, 0.2

majority needs to be wrong (at least 9 out of 17 are wrong), so our probability of error is low
- 0.0025 accuracy is much lower when we combine these accuracies 

### Bootstrapping
**baggging** classifer: automatically generate classifiers based on samples (sample with replacement to generate datasets)
- computationally expensive

**boosting**: change classifiers based on how well they can predict certain examples
- expose it to ideas that it doesn't work with / bad examples

Then, give each classifier a particular weight. (AdaBoost)


# Lecture 10 (10/15)
From last lecture: intro to classification, relationship between features and class (category)

**Instance-based classifiers**: if the instance appears in the dataset, just match the output, 

What if it is not in dataset? If it is close (some distance, has some neighbor, k-closest one), then take that output!
- "points that are further away has lower power"

--> **K-closest Neighbor Model**: output/decision determined by nearby neighbor of the point, has a decision boundary

## Decision Trees

Start at root of a tree, look at value, traverse corresponding branch, etc. (pretty intuitive)

What if a combination of features never appeared in the dataset? Define some default class

**Hunt's algorithm**: repeatedly split dataset based on attribute values (recursive)
- Base case:
    - if split and all data points in the same class, predict that class 
        - So all REFUND=YES, then CLASS=NO
    - if split and no datapoints, predict reasonable default 
        - So REFUND=NO, different classes, split MARTIAL STATUS 
        - MARTIAL STATUS=MARRIED, all classes are NO
        - etc.

Order of features to splits matter, how do we define that order? 
- find one with more "even" split rather than ones that are not even
- alternatively, do bins until they're pretty even
- binary split or multi-split? 

For **continuous variables**: use binning before running (pre-processing) OR compute threshold while running (A>t or A<t, scan for best threshold)

**GINI Index**: p(j | t) as relative frequency of class *j* for node *t*
- ideal GINI is 0 (data is perfectly separated), worse is (1/2) (50% separated, no valuable info)
- Get GINI of the ENTIRE node recursively

**over-fitting**: too specific to our training set. will not apply to the test data
- if we apply GINI continuously, too many splts
- fix: early termination before tree is fully grown or prune the full tree

Other metrics for node purity:
- **Entropy** -sum of freq * log(freq)
- **Misclassification Error** 1-max(freq)


# Lecture 7 (10/02)
Large number of features in dataset that determines a result, needs to be independent of each other

**Rank of matrix / degrees of freedom**: how much information is actually present in the matrix, non-redundancy 

Ideally, isolate effect of each feature 

## Singular Value Decomposition
Transfer set of features into something that is not related / lowering the rank (remove any relationships and make them **linearly independent**)

**Dimension reduction** through projection onto any of the features, also called feature extraction

n x m: n data points for m features 

Full-rank matrices: max information stored in matrix, so rank is either n or m

## Matrix Factorization
Or matrix decomposition, where A=UV where U is *n x k* and V is *k x m* which is less to store by just A by itself (k(m+n) instead of (m x n)).

*k* being small means there is a lot of redudant information, not a lot of features

Benefit of low rank matrix, but cost of lower resolution 

### How do we find A=UV?

**Adding the factorization aspect:** Not just $A=UV$ but also $A=U\Sigma V^T$
- U is n x r
- V is m x r
- $\Sigma$ is diagonal matrix that is r x r

Approximate $A$ with $A^k$ as minimizing pairwise distances 

**Frobenius Distance**: the pairwise sum of squares difference in values of two matrices

Singular values are ORDERED by weight, also represents importance

**Mean centering** ensures values are properly scaled / on the same scale
- Ex. Age is in tens and income is in thousands, in different scales, so it is hard to see relationship on just plot, so they need to be scaled around the mean
- need to normalize data in some way

## Anomaly Detection


# Lecture 6 (09/30)
Clusters from a probabilistic perspective

Problem statement: given *k* species and *n* number of different weights, can we figure out which species belong to which group of weights?

P(S_j | X_j) probability of belonging to S_j species given weight X_j

Use Baye's rule:
- P(S_j) prior probability of seeing S_j in general
- P(X_j | S_j) the PDF (probability) of weight X_j given its species (seeing a Sauropod that weighs 100 tons is way more likely than seeing a Raptor that weighs 100 tons)
- P(X_j) probability of any species being a weight, so it is combined weight distributions (weighted to number of species) 

## Mixture Model
X comes from a mixture model with k mixture components if the probability distribution of X is 
P(X) = summation (of j data points) of P_j (mixture component) * P(X| S_j)

### Gaussian Mixture Model
P(X) is normally distributed, defined by mean and variance

## Maximum Likelihood Estimation 
Find *p* that maximizes the chance of seen data sequence happening (a naive approach)

Find critical points by transforming function to preserve critical points (not changing variance)


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

Cluster points that are densely packed together (allow for weird-shaped clusters)

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

## Problems
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


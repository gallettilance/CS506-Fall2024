## 9/11 notes
Git
A tool to help us manage the timeline(s) of a project (also called repository)
Formally called a version control system or source control management
Fundamental Workflow
As we change the project over time
Create save points (called commits) that track the timeline of the project’s evolution

Creates a timeline of your code/files

Demo
-mkdir git-demo
- cd git-demo
- git init //initializes the git repository
- ls -a //shows . .. .git
- if you do rm -r .git you are no longer a git repository
- can add a README.md file into the folder of git-demo
- when doing ls you will get README.md
- if you do git status it says the untracked files and stuff
- git add README.md or git add . will add everything in the current directory
- git commit -m “message”
- git log (to look at commit messages, to exit press Q)
- git diff (shows you the differences you made in the commit)
- git log (shows you the log)
- git checkout [with specific id] //goes back in time in the repo
- git checkout main //goes back to the current state
GitHub vs Git
- Git → [terminal] a version control system
- GitHub → [browser] a website to backup and host the timeline(s) of your project
Fundamental Workflow
- Create save points (called commits)
- Push the updates to GitHub (from your laptop) to back up your work

Initialize a repository
git init
git add <files>
git commit -m “some message”

Adding a remote that points to GitHub
git remote add origin <link>



DEMO
git remote add origin [e.x. git@github.com:gallettilance/git-demo.git
git remote -v //shows the origin fetch and push


Motivation
For each project (repository) I own, I want to write code where:
1. Iterating on (+keeping track of) different versions of the code is easy
2. Work is backed up to and hosted on the cloud
3. Collaboration is productive
Iterating on different versions
The ease or difficulty of adding a new feature to the codebase might depend on the state/version of the codebase
It may be easiest to add this feature at a specific commit



This won’t work
Another way: (will branch off that particular commit to create a new timeline)
Can push commits per branch

Can create a lot of branches
But one branch typically needs to be chosen as the primary, stable branch
This branch is typically called the “main” branch
At some point we will want to clean up certain branches by merging them with the master/main branch or with each other
Merging is trivial if the base of one branch is the head of the other, the changes are “simply” appended
When this is not the case, commits can conflict with each other

We need to change the base of the login-page branch (rebase) to be at the head of the master branch


This is not a simple operation! It will often require manual intervention to resolve the conflicts


## 9/16 notes
Each of the commits should be somewhat meaningful
The more organized/clear the commits, the easier to look through the history and figure out what was going on
Can always git clone a repo to get a fresh copy of your work

When working with multiple people on a repo:
Create a copy/branch, work on whatever you need to, then submit a pull request onto the main
After a vetting process, you either get to merge your code or you don't
Repository as branches

GitHub main repository -(fork)-> GitHub my repository -(git clone)-> my repository on laptop (add+commit)
If new changes are made to the main repository you won't get access to them unless you specifically specify that you want to get those changes
Remote "ORIGIN" is my repository that was forked off
Remote "UPSTREAM" is the main repository on GitHub
Git push ORIGIN -> pushes to my repo

In GitHub you need to visualize the timelines
Different repos as different branches part of the same whole (like with the above example)
There is upstream/main on the main repo, origin/main on the github site my repo, and main on my repo on the laptop
If the upstream/main updates after you have forked, you can do the following to update your copy:
- Git pull upstream main
- Git push origin main (pushing to your personal repo on the Github website)

Best practices:
Never really commit to the main branch (its supposed to be special and stable, not where you do development) -> want to do development in an isolated area

Good first issue tag are good first issue things on repos (like on the tensorflow repo)
(END OF GIT SLIDES)

Data science is hard (likely an impossible task most of the time)
Sometimes not clear if you've found all the factors that influence something
Sometimes impossible to quantify/capture everything; might be inherent randomness to things
Very rare to find things that are perfectly related to each other

Example of linear relationship between x and y (y=f(x)) -> Pressure and temperature
- if there is all of a sudden one singular outlier, your hypothesis y=f(x) (perfectly linear relationship) is now completely wrong

Can create models that are somewhat useful
- EX: y=f(x) could give us a general guide on how x and y vary

Confirmation Bias
EX: think of a rule that governs triples, and then you have to guess what the rule is (given yes-or-no answers to questions you ask).
- announce "(2,4,6) follows the rule"
- examples submitted by participants: (2,4,3) -> NO; (6,8,10) -> YES; (1,3,5) -> YES
- after submitting these examples, they wrote their hypothesized rule. would you have wanted to try more examples? if so, which and for what reason?
- poll: A. (100,102,104) B. (5,7,9) C. (1,2,3)
challenges of data science
- a set of exdamples may not always be representative of the underlying rule
- there may be infinitely many rules that match the examples provided
- rules and/or examples may change over time

data science is about doing the best with the data that you have a lot of the time, and sometimes the data will be biased in one way
data science is very difficult! all models are wrong but some are useful

positive examples vs negative examples
- assuming the hypothesis h is (x, x+2, x+4) which type of examples are the following?
- (2,4,3) -> negative
- (6,8,10) -> positive
- (1,3,5) -> positive

if you just try positive examples then you will only get positive responses and won't be aware of possible constraints

data science workflow (simplified)
process data -> explore data -(either loop back or continue) -> extract features (loop back or continue) -> create model (loop back)

use common sense first, think about if there's actually an relationship between the things i'm looking at
putting the data in the right format and quantifying it properly is important
modeling part is a very small part of data science --> get the biggest bang for your buck in the first couple of steps (process data and explore data and extract features)
can visualize the relationships between features (if i change the predictor does it also change the outcome? --> if you change the predictor and there's no change that means there is no relationship)

types of data - records
- m-dimensional points/vectors; EX: (name, age, balance) -> ("John", 20, 100)
types of data - graphs
- nodes connected by edges; EX: triangle with nodes and edges that we can represent as an adjacency matrix (says is there an edge btwn the two nodes) or an adjacency list
types of data - images
- image that can be split up into a collection of pixels (usually a triple of red, green, blue intensities)
types of data - text
- list of words that we can extract more info from; like with a corpus of documents and figure out what words are important and which are not

## 9/18 notes
on clustering - kmeans

what is it about the data that is interesting and what are patterns that naturally exist in the data

clustering is about finding a partition of our dataset such that points are together in groups and those points in groups are similar to each other and those not in the same group are dissimilar to each other
not really clear usually what the "right answer" is --> many of the cluster results/solutions could be right

types of clusterings
- partitional - each object belongs to exactly one cluster
- hierarchical - set of nested clusters organized in a tree
- density-based - defined basd on the local density of points
- soft clustering - each point is assigned to every cluster with a certain probability

partitional clustering
- spread of variance around the mean is smaller on the left than on the right with the example
- mathematical way of evaluating the parition (one cluster better than the other because sum of the variances is smaller in that better cluster)
- $\sum_{i}^{k} \sum_{x \in C_i} d(x,\mu_i)^2$  <- cost function. way to evaluate and compare solutions. hope: can find some algorithm that finds solutions that makes the cost small. sum of the variance of the particular cluster $C_i$. outside sum is over ALL clusters. cardinality $\frac{1}{|C_i|}$ is omitted. $\mu_i$ is the mean of cluster i.
- if we have a button that can generate a partition we now have a mathematical formula that can evaluate if this is a good or bad partition (with the cost function)

k-means
- given $X = x_1, ..., x_n$ our dataset d, the euclidean distance, and k
- find k centers $\mu_1, ..., \mu_k$ that minimizes the cost function: $\sum_{i}^{k} \sum_{x \in C_i} d(x,\mu_i)^2$
- if k=1 this is easy bc just 1 cluster. k=n is easy bc every point is in its own cluster
- when $x_i$ lives in more than 2 dimensions this is very difficult

- EX: given a partition, can give points to one cluster or the other depending on which cluster its closer to and then eventually have clusters of points that dont really overlap too much anymore

could start with a random set of centroids (or middles) to clusters and then start assigning points to the closest centroids which will generate partitions. 

k-means - llyod's algorithm
- 1. randomly pick k centers $\mu_1, ..., \mu_k$
- 2. assign each point in the dataset to its closest center
- 3. compute the new centers as the means of each cluster
- 4. repeat 2 & 3 until convergence

 where does the cost function come into play for k-means? -> each time we change the centers and moving the points we are making the spreads smaller and therefore the variance smaller and we are minimizing the cost function

in-class slides questions
1. is this a possible output of kmeans? -> yes. this is one of the best outputs you can have for k=4 in this case (chosing k)
2. is this a possible output of kmeans? -> no. there are points that would need to be a part of the blue to minimize the spreads of the clusters. if the clusters were farther apart the answer would be potentially yes. (elongated)
3. is this a possible output of kmeans? -> no. the clusters are not typical circular shape and k-mean cannot make that shape. (non-globular)
4. is this a possible output of kmeans? -> yes. the initial placement of the k-centers matters and the algorithm is NOT optimal so we won't find the best solution everytime. (not optimal solution)
5. is this a possible output of kmeans? -> yes. the outliers really throw off the algorithm so if you happen to be unlucky enough to pick centroids that are outliers you can have a result like this. (outliers)


## 9/23 notes
k-means - llyod's algorithm
- 1. randomly pick k centers $\mu_1, ..., \mu_k$
- 2. assign each point in the dataset to its closest center
- 3. compute the new centers as the means of each cluster
- 4. repeat 2 & 3 until convergence

k-means lloyd's algorithm questions
1. will this algorithm always converge? -> yes
- proof (by contradiction): suppose it doesn't converge. then, either 1 the minimum oft he cost function is only reached in the limit (i.e. after an infinite number of iterations) which is impossible because we are iterating over a finite set of partitions. 2 the algorithm gets stuck in a cycle/loop this isn't possible because  this would require having a clustering that has a lower cost...
2. will this always converge to the optimal solution? -> no (how can we make this better? --> use farthest first traversal, first point is random and the rest are as far as possible from the other ones that are picked)
3. the black box returns "12" as the random number generated. which point do we choose for the next center (x,y, or z)? -> x
4. the black box returns "4" as the random number generated. which points do we choose for the next center (x,y, or z)? -> y
5. given the examples of the sihlouette scores, with 200 data points on the y-axis, x-axis with the scores, and the red dotted line as the average. looking for the silhouette score where the average is high and most clusters have points beyond that average. which two would you completely rule out? -> 3 and 4

farthest first traversal doesn't always work b/c like in one example we could have far outliers

k-means++
- initialize with a combination of the two methods (points that are farther have a higher chance of being picked but not for sure)
- 1. start with a random center
- 2. let D(x) be the distance between x and the closest of the centers picked so far. choose the next center with probability proportional to $D(x)^2$

suppose we are given a black box that will generate uniform random number between 0 and any N. how do we use this black box to select points with probability proportional to $D(x)^2$?
- $D(x)^2 = 3^2 = 9$, $D(y)^2 = 2^2 = 4$, $D(z)^2 = 1^2 = 1$
- x has a high chance of being picked
- N = $D(x)^2 + D(y)^2 + D(z)^2 = 14$

k-means/kmeans++
- doesn't do good in different density clusters, non globular clusters, sparse clusters, etc.

how to choose the right k?
1. iterate through different values of k (elbow method, elbow is the point of diminishing returns)
2. use empirical/domain-specific knowledge (example: if there a known approximate distribution of the data? (k-means is good for spherical gaussians))
3. metric for evlauating a clustering output

evaluation
- recall our goal: find a clustering such that 
- similar data points are in the same cluster (kmeans does do this)
- dissimilar data points are in different clusters (kmeans doesnt really evaluate this)

k-means cost function tells us the within-cluster distances between points will be small overall
- but what about intra-cluster distance? are the clusters we created far? how far? relative to what?

discussion -> define a metric that evaluates how spread out the clusters are from one another -> distance between the centroids, average distance (between pairwise distances)

example
- a: average within-cluster distance (spread of the clustering)
- b: average intra-cluster distance (how far the two clusters are from each other)
- what does it mean for (b-a) to be 0? -> they're mixed up together/overlapping, very close to each other
- what does it mean for (b-a) to be large? -> they're spread out
- the value of (b-a) doesn't mean much by itself. can we compare it to something so that the ratio becomes a value between 0 and 1? 
- (b-a) / max(a,b)
- what does it mean for (b-a)/max(a,b) to be close to 1? -> almost perfect separation
- what does it mean for (b-a)/max(a,b) to be close to 0? -> that means that (b-a) is probably 0 so we probably have overlapping clusters

silhouette scores
- for each data point i:
- $a_i$: mean distance from point i to every other point in its cluster
- $b_i$: smallest mean distance from point i to every point in another cluster
- $s_i = (b_i - a_i) / max(a_i, b_i)$, if all of them are close to 1 that means we have done a good job of clustering
- silhouette score plot
- OR return the mean $s_i$ over the entire dataset as a measure of the goodness of fit

k-means variations
1. k-means (uses the $L_1$ norm/manhattan distance)
2. k-medoids (any distance function + the centers must be in the dataset)
3. weighted k-means (each point has a differentw weight when computing the mean)

## 9/25 notes
hierarchical cluistering
- at every step, we record which clusters were merged in order to produce a dendrogram
- we can "cut" the dendrogram at any threshold to produce any number of clusters

two types of hierarchical clustering
- agglomerative:
1. start with every point in its own cluster
2. at each step, merge the two closest clusters
3. stop when every point is in the same cluster
- divisive:
1. start with every point in the same cluster
2. at each step, split until every point is in its own cluster

agglomerative clustering algorithm
1. let each point in the dataset be its own cluster
2. compute the distance between all pairs of clusters
3. merge the two closest clusters
4. repeat 3 & 4 until all points are in the same cluster

can we implement this? are we missing anything?

how would you define distance between clusters?

hierarchical clustering - distance functions
- lets first define:
- distance between points: $d(p_1,p_2)$
- distance between clusters: $D(C_1,C_2)$

single-link distance
- is the minimum of all pairwise distances between a point from one cluster and a point from the other cluster
- $D_{SL}(C_1,C_2) = min(d(p_1,p_2) | p_1 \in C_1, p_2 \in C_2)$
- depends on choice of d
- can handle clusters of different sizes
- but... sensitive to noise points, tends to create elongated clusters

Q. is C or D closer to {A,B}?
- need to define 3 distances: the distance between A,B, and D (defined by (A,D)=2) and between A,B, and C (defined by (B,C)=sqrt(5)).
- sqrt(5) is longer than 2, so we should merge D into cluster {A,B}

complete-link distance
- is the maximum of all pairwise distances between a point from one cluster and a point from the other cluster.
- $D_{CL}(C_1,C_2) = max(d(p_1,p_2) | p_1 \in C_1, p_2 \in C_2)$
- less susceptible to noise
- creates more balanced (equal diameter) clusters
- but... tends to split up large clusters. all clusters tend to have the same diameter

Q. is C or D closer to {A,B}?
- distance betweeen A,B,D defined by B,D distance which is sqrt(10) and distance between A,B,C is defined by A,C distance which is 3
- we merge C (because its the smallest distance)

average-link distance
- is the average of all pairwise distances between a point from one cluster and a point from the other cluster
- $D_{AL}(C_1,C_2) = \frac{1}{|C_1| \dot |C_2|} \sum_{p_1 \in C_1, p_2 \in C_2} d(p_1,p_2)$
- less susceptible to noise and outliers
- but... tends to be biased towards globular clusters

Q. is C or D closer to {A,B}?
(didn't calculate in-class)

centroid distance
- the distance between the centroids of clusters
- $D_C(C_1,C_2) = d(\mu_1, \mu_2)$

Q. is C or D closer to {A,B}?
(didn't calculate in-class)

ward's distance
- is the difference between the spread/variance of points in the merged cluster and the unmerged clusters
- $D_{WD}(C_1,C_2) = \sum_{p \in C_{12}} d(p,\mu_{12}) - \sum_{p \in C_{1}} d(p_1,\mu_{1}) - \sum_{p \in C_{2}} d(p_2,\mu_{2})$
- whats the penalty in terms of spread if we merge the clusters
- we dont want a high variance with the clusters

Q. is C or D closer to {A,B}?
(didn't calculate in-class)

example with distance matrix, where d = Euclidean, D = Single-Link
- matrix has all the distances between the different points
- choose the smallest distance and merge those two as a cluster
- now A and B are together, and we update the distances using the linked distances with the other points C and D
- choose the two closest clusters -> merge A,B cluster with D
- update matrix again with the A,B,D and merge together with point C
- if you have a higher distance you're willing to accept for clusters, that means you have a lot of clusters merged together and vice versa

hierarchical clustering
- finding the threshold with which to cut the dendrogram requires exploration and tuning
- but in general hierarchical clustering is used to expose a hierarchy in the data (EX: finding/defining species via DNA similarity)
- to capture the difference between clustering you can use a cost function, or methods that we'll discuss later when looking at clustering aggregation

density-based clustering
- goal: cluster together points that are densely packed together
- how should we define density? number of points in a given area
- given a fixed radius $\epsilon$ around a point, if there are at least min_pts number of points in that area, then this area is dense

example
- min_pts = 3
- $\epsilon$-neighborhood of this point is within that radius
- if more than three points we classify as dense

we need to distinguish between points at the core of a dense region and points at the border of a dense region
- let's define:
- core point: if its $\epsilon$-neighborhood contains at least min_pts
- border point: if its in the $\epsilon$-neighborhood of a core point
- noise point: if its neither a core nor border point

DBScan Algorithm
- $\epsilon$ and min_pts given:
1. find the $\epsilon$-neighborhood of each point
2. label the point as core if it contains at least min_pts
3. for each core point, assign to the same cluster all core points in its neighborhood (crux of the algorithm)
4. label points in its neighborhood that are not core as border
5. label points as noise if they are neither core nor border
6. assign border points to nearby clusters

DBScan visualized
- iterate through the dataset
- if core point - iterate through its neighborhood to find more core points that should also be part of the cluster
- go to next point in the dataset
- iterate over its neighborhood since its a core point
- found another core point so we need to iterate over its neighborhood too


DBScan benefits
1. can identify clusters of different shapes and sizes
2. resistant to noise

DBScan limitations
1. can fail to identify clusters of varying densities
2. tends to create clusters of the same density
3. notion of density is problematic in high-dimensional spaces

## 9/30 notes
dino example
- have k different species in the park, weighing them
- trying to find which species corresponds to which weight

problem statement
- given a dataset of weights smapled from N different animals
- can we determine which weight belongs to which animal?

output
- makes more senes to provide, for each data point (weight) the probability that it came from each species
- $P(S_j|X_i)$
- where $S_j$ is species j and $X_i$ is the ith weight in the dataset

things to consider
1. there is a prior probability of being one species (i.e. we could have an imbalanced dataset or there oculd just be more of one species than the other)
    - some dinosaurs are more common than others: for example there are many more Stegosauruses than Raptors in the park. this means a given data point, knowing nothing about it would just have a higher chance of being a Stegosaurus than a Raptor
2. weights vary differently depending on the species (i.e. each species could have a different weight distribution)

how to compute $P(S_j|X_i)$?
- $P(S_j|X_i) = \frac{P(X_i|S_j)P(S_j)}{P(X_i)}$
- $P(S_j)$ is the prior probability of seeing species $S_j$ (that probability would be higher for Stegosauruses than the Raptors for example)
- $P(X_i|S_j)$ is the PDF of species $S_j$ weights evaluated at weight $X_i$ (seeing a Sauropod that weighs 100 tons is way more likely than seeing a Raptor that weights 100 tons)

what about $P(X_i)$?
- $P(X_i) = \sum_j P(S_j)P(X_i|S_j)$

mixture model
- X comes from a mixture model with k mixture components if the probability distribution of X is:
- $P(X) = \sum_j P(S_j)P(X|S_j)$
    - $P(S_j)$ is the mixture proportion that represents the probability of belonging to $S_j$
    - $P(X|X_j)$ is the probability of seeing x when sampling from $S_j$

gaussian mixture model
- a gaussian mixture model (GMM) is a mixture model where $P(X|S_j) ~ N(\mu, \sigma)$

maximum likelihood estimation (intuition)
- suppose you are given a dataset of coin tosses and are asked to estimate the parameters that characterize that distribution - how would you do that?
- MLE: find the parameters that maximized the probability of having seen the data we got
- example: assume Bernoulli(p) iid coin tosses. goal: find p that maximized that probability
    - P(having seen the data we saw) = P(H)P(T)P(T)P(H)P(T) = $p^2(1-p)^3$
    - the sample proportion $2/5$ is what maximizes this probability

GMM clustering
- goal: find the GMM that maximizes the probability of seeing the data we have
- recall: $P(X_i) = \sum_j P(S_j)P(X_i|S_j)$
- finding the GMM means finding the parameters that uniquely characterize it. what are these parameters?
    - $P(S_j)$ & $\mu_j$ & $\sigma_j$ for all k components
    - lets call $\theta = (\mu_1,..., \mu_k, \sigma_1, ...., \sigma_k, P(S_1), ..., P(S_k))$
- the probability of seeing the data we saw is (assuming each data point was sampled independently) the product of the probabilities of observing each data point
- goal: $\Pi_i P(X_i) = \Pi_i \sum_j P(S_j)P(X_i|S_j)$
- how do we find the critical points of this function?
    - notice: taking the log-transform does not change the critical points
    - define:
        - log($\Pi_i \sum_j P(S_j)P(X_i|S_j)$) = $\sum_i$ log ($\sum_j P(S_j)P(X_i|S_j)$)
- to get:
    - $\hat{\mu_j} = \frac{\sum_i P(S_j|X_i)X_i}{\sum_i P(S_j|X_i)}$
    - $\hat{\sum_j} = \frac{\sum_i P(S_j|X_i)(X_i - \hat{\mu_j})^T(X_i - \hat{\mu_j})}{\sum_i P(S_j|X_i)}$
    - $\hat{P}(S_j) = \frac{1}{N} \sum_i P(S_j|X_i)$

expectation maximization algorithm
1. start with random $\mu, \sum, P(S_j)$
2. compute $P(S_j|X_i)$ for all $X_i$ using $\mu, \sum, P(S_j)$
3. compute/update $\mu, \sum, P(S_j)$ from $P(S_j|X_i)$
4. repeat 2 and 3 until convergence

## 10/2 notes
characteristics of a dataset to look for
- all information contained in a is contained in b and vice versa (linear relationship); dimension of this dataset is 2, but there's only 1 dimensional information because there's redunancy. we call that the rank/span. not a desirable relationship between a and b.
- second graph is more realistic, but we see that changing a changes b so we can't isolate. not a desirable relationship between a and b.
- third graph we have that as a changes b does not change likely.

SVD a tool to transforme dataset from a set of features that are related to ones that are not linearly related


left graph approximates the data the best (with the straight line of red dots)

goal
- examine this matrix and uncover its linear algebraic properties to:
    - approximate A with a smaller matrix B that is easier to store but contains similar information as A
    - dimensionality reduction / feature extraction
    - anomaly detection and denoising

linear algebra reviews
- definition: the vectors in a set $V = {v_1, v_2, ..., v_n}$ are linearly independent if:
    - $a_1v_1 + ... + a_nv_n = o$
    - can only be satisifed by $a_i = 0$
    - note: this means no vector in that set can be expressed as a linear combination of other vectors in the set
- definition: the determinant of a square matrix A is a scalar value that encodes properties about the linear mapping described by A.
    - 2x2: det(A) = ad - bc
    - 3x3: det(A) = $a*det(e,f,h,i) - b*det(d,f,g,i) + c*det(d,e,g,h)$
- defenition: of a square matrix A is a scalar value that encodes properties about the linear mapping described by A. n x n can recursively compute it. how?
- property: n vectors ${v_1, ..., v_n}$ in an n-dimensional space are linearly independent iff the matrix A: $A = [v_1, ..., v_n] (n x n)$ has a non-zero determinant.
    - Q. can m > n vectors in an n-dimensional space be linearly independent?
- definition: the rank of a matrix A is the dimension of the vector space spanned by its column space. this is equivalent to the maximal number of linearly independent columns/rows of A. 
- definition: a matrix A is full-rank iff rank(A) = min(m,n)
- note: get the rank of a matrix through the Gram-Schmidt process

matrix factorization
- any matrix A of rank k can be factored as A = UV where U = n x k and V is k x m
- to store an n x m matrix A requires storing $m \dot n$ values. however, if the rank of the matrix of A is k, since A can be factored as A=UV which requires storing k(m+n) values

in practice
- most datasets are full rank despite containing a lot of redunant / similar information
- but we might be able to approximate the dataset with a lower rank one that contains similar information

approximation
- goal:
    - approxiamte A with $A^{(k)}$ (low-rank matrix) such that
    1. $d(A, A^{(k)})$ is small
    2. k is small compared to m & n

frobenius distance
- $d_F(A,B) = ||A-B||_F= \sqrt{\sum_{i,j}(a_{ij}-b_{ij})^2}$ i.e. the pairwise sum of squares difference in values of A and B

approximation
- definition: when k < rank(A), the rank-k approximation of A (in the least squares sense) is $A^{(k)}=argmin_{(B|rank(B)=k)} d_F(A,B)$

matrix factorization improved
- not only can we factorize a matrix A of rank k as A = UV. but we can factorize A using a process called singular value decomposition where $A = U\Sigma V^T$

approximation
- definition: the singular value decomposition of a rank-r matrix A has the form $A = U\Sigma V^T$ where U is n x r, the columns of U are orthogonal and unit length $(U^TU=I)$ and V is m x r, the columns of V are orthogonal and unit length $(V^TV=I)$
- definition: the singular value decomposition of a rank-r matrix has the form $A = U\Sigma V^T$ where $\Sigma$ is a diagonal matrix with $\sigma_1,...,\sigma_r$ on the diagonal with $\sigma_1 \ge \sigma_2 \ge ... \ge \sigma_r \ge 0$ and $\sigma_i$ is the square root of the eigenvalues of $A^TA$ and are called singular values
- find $A^{(k)}$ by decomposing A:
    - $A^{(k)} = U_1\Sigma_1V_1^T$
    - where $U_1$ is n x k, $\Sigma_1$ is k x k, $V_1$ is m x k
- the ith singular vector represents the direction of the ith most variance. singular values express the importance / significance of a singular vector
- property: $d_F(A,A^{(k)})^2 = \sum_{i=k+1}^{r}\sigma_i^2$
    - note: the larger k is, the smaller the distance
- to find the right k, you can
    1. look at the singular value plot to find the elbow point
    2. look at the residual error of choosing different k

related to principal component analysis (PCA)
- SVD and PCA are related

dimensionality reduction
- idea: project the data onto a subspace generated from a subset of singular vectors / principal components
- we want to project onto the components that capture most of the variance / information in the data
- which principal component should we project on, A or B?

anomaly detection
- define O = $A - A^{(k)}$
- the largest rows of O could be considered anomalies

features that have the same units you don't want to mean-center because you want to see difference in scale

features that are not the same units and on different scales (like age and income) you want to normalize to see better

## 10/7 notes
doc-to-term similarity X term-to-concept similarity = doc-to-concept similarity

latent semantic analysis
- inputs are documents. each word is a feature. we can represent each document by:
    - the presence of each word (0/1)
    - count of the word (0, 1, ...)

in theory
- we would have that the lung term would be close to the biology concept and data closer to computer concept while the neuron is in the middle

in practice
- we don't know the axes -- don't know if we have a "computer science" topic exactly or a "biology" topic/axes
- we know in the embedding of the semantic meaning of these words lung will be close to brain and data close to information
- we would have king, queen close and car, bottle close together (why? not sure, we don't know what the axes are)

words with similar semantic meanings should be close

lots of ways to generate embeddings. SVD is one of them

example 1 (with the presence of each word)
- we have the CS paper on the left, and then multiply by the embedding and term-to-concept similarity = doc-to-concept similarity / CS feature (stronger the similarity the bigger number)

example 2 (with the count of the word)
- have the CS paper on the left, then term-to-concept similarity, and doc-to-concept similarity (stronger similarity)

how we represent the documents affects the output

how do we get the embedding using SVD?
- first matrix as doc-to-term similarity
- first singular vector as CS concept/topic, second vector as MD (medical) concept/topic
    - the values are the doc-to-concept similarity
- second matrix has "strength" of each concept/how prevalent they are
- last matriox as made up of term-to-concept similarities

latent semantic analysis
- we can better represent each document by:
    - frequency of the word ($n_i / \sum n_i$)
    - TfiDf
    - tf * idf where tf is term frequency in the document and idf is the log(number of documents / number of documents that contain the term)


## 10/9 notes
what is classification?
- there are classes, and predictors/features/attributes
- classes, predictors/features/attributes -> learn model -> model f: age x tumor size --> {yes, no}
- what is property/combination of age and tumor size is unique to malignant tumors?
    - if 2*tumor size > age, it's malignant (older you are, higher tumor threshold)

example with alien -> what is it that allows someone in the U.S. to legally drink?
- get a dataset and ask people: how old are you? and can you legally drink?
- what distinguishes points between those in the can't drink class and the can drink class
- figure out that the threshold between classes is 21 years old
- sometimes there are many correct answers

- sometimes there are no correct answers (like with the example where we are trying to understand whether or not a student will pass an exam given how many hours they studied)
- could be because we have wrong or insufficient attributes for the task (could add the exam length as an attribute)
- could be because the problem just doesn't have an exact solution
- all models are wrong but some are useful

feasability of a classification task completely depends on the relationship between the attributes (or predictors) and the class

for example if we used age instead of weight for elephants and rhinos
- age cannot distinguish rhinos and elephants

takeaways
- there could be many correct answers
- there could be no correct answers
    - but the model could still be useful if it's more or less correct most of the time
- whether a task is feasible depends on:
    - the relationship between the predictors and the class

lots of questions
- how do we know if we have good predictors for a task?
- how do we know if we have done a good job at classification?

how do we know if we have good predictors?
- what constitutes a good feature/predictor?
    - with the example, the first one has no discrepancy really between 1,2, and 3 star reviews (not good), second case has more discrepancy but not really, and the third scenario is the most ideal (where there are more distinct review lengths for each star)
- what constitutes a good set of features/predictors?
    - we want to see some relationships between the features and the class, but we don't want redundant relationships between features (where if we included feature 1 it would include feature 2 basically if it was linear relationship it wouldnt be useful)
- BUT...
    - correlation is not causation
- correlation vs causation
    1. temperature and ice cream sales are positively correlated
        - temperature increases cause ice cream sales to spike
            - BUT in the desert where there is no ice cream, there is no spike in sales
        - ice cream sale increases do not cause the temperature to reviews
    2. sleeping with shoes on is strongly correlated with waking up with a headache
        - but neither causes the other...
        - there's a third common factor causing this correlation: going to bed drunk
- testing for causality requires specific testing/experimentation with a control group

how do we know we've done well at classification?
- testing without cheating. learning not memorizing.
    - split up our data into a training set and a separate testing set
    - use the training set to find patterns and create a model
    - use the testing set to evaluate the model on data it has not seen before
- also allows us to check that we have not learned a model TOO SPECIFIC to the dataset
    - overfitting vs underfitting
    - goal is to capture general trends
        - watch out for outliers and noise
- the types of mistakes made matters
- types of mistakes
    - testing for a rare disease
        - out of 1000 data points, only 10 have this rare disease. a model that simply tells folks they don't have the disease will have an accuracy of 99%

classification
- training step
    - create the model based on the examples/data points in the training set
- testing step
    - use the model to fill in the blanks of the testing set
    - compare the result of the model to the true values

instance-based classifiers
- use the stored training records to predict the class label of unseen causes
- rote-learners:
    - perform classification only if the attributes of the unseen record exactly match a record in our training set

instance-based classifiers: training step
- data -> learn model -> there is no training step per se. the dataset itself is the model

instance-based classifiers: applying the model
- take a given age and tumor size and look and see if the malignant is yes or no

nearest neighbor classifier
- use SIMILAR records to perform classification

K nearest neighbor classifier
- requires:
    - training set
    - distance function
    - value for k 
- how to classify an unseen record:
    1. compute distance of unseen record to all training records
    2. identify the k-nearest neighbors
    3. aggregate the labels of these k neighbors to predict the unseen record class (EX: majority rule)
- aggregation methods:
    - majority rule
    - weighted majority based on distance $w=1/d^2$
- scaling issues:
    - attributes should be scaled to prevent distance measures from being dominated by one attribute. example:
        - age: 0 -> 100
        - income: 10k -> 1 million

scaling attributes

k nearest neighbor classifier
- choosing the value of k:
    - if k is too small ->
        - sensitive to noise points + doesn't generalize well
    - if k is too big ->
        - neighborhood may include points from other classes

k nearest neighbor classifier
- pros:
    - simple to understand why a given unseen record was given a particular class
- cons:
    - expensive to classify new points
    - KNN can be problematic in high dimensions (curse of dimensionality)

# 10/15 notes
how a decision tree works
- given some info like refund=No, martial status=single, income=70k we can get an answer for class
    - we start at the root of the tree. we see that its refund=No so we go to the right of the tree for that
    - then we look at marital status, and go down the single side on the left
    - then we look at income and go to the less than 80k side
    - then we get the answer

how do we learn it?
- what happens if marital status == married? --> we always have that the class = no


hunt's algorithm
- recursive algorithm
    - repeatedly split the dataset based on attributes
- base cases:
    - IF split and all data points in the same class
        - great! predict that class!
    - IF split and no data points
        - no problem! predict a reasonable default
- the recursion (IF split and data points belongs to more than one class)
    - find the attribute (and best way to split that attribute) that best splits the data

example

many ways to split a given attribute
- binary split
- multi-way split


continuous variables
- use binning before running the decision tree
    - can use clustering for that example
- compute a threshold while building the tree
    - A > t vs A < t 

need a metric
- that favors nodes like this: NO = 1, YES = 7
- over nodes like this: NO = 4, YES = 4

GINI index
- denote $p(j|t)$ as the relative frequency of class j at node T
- GINI(t) = $1 - \sum_j p(j|t)^2$
- best possible GINI would be 0
- worst possible GINI would be 1/2
- $GINI_split = \sum_{t=1}^{k} \frac{n_t}{n} GINI(t)$
    - where $n_t$ = number of data points at node t and n = number of data points before the split (parent node)

limitations
- easy to construct a tree that is too complex and overfits the data
- solutions:
    - early termination (stop before the tree is fully grown - use majority vote at leaf node)
        - stop at some specified depth
        - stop if size of node is below some threshold
        - stop if GINI does not improve
    - pruning (create fully grown tree then trim)

other measures of node purity
- entropy
    - $Entropy(t) = -\sum_j p(j|t)log(p(j|t))$
- misclassification error
    - $Error(t) = 1 - max_j (p(j|t))$


## 10/16 notes
naive bayes

conditional probability
- recall $P(A|C) = \frac{P(A \cap C)}{P(C)}$

bayes theorem
- $P(A|C) = \frac{P(C|A)P(A)}{P(C)}$
- $P(C|A) = \frac{P(A \cap C)}{P(A)}$
- $P(A|C) = \frac{P(A \cap C)}{P(C)}$

example
- given
    - meningitis causes a stiff neck 50% of the time
    - prior probability of any patient having meningitis is 1/50,000
    - prior probability of any having a stiff is 1/20
- if a patient has a stiff neck, what is the probability that they have meningitis?
- $P(M|S) = \frac{P(S|M)P(M)}{P(S)}$

bayesian classifiers
- given an unknown example:
    - $(A_1 = a_1, A_2 = a_2, ..., A_m = a_m)$
- predict the class C that maximizes $P(C|A_1 = a_1, A_2 = a_2, ..., A_m = a_m)$
- example: binary class {yes, no}
- to classify unseen record (marital status = "married", income=100k)
1. Compute P(yes | marital status = "married" and income = 100k)
2. Compute P(no | marital status = "married" and income = 100k)
3. Compare and predict the class that has the highest probability given the attribute values
- how do we estimate $P(C|A_1 = a_1, A_2 = a_2, ..., A_m = a_m)$ from the data?
    - $P(C|A_1 \cap A_2 \cap ... \cap A_n) = \frac{P(A_1 \cap A_2 \cap ... \cap A_n | C)P(C)}{P(A_1 \cap A_2 \cap ... \cap A_n)}$
    - $P(A_1 \cap A_2 \cap ... \cap A_n)$ does not depend on C 
    - maximizing $P(C|A_1 \cap A_2 \cap ... \cap A_n)$ is equivalent to maximizing the numerator $P(A_1 \cap A_2 \cap ... \cap A_n | C)P(C)$
- so how do we estimate $P(A_1, A_2, ..., A_n | C)P(C)$ from the data?
    - $P(C)$ is easy - why? -> we can just count how many instances of each class we have
    - but $P(A_1,A_2,...,A_n|C)$ is tricky because it requires knowing the joint distribution of the attributes...
- can we make some assumptions about the attributes in order to simplify the problem?
    - assume that $A_1, A_2, ..., A_n$ are independent!
    - then..
        - $P(A_1,A_2,...,A_n|C) = P(A_1|C)P(A_2|C)...P(A_n|C)$
        - can we estimate $P(A_j|C)$ from the data?
            - yes! just count the occurrence of $A_j$ for that class!

example
- refund, marital status, income, class table
- P(class = Yes) = 3/10
- P(marital status = "single" | class = yes) = 2/3
- P(marital status = "married" | class = no) = 4/7
- P(income = 120k | class = No) = 1/7

continuous attributes
- binning / 2-way or multi-way split
    - create new attribute for each binary
    - issue is that these attributes are no longer independent
- PDF estimation
    - assume attribute follows a particular distribution (example: normal)
    - use data to estimate parameters of the distribution

example
- assume normal distribution
- P(income = 120k | class = No)
    - sample mean = 110
    - sample variance = 2975
    - $P(Income = 120|No) = \frac{1}{\sqrt{2\pi}(54.54)}e^{-\frac{(120-110)^2}{2(2975)}} = 0.0072$

putting it all together
- test record: X = (Refund = No, Married, Income = 120k)
    - P(X|No) = P(Refund = No | No)P(Married|No)P(Income=120k|No) = $4/7 * 4/7 * 0.0072 = 0.0024$
    - P(X|Yes) = P(Refund=No|Yes)P(Married|Yes)P(Income=120k|Yes) = $1 \dot 0 \dot 1.2 \dot 10^{-9} = 0$
    - since P(X|No)P(No) > P(X|Yes)P(Yes) -> predict No

limitations
- if one of the conditional probabilities is zero, the entire expression becomes zero..
- original estimate of $P(A_i|C) = \frac{N_{ic}}{N_c}$
- laplace estimate: $P(A_i|C) = \frac{N_{ic} + 1}{N_c + constant}$

question
- can you use naive bayes to predict class C based on the following two features?
    1. weight
    2. height


model evaluation

confusion matrix
- predicted class on top, actual class on the side

accuracy can be misleading
- binary classification problem where:
    - number of class 0 examples: 9990
    - number of class 1 examples: 10
- a model that predicts everything to be class 0 will have an accuracy of 99.9%

cost matrix
- have costs associated with C(Yes|Yes) = -1, C(No|Yes) = 100, C(Yes|No) = 1, C(No|No) = 0

other metrics
- precision $\frac{a}{(a+c)}$
- recall $\frac{a}{(a+b)}$
- F-measure: $2RP / (R+P)$

methods of estimation
- goal: get a reliable estimate of the performance of the model on unseen data
- holdout:
    - ex: reserve 1/4 of the dataset for testing and use 3/4 for training
- cross validation:
    - partition into K disjoint subsets
    - K-fold: train on K-1 partitions, test the remaining on
    - K = n: leave one out

validation set
- for tuning parameters

ensemble methods

the idea
- suppose you have trained 17 different classifiers on a dataset
    - every classifier has error rate e = 0.2
    - assume all classifiers are independent
- in order to classify a new record we poll all 17 classifiers and take the class that the majority agrees on
- what is the probability that this ensemble classifier makes a wrong prediction?
- the majority needs to make a mistake (i.e. at least 9 out of 17 make mistakes)
    - $P(X \geq 9) = sum_{k=9}^{17} C(17,k)(0.2)^k(1-0.2)^{17-k}=0.002581463$


how to generate independent classifiers?
- by generating samples of data to train on
    - bagging
    - boosting

bagging
- build a classifier on each bootstrapped sample

boosting
- an adaptive sampling process to chagne the sampling distribution based on difficult-to-classify examples
- start with all samples having equal probability of being selected. next boosting round, increase the weights of those samples that were misclassified, decrease the weights of those samples that were correctly classified


## 10/21 notes
support vector machines

example
- positive class and a negative class
- nearest neighbor decision boundary
- decision tree split on attribute x_1
- $P(C|X_1X_2) = 1/2$ (Naive Bayes)

SVM: find the widest street that separates our classes - the dotted line in the middle is our decision boundary

how do we define this street? what is the format of the equation of this line/decision boundary?
- $w_1x_1 + w_2x_2 + b = 0$
- $w^Tx + b = 0$

suppose we found this decision boundary, how would we classify an unknown point u?
- $[u_1, u_2]$
- for the line, $w_1x_1 + w_2x_2 + b = 0$ -> $w^Tx + b = 0$
- $w_1u_1 + w_2u_2 + b$ -> $w^Tu + b$
- DECISION RULE: $\vec{w} * \vec{u} + b \geq 0$ then +
- $w^Tx + b = -1$, $w^Tx + b = 1$, $w^Tx + b = 0$

there are many w's and b's 
- $c*w^Tx + c*b = 0$

what happens if c > 1?
- $c*w^Tx + c*b = 0$, $c*w^Tx + c*b = -1$, $c*w^Tx + c*b = 1$
- get a smaller street

what happens if 0 < c < 1?
- $c*w^Tx + c*b = 0$, $c*w^Tx + c*b = -1$, $c*w^Tx + c*b = 1$
- get a wider street

wide as possible, means $w$ as small as possible

assuming our data is linearly separable, we want to impose the constraint that none of our samples can be in the street. that is:
- $\vec{w} * \vec{x_+} + b \geq 1$
- $\vec{w} * \vec{x_-} + b \leq -1$

to move the street in the direction of a point
- pick a step size a, in order to move a steps in the direction of x 
- $w_{new} = w_{old} + y_i * x * a$
- $b_{new} = b{old} + y_i * a$

full algorithm (perceptron algorithm)
- start with random line $w_1x_1 + w_2x_2 + b = 0$
- define:
    - a total number of iterations (ex: 100)
    - a learning rate a (not too big not too small)
    - an expaqnding rate c (< 1 but not too close to 1)
- repeat number of iterations times:
    - pick a point $(x_i, y_i)$ from the dataset
    - if correctly classified: do nothing
    - if incorrectly classified:
        - adjust $w_1$ by adding $(y_i * a * x_1), w_2$ by adding $(y_i * a * x_2)$, and b by adding $(y_i * a)$
    - expand or retract the width c (multiply the new line by c)

what contributes to the widest street?
- intuitively:
- this point is called a support vector

find the widest street subject to...
- we want our samples to lie beyond the street. that is:
    - $\vec{w} * \vec{x_+} + b \geq 1$
    - $\vec{w} * \vec{x_-} + b \leq -1$
- note: for unknown u, we can have 
    - $-1 < \vec{w} * \vec{u} + b < 1$
- lets introduce a variable:
    - $y_i$ = +1 if $x_i$ is a + sample, -1 if $x_i$ is a - sample
    - note: this is effectively the class label of $x_i$
- if we multiple by our sample decision rules by this new variable:
    - $y_i(\vec{w} * \vec{x_i} + b) \geq 1$
- meaning for $x_i$ on the decision boundary we want:
    - $y_i(\vec{w} * \vec{x_i} + b) - 1 = 0$


how to find the width of the street
- we know that WIDTH = $(\vec{x_+} - \vec{x_-}) * \frac{\vec{w}}{||\vec{w}||}$ for $\vec{x_-}$ and $\vec{x_+}$ points on the boundary
- and since they are on the boundary we know that $y_i(\vec{w} * \vec{x_i} + b) - 1 = 0$
- hence, WIDTH = $\frac{2}{||\vec{w}||}$

what does that mean?
- size of w is inversely proportional to the width of the street
- aligns with what we found previously

how to find the widest street
- goal is to maximize the width $max(\frac{2}{||\vec{w}||}) = min(||\vec{w}||) = min(\frac{1}{2}||\vec{w}||^2)$
- subject to: $y_i(\vec{w} * \vec{x_i} + b) - 1 = 0$
- can use Lagrange multipliers to form a single expression to find the extremum of:
    - $L = \frac{1}{2}||\vec{w}||^2 - \sum_i a_i[y_i(\vec{x_i} * \vec{w} + b) - 1]$
    - where $a_i$ is 0 and $x_i$ is not on the boundary
- let's take the partial derivative of L with respect to w to see what w looks like at the extremum of L
- $\frac{\partial L}{\partial \vec{w}} = \vec{w} - \sum_i a_iy_i\vec{x_i} = 0$
    - $\vec{w} = \sum_i a_iy_i\vec{x_i}$
- means w is a linear sum of vectors in our sample/training set!

$\sum_i a_i < x_i, x > +b \geq 0$ then  +

to move the street in the direction of a point
- $a_{i, new} = a_{i, old} + y_i * a$
- $b_{new} = b_{old} + y_i * a$

how to find the widest street
- goal to maximize the width
    - $min(\frac{1}{2}||w||^2 + \lambda \sum_i e_i)$
- subject to:
    - $y_i(\vec{w} * \vec{x_i} + b) \geq 1 - e_i$

option 1: soft margins
- can allow for some points in the dataset to be misclassifeid

option 2: change perspective
- use $\phi$

but how to find $\phi$?
- turns out we don't find or define a transformation $\phi$
- recall: $\sum_i a_i < x_i, x > +b \geq 0$ then  +
- we only need to define $K(\vec{x_i}, \vec{x_j}) = \phi(\vec{x_i}) * \phi(\vec{x_j})$
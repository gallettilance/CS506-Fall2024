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

# 9/25/2024
## Heirarchical Clustering 
- Dendrogram -- tells you at what distance do you start to merge points together becuase they are so similar 
  - go to different distances (y axis) and it allows us to see how where the different clusters are merged together 
  - can cut the dendrogram and a particular point andd see what are bunched together 
- Two types: Agglomerative & Divisive 
  - Agglomerative:
    - Start with every poit in its own cluster
    - At each step, merge the two closest clusters
    - Stop when every point is in the same cluster
  - Divisive: 
    - Start with every point in the same cluster
    - At each step, split until every point is its own cluster 
- How would you define distance between clusters? 
  - Single-Link Distance -- minimum of all pairwise distances between a point from one cluster and a point from the other cluster 
    - $D_{SL}(C_1, C_2) = min\{d(p_1, p_2) | p_1 \in C_1, p_2 \in C_2\}$
    - can handle more elongated clusters which couldnt be handled with k-means 
    - but is very sensitive to noise and imperfections between clusters
  - Complete-Link Distance -- max of all pairwise distacnes between a point from one cluster and a point from the other 
    - $D_{ML}(C_1, C_2) = max\{d(p_1, p_2) | p_1 \in C_1, p_2 \in C_2\}$
    - less susceptible ot noise, creates more balanced clusters
    - But, tends to split up larger clusters and all clusters have similar radius 
  - Average-Link Distance -- is the average of all pariwise distances between a point from one cluster a point form the other cluster 
    - $D_{AL}(C_1, C_2) = \frac{1}{|C_1|*|C_2|}\sum_{p \in C_1, p_2 \in C_2}{d(p_1, p_2)}$
  - Ward's Distance -- difference between the spread/veraicne of points in the merged cluster and the unemerged clusters 
    - after we merge them together, will the variabce increase a lot or will it still be similar to the smaller variances while separate
    - *include the equation later*
## Density-Based Clustering 
- Goal: cluster together points that are densely packed together 
- you can grow a radius and see how many points are in it and if there are at least "min_pts" number of points then it is dense, otherwise not 
  - can be in a dense region but not core to a dense region 
- Need to distinguash between points at the core of a dense region and points at the border of a dense region 
  - Core points - if its E-neighborhood contains at least min_pts 
  - Border poitns - if its in the E-neighborhood of  acore point but not core point 
  - Noise point - neither core nor border 
- DBScan Algorithm: 
  1. Find the E-neighborhood of each point 
  2. later the point as core if contains at least min_pts
  3. for each core point, assign to the same cluster all core points in the neighborhood 
  4. label points in its neighborhood that are not core as border 
  5. label points as noise if they are neither core nor border 
  6. Assign border poitns to nearby clusters 
- Limitations: 
  - can fail to identify clusters of varying densities 
  - Tends to create clusters of the same density 
  - Notion of density is problematic is high-dimentional spaces


# 9/30/2024
## Soft Clustering 
- another form of clustering using probabiltiy and weighting 
- in your dataset there could be the possibiltiy that one value from a point is more probable than another 
- consider the probabiltiy of ebign some species given that we had previously been different species 
- $P(S_j|X_i) = \frac{P(X_i|S_j)P(S_j)}{P(X_i)}$
  - $P(S_j)$ is the prior probability of seeing species $S_j$ 
  - $P(S_j|X_i)$ is the PDF of species $S_j$ weights evaluated at weight $X_i$ 
- $P(X_i) = \sum_{j}{P(S_j)P(X_i|S_j)}$
  - Mixture model -- X comes from a mixture model with k mixture componenets if probability is above 
- Gaussian Mixture model -- if they are Gaussian mixture for all distributions 
- Maximum Likelihood Estimation (intuition)
  - if we flip coid 5 times and get HTTHT, then we would htink that Pr(H) = 0.4
  - But we actually did was that we found a p (bernoulli param) such that we can maximize the probability of htis specific sequence happening 
  - the data does not perfectly represent the probability of something happening 
- With GMM - now we have more parameters to figure out 
  - need the math to reason about this and solve it 
- $\Pi_{i}{P(X_i)}=\Pi_{i}{\sum_{j}{P(S_j)P(X_i|S_j)}}$
  - to find the probabiltiy of this sequence happening again 
- Expectation Maximization algorithm 
  - Start with random $\mu, \sum, P(S_j)$
  - Computer $P(S_j|X_i)$ for alll $X_i$ by using $\mu, \sum, P(S_j)$
  - computer/update $\mu, \sum, P(S_j)$ from $P(S_j|X_i)$
  - Repease 2 & 3 until convergence 
- usually k-means is used to initialize this algorithm 


# 10/1/2024
## Singular Value Decomposition
- characterics of a dataset to look for: 
  - what do we want the relationship to look like between the features of our dataset? 
  - We want the two attributes to be independent from each other 
  - A is not a function of B, but we know that if A changes then B changes as well -- we don't want this because if they are related we cannot isolate the changes 
  - dont want A and B' to not be related -- straight line or scatter plot
- representing attributes a and b - 2 dimensional, but the dataset is only 1 dimensional -- this is the rank 
- our goal is to transform a set of data that is linearly related to make then not linearly related 
  - only removing linear relationships; if we had quadratic relation, we would just be removing the linear component 
- purposes: 
  - denoise our data 
  - dimension reduction -- project onto A or B 
- data set: n data points (rows) x m features (columns) 
- Goal: 
  - Approximate A with a smaller (rank) matrix B that is easier to store but contains similar information as A 
  - dimensionality reduction/feature extraction (from m features to j features)
  - Anomaly Detection and Denoising 
- Matrix Factorization: A = UV --> A is nxm so now it can just be stored as nxk and mxk (with relatively small k) 
  - but this also means that the amount of information stored in this is relatively small 
  - saves a lot of time in running our algorithms 
- Singular Value Decomposition: $A = U\sum{V}^T$
- find the Frobenius Distance -- pairwise sum of square differences between values of A and B 
- approxiamtion - when K < rank(A), the rank-k approximation of A is: 
  $A^k = argmin_{B|rank(B)=k}d_F(A, B)$
  - the ith singlar value vector represents the direction of the ith most variance 

# 10/7/2024
## Latent Semantic Analysis 
- another applicattion for SVD
- in a matrix, hold a count of words that appears in different papers
- what concepts best represent these documents? Can we write each document as a vector in a space of concepts? 
  - doc-to-term similarity X term-to-concept similary = doc-to-concept similarity 
- Inputs are documents, each word is a feature, we can represent each doc as: 
  - Represent paper as presence of a word (0/1)
    - may have to do some preprocessing -- stemming, filler, etc
- term-to-concept similarity 
  - in theory, there is biology concept on y-axis, computer science on x-axis --> each term is a vector between biology and cs 
  - worsd with similar semantic meanings should be close together 
- another option for representing document: Count of word (0, 1, 2, ...)
  - when multipky by term-to-concept matrix, get higher similarity
- doc-to-concept similarity matrix X "strength" of each concept X term-to-concept similarity matrix 
- one more way of representing docuemnts: most popular!!
  - TfiDf --> tf*idf
  - tf = term frequency in the document 
  - idf = log(number of documents/number of documents that contain the term)
  - if word is very present, then we wan tot penalize by its freqeuncy across other documents..?
    - eg. "the" would be super frequent, bit not important, so penalize so it won't be repressented 
  
# 10/9/2024
## Classification
- assigning points to a class or category based on certain attributes 
- Can have many classes (needs to be finitely many) and predictors/features 
- through classification, build a model which is mathematical function that allows us to classify the data 
- sometimes no correct answers -- no way to distinguish between groups
  - could be because we have wrong or insufficient attributes for the task 
  - could be because the problem just doesnt have an exact solution (some overlap)
    - but that would also ive us some useful information 
- the feasability of a classification task completely depends on the relationship between attributes and the class
- How do we know if we have good predictors for a task? 
  - look for correlation between the predictor and categories 
  - BUT correlation does not = causation 
- How do we know we have done a good job at classification?
  - testing without cheating -- learning not memorizing 
  - split dataset into training and testing 
    - use training to find patters and create model 
    - use testing to evaluate the model on data it has not seen 
  - allows us to check that we have not learned a model TOO specific to the dataset 
    - overfitting vs. underfitting 
- underfitting = too loose to the data 
- overfitting = too tight to the training 
- how do we know we've done well at classification 
  - Testing for a rare disease: our of 1000 data points, only 10 have this rare disease. A model that simply tells everyone they don't have diease will have accuracy of 99%
- Classification tools:
  - training step - create the odel based on the examples/data points in the training set 
  - testing step - use the model to fill in the blanks of the testing set and compare the results to the true values 
- Instance-Based Classifiers
  - Use the stored training records to predict the class label of unseeen cases 
  - Rote learners: perform classification only if the attributes of the unseen record exactly match a record in our training set 
    - next best: find something close to the point thats already there 
- K Neaerest Neighbor Classifier 
  - pick K similar records and aggregate those records to create classificationi 
  1. compute distance of unseen record to al training 
  2. identify the k nearest neighbors
  3. aggregate the labels of these k neighbors to predict the unseen record class (ex. majority rule)
  - if there is a tie, we could do a weighted majority instead to take into acount the distance of each point from the data we are looking at 
  - varying K varies our model 
    - if too small -> sensitive to noise points + doesn't generalize well
    - if too large -> neighborhood may include points from other classes 
  - need to scale if they have different units 
  - pros: simple to nderstand why unseen record was ggiven particular class
  - cons: expensive; can be problematic in high dimensions


# 10/15/2024
## Decision Trees 
- Decision Tree - given information about an entity, we can see what the result should be 
  - look at the value of each attribute and follow which value matched that in the decision tree 
- Hunt's Algorithm 
  - recursive algorithm -- repeatedly splot the dataset based on attributed 
  - base case:
    - if splot and all data poitns in the same class -- predict that class
    - if split and no data points -- predict a reasonable default 
  - can have binary splits or multi-splits, but jut need to be clear how to classify after the splits 
  - determining bins can be a preprocessing step or do it as you are building the trees 
- GINI index - denote p(j|t) as the relative frequency of p at node t 
  - GINI(t) = 1 - sum(p(j|t)^2)
  - best possible GINI = 0; worst possible GINI = 1/2 
  - to find the GINI of the whole node, then we can average the GINIs of each of the individual nodes 

# 10/16/2024
## Naive Baye's 
- conditional probablility 
- bayein clsssifierd
- cost matrix


# 10/21/2024
## Support Vector Machine
- we want to find the best line to separate the data points into groups 
  - add 2 lines on either side of boundary to show the distance from the data to the boundary line; want the widest set of lines -- call this a street 
- how do we define this street? 
  - equation of line (like y = mx + b)
  - have an equation for central line, and then define the equations for the lines on either side 
- to classify an unknown point, you can plug the values into the equation 
- if you multiply by a constant, then the decision boundary doesn't change 
  - the boundary does not change, but the width of the street cahnges 
  - multiplying by big constant makes street smaller, smaller constant makes street bigger 
- so if magnitude of w is larger, then the street is smaller 
- to move street in the direction of a point, pick a step size a, in order to move a steps in the ddirection of x
  - $w_{new} = w_{old} + y_i + x + a$
  - $b_{new} = b_{old + y_i + a}$
- find the widest street subject to...
  - y_i = 1 if it is positive example; -1 if it is a negative example
  - x+ - x- is not enough because we ned it to be perpendicular
  - we project it onto w, which is perpendicular, and then normalize by the length of w. 
  - (makes sense for when we said that w is inversely proportional to the width of the street)


# 10/30/2024
## Linear Regression
- variation around linear regression is same as variation if you collect data at one specific x value 
- linear matters for beta and not x 

# 11/4/2024
## Linear Model Evaluation
- evaluation of a model: 
  - Total sum of squares (TSS) -- measure of the spread of $y_i$ around the main of y 
  - ESS -- measure of the spread of our model's estimates of $y_i$ aroudn the mean of y
  - $R^2$ = $\frac{ESS}{TSS}$ -- measures fraction of variance that is exokained by your model 
    - if $R^2$ is 1, that means that every single point is on the line 
  - adjusted $R^2$ accounts for the fact that the more features we have, the more we will be artificially inflating our $R^2$ value 
- Hypothesis Testing 
  - if we repeat  give number of times, how many times will we see what was observed 


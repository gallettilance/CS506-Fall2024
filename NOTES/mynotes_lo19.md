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
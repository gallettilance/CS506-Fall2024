### 9/23/24
## finishing lloyd's algorithm
    * prof way: pseudo code with what methods you'll need, then implement those functions
    * see workseet 3 for lloyds
* will lloyd's always converge? let's prove by contradiction
    * suppose it doesn't converge, then either
        * the minimum of the cost function is only reached in the limit (i.e. an infinite number of iterations, like you tryign to look at every partition ever and there are infinite??)
            * but impossibel since we are iterating over a finite set of partitions
        * the algo gets stuck in a cycle or a loop
            * not possible since this would only be possible when if you have 2 overlapping points and a randomized assignment of points to clusters, but our algo can spot that these are the same points
* does it always converge to optimal solution though?
    * no, there can be clusters that are close but don't necessarily count as an organic organic cluster as one but maybe as two
* i misssed some stufff
* k-means++: this is a combination of the two methods (decreasing randomization and choosing a point farther from ????????????????????????)
    * start with random center
    * let d(x) be the distance between x and the closest of the centers picked so far
        * choose the next center with probability proportional to $d(x)^2$
            * this allows for less randomization and for greater distance point to be more likely to be picked
* bro im crashing out
* so how do we choose the right k?
    * iterate through different values of k
    * use empirical/domain specific knowledge
    * metric for evaluating a clustering output
* so remember our goal to find a clustering such that
    * similar dtaa points are in the sae cluster
    * dissimilar data points are in different clusters
    * we wanna create clusters that are far from each other, clusters should be distinct from each other
* how could we try to define this metric that evaluates how spread out the clusters are from one another
    * maybe distance between centers
    * maybe minimum distance between points across clusters
    * make the cneter distnce but then subract the sread within one cluster, since maybe need to account for that?
* let b be the distance between centers and a be the dustanc across one center
    * if b - a = 0, this means that the clusters overlao
    * so ideally we want b - a to be large!! this means clusters are spread far apart relative to the compactness of the clusters
* but the value of b - a doesn;t really mean anything so how do we really get a meaning
    * like b - a = 5 doesn't tell us about anything
    * so do $\frac{(b - a)}{max(a, b)}$
        * if $\frac{(b - a)}{b} = 1$ then b - a is basically b, so a is really small, so that means clusters are small and there is good spread
        * if $\frac{(b - a)}{max(a, b)} = 0$, then there may be overlap
* sillhouette scores
    * for each data point i, define:
        * $a_i$: ean distnce from point i to ever other point in its cluster
        * $b_i$ smallest mean distance from point i to every point in another cluster
    * so silhouette score $s_i$ = $\frac{b_i - a_i}{max(a_i, b_i)}$
        * low silhouette score is bad...
        * you can plot silhouette score of plo to get an idea
    * how to tell if silhouette score is better...?
        * avg silhouette score across clusters good
        * clusters are all kind of similar
* point of diminishing return...
    * silhouette score plot bad if pqast the point of diminishig return
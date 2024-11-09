### 10/14/24
* these notes may be a little sparse
* how a decision tree works
    * given a classifier (liek the categoriues and values for the), you can use the tree to figure out what class they belong to
    * ![image info](./assets/dtree.png)
    * nbasically just follow the tree
* then we can use certain classifiers ot build classes, like all maried ends up being no
* algo for decision trees -> hunt's algorithm
    * recursive
        * repeatedly split the dataset on attributs
    * base cases
        * if split and all data points in same class, then can predict that class
        * if snplit and no data pitns, then predict a reasonable fault
* when we split, we want the split that provides with the most distinguishable results
    * like if one attribute gives a more dstinguishable split, then better to split on that one
* GINI index...
    * $GINI(t) = 1 - \sum_jp(j |t)^2$
    * best possible GINI is 0
    * worst GINI is 1/2, since that's when data is evenly split, not distinguishable
* combine all GINIs with a weighted average/sum
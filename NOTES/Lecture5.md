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
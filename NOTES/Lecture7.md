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

# Temporal Dietary Patterns of Iron Intake

## About


## Dataset
The dataset used is NHANES 2017-2018 Dietary Data (Dietary Interview - Individual Foods, First Day). Link: https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/DR1IFF_J.XPT

## Algorithm
The algorithm used is kernel k-means clustering.
K-means clustering is an unsupervised machine learning algorithm that partitions data into n clusters.
Because the algorithm can only form linear decision boundaries, it is not suited for complex data.
Kernel k-means clustering, on the other hand, can form nonlinear decision boundaries and is suitable for more complex data.
It uses the same algorithm but modifies the distance calculation.
I will be following the custom distance function presented in the paper "Temporal Dietary Patterns Using Kernel k-Means Clustering."

## To do
* Implement the custom distance function for kernel kmeans
* Change the centroid initialization technique

## Sources
* https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4171949/

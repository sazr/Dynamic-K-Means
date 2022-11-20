### How's it work?

The benefit of this algorithm opposed to normal K-Means is we don't need to 
specify X number of clusters to find.

Instead we...

    For each data point x:
	- If no clusters exist yet:
		- x becomes a cluster
		- continue to next data point

	- Find x's euclidean/etc. distance to each cluster; closest cluster is c
	- If distance between x and c is within a tolerance t:
		- x is assigned to cluser c
		- update c cluster centre now that its data-set has changed.
	- Else:
		- x becomes a new cluster

### Applications:

- Find the dominant colour's in an image.

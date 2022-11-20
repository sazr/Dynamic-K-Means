"""
How's it work?

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

Applications:

- Find the dominant colour's in an image.

"""

import cv2
import numpy as np 
import numpy.linalg as la
import imutils
import math
from scipy.spatial.distance import cdist, euclidean
from sklearn.metrics import pairwise
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib.pyplot as plt
import sys
sys.path.append('../../algorithms')
import _util

np.set_printoptions(precision=3, threshold=np.inf, linewidth=np.inf, suppress=True)

CONST_WIDTH = 300

# Variability:
# Allow ignore of any channel (not just channel 0). That way we can consider Hue (0) and Saturation (1) only and not channel 2 (Value)
# Update interval logic; update when total size % x == 0 or only when this.cluster.size % x == 0
# Update sum_dists after every added colour or only when we create a new cluster
# Use a different method to find the cluster centroid (not mean or median)
# Use different cluster distance measure (not euclidean but chi^2, manhattan, cosine dist, etc.)

class DynamicKMeans:
	def __init__(self, compare_cspace=cv2.COLOR_BGR2LAB, ignore_luminocity_channel=False, seed_max_dist=75, max_dist_ratio=0.95, cluster_update_interval=1, measure=np.median, **kwargs):
		self.compare_cspace = compare_cspace
		self.clusters = []
		self.colour_map = []
		self.n_colours = 0
		self.dists_sum = 0
		self.max_dist = seed_max_dist
		self.max_dist_ratio = max_dist_ratio
		self.ignore_luminocity_channel = ignore_luminocity_channel
		self.cluster_update_interval = cluster_update_interval
		self.measure = measure
		self.total_size = 0

	def update_cluster(self, cluster_index):
		# TODO: how to find the centroid of a series of colours faster
		ccolours = [_util.change_colour_space(c, self.compare_cspace) for c in self.colour_map[cluster_index]]
		measure = self.measure(ccolours, axis=0)
		if self.ignore_luminocity_channel:
			measure = measure[1:]
		self.clusters[cluster_index] = measure

	def add(self, colour):
		ccolour = _util.change_colour_space(colour, self.compare_cspace)
		if self.ignore_luminocity_channel:
			ccolour = ccolour[1:]
		if len(self.clusters) <= 0:
			self.clusters.append(ccolour)
			self.colour_map.append( [colour] )
			self.n_colours = 1
			return

		ccolours = [ccolour for i in range(len(self.clusters))]
		# dists = cdist([ccolour], self.clusters) #, 'euclidean')
		dists = [euclidean(ccolour, c) for c in self.clusters]
		argmin = np.argmin(dists)
		dist = dists[argmin]

		# if this colour's distance to it's most similar cluster is within tolerance: add this colour to that cluster.
		if dist <= self.max_dist:
			# cluster = self.clusters[argmin]
			self.colour_map[argmin].append(colour)
			
			# update cluster centre
			# if len(self.colour_map[argmin]) % self.cluster_update_interval == 0:
			# if self.n_colours % self.cluster_update_interval == 0:
				# self.update_cluster(argmin)
		else:
			# This colour becomes a new cluster centre.
			self.clusters.append(ccolour)
			self.colour_map.append( [colour] )
			self.dists_sum += dist
			self.n_colours += 1
			self.max_dist = self.dists_sum / self.n_colours * self.max_dist_ratio
			# self.max_dist = self.dists_sum / len(self.clusters) * self.max_dist_ratio

		# self.dists_sum += dist
		# self.n_colours += 1
		self.total_size += 1
		# self.max_dist = self.dists_sum / self.n_colours * 3
		if self.total_size % self.cluster_update_interval == 0:
			for i in range(len(self.clusters)):
				self.update_cluster(i)

	def get_clusters(self):
		clusters = [self.measure(c, axis=0) for c in self.colour_map]
		return clusters

	def display(self, title='colour_map', sort=False, dim=25, per_row=25, to_file=False):
		colour_lists = sorted(self.colour_map, key=len, reverse=True)
		if sort:
			# colour_lists = [np.sort(l, axis=0) for l in colour_lists]
			colour_lists = [sorted(l) for l in colour_lists]
		swatch = _util.build_colour_swatches(colour_lists, dim, per_row)
		cv2.imshow(title, swatch)
		if to_file:
			cv2.imwrite(title+'.png', swatch)
		return swatch

	def plot_data(self):
		fig = plt.figure()
		ax = fig.add_subplot(111, projection='3d')
		ax.set_xlim3d(0, 255)
		ax.set_ylim3d(0, 255)
		ax.set_zlim3d(0, 255)

		for c_list in self.colour_map:
			c_list = [_util.change_colour_space(c, self.compare_cspace) for c in c_list]
			x = [c[0] for c in c_list]
			y = [c[1] for c in c_list]
			z = [c[2] for c in c_list]
			ax.scatter(x, y, z, marker='o', c=np.random.rand(3,))
			# plt.plot(c_list, c=np.random.rand(3,))

		for cluster_centre in self.get_clusters():
			cluster_centre = _util.change_colour_space(cluster_centre, self.compare_cspace)
			ax.scatter([cluster_centre[0]], [cluster_centre[1]], [cluster_centre[2]], s=200, marker='^', c=np.random.rand(3,))

		plt.show() #block=False)

def get_dominant_colours(src, process_opts, **kwargs):
	src = _util.preprocess(src, process_opts)

	if not 'sample' in process_opts:
		process_opts['sample'] = 1000

	colours = src.reshape(-1,3)
	colours = colours[::process_opts['sample']]
	dkmeans = DynamicKMeans(**process_opts)

	for i,col in enumerate(colours):
		dkmeans.add((int(col[0]), int(col[1]), int(col[2])))
		# if 'debug' in process_opts and process_opts['debug'] == True:	
		# 	dkmeans.plot_data()

	# NEW ALGORITHM: at this point we have found the correct number of clusters
	# But each cluster has outliers - colours that should be part of a different 
	# cluster.
	# So for each cluster: calculate the median dist of all colours to the cluster
	# centre AND/OR find the outliers that are too far from the cluster orbit. 
	# For these outliers: either put them in a different cluster or throw away

	if 'debug' in process_opts and process_opts['debug'] == True:	
		dkmeans.display(sort=True, to_file=False)
		dkmeans.plot_data()

	return dkmeans.get_clusters()

if __name__ == "__main__":
	src = cv2.imread('../../images/12.jpg') 
	src = imutils.resize(src, width=CONST_WIDTH)

	# opts = { 'const_width': 360, 'ignore_luminocity_channel': False, 'seed_max_dist': 21, 'max_dist_ratio': 1.5500000000000003, 'cluster_update_interval': 61 }
	opts = {'debug': True, 'sample': 100, 'cluster_update_interval': 3, 'measure': np.median } #, 'seed_max_dist':5, 'max_dist_ratio':0.95}
	
	dominant_colours, timer = _util.time_func( get_dominant_colours, src, opts )
	kmeans_dc, kmeans_timer = _util.time_func( _util.kmeans, np.array([src.reshape(-1,3)[::opts['sample']]]), n_clusers=len(dominant_colours))

	print('Execution: {} seconds'.format(timer))
	print('Execution: {} seconds'.format(kmeans_timer))

	cv2.imshow('src', src)
	cv2.imshow('dominant colours', _util.build_colour_swatch(dominant_colours, 30, 30)) 
	cv2.imshow('kmeans', _util.build_colour_swatch(kmeans_dc, 30, 30))
	cv2.waitKey(0)


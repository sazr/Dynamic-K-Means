import cv2
import math
import numpy as np 
import numpy.linalg as la
import imutils
import itertools
import glob
import json
import os
from sklearn.cluster import KMeans
from skimage import exposure
from timeit import default_timer as timer
from skimage.morphology import skeletonize
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
import scipy.misc
import skimage.transform
import skimage.color
# import _util

CONST_WIDTH = 300

def log(img):
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	b1 = cv2.GaussianBlur(gray, (7,7), 0)
	lap = cv2.Laplacian(b1, cv2.CV_64F)
	lap = lap.astype('uint8')
	return lap

def dog(img, equalise=False):
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	if equalise:
		gray = cv2.equalizeHist(gray)
	b1 = cv2.GaussianBlur(gray, (1,1), 0)
	b2 = cv2.GaussianBlur(gray, (3,3), 0)
	return np.subtract(b1, b2)

def quantize(img, n_bins=8):
	# Pre: ALL channels must have the same max, ie, 255. So you cant use the function on HSV
	# because the maxes for each channel differ (180,255,255)
	return np.multiply(np.round(np.multiply(img, (n_bins/255.0))), (255.0/n_bins)).astype('uint8')

def quantize_channels(img, n_bins=(8,8,8), n_ranges=(255,255,255)):
	channels = cv2.split(img)
	if n_ranges[0] > 0:
		channels[0] = np.multiply(np.round(np.multiply(channels[0], (n_bins[0]/float(n_ranges[0])))), (float(n_ranges[0])/n_bins[0])).astype('uint8')
	if n_ranges[1] > 0:
		channels[1] = np.multiply(np.round(np.multiply(channels[1], (n_bins[1]/float(n_ranges[1])))), (float(n_ranges[1])/n_bins[1])).astype('uint8')
	if n_ranges[2] > 0:
		channels[2] = np.multiply(np.round(np.multiply(channels[2], (n_bins[2]/float(n_ranges[2])))), (float(n_ranges[2])/n_bins[2])).astype('uint8')
	img = cv2.merge(channels)
	return img

def hue_quantize(img, is_hsv=False, n_bins=8):
	if not is_hsv:
		img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

	channels = cv2.split(img)
	channels[0] = np.multiply(np.round(np.multiply(channels[0], (n_bins/180.0))), (180.0/n_bins)).astype('uint8')
	channels[1] = np.multiply(np.round(np.multiply(channels[1], (n_bins/255.0))), (255.0/n_bins)).astype('uint8')
	# channels[2] = np.multiply(np.round(np.multiply(channels[2], (n_bins/255.0))), (255.0/n_bins)).astype('uint8')
	
	# channels[0] = bin_values(channels[0], n_bins, 180) #float(180/n_bins))
	# channels[1] = bin_values(channels[1], int(255/n_bins))
	# channels[2] = bin_values(channels[2], int(255/n_bins)) #np.full(channels[2].shape, 125, dtype='uint8')
	img = cv2.merge(channels)

	if not is_hsv:
		img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
	return img

def darken(img, is_hsv=False, ratio=0.75):
	# ALTERNATE way to darken/brighten image is to use a blur/mean kernel but the sum of the kernel
	# is < 1 (for darker) or > 1 (for brighter)
	# 
	# Darken an image by dividing the Value channel by a factor (in a HSV image)
	# More light equals more information, thus a darker image should mean the inverse
	# right?
	if not is_hsv:
		img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	img[...,2] = img[...,2]*ratio
	if not is_hsv:
		img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
	return img

def normalise_contrast(img):
	img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV) #COLOR_BGR2YCrCb)
	c1,c2,c3 = cv2.split(img)
	c1 = cv2.equalizeHist(c1)
	# c2 = cv2.equalizeHist(c2)
	# c3 = cv2.equalizeHist(c3)
	img = cv2.merge((c1,c2,c3))
	img = cv2.cvtColor(img, cv2.COLOR_YUV2BGR) #COLOR_YCR_CB2BGR)
	return img
	# img = exposure.equalize_hist(img.copy()) # good alternative to above
	# img = exposure.rescale_intensity(img.copy(), out_range=(0, 255)) # no different results
	
def bin_values(colour, bin_rng=6, channel_maxes=[255,255,255]):
	steps = np.divide( np.array(channel_maxes), float(bin_rng) )
	colour = np.divide(colour, steps)
	colour = np.round(colour)
	colour = np.multiply(colour, steps)
	return colour.astype('uint8')

def bin_image(src, bin_rng=6, channel_maxes=[255,255,255]):
	steps = np.divide( np.array(channel_maxes), float(bin_rng) )
	src = np.divide(src, steps)
	src = np.round(src)
	src = np.multiply(src, steps)
	return src.astype('uint8')

def colour_blocks_by_dilation(src, shape=cv2.MORPH_ELLIPSE, kernel=(15,15), iterations=1):
	return cv2.dilate(src.copy(), cv2.getStructuringElement(shape, kernel), iterations=iterations)

def colour_blocks_by_erosion(src, shape=cv2.MORPH_ELLIPSE, kernel=(15,15), iterations=1):
	return cv2.erode(src.copy(), cv2.getStructuringElement(shape, kernel), iterations=iterations)

def colour_simplify(src, shape=cv2.MORPH_ELLIPSE, kernel=(15,15), iterations=1):
	src = colour_blocks_by_dilation(src, shape, kernel, iterations)
	src = colour_blocks_by_erosion(src, shape, kernel, iterations)

	src = colour_blocks_by_erosion(src, shape, kernel, iterations)
	src = colour_blocks_by_dilation(src, shape, kernel, iterations)
	return src

def count_colours(src):
	unique, counts = np.unique(src.reshape(-1, src.shape[-1]), axis=0, return_counts=True)
	return counts.size
	
def change_colour_space(bgr_colour, colour_space):
	m = np.full((1, 1, 3), bgr_colour, dtype='uint8')
	m = cv2.cvtColor(m, colour_space)
	return (int(m[0,0][0]), int(m[0,0][1]), int(m[0,0][2]))

def get_mode(values, ignore_blk=None, get_all=False):
	mode_map = {}

	for r in range(0, values.shape[0]):
		for c in range(0, values.shape[1]):
			value = values[r,c]
			if ignore_blk and value.sum() <= 0:
				continue
			binned_value = bin_values(value, bin_rng=12)
			value = tuple(binned_value.tolist()) 
			if not value in mode_map:
				mode_map[value] = 1
			else:
				mode_map[value] += 1

	s = zip(mode_map.values(), mode_map.keys())
	s = sorted(s, key = lambda x: x[0], reverse=True) 
	# mode = max(s)
	# return mode[1] if not get_all	else s 
	return s[0][1] if not get_all	else s 

def get_lab_distance(lab1, lab2):
	L = int(lab1[0]) - int(lab2[0])
	A = int(lab1[1]) - int(lab2[1])
	B = int(lab1[2]) - int(lab2[2])
	return math.sqrt((L * L) +  (A * A) +  (B * B))

def get_hsv_distance(hsv1, hsv2):
	return 1

def round_to(val, round_to=1):
	return int(round_to * round(val/round_to))

def vector_angle(v1, v2):
	""" Returns the angle in radians between vectors 'v1' and 'v2'    """
	cosang = np.dot(v1, v2)
	sinang = la.norm(np.cross(v1, v2))
	return math.degrees( np.arctan2(sinang, cosang) )

def rotate(origin, point, angle):
	# Rotate a point counterclockwise by a given angle around a given origin
	# The angle should be given in radians
	ox, oy = origin
	px, py = point
	qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
	qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
	return int(qx), int(qy)

def rotate_rect(origin, poly, angle_degrees):
	rotated = []
	angle = math.radians(angle_degrees)
	for pt in poly:
		rot = rotate(origin, pt, angle)
		rotated.append(rot)
	return rotated

def mask_poly(src, poly):
	mask = np.ones((src.shape[0], src.shape[1], 3), dtype='uint8')
	cv2.drawContours(mask, [np.array(poly)], -1, (0,0,0), -1)
	msrc = np.ma.array(src, mask=mask) #, dtype='uint8')
	return msrc
	# masked = cv2.bitwise_and(src, src, mask=mask)
	# return masked

def mask_poly_cv2(src, poly):
	mask = np.zeros((src.shape[0], src.shape[1], 1), dtype='uint8')
	cv2.drawContours(mask, [np.array(poly)], -1, (255,), -1)
	masked = cv2.bitwise_and(src, src, mask=mask)
	return masked

def visualise_segment_labels_colour(segments):
	# colours = {
	# 	1: (255,0,0), 
	# 	2: (0,255,0), 
	# 	3: (0,0,255), 
	# 	4: (255,255,0), 
	# 	5: (255,0,255), 
	# 	6: (0,255,255), 
	# 	7: (125,0,255), 
	# 	8: (255,0,125), 
	# 	9: (125,255,0), 
	# 	10: (0,255,125), 
	# 	11: (45,255,125), 
	# 	12: (125,255,45), 
	# 	13: (45,255,45), 
	# 	14: (255,255,125)
	# }
	# colour_segments = np.zeros((segments.shape[0], segments.shape[1], 3), dtype='uint8')
	# for r in range(segments.shape[0]):
	# 	for c in range(segments.shape[1]):
	# 		colour_segments[r,c] = colours[int(segments[r,c])]
	# return colour_segments

	# return cv2.applyColorMap(cv2.equalizeHist(segments), cv2.COLORMAP_JET)

	# return cv2.applyColorMap(visualise_segment_labels(segments), cv2.COLORMAP_JET)

	# label_range = np.linspace(0, 1, 256)
 #  lut = np.uint8(plt.cm.viridis(label_range)[:,2::-1]*256).reshape(256, 1, 3) 
 #  return cv2.LUT(cv2.merge((labels, labels, labels)), lut)

	# label_hue = np.uint8(179*labels/np.max(segments))
	label_hue = np.uint8((53 * segments.astype(int)) % 180)
	# golden_angle = 180 * (3 - np.sqrt(5))
	# label_hue = segments * golden_angle % 360
	blank_ch = 255*np.ones_like(label_hue)
	labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

	# cvt to BGR for display
	labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)
	return labeled_img

def visualise_segment_labels(segments):
	max_label = np.amax(segments)
	step = math.floor(255/max_label)
	return np.multiply(segments, step)

# def visualise_segment_labels(segments):
# 	return cv2.equalizeHist(segments)

def kmeans(src, n_clusers=7):
	# TODO: sort from most frequent to least frequent. See this for how: https://www.pyimagesearch.com/2014/05/26/opencv-python-k-means-color-clustering/
	src = src.reshape((src.shape[0] * src.shape[1], 3))
	clt = KMeans(n_clusters=n_clusers)
	clt.fit(src)
	return clt.cluster_centers_

def mean_shift(src):
	return cv2.pyrMeanShiftFiltering(src, 30, 30, 3)

def gen_permutations(x):
	return itertools.product(*x)

def dict_permutations(d):
	# Example usage:
	# tuner_params = {
	#    'darken': (True, False),
	#    'hue_quantize': (True, False),
	#    'quantize': (True, False),
	#    'blur': gen_permutations([range(1,26), range(1,26)]),
	#    'h_samples': range(2, 21),
	#    'v_samples': range(2, 21), 
	#    'offset': np.arange(0.05, 0.95, 0.05), 
	#    'kernel': gen_permutations([range(1,26), range(1,26)])
	# }
	# print(dict_permutations(tuner_params))

	keys = d.keys()
	permutations = list(gen_permutations(d.values()))
	d_permutations = [dict(zip(keys, x)) for x in permutations]
	return d_permutations

def build_colour_swatch(colours, dim=25, per_row=25):
	per_row = len(colours) if per_row > len(colours) else per_row 
	swatch = np.zeros((math.ceil(len(colours)/per_row)*dim, dim*per_row, 3), dtype='uint8')
	x = 0
	y = 0
	for colour in colours:
		print('888')
		print(colour)
		cv2.rectangle(swatch, (x,y), (x+dim,y+dim), np.array(colour).tolist(), -1)
		x+=dim
		if x % (dim*per_row) == 0:
			y += dim
			x = 0

	return swatch

def build_colour_swatches(colour_lists, dim=25, per_row=25):
	per_rows = []
	offset = 15
	width = 0
	height = -offset
	x = 0
	y = 0

	biggest = max([len(x) for x in colour_lists])
	per_row = biggest if biggest < per_row else per_row 
	width = dim*per_row

	for col_list in colour_lists:
		height += (math.ceil(len(col_list)/per_row) * dim) + offset

	swatch = np.zeros((height, width, 3), dtype='uint8')
	# cv2.rectangle(swatch, (0,0), (width,y), (0,255,0), -1)
	for col_list in colour_lists:
		for colour in col_list:
			cv2.rectangle(swatch, (x,y), (x+dim,y+dim), (int(colour[0]), int(colour[1]), int(colour[2])), -1)
			x+=dim
			if x % (dim*per_row) == 0:
				y += dim
				x = 0
		if x > 0:
			y += dim
		y += offset
		x = 0
		# cv2.rectangle(swatch, (0,y-offset), (width,y), (0,255,0), -1)

	return swatch

def simplify_to_cells(src, kernel_w=7, kernel_h=7, slide_h=7, slide_v=7, measure=np.mean, axis=(0,1), output=None):
	res = np.zeros_like(src)
	output = output if output else np.array([kernel_w, kernel_h])
	# for r in range(0, src.shape[0]-kernel_h, slide_v):
	# 	for c in range(0, src.shape[1]-kernel_w, slide_h):
	# 		centre = (r + math.floor(kernel_h/2), c + math.floor(kernel_w/2))
	# 		roi = src[r:r+kernel_h, c:c+kernel_w]
	# 		m = measure(roi, axis=axis)
	# 		cv2.rectangle(res, (c,r), tuple(np.add((c,r), output)), m, -1)

	for (roi, pos, strt, end, padded) in strt_end_centre_gen(src, kernel=(kernel_w, kernel_h), slide=(slide_h,slide_v)):
		m = measure(roi, axis=axis)
		cv2.rectangle(res, tuple(np.subtract(pos, output)), tuple(np.add(pos, output)), m, -1)

	return res
			

def remove_outliers(data, thresh=1.5, axis=(0,1), use_median=False, **kwargs):
	# Post: Remove outlier values from data. A value in data is considered an outlier if it is NOT mean-std_deviation*thresh < value < mean+std_deviation*thresh

	inliers 			= []
	indexes 			= []
	median 				= np.median(data, axis) 
	mean, std_dev = cv2.meanStdDev(data)
	measure 			= median if use_median else mean
	lower_thresh 	= np.subtract(measure, np.multiply(std_dev, thresh))
	upper_thresh 	= np.add(measure, np.multiply(std_dev, thresh))

	# Handle arrays that are n dimensional
	if len(data.shape) == 3:
		data = data.reshape((int(data.size/3), 3))

	for v in data:
		if np.all(v > lower_thresh) and np.all(v < upper_thresh):
			inliers.append(v)
			indexes.append(True)
		else:
			indexes.append(False)

	return (inliers, indexes)

def skeletonise(gray, method='zhang'): #'lee'):
	gray[gray>0] = 1
	gray[gray<1] = 0
	skeleton = skeletonize(gray, method=method).astype('uint8')
	skeleton[skeleton > 0.0] = 255
	return skeleton

def time_func(func, *args, **kwargs):
	start = timer()
	res = func(*args, **kwargs)
	end = timer()
	return (res, end - start) # Time in seconds, e.g. 5.38091952400282

def rename_imgs(dir_path, n=1):
	# Pre: dir_path must end in '/'
	img_exts = (
		os.path.join(dir_path, '*.gif'), 
		os.path.join(dir_path, '*.png'), 
		os.path.join(dir_path, '*.jpg'), 
		os.path.join(dir_path, '*.jfif'), 
		os.path.join(dir_path, '*.jpeg')
	)
	files = []
	for ext in img_exts:
		files.extend(glob.glob(ext))
	
	for (i, path) in enumerate(files):
		file_path = os.path.dirname(path)
		file_ext = os.path.splitext(path)[1]
		new_path = os.path.join(file_path, str(n) + file_ext)
		n += 1

		while os.path.exists(new_path):
			new_path = os.path.join(file_path, str(n) + file_ext)
			n += 1

		print('Renaming: {} to {}'.format(path, new_path))
		os.rename(path, new_path)

def change_img_type(dir_path, extension='.png'):
	# Pre: dir_path must end in '/'
	img_exts = (
		os.path.join(dir_path, '*.gif'), 
		os.path.join(dir_path, '*.png'), 
		os.path.join(dir_path, '*.jpg'), 
		os.path.join(dir_path, '*.jfif'), 
		os.path.join(dir_path, '*.jpeg')
	)
	files = []
	for ext in img_exts:
		files.extend(glob.glob(ext))
	
	for (i, path) in enumerate(files):
		src = cv2.imread(path)
		file_path = os.path.dirname(path)
		file_name, file_ext = os.path.splitext(path)
		new_path = os.path.join(file_path, file_name + extension)
		print('Changeing file type from: {} to {}'.format(path, new_path))
		cv2.imwrite(new_path, src)
		os.remove(path)

def pad_mat(src, kernel):
	x_pad = (math.ceil(src.shape[1]/kernel[0]) * kernel[0]) - src.shape[1]
	y_pad = (math.ceil(src.shape[0]/kernel[1]) * kernel[1]) - src.shape[0]
	padded = np.zeros(np.add(src.shape, (y_pad*2, x_pad*2, 0)), dtype=src.dtype)
	padded[y_pad:padded.shape[0]-y_pad, x_pad:padded.shape[1]-x_pad]= src
	return (padded, (x_pad, y_pad))

def strt_end_gen(src, kernel=(5,5), slide=(1,1)):
	for r in range(0, src.shape[0], slide[1]):
		for c in range(0, src.shape[1], slide[0]):
			roi = src[r:r+kernel[1], c:c+kernel[0]]
			yield (roi, (c,r))

def strt_end_centre_gen(src, kernel=(5,5), slide=(1,1)):
	half = (float(kernel[0]/2), float(kernel[1]/2))
	padded = cv2.copyMakeBorder(src, int(half[1])*2, int(half[1])*2, int(half[0])*2, int(half[0])*2, cv2.BORDER_CONSTANT, None, (0,0,0))
	for r in range(int(half[1]), padded.shape[0]-kernel[1], slide[1]):
		for c in range(int(half[0]), padded.shape[1]-kernel[0], slide[0]):
			strt 				= np.ceil(np.subtract((c,r), half)).astype(int)
			end 				= np.floor(np.add((c,r), half)).astype(int)
			adj_strt 		= tuple(np.subtract(strt, np.floor(half).astype(int)))
			adj_end 		= tuple(np.subtract(end, np.floor(half).astype(int)))
			adj_centre 	= tuple(np.subtract((c,r), np.floor(half).astype(int)))
			roi 				= padded[strt[1]:end[1], strt[0]:end[0]]
			# print('c: {}, r: {}'.format(c,r))
			yield (roi, adj_centre, adj_strt, adj_end, padded)

def centre_out_gen(src, kernel=(5,5), slide=(1,1), pad=True):
	if pad:
		src, pad_offset = pad_mat(src, kernel)
	centre = (math.floor(src.shape[1]/2)-kernel[0], math.floor(src.shape[0]/2)-kernel[1])
	current = centre
	it = 0

	yield (src, src[centre[1]:centre[1]+kernel[1], centre[0]:centre[0]+kernel[0]], centre)

	while True:
		it += 1
		strt = np.subtract(centre, np.multiply(kernel, (it,it)))
		end = np.add(centre, np.multiply(kernel, (it,it)))

		if strt[0] < 0 and strt[1] < 0 and end[0] >= src.shape[1] and end[1] >= src.shape[0]:
			break

		# top
		if strt[1] >= 0:
			r = strt[1]
			for c in range(strt[0], end[0], slide[0]):
				roi = src[r:r+kernel[1], c:c+kernel[0]]
				yield (src, roi, (c,r))
		# right
		if end[0] < src.shape[1]:
			c = end[0]
			for r in range(strt[1], end[1], slide[1]):
				roi = src[r:r+kernel[1], c:c+kernel[0]]
				yield (src, roi, (c,r))
		# bottom
		if end[1] < src.shape[0]:
			r = end[1]
			for c in range(end[0], strt[0], -slide[0]):
				roi = src[r:r+kernel[1], c:c+kernel[0]]
				yield (src, roi, (c,r))
		# left
		if strt[0] < src.shape[1]:
			c = strt[0]
			for r in range(end[1], strt[1], -slide[1]):
				roi = src[r:r+kernel[1], c:c+kernel[0]]
				yield (src, roi, (c,r))

def selectPoly(src):
	poly = []
	cur_src = src.copy()
	backup_src = src.copy()
	def callback(evt, x, y, flags, user_data):
		if evt == cv2.EVENT_LBUTTONDOWN:
			if x >= 0 and y >= 0 and x < backup_src.shape[1] and y < backup_src.shape[0]:
				poly.append( (x,y) )
				cv2.circle(cur_src, (x,y), 2, (0,0,255), -1)
				cv2.imshow('Select polyline', cur_src)

	while True:
		cv2.namedWindow('Select polyline')
		cv2.moveWindow("Select polyline", 600, 100)
		cv2.setMouseCallback("Select polyline", callback)
		cv2.imshow('Select polyline', cur_src)
		key = cv2.waitKey(0)
		if key == 27 or key == 13:
			break

	return poly

def selectROIEx(src, fromCenter=False):
	rois = cv2.selectROI('Select ROIs', src.copy(), fromCenter)
	return [[rois[0],rois[1],], [rois[0]+rois[2],rois[1],], 
		[rois[0]+rois[2],rois[1]+rois[3],], [rois[0],rois[1]+rois[3],], ]

def as_heightmap(grayscale):
	# grayscale = imutils.resize(grayscale, width=100)
	# grayscale = skimage.color.rgb2gray(grayscale)

	# create the x and y coordinate arrays (here we just use pixel indices)
	xx, yy = np.mgrid[0:grayscale.shape[0], 0:grayscale.shape[1]]
	fig = plt.figure()
	ax = fig.gca(projection='3d')
	ax.plot_surface(xx, yy, grayscale, rstride=1, cstride=1, cmap=plt.cm.gray, linewidth=0)
	# ax.set_zticks(np.arange(0,200000,100))
	ax.set_xlabel('X Label')
	ax.set_ylabel('Y Label')
	ax.set_zlabel('Z Label')
	plt.show()

def preprocess(src, options):
	if 'const_width' in options and int(options['const_width']) > 0:
		src = imutils.resize(src, width=options['const_width'])
	if 'simplify_to_cells' in options and options['simplify_to_cells']:
		src = simplify_to_cells(src)
	if 'blur' in options and sum(options['blur']) > 0:
		src = cv2.GaussianBlur(src, options['blur'], 0)
	if 'median_blur' in options and int(options['median_blur']) > 0:
		src = cv2.medianBlur(src, options['median_blur'])
	if 'mean_blur' in options and sum(options['mean_blur']) > 0:
		src = cv2.blur(src, options['mean_blur'])
	if 'darken' in options and options['darken']:
		src = darken(src)
	if 'normalise_contrast' in options and options['normalise_contrast']:
		src = normalise_contrast(src)
	if 'hue_quantize' in options and int(options['hue_quantize']) > 0:
		src = hue_quantize(src, n_bins=options['hue_quantize'])
	if 'colour_space' in options and options['colour_space']:
		src = cv2.cvtColor(src, options['colour_space'])
	if 'quantize' in options and int(options['quantize']) > 0:
		src = quantize(src, n_bins=options['quantize'])
	return src

class GroundTruthEle:
	def __init__(self, path, gt, results=[]):
		self.path = path
		self.ground_truth = gt or {} # dict(gt or {})
		self.results = results
		# self.results = [{
		# 	'params': {},
		# 	'result': [],
		# 	'accuracy': -1
		# }]

	def add_result(params, result, accuracy):
		self.results.append({
			'params': params,
			'result': result,
			'accuracy': accuracy
		})

	def __json__(self):
		return self.__dict__
	def __repr__(self):
		return str(self.__dict__)
	def __str__(self):
		return str(self.__dict__)

class _UtilEncoder(json.JSONEncoder):
	def default(self, obj):
		if isinstance(obj, GroundTruthEle): 
			return obj.__json__()
		return json.JSONEncoder.default(self, obj)

def _UtilDecoder(dct):
	if 'path' in dct and 'ground_truth' in dct and 'results' in dct:
		return GroundTruthEle(dct["path"], dct["ground_truth"], dct["results"])
	return dct

def main():
	src = cv2.imread('../images/ls9.jpg')
	src = imutils.resize(src, width=CONST_WIDTH)
	# src = cv2.medianBlur(src, 11)
	# src = cv2.GaussianBlur(src, (11,11), 0)
	cv2.imshow('src', src)
	cv2.imshow('quantized', quantize(src, 8))
	cv2.imshow('quantized_hue', hue_quantize(src, False, 8))
	# cv2.imshow('colours', build_colour_swatch( kmeans(src) ))
	
	# rename_imgs('C:/Users/admin\Desktop/raspberry_pi/grass_detector/data/testing')
	# change_img_type('C:/Users/admin\Desktop/raspberry_pi/grass_detector/data/soccer_ball')
	cv2.waitKey(0)

if __name__ == "__main__":
	main()
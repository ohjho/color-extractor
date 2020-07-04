import os, io, validators, colorsys, webcolors, warnings, time
import numpy as np
from PIL import Image, ImageDraw
import urllib.request as urllib
from tqdm import tqdm

def get_im(im, verbose = False):
    '''
    return a PIL image object
    Args:
        im: np.array, filepath, or url
    '''
    pil_im = None
    if type(im) == np.ndarray:
        pil_im = Image.fromarray(im)
    elif os.path.isfile(im):
        pil_im = Image.open(im)
    elif validators.url(im):
        r = urllib.urlopen(im)
        pil_im = Image.open(io.BytesIO(r.read()))
    else:
        raise ValueError(f'get_im: im must be np array, filename, or url')

    if verbose:
        print(f'Find image of size {pil_im.size}')
    return pil_im

def resize_im(im, max_h = 100):
    o_w, o_h = im.size
    h = min(o_h, max_h)
    w = int(o_w * h/ o_h)
    im_small = im.resize((w,h), Image.ANTIALIAS) #best downsize filter
    return im_small

def remove_alpha(pixels_array, verbose = False):
    '''
    Args:
        pixels_array: an X x 4 np array where the 4th channel
    '''
    if pixels_array.shape[1] != 4:
        raise ValueError(f'input pixels_array needs 4 channels')

    alpha = pixels_array[:,3]
    index_to_remove = np.where( alpha == 0)[0]
    out_array = np.delete(pixels_array, index_to_remove, axis = 0)
    out_array = np.delete(out_array, 3, axis =1)

    if verbose:
        print(f'Original array shape: {pixels_array.shape}')
        print(f'transparent pixels found: {index_to_remove.shape} ({"{:.2f}".format(index_to_remove.shape[0]/ pixels_array.shape[0]*100)}%)')
        print(f'resulting pixel array shape: {out_array.shape}')
    return out_array

def kmeans(data, k=3, normalize=False, max_iter = 5000, min_iter = 20, verbose = False):
    """
    Basic k-means clustering algorithm.
    Args:
        min_iter: prevents early stops (lower will be faster)
        normalize: optionally normalize the data. k-means will perform poorly or strangely if the dimensions don't have the same ranges.
    """
    if normalize:
        stats = (data.mean(axis=0), data.std(axis=0))
        data = (data - stats[0]) / stats[1]

    # initialize centroids
    #centers = data[:k]
    centers = data[np.random.choice(np.arange(len(data)), k)]

    pbar = tqdm(range(max_iter), desc = 'Clustering') if verbose else range(max_iter)
    for i in pbar:
        # core of clustering algorithm...
        # first, use broadcasting to calculate the distance from each point to each center, then
        # classify based on the minimum distance.

        #classifications = np.argmin(((data[:, :, None] - centers.T[None, :, :])**2).sum(axis=1), axis=1)
        classifications = np.array([np.argmin([np.dot(x_i-y_k, x_i-y_k) for y_k in centers]) for x_i in data])

        # next, calculate the new centers for each cluster.

        #new_centers = np.array([data[classifications == j, :].mean(axis=0) for j in range(k)])
        new_centers = np.array([data[classifications == i].mean(axis = 0) for i in range(k)])

        if len(np.unique(classifications)) < k:
            # there are fewer than K clusters, start again
            centers = data[np.random.choice(np.arange(len(data)), k)]
        elif (new_centers == centers).all() and i > min_iter:
            # if the centers aren't moving anymore it is time to stop.
            break
        else: # keep looking
            centers = new_centers
    else:
        # this will not execute if the for loop exits on a break.
        #if not force_output:
        #    raise RuntimeError(f"Clustering algorithm did not complete within {max_iter} iterations")
        warnings.warn(f'Clustering algorithm did not complete within {max_iter + 1}')
        return None, None

    if normalize:
        centers = centers * stats[1] + stats[0]

    if verbose:
        print(f"Clustering completed after {i+1} iterations")

    return classifications, centers

def find_histogram(labels, centers, bDict = False):
    """
    create a histogram with k clusters
    all this function is doing is counting the number of pixels that belong to each cluster
    :param: clt
    :return:hist
    """
    numLabels = np.arange(0, len(np.unique(labels)) + 1)
    (hist, _) = np.histogram(labels, bins=numLabels)

    hist = hist.astype("float")
    hist /= hist.sum()

    centroids =  list(centers) #[ list[center] for center in list(clt.cluster_centers_)]
    if bDict:
        return { pct : np.array(color, dtype = int) for pct , color in zip(hist, centers)}
    else:
        return hist

def plot_colors(hist, centroids, w= 300 , h = 50):
    # initialize the bar chart representing the relative frequency of each of the colors
    bar = np.zeros((50, 300, 3), dtype = "uint8")
    startX = 0

    im = Image.new('RGB', (300, 50), (128, 128, 128))
    draw = ImageDraw.Draw(im)

    # loop over the percentage of each cluster and the color of
    # each cluster
    for (percent, color) in zip(hist, centroids):
        # plot the relative percentage of each cluster
        endX = startX + (percent * 300)
        xy = (int(startX), 0, int(endX), 50)
        fill = tuple(color.astype('uint8').tolist())
        draw.rectangle(xy, fill)
        startX = endX

    # return the bar chart
    im.resize( (w,h))
    return im

def get_xkcd_color(url = 'http://xkcd.com/color/rgb.txt'):
	'''
	Get xkcd_color hex code to name map
	assuming that the file is tsv
	'''
	page = urllib.urlopen(url)
	lines = [line.decode('UTF-8').strip() for line in page.readlines() ]
	xkcd_colors = [line.split('\t') for line in lines if not line.startswith('#')]
	xkcd_colors = {hex_str.strip() : name.strip() for name, hex_str in xkcd_colors}
	return xkcd_colors

def closest_colour(rgb_tuple, hex_name_map_dict = webcolors.CSS2_HEX_TO_NAMES):
	'''
	Take a tuple of RGB code and return the closest color name
	 within the given hex_name_map_dict
	reference: https://stackoverflow.com/questions/54242194/python-find-the-closest-color-to-a-color-from-giving-list-of-colors?noredirect=1&lq=1
	Args:
		rgb_tuple: tuple of three integers representing RGB
		hex_name_map_dict: a dictionary of hex code to color name
	'''
	map_rgbs = np.array([tuple(webcolors.hex_to_rgb(hex_code))
						 for hex_code in hex_name_map_dict.keys()])
	target_rgb = np.array(rgb_tuple)
	distances = np.sqrt(np.sum((map_rgbs - target_rgb)**2, axis = 1))
	index_of_smallest = np.argmin(distances)
	return list(hex_name_map_dict.values())[index_of_smallest]

def im_kmeans(im, k = 3, get_json = False, verbose = False,
                kmeans_kargs = {'normalize': True}, color_name_space = 'css2'):
    '''
    return centers and pct match for each for the given image
    Args:
        max_retry: if kmeans fails , rerun
    '''
    im = get_im(im, verbose = verbose)
    im_small = resize_im(im)
    im_arr = np.array(im_small)
    h , w, c = im_arr.shape
    im_data = im_arr.reshape((h * w, c))
    has_alpha = False
    s_time = time.time()

    if c == 4:
        has_alpha = True
        im_data = remove_alpha(im_data, verbose = verbose)

    labels, centroids = kmeans(im_data, k = k, verbose = verbose, **kmeans_kargs)
    if type(labels) != np.ndarray and type(centroids) != np.ndarray:
        if get_json:
            return None
        else:
            return None, None

    hist = find_histogram(labels, centroids)

    if get_json:
        cname_space_dict = {
            'xkcd': get_xkcd_color(),
            'css2': webcolors.CSS2_HEX_TO_NAMES
        }
        if color_name_space not in cname_space_dict.keys():
            raise ValueError(f'im_kmeans: color_name_space {color_name_space} must be one of {cname_space_dict.keys()}')

        color_data = {
            i : {
                'rbg': list(h_rbg[1]),
                'hsv': list(colorsys.rgb_to_hsv(
                    h_rbg[1][0]/255, h_rbg[1][1]/255, h_rbg[1][2]/255
                    )),
                'name': closest_colour(tuple(h_rbg[1]), hex_name_map_dict = cname_space_dict[color_name_space]),
                'percentage': h_rbg[0]
            }
            for i, h_rbg in enumerate(sorted(zip(hist, centroids), reverse = True))
        }
        out_json = {
            'centroids': color_data,
            'meta': {
                'has_alpha': has_alpha,
                'k': k,
                'color_space': color_name_space,
                'kmeans_kargs': kmeans_kargs,
                'compute_time': "{:.2f}".format(time.time() - s_time) + 's'
            }
        }
        return out_json
    else:
        return centroids, list(hist)

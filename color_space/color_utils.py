import os, webcolors, argparse, json
import urllib.request
import numpy as np
from tqdm import tqdm

def get_xkcd_color(url = 'http://xkcd.com/color/rgb.txt'):
	'''
	Get xkcd_color hex code to name map
	assuming that the file is tsv
	'''
	page = urllib.request.urlopen(url)
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

def Main(color_space = 'css2', sample_npz = 'color_names.npz', output_path = None):
	if not os.path.isfile(sample_npz):
		raise ValueError(f'sample npz is not found: {sample_npz}')

	valid_conv = ['css2', 'xkcd']
	if color_space not in valid_conv:
		raise ValueError(f'{color_space} not in supported conventions: {valid_conv}')

	hex_name_map_dict = None
	if color_space == 'css2':
		hex_name_map_dict = webcolors.CSS2_HEX_TO_NAMES
	elif color_space == 'xkcd':
		hex_name_map_dict = get_xkcd_color()

	npz = dict(np.load(sample_npz))
	npz['labels'] = [closest_colour(rgb, hex_name_map_dict)
					for rgb in tqdm(npz['samples'], desc = f'mapping colors to {color_space}')]
	if output_path:
		if output_path.endswith('.json'):
			outdict = {'samples': npz['samples'].tolist(), 'labels': list(npz['labels'])}
			with open(output_path, 'w') as fh:
				json.dump(outdict, fh)
		elif output_path.endswith('.npz'):
			np.savez( output_path, samples = npz['samples'], labels = np.array(npz['labels']))
		else:
			raise ValueError(f'file type not recognized: {output_path}')
		print(f'{color_space} file saved to {output_path}')

	return npz

if __name__ == '__main__':
	parser = argparse.ArgumentParser(
		description = f'Color Utility: Generating color space npz files'
	)
	parser.add_argument('--color_space_name', type = str, help = 'color space name: css2 or xkcd', required = True)
	parser.add_argument('--sample_npz', type = str, help = 'path of color samples npz', required = True)
	parser.add_argument('--output_path', type = str, help = 'output path of color space file in npz format', default = None)
	args = parser.parse_args()

	Main(color_space = args.color_space_name, sample_npz = args.sample_npz, output_path = args.output_path)

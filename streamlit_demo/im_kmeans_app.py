import streamlit as st

import os, sys, json
import numpy as np
from PIL import Image

from st_utils import file_selector, show_miro_logo, get_image

#Paths
cwdir = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(1, os.path.join(cwdir, "../"))
from img_color import im_kmeans, plot_colors

def Main():
	app_desc = '''
	fashion color detection without deep learning \n
	* [docs on extractor's settings](https://github.com/ohjho/color-extractor#passing-settings)
	* [xkcd color space](https://xkcd.com/color/rgb/)
	* [webcolors' color space](https://webcolors.readthedocs.io/en/1.11.1/colors.html)
	'''
	show_miro_logo()
	st.sidebar.header('image color kmeans')
	#st.sidebar.markdown(app_desc)

	# user input
	default_settings = {'normalize': True, 'max_iter': 5000, 'min_iter': 50}
	settings_dict = json.loads(
 		st.sidebar.text_area('kmeans settings', value = json.dumps(default_settings))
 	)
	im = get_image(st_asset = st.sidebar, as_np_arr = True)
	num_clusters = st.sidebar.number_input('number of clusters', min_value = 1, max_value = 10, value = 3)
	color_name_space = st.sidebar.selectbox('color name space', options = ['xkcd', 'css2'])

	if type(im) == np.ndarray:
		result = im_kmeans(im, k = num_clusters, get_json = True, verbose = False,
						kmeans_kargs = settings_dict, color_name_space = color_name_space
						)
		if result:
			st.success(f'Color found. Has Alpha: `{result["meta"]["has_alpha"]}`')
			st.json(result)

			hist = [v['percentage'] for k, v in result['color'].items() ]
			centroids = [v['rbg'] for k, v in result['color'].items() ]
			cnames = [v['name'] for k, v in result['color'].items() ]
			l_caption = [ f'{c}: {"{:.0f}".format(p*100)}%'for c, p in zip(cnames, hist)]
			
			plot = plot_colors(np.array(hist), np.array(centroids))
			st.write('#### result')
			st.image(plot, f'color found {l_caption}', use_column_width = True)
			st.write('#### original')
			st.image(im, width = 500, use_column_width = True)
		else:
			st.warning(f'Clustering algo failed to find {num_clusters}')
if __name__ == '__main__':
	Main()

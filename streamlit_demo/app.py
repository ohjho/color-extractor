import streamlit as st

import os, sys, json
import numpy as np
from PIL import Image

from st_utils import file_selector, show_miro_logo, get_image

#Paths
cwdir = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(1, os.path.join(cwdir, "../"))
from color_extractor import ImageToColor

@st.cache
def get_color_samples_labels(npz_fn):
	npz = np.load(npz_fn)
	return npz['samples'], npz['labels']

# def get_color_names(convention = 'css2', data_dir = '../color_space/'):
# 	valid_conv = ['css2', 'xkcd']
# 	if convention not in valid_conv:
# 		raise ValueError(f'{convention} not in supported conventions: {valid_conv}')
#
# 	npz_fn = f'{data_dir}{convention}.npz'
# 	if os.path.isfile(npz_fn):
# 		npz = np.load(npz_fn)
# 	else:
# 		raise ValueError(f'{npz_fn} not found.')
#
# 	return npz['samples'], npz['labels']

#@st.cache
def get_extractor(rbg_arr, name_arr, settings = {'debug': {}}):
	extractor = ImageToColor(rbg_arr, name_arr, settings)
	return extractor

def Main():
	app_desc = '''
	fashion color detection without deep learning \n
	* [docs on extractor's settings](https://github.com/ohjho/color-extractor#passing-settings)
	* [xkcd color space](https://xkcd.com/color/rgb/)
	* [webcolors' color space](https://webcolors.readthedocs.io/en/1.11.1/colors.html)
	'''
	show_miro_logo()
	st.sidebar.header('color_extractor demo')
	st.sidebar.markdown(app_desc)

	# user input
	default_settings = {
		'debug': {},
		'resize':{'crop': 1},
		'back':{},
		'skin': {'skin_type': 'general'},
		'cluster':{'min_k': 3, 'max_k': 7},
		'selector':{'strategy': 'ratio', 'ratio.threshold': 0.5},
		'name':{}
	}
	settings_dict = json.loads(
		st.text_area('color extractor settings', value = json.dumps(default_settings))
		)
	#color_name_space = st.sidebar.selectbox('color name space', options = ['css2', 'xkcd'])
	color_space_npz = file_selector(os.path.join(cwdir, '../color_space/'), st_asset = st.sidebar, str_msg = 'select color space npz file', extension_tuple = ('.npz'))
	im = get_image(st_asset = st.sidebar, as_np_arr = True)

	if type(im) == np.ndarray:
		samples, names = get_color_samples_labels(color_space_npz)
		extractor = get_extractor(rbg_arr = samples, name_arr= names, settings = settings_dict)
		result = extractor.get(im)

		st.success(f'Color found: {result[0]}')

		# Show background, skin, and cluster mask
		resize_im = result[1]['resized']* 255
		resize_im = resize_im.astype(np.uint8)

		back_mask = result[1]['back'][:,:,np.newaxis]
		back_mask_im = resize_im * back_mask

		skin_mask = result[1]['skin'][:,:,np.newaxis]
		skin_mask_im = resize_im * skin_mask

		cluster_im = result[1]['clusters'] * 255
		cluster_im = cluster_im.astype(np.uint8)

		l_im = [resize_im, back_mask_im, skin_mask_im, cluster_im]
		l_caption = ['resized image seen by model', 'background detected',
					'skin detected', 'clusters of color found'
					]
		st.write('#### original')
		st.image(im, 'original', width = 200)
		st.write('#### result')
		st.image(l_im, l_caption, width = 200)

if __name__ == '__main__':
	Main()

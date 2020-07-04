import os, sys, io
import streamlit as st
import numpy as np
from PIL import Image
import urllib.request as urllib

def show_miro_logo(use_column_width = False, width = 100, st_asset= st.sidebar):
	logo_url = 'https://miro.medium.com/max/1400/0*qLL-32srlq6Y_iTm.png'
	st_asset.image(logo_url, use_column_width = use_column_width, channels = 'BGR', format = 'PNG', width = width)

def file_selector(folder_path='.', st_asset = st, extension_tuple = None,
	str_msg = "Select a file", get_dir = False):
	'''
	using streamlit selectbox to return a filepath
	'''
	if not folder_path:
		return None
	else:
		if not os.path.isdir(folder_path):
			st_asset.warning(f'`{folder_path}` is not a valid directory path')
			return None

		filenames = os.listdir(folder_path)
		if get_dir:
			filenames = [f for f in filenames if os.path.isdir(os.path.join(folder_path, f))]
		elif extension_tuple:
			filenames = [f for f in filenames if f.endswith(extension_tuple) and os.path.isfile(os.path.join(folder_path, f))]
		selected_filename = st_asset.selectbox(str_msg, sorted(filenames))
		return os.path.join(folder_path, selected_filename)

def get_image(st_asset = st.sidebar, as_np_arr = False, extension_list = ['jpg', 'jpeg', 'png'],
				default_url = None):
	image_url = st_asset.text_input("Enter Image URL", value = default_url)
	image_fh = st_asset.file_uploader(label = "Update your image", type = extension_list)

	if image_url and image_fh:
		st_asset.warning(f'image url takes precedence over uploaded image file')

	im = None

	if image_url:
		response = urllib.urlopen(image_url)
		im = Image.open(io.BytesIO(bytearray(response.read())))
	elif image_fh:
		im = Image.open(image_fh)

	if im and as_np_arr:
		im = np.array(im)
	return im

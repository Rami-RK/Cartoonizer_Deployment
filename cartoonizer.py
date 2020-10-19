import streamlit as st
from PIL import Image
from datetime import datetime
import time
import random 
import pandas as pd
import numpy as np
import cv2
from cv2 import cvtColor, COLOR_BGR2RGB
import tensorflow as tf
import os
import tqdm
import io
import base64
st.set_option('deprecation.showfileUploaderEncoding', False)

import network
import guided_filter


status = False


@st.cache(suppress_st_warning=True)

def resize_crop(image):
  try:
    h, w, c = np.shape(image)
    if min(h, w) > 720:
        if h > w:
            h, w = int(720*h/w), 720
        else:
            h, w = 720, int(720*w/h)
    image = cv2.resize(image, (w, h),
                        interpolation=cv2.INTER_AREA)
    h, w = (h//8)*8, (w//8)*8
    image = image[:h, :w, :]
    return image
  except:
    return "ERROR"

def load_model():

  ss = time.time()
  model_path = 'saved_models'

  input_photo = tf.placeholder(tf.float32, [1, None, None, 3])
  network_out = network.unet_generator(input_photo)
  final_out = guided_filter.guided_filter(input_photo, network_out, r=1, eps=5e-3)

  all_vars = tf.trainable_variables()
  gene_vars = [var for var in all_vars if 'generator' in var.name]
  saver = tf.train.Saver(var_list=gene_vars)

  print("loading model...............")
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  sess = tf.Session(config=config)

  sess.run(tf.global_variables_initializer())
  saver.restore(sess, tf.train.latest_checkpoint(model_path))
  print("model loaded.................")
  print("model loading time ",str(time.time()-ss))

  return sess, input_photo, final_out 

def cartoonize(image):
  try:
    batch_image = image.astype(np.float32)/127.5 - 1
    batch_image = np.expand_dims(batch_image, axis=0)

    sess, input_photo, final_out = load_model()

    output = sess.run(final_out, feed_dict={input_photo: batch_image})
    output = (np.squeeze(output)+1)*127.5
    output = np.clip(output, 0, 255).astype(np.uint8)
    return output
  except:
    st.markdown("error from cartoonize...")
    return "ERROR"
         
@st.cache(suppress_st_warning=True)
def get_image_download_link(img):
  buffered = io.BytesIO()
  img = Image.fromarray(np.uint8(img)).convert('RGB')
  img.save(buffered, format="JPEG")
  img_str = base64.b64encode(buffered.getvalue()).decode()
  href = f'<a href="data:file/jpg;base64,{img_str}" download="cartoon.jpg">Download this image</a>'
  return href

@st.cache(suppress_st_warning=True)
def load_image(image):
  try:
    global status
    status = True
    image = Image.open(image)
    open_cv_image = np.asarray(image)
    return open_cv_image
  except:
    return "ERROR"


def cartoon_main(uploaded_file):         

  if uploaded_file is not None:

    try:
	
      image = load_image(uploaded_file)
      if image != "ERROR":
        start = time.time()
        image = resize_crop(image)
        with st.spinner('Hang on, cartoonizing your image....'):

          result = cartoonize(image)
          if result != "ERROR":
            end = time.time()
            st.image(result, use_column_width=True)
            st.markdown(get_image_download_link(result), unsafe_allow_html=True)  
          else:
            global status
            status = False

        shapes = image.shape
        inptype = "Image"
        
      else:
        start = time.time() 	 								
        inptype = "Not_Image"
        shapes = "None"
        end = time.time()
        st.markdown("Invalid Input : Please upload image only")
                  
    except:
      st.markdown("ERROR : Something went wrong ")

page = st.sidebar.selectbox("Select a page", ["Cartoonize", "About Me"])

if page == "Cartoonize":
  
  st.title('Image Cartoonizer')

  uploaded_file = None

  uploaded_file = st.file_uploader("Choose your image...")

  cartoon_main(uploaded_file)

elif page == "About Me":
  st.title('About Me')
  st.markdown("Hi! Myself Ramendra Kumar, MS(R), IITD, not just a learner, an avid learner for solving a problem of interest.")
  st.markdown(" I am a Machine Learning Engineer/Trainer/Content Creater(Pure Technical). Mechanical Engineer by degree.")
  st.markdown("You can connent with me at  https://www.linkedin.com/in/ramendra-kumar-57334478/")
  st.markdown("https://github.com/Rami-RK")
  st.markdown("Mail me at karna.ramenk@gmail.com")
  st.markdown("Pretrained model and weights are used and deployed based on a 'Generative Adversarial Network' (GAN) framework ")
  st.markdown("Reference : https://github.com/SystemErrorWang/White-box-Cartoonization")
else:
	pass

import streamlit as st
import surface_reconstruction
import mde_model
from PIL import Image, ImageOps
import cv2
import tensorflow as tf
import numpy as np
import skimage.io
import matplotlib.pyplot as plt


PAGE_CONFIG = {"page_title":'Streamlit Prime', 'page_icon':'smiley:', 'layout':'centered'}
# st.beta_set_page_config(**PAGE_CONFIG)

@st.cache
def load_model():
    model = mde_model.res_unet()
    model.load_weights(r'model/mde0.h5')
    model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005, amsgrad=True),
                    loss = mde_model.depth_loss_function
                    )
    print('Model Loaded')
    return model
model = load_model()

def main():
    st.title('Monocular Depth Estimation - Group C55')
    st.subheader('By Shubham, Rohit and Aaryamaan under the guidance of Prof. Mandar.')

    menu = ['Home', 'About']
    choice = st.sidebar.selectbox('Menu',menu)
    if choice == 'Home':
        st.subheader('Streamlit from Colab')

        uploaded_img = st.file_uploader("Upload Input RGB Image",type=['png','jpg','jpeg'])
        if uploaded_img is not None:
            img_details = {"FileName":uploaded_img.name,"FileType":uploaded_img.type,"FileSize":uploaded_img.size}
            st.write(img_details)

            rgb = Image.open(uploaded_img)
            st.image(rgb)
            st.success('Image uploaded!')
            depth_button=st.sidebar.button('  Depth Prediction  ')
            surface_button=st.sidebar.button('Surface Reconstruction')
            

            if depth_button:
                depth_click = True
                rgb = ImageOps.fit(rgb, (1280, 384), Image.ANTIALIAS)
                rgb = np.asarray(rgb).astype(np.float32) 
                rgb_p = rgb[np.newaxis,...] / 255.0

                depth = mde_model.model_predict(rgb_p, model)[0]*255
                print('done')
                cv2.imwrite('depth55.png', depth)
                cv2.imwrite('rgb55.png', rgb)

                plt.figure(figsize=(19, 64))
                plt.imshow(depth[:,:,0], cmap='jet')
                plt.axis('off')
                plt.show()
                plt.savefig('depth_jet_colorspace.png', bbox_inches='tight', pad_inches=0)

            # st.image(depth)

                depth_jet = skimage.io.imread('depth_jet_colorspace.png')
                st.image(depth_jet)
                st.success('Depthmap created successfully!')

            #surface reconstruction
            if surface_button:
                depth_gray = skimage.io.imread('depth55.png')
                rgb1 = skimage.io.imread('rgb55.png')

                depth1 = skimage.io.imread('depth55.png')
                rgb1 = skimage.io.imread('rgb55.png')
                fig = surface_reconstruction.create_3d_surface(rgb_img=rgb1,
                                    depth_img=depth_gray
                                    )
                st.plotly_chart(fig)
                st.success('Surface Reconstruction successful!')
            
if __name__ == "__main__":
    main()
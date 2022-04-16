import streamlit as st
from PIL import Image
import tensorflow as tf
from tensorflow import keras
import keras
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image, ImageOps
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from keras.models import load_model
model = tf.keras.models.load_model('model.h5')




hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """



st.set_page_config(page_title="Mnemo Classifier                 ", 
                    page_icon="🤖", 
                    layout="centered", 
                    initial_sidebar_state="collapsed",
                    menu_items=None)




st.markdown(hide_streamlit_style, unsafe_allow_html=True) 



col1, col2, col3, col4, col5 = st.columns([1,1,5,1,1])



#create a sidebar
with st.sidebar:

    st.header("You don't have time to take screenshots of posts? Here are some example PNGs")

    #create an expander
    with st.expander("Mnemo Posts"):
        st.markdown(f'<a href="{"https://i.imgur.com/Yx7CEPB.png"}" download>Image</a>', unsafe_allow_html=True)


    with st.expander("Non-Mnemo Posts"):
        st.markdown(f'<a href="{"https://i.imgur.com/RvKb5Sf.png"}" download>Image</a>', unsafe_allow_html=True)   

    st.write(' ')
    st.write(' ')

    st.header('How does my Ai work?')                    
    st.write('I used the library Tensorflow and collected 125 images of my instagram page to train my Ai on my style of posts.') 
    st.write('Since the Ai also needs counter examples I also took 200 images from my front page and used these to help out my model  ')
    

    st.write("My Instagram page [link](https://www.instagram.com/max_mnemo/)")
            
            
            

with col3:
            
         
            
    def teachable_machine_classification(img, uploaded_file):
        # Disable scientific notation for clarity
        np.set_printoptions(suppress=True)

        # Load the model
        model = keras.models.load_model(uploaded_file)

        # Create the array of the right shape to feed into the keras model
        # The 'length' or number of images you can put into the array is
        # determined by the first position in the shape tuple, in this case 1.
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

        # Replace this with the path to your image
        image = img
        # image = Image.open(img_name).convert('RGB')
        # image = cv2.imread(image)

        # resize the image to a 224x224 with the same strategy as in TM2:
        # resizing the image to be at least 224x224 and then cropping from the center
        size = (224, 224)
        image = ImageOps.fit(image, size, Image.ANTIALIAS)

        # turn the image into a numpy array
        image_array = np.asarray(image)

        # display the resized image
        image.show()

        # Normalize the image
        normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

        # Load the image into the array
        data[0] = normalized_image_array

        # run the inference
        prediction = model.predict(data)
        #print(prediction)
        return np.argmax(prediction)



    st.header("Mnemo Post Classifier")
    st.write("Take a screenshot of any Instagram post on your home page and see if the post was created by max_mnemo (me) or not.")
    
    
    st.write(' ')          
    img_title = Image.open("123456.png")    
    st.image(img_title)
    st.write(' ')        
            
        
    # file upload and handling logic
    uploaded_file = st.file_uploader("Upload your screenshot here", type="png")
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
    #image = Image.open(img_name).convert('RGB')
        st.image(image, caption='This is your image', use_column_width=True)
        st.write(" ")
        
        data_load_state = st.text('Please wait...')  
        label = teachable_machine_classification(image, 'model.h5')
        data_load_state.text('Loading data done!')    
            
        st.write(" ")    
            
        if label == 1:
            st.write("This does not look like a MNEMO post 👎 ")
        else:
            st.write("This looks like a MNEMO post 👍  ")
            
    st.write(' ')
    st.write(' ')
              
                
                
                
                
                
                
                
                
                
                


footer="""<style>
a:link , a:visited{
color: red;
background-color: transparent;
text-decoration: underline;
}
a:hover,  a:active {
color: LightBlue;
background-color: transparent;
text-decoration: underline;
}
.footer {
position: fixed;
left: 0;
bottom: 0;
width: 100%;
background-color: transparent;
color: grey;
text-align: center;
}
</style>
<div class="footer">
<p>Entwickelt mit ❤️  von <a style='display: inline-block;' href="https://www.instagram.com/max_mnemo/" target="_blank">Max Mnemo </a> // <a style='display: block-inline; text-align: center;' href="https://github.com/MaximilianFreitag/mnemo-post-classifier" target="_blank">Github </a> // <a style='display: block-inline; text-align: center;' href="https://mnemo.uk/contact/" target="_blank">Contact </a></p>
</div>
"""
st.markdown(footer,unsafe_allow_html=True)                

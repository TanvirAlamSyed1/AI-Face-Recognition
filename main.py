from keras.models import load_model  # TensorFlow is required for Keras to work
from streamlit_extras.add_vertical_space import add_vertical_space
import cv2  # Install opencv-python
import numpy as np
import streamlit as st
from PIL import Image


# Sidebar contents
with st.sidebar:
    st.title('🤖 Item Detection')
    st.markdown('''
    ## About
    This app uses a kersa model to detect items using:
    - [Streamlit](https://streamlit.io/)
    - [Teachable Machine](https://teachablemachine.withgoogle.com/)
    ''')
    add_vertical_space(3)
    st.write('This is my linkedin🤗: (https://www.linkedin.com/in/tanviralamsyed/)')
 


    
def main():
     np.set_printoptions(suppress=True)
     image = st.camera_input(label ="Capture Image", key="First Camera", label_visibility="hidden")# this captures the image 
     if image:
        np.set_printoptions(suppress=True)
        model = load_model("keras_Model.h5", compile=False) # this section loads the  model and labels that are used
        class_names = open("labels.txt", "r").readlines()
        
        img = Image.open(image) # stores the natural image so that itcan be manipulated
        img_array = np.array(img)
        
        image = cv2.resize(img_array, (224, 224),interpolation=cv2.INTER_AREA)#resizes it so that it is appropiate for the model to consume theimage
        # st.image(image)
        image = np.asarray(image).reshape(1,224, 224, 3)
        image = (image / 127.5) - 1
        # Predicts the model
        prediction = model.predict(image)
        index = np.argmax(prediction)
        class_name = class_names[index]
        confidence_score = prediction[0][index]
        print(class_name)
        print(confidence_score)
        if confidence_score > 0.9:
            if index == 0:
                st.subheader("Ferrari")
                st.write("Ferrari is an iconic Italian luxury sports car manufacturer renowned for its high-performance vehicles and rich racing heritage. Founded by Enzo Ferrari in 1939, the company has become a symbol of speed, elegance, and exclusivity. With a strong presence in Formula One, Ferrari continues to push the boundaries of automotive engineering and design, creating some of the most coveted cars in the world.")
            if index == 1:
                st.subheader("Pagani")
                st.write("Pagani is an Italian luxury sports car manufacturer renowned for crafting exquisite, high-performance vehicles. Founded by Horacio Pagani, the company is celebrated for its meticulous attention to detail, innovative use of carbon fiber, and powerful engines. Pagani cars are considered works of art, with each model a limited-production masterpiece.")
            if index ==  2:
                st.subheader("Batmobile")
                st.write("The Batmobile is Batman's iconic vehicle, a high-tech masterpiece housed in the Batcave. It's more than just a car; it's a heavily armored, versatile tool used for pursuit, capture, and combat. Packed with advanced weaponry and gadgets, it's a symbol of Batman's technological prowess and a fearsome deterrent to Gotham City's criminals. Its design has evolved over the years, but its core purpose remains the same: to aid the Dark Knight in his crusade for justice.")
            print("Class:", class_name[2:], end="")
            st.write("Confidence Score:", str(np.round(confidence_score * 100))[:-2], "%")
        else:
            st.warning("unsure of item, try another one")


if __name__ == '__main__':
    main()
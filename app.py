from datetime import datetime
import streamlit as st
from PIL import Image
import pandas as pd
import os
from datetime import datetime
import numpy as np
from centernet_model_app import run_cunternet_model_app
from fasterR_cnn_model_app import run_faster_model_app

from ssdmodel_app import run_ssdmodel_app



# 디렉토리 정보와 파일을 알려주면 해당 디렉토리에 파일을 저장하는 함수를 만들겁니다.
def save_uploaded_file(directory,file):
    # 1. 디렉토리가 있는지 확인하여 없으면 디렉토리부터 만든다.
    if not os.path.exists(directory) :
        os.makedirs(directory)
    # 2. 디렉토리가 있으니 파일을 저장
    with open(os.path.join(directory,file.name), 'wb') as f :
        f.write(file.getbuffer())
    return st.success('Saved file : {} in {}'.format(file.name,directory))

title_img = Image.open('data/1.png')
st.set_page_config(page_title='Image Detection',page_icon=title_img,
                    layout='wide',initial_sidebar_state='collapsed')


def main():
    st.title('Tensorflow Object Detection Project')

    logo = Image.open('data/tensorflow.png')
    st.sidebar.image(logo)
    
    tf_models = ['SSD MobileNet','CenterNet HourGlass','EfficientDet','Faster R-CNN ResNet']

        
    model_choice = st.sidebar.selectbox('TensorFlow Model Select',tf_models)
    boxes = st.sidebar.slider('Max Boxes',0,200,value=50,step=5)
    min_score = st.sidebar.slider('Score',1,100,value=50)

    if model_choice == 'SSD MobileNet':

        image_file = st.file_uploader('Upload your image',type=['png','jpg','jpeg'])
        if image_file is not None :
          

            #파일명 변경 후 저장
            image_file.name = 'test.jpg'

            #파일명을 현재시간의 조합으로 해서 만들어보세요.(안겹치려고)
            #예) 현재시간.jpg
            current_time = datetime.now()
            current_time = current_time.isoformat().replace(':','_')
            image_file.name = current_time+'.jpg'

            #파일을 저장할 수 있도록 위의 함수를 호출하자
            #save_uploaded_file('temp',image_file)
            
            #오브젝트 디텍션을 여기서 한다
            img = Image.open(image_file)

            img = np.array(img)
            #넘파이 어레이를 오브젝트 디택션함수에 넘겨준다
            run_ssdmodel_app(img,boxes,min_score)
            st.success('Detection Success!')


    elif model_choice == 'CenterNet HourGlass':
        image_file = st.file_uploader('Upload your image',type=['png','jpg','jpeg'])
        if image_file is not None :
            #프린트문은 디버깅용 터미널에 출력된다.
            
            #파일명 변경 후 저장
            image_file.name = 'test.jpg'

            #파일명을 현재시간의 조합으로 해서 만들어보세요.(안겹치려고)
            #예) 현재시간.jpg
            current_time = datetime.now()
            current_time = current_time.isoformat().replace(':','_')
            image_file.name = current_time+'.jpg'

            #파일을 저장할 수 있도록 위의 함수를 호출하자
            #save_uploaded_file('temp',image_file)
            
            #오브젝트 디텍션을 여기서 한다
            img = Image.open(image_file)

            img = np.array(img)
            #넘파이 어레이를 오브젝트 디택션함수에 넘겨준다
            run_cunternet_model_app(img,boxes,min_score)
            st.success('Detection Success!')

    elif model_choice == 'EfficientDet':
        image_file = st.file_uploader('Upload your image',type=['png','jpg','jpeg'])
        if image_file is not None :
            #프린트문은 디버깅용 터미널에 출력된다.
            
            #파일명 변경 후 저장
            image_file.name = 'test.jpg'

            #파일명을 현재시간의 조합으로 해서 만들어보세요.(안겹치려고)
            #예) 현재시간.jpg
            current_time = datetime.now()
            current_time = current_time.isoformat().replace(':','_')
            image_file.name = current_time+'.jpg'

            #파일을 저장할 수 있도록 위의 함수를 호출하자
            #save_uploaded_file('temp',image_file)
            
            #오브젝트 디텍션을 여기서 한다
            img = Image.open(image_file)

            img = np.array(img)
            #넘파이 어레이를 오브젝트 디택션함수에 넘겨준다
            run_cunternet_model_app(img,boxes,min_score)
            st.success('Detection Success!')

    elif model_choice == 'Faster R-CNN ResNet':
        image_file = st.file_uploader('Upload your image',type=['png','jpg','jpeg'])
        if image_file is not None :
            #프린트문은 디버깅용 터미널에 출력된다.
            
            #파일명 변경 후 저장
            image_file.name = 'test.jpg'

            #파일명을 현재시간의 조합으로 해서 만들어보세요.(안겹치려고)
            #예) 현재시간.jpg
            current_time = datetime.now()
            current_time = current_time.isoformat().replace(':','_')
            image_file.name = current_time+'.jpg'

            #파일을 저장할 수 있도록 위의 함수를 호출하자
            #save_uploaded_file('temp',image_file)
            
            #오브젝트 디텍션을 여기서 한다
            img = Image.open(image_file)

            img = np.array(img)
            #넘파이 어레이를 오브젝트 디택션함수에 넘겨준다
            run_faster_model_app(img,boxes,min_score)
            st.success('Detection Success!')

if __name__ == '__main__':
    main()
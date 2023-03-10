import streamlit as st
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from datetime import datetime
import os

st.set_page_config(layout='wide')

st.write('''
# Обнаружение объектов на стройплощадке
''')
         
st.write('''
Краткое описание текст текст текст текст текст текст текст текст текст текст 
''')

st.header('Загрузите фото')


image_file = st.file_uploader(
    "Можно перетащить (драг-н-дроп)", 
    type=["jpg", "jpeg", "png"]
    )

new_names = {0: 'Падение',
 1: 'Перчатки',
 2: 'Очки',
 3: 'Каска',
 4: 'Лестница',
 5: 'Маска',
 6: 'НЕТ перчаток',
 7: 'НЕТ очков',
 8: 'НЕТ каски',
 9: 'НЕТ маски',
 10: 'НЕТ защитного жилета',
 11: 'Человек',
 12: 'Разметочный конус',
 13: 'Защитный жилет'}

if image_file is not None:
    img = Image.open(image_file)
    col1, col2 = st.columns(2)
    with col1:
        st.image(img, caption='Uploaded Image', use_column_width='always')
        ts = datetime.timestamp(datetime.now())
        #imgpath = os.path.join('data/uploads', str(ts)+image_file.name)
        imgpath = os.path.join('Personal-Protective-Equipment---Combined-Model-4/valid/images', image_file.name)
        #st.write('this is name', image_file.name)

        #outputpath = os.path.join('data/outputs', os.path.basename(imgpath))
        outputpath = os.path.basename(imgpath)
        with open(imgpath, mode="wb") as f:
            f.write(image_file.getbuffer())

        #call Model prediction--
        model = torch.hub.load('ultralytics/yolov5', 'custom', path='best170.pt', force_reload=True, device='cpu') 
        model.names = new_names
        #model.conf = 0.7
        #model.cuda() if device == 'cuda' else model.cpu()
        model.cpu()
        pred = model(imgpath)
        pred.render()  # render bbox in image
        for im in pred.ims:
            im_base64 = Image.fromarray(im)
            im_base64.save(outputpath)

            #--Display predicton
            
        img_ = Image.open(outputpath)
        with col2:
            st.image(img_, caption='Model Prediction(s)', use_column_width='always')
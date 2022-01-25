# coding: utf-8
import sys, os
sys.path.append(os.pardir) # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import numpy as np
from PIL import Image
from deep_convnet import DeepConvNet

network = DeepConvNet()
network.load_params() # 이미 학습된 가중치(W)와 편향(b)값 불러오기

PATH = '../input_image'
file_list = os.listdir(PATH)

for name in file_list:
    image_pill = Image.open('../input_image/' + name).convert('L') # 흑백 이미지로 변환
    image = np.array(image_pill) # 이미지를 numpy array로 변환
    image = image.reshape(1, 1, 28, 28) # 이미지 형상을 (1, 1, 28, 28)로 변환

    res = network.predict(image) # 예측 수행
    s = np.argmax(res, axis=1) # 예측된 결과값들의 최댓값을 예측된 값으로 선택
    print(f"{name} : {s}") # 결과 출력
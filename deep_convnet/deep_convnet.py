# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import pickle
import numpy as np
from collections import OrderedDict
from common.layers import *


class DeepConvNet:
    """정확도 99% 이상의 고정밀 합성곱 신경망

    네트워크 구성은 아래와 같음
        conv - relu - conv- relu - pool -
        conv - relu - conv- relu - pool -
        conv - relu - conv- relu - pool -
        affine - relu - dropout - affine - dropout - softmax
        
        합성곱 계층 6개 + 완전연결 계층 2개
    """
    def __init__(self, input_dim=(1, 28, 28), # 입력받는 이미지의 크기는 28X28 사이즈의 채널은 1개인 흑백 이미지
                 # 합성 곱 계층의 파라미터 6개 (필터의 개수는 출력의 채널수를 결정)
                 conv_param_1 = {'filter_num':16, 'filter_size':3, 'pad':1, 'stride':1},
                 conv_param_2 = {'filter_num':16, 'filter_size':3, 'pad':1, 'stride':1},
                 conv_param_3 = {'filter_num':32, 'filter_size':3, 'pad':1, 'stride':1},
                 conv_param_4 = {'filter_num':32, 'filter_size':3, 'pad':2, 'stride':1},
                 conv_param_5 = {'filter_num':64, 'filter_size':3, 'pad':1, 'stride':1},
                 conv_param_6 = {'filter_num':64, 'filter_size':3, 'pad':1, 'stride':1},
                 # 완전연결 계층 2개중 은닉층과 출력층의 크기
                 hidden_size=50, output_size=10):
        # 가중치 초기화===========
        # 각 층의 뉴런 하나당 앞 층의 몇 개 뉴런과 연결되는가（TODO: 자동 계산되게 바꿀 것) --> 자동화가 가능함
        # pre_node_nums[0] : 첫 입력 채널 크기가 1이고 필터의 사이즈가 3*3이기 때문에 1*3*3 (흑백 이미지를 입력 받기 때문에 채널의 크기는 1)
        # pre_node_nums[1] : 입력 채널 크기가 16이고 필터의 사이즈가 3*3이기 때문에 16*3*3 (앞에서 필터 16개로 합성곱을 했으므로 출력 채널 크기는 16)
        # pre_node_nums[2] : 입력 채널 크기가 16이고 필터의 사이즈가 3*3이기 때문에 16*3*3 (앞에서 필터 16개로 합성곱을 했으므로 출력 채널 크기는 16)
        # pre_node_nums[3] : 입력 채널 크기가 32이고 필터의 사이즈가 3*3이기 때문에 32*3*3 (앞에서 필터 32개로 합성곱을 했으므로 출력 채널 크기는 32)
        # pre_node_nums[4] : 입력 채널 크기가 32이고 필터의 사이즈가 3*3이기 때문에 32*3*3 (앞에서 필터 32개로 합성곱을 했으므로 출력 채널 크기는 32)
        # pre_node_nums[5] : 입력 채널 크기가 64이고 필터의 사이즈가 3*3이기 때문에 64*3*3 (앞에서 필터 64개로 합성곱을 했으므로 출력 채널 크기는 64)
        # pre_node_nums[6] : 입력 형상이 16*4*4이기 때문에 16*4*4 (Affine 계층의 완전연결은 필터를 사용하지 않고 출력 노드 하나당 입력의 모든 노드와 연결됨)
        # pre_node_nums[7] : 입력 형상이 50이기 때문에 50 (Affine 계층의 완전연결은 필터를 사용하지 않고 출력 노드 하나당 입력의 모든 노드와 연결됨)
        pre_node_nums = np.array([1*3*3, 16*3*3, 16*3*3, 32*3*3, 32*3*3, 64*3*3, 64*4*4, hidden_size])

        wight_init_scales = np.sqrt(2.0 / pre_node_nums)  # ReLU를 사용할 때의 권장 초기값 ( He 초기값 )
        
        self.params = {} # 각 레이어의 가중치(W)와 편향(b)를 저장하기 위한 딕셔너리 변수
        # 저장해야 할 가중치와 편향은 Convolution, Relu 레이어만 존재

        # 합성곱 가중치(W)의 형상을 정하기 위해 이전 채널의 크기를 정함
        # 가중치(W)의 채널의 크기는 항상 입력 채널의 크기와 같아야 하기 때문
        # 필터의 개수는 출력의 채널수를 결정
        pre_channel_num = input_dim[0]

        # 합성 곱 계층을 전부 돌면서 self.params 변수 초기화
        # 편향(b)는 기본적으로 0으로 초기화를 한다.
        # 가중치(W)는 활성화 함수 ReLu 사용에 최적화된 표본편차가 sqrt(2/입력층의 수)인 표준정규분포를 따르는 난수로 설정 ( He 초기값 )
        for idx, conv_param in enumerate([conv_param_1, conv_param_2, conv_param_3, conv_param_4, conv_param_5, conv_param_6]):
            # 합성곱 가중치(W)의 형상은 필터의 크기이므로 필터의개수 X 채널 수 X 세로 X 가로 --> (필터의 개수, 채널 수, 세로 크기, 가로 크기)
            self.params['W' + str(idx+1)] = wight_init_scales[idx] * np.random.randn(conv_param['filter_num'], pre_channel_num, conv_param['filter_size'], conv_param['filter_size'])

            # 합성곱 편향(b)의 형상은 필터의 크기만큼 필요함
            # 각 필터마다 일괄적으로 편향을 더해주기 때문
            self.params['b' + str(idx+1)] = np.zeros(conv_param['filter_num'])

            # 이전 채널의 수 갱신
            pre_channel_num = conv_param['filter_num']
        
        # Affine 계층의 가중치와 편향 초기화
        # 가중치(W)의 형상은 (입력크기, 출력크기)
        # 행렬곱으로 바로 출력을 하기위해 이렇게 지정
        # 초기값은 전부 He 초기값 사용
        self.params['W7'] = wight_init_scales[6] * np.random.randn(64*4*4, hidden_size)
        self.params['b7'] = np.zeros(hidden_size)
        self.params['W8'] = wight_init_scales[7] * np.random.randn(hidden_size, output_size)
        self.params['b8'] = np.zeros(output_size)

        # 계층 생성===========
        self.layers = []
        self.layers.append(Convolution(self.params['W1'], self.params['b1'], 
                           conv_param_1['stride'], conv_param_1['pad'])) # 합성곱 계층
        self.layers.append(Relu()) # 활성화 함수
        self.layers.append(Convolution(self.params['W2'], self.params['b2'], 
                           conv_param_2['stride'], conv_param_2['pad'])) # 합성곱 계층
        self.layers.append(Relu()) # 활성화 함수
        self.layers.append(Pooling(pool_h=2, pool_w=2, stride=2)) # 풀링 계층
        self.layers.append(Convolution(self.params['W3'], self.params['b3'], 
                           conv_param_3['stride'], conv_param_3['pad'])) # 합성곱 계층
        self.layers.append(Relu()) # 활성화 함수
        self.layers.append(Convolution(self.params['W4'], self.params['b4'],
                           conv_param_4['stride'], conv_param_4['pad'])) # 합성곱 계층
        self.layers.append(Relu()) # 활성화 함수
        self.layers.append(Pooling(pool_h=2, pool_w=2, stride=2)) # 풀링 계층
        self.layers.append(Convolution(self.params['W5'], self.params['b5'],
                           conv_param_5['stride'], conv_param_5['pad'])) # 합성곱 계층
        self.layers.append(Relu()) # 활성화 함수
        self.layers.append(Convolution(self.params['W6'], self.params['b6'],
                           conv_param_6['stride'], conv_param_6['pad'])) # 합성곱 계층
        self.layers.append(Relu()) # 활성화 함수
        self.layers.append(Pooling(pool_h=2, pool_w=2, stride=2)) # 풀링 계층
        self.layers.append(Affine(self.params['W7'], self.params['b7']))
        self.layers.append(Relu()) # 활성화 함수
        self.layers.append(Dropout(0.5)) # 드랍 아웃 : 학습 시 랜덤으로 50프로의 노드 비활성화
        self.layers.append(Affine(self.params['W8'], self.params['b8'])) # 완전연결 계층
        self.layers.append(Dropout(0.5)) # 드랍 아웃 : 학습 시 랜덤으로 50프로의 노드 비활성화
        
        self.last_layer = SoftmaxWithLoss() # 학습 시 이용되는 확률 계산 후 교차 엔트로피 오차를 이용하여 손실 계산 

    # 이미지 입력 x를 모든 레이어를 통과시키고 마지막 출력값을 반환
    def predict(self, x, train_flg=False):
        for layer in self.layers:
            # Dropout 레이어일 경우 학습 시에만 랜덤으로 노드를 비활성화 시키고 다른 경우에는 활성화 비율을 곱해줌
            if isinstance(layer, Dropout):
                x = layer.forward(x, train_flg)
            else:
                x = layer.forward(x)
        return x

    # x : 이미지, t : 정답 레이블
    # 예측을 수행하고 교차 엔트로피 오차를 이용하여 정답 레이블과의 손실 계산
    def loss(self, x, t):
        y = self.predict(x, train_flg=True)
        return self.last_layer.forward(y, t)

    # x : 이미지, t : 정답 레이블, batch_size : 배치 크기
    def accuracy(self, x, t, batch_size=100):

        # 원-핫 인코딩 이라면 정답만 나열되어있는 배열로 변경
        if t.ndim != 1 : t = np.argmax(t, axis=1)

        acc = 0.0

        # 입력된 이미지들을 배치 크기만큼 여러개로 분할해서 예측 수행
        for i in range(int(x.shape[0] / batch_size)):
            tx = x[i*batch_size:(i+1)*batch_size] # 이미지를 배치크기만큼 자르기
            tt = t[i*batch_size:(i+1)*batch_size] # 정답을 배치크기만큼 자르기
            y = self.predict(tx, train_flg=False) # 예측 수행
            y = np.argmax(y, axis=1) # 예측된 값들중 가장 높은 값을 선택하여 예측
            acc += np.sum(y == tt) # 예측된 값과 정답과 비교해서 같은 개수만 카운트해서 acc에 더함

        return acc / x.shape[0] # acc를 전체 이미지 개수로 나눠서 정답률 반환

    def gradient(self, x, t):
        # forward
        self.loss(x, t)

        # backward
        dout = 1
        dout = self.last_layer.backward(dout)

        tmp_layers = self.layers.copy()
        tmp_layers.reverse()
        for layer in tmp_layers:
            dout = layer.backward(dout)

        # 결과 저장
        grads = {}
        for i, layer_idx in enumerate((0, 2, 5, 7, 10, 12, 15, 18)):
            grads['W' + str(i+1)] = self.layers[layer_idx].dW
            grads['b' + str(i+1)] = self.layers[layer_idx].db

        return grads

    def save_params(self, file_name="params.pkl"):
        params = {}
        for key, val in self.params.items():
            params[key] = val
        with open(file_name, 'wb') as f:
            pickle.dump(params, f)

    def load_params(self, file_name="params.pkl"):
        with open(file_name, 'rb') as f:
            params = pickle.load(f)
        for key, val in params.items():
            self.params[key] = val

        for i, layer_idx in enumerate((0, 2, 5, 7, 10, 12, 15, 18)):
            self.layers[layer_idx].W = self.params['W' + str(i+1)]
            self.layers[layer_idx].b = self.params['b' + str(i+1)]

# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import numpy as np
from common.optimizer import *

class Trainer:
    """신경망 훈련을 대신 해주는 클래스
    """
    def __init__(self, 
        network,    # 모델 종류
        x_train,    # 훈련용 이미지
        t_train,    # 훈련용 정답 레이블
        x_test,     # 테스트용 이미지
        t_test,     # 테스트용 정답 레이블
        epochs=20,  # 전체 데이터 학습 횟수
        mini_batch_size=100,    # 학습 시 배치 크기
        optimizer='SGD',        # 가중치와 편향을 조정하는 방법
        optimizer_param={'lr':0.01},        # 학습률
        evaluate_sample_num_per_epoch=None,
        verbose=True # 로깅 여부
    ):
        self.network = network
        self.verbose = verbose
        self.x_train = x_train
        self.t_train = t_train
        self.x_test = x_test
        self.t_test = t_test
        self.epochs = epochs
        self.batch_size = mini_batch_size 
        self.evaluate_sample_num_per_epoch = evaluate_sample_num_per_epoch

        # optimzer
        # 기울기에 대해 가중치와 편향을 조정하는 방법
        optimizer_class_dict = {'sgd':SGD, 'momentum':Momentum, 'nesterov':Nesterov,
                                'adagrad':AdaGrad, 'rmsprpo':RMSprop, 'adam':Adam}
        self.optimizer = optimizer_class_dict[optimizer.lower()](**optimizer_param)
        
        self.train_size = x_train.shape[0] # 학습 크기를 학습용 이미지 개수로 설정

        # 1 epoch 당 학습해야 할 횟수
        # 1번 학습에 배치크기만큼 학습하므로 전체 데이터를 1번 학습하기 위한 반복 횟수
        self.iter_per_epoch = max(self.train_size / mini_batch_size, 1)

        # 모든 데이터를 설정한 epoch만큼 학습시키기 위한 반복 횟수
        self.max_iter = int(epochs * self.iter_per_epoch)
        
        self.current_iter = 0
        self.current_epoch = 0
        
        self.train_loss_list = []
        self.train_acc_list = []
        self.test_acc_list = []

    def train_step(self):
        batch_mask = np.random.choice(self.train_size, self.batch_size)
        x_batch = self.x_train[batch_mask]
        t_batch = self.t_train[batch_mask]
        
        grads = self.network.gradient(x_batch, t_batch)
        self.optimizer.update(self.network.params, grads)
        
        loss = self.network.loss(x_batch, t_batch)
        self.train_loss_list.append(loss)
        if self.verbose: print("train loss:" + str(loss))
        
        if self.current_iter % self.iter_per_epoch == 0:
            self.current_epoch += 1
            
            x_train_sample, t_train_sample = self.x_train, self.t_train
            x_test_sample, t_test_sample = self.x_test, self.t_test
            if not self.evaluate_sample_num_per_epoch is None:
                t = self.evaluate_sample_num_per_epoch
                x_train_sample, t_train_sample = self.x_train[:t], self.t_train[:t]
                x_test_sample, t_test_sample = self.x_test[:t], self.t_test[:t]
                
            train_acc = self.network.accuracy(x_train_sample, t_train_sample)
            test_acc = self.network.accuracy(x_test_sample, t_test_sample)
            self.train_acc_list.append(train_acc)
            self.test_acc_list.append(test_acc)

            if self.verbose: print("=== epoch:" + str(self.current_epoch) + ", train acc:" + str(train_acc) + ", test acc:" + str(test_acc) + " ===")
        self.current_iter += 1

    def train(self):
        for i in range(self.max_iter):
            self.train_step()

        test_acc = self.network.accuracy(self.x_test, self.t_test)

        if self.verbose:
            print("=============== Final Test Accuracy ===============")
            print("test acc:" + str(test_acc))


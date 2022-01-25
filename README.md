# Mnist 데이터 세트를 이용한 이미지 분류 CNN 모델  
## 테스트 방법  
```
git clone https://github.com/pseong/mnist-cnn.git
cd mnist-cnn/deep_convnet  
python classification.py  
```
미리 학습된 모델로 input_images에 있는 숫자 손글씨를 인식해서 출력합니다.  

이미지를 직접 만들려면 미리 학습된 모델은 아래와 같은 손글씨로 학습되었기 때문에 아래와 비슷하게 만들어야 합니다.  

<img width="372" alt="fig 3-24" src="https://user-images.githubusercontent.com/76799354/151030971-a6b13e9a-3b26-41df-95a4-3a2f7aae5af9.png">  

가장 비슷하게 이미지 파일 만드는 방법

1. 그림판에서 28x28 크기로 새로운 파일을 생성
2. ctrl+마우스휠로 최대로 확대
3. 배경색을 검정색으로 색칠
4. 흰색 브러시를 사용하여 아무 숫자 적기
5. 그리고 이미지를 mnist-cnn/input_image 에다가 저장

이렇게 안하고 모든 손글씨를 인식하고 싶으면 새롭게 모델을 재학습 시켜야 합니다.  
## 학습 방법
```
cd mnist-cnn/deep_convnet
python train_deepnet.py
```
## 학습 데이터 수정 방법
mnist-cnn/deep_convnet/train_deepnet.py에 10번째 줄 변경  

블로그 정리 : https://pseong.tistory.com/17  

참고 코드 : https://github.com/WegraLee/deep-learning-from-scratch

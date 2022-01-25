# Mnist 데이터 세트를 이용한 이미지 분류 CNN 모델  
## 패키지 설치
```
pip3 install numpy
pip3 install image
```
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

모든 손글씨를 인식하고 싶으면 새롭게 모델을 재학습 시켜야 합니다.  
## 학습 방법
```
cd mnist-cnn/deep_convnet
python train_deepnet.py
```
## 학습 데이터 수정 방법
mnist-cnn/deep_convnet/train_deepnet.py에 10번째 줄 변경  

## 그림판으로 그린 이미지 테스트 결과
img.png : ![img](https://user-images.githubusercontent.com/76799354/151032964-2eea029a-1673-4dcc-a256-3d6cc9f2c99b.png)
img2.png : ![img2](https://user-images.githubusercontent.com/76799354/151032969-ee37a81c-a87b-4944-9591-54a3d89e03e6.png)
img3.png : ![img3](https://user-images.githubusercontent.com/76799354/151032977-be3465b6-b489-4d18-93b2-5b5a981d7573.png)

출력 결과 : ![출력 결과](https://user-images.githubusercontent.com/76799354/151033316-6d38beea-be2b-4a03-bd93-334dfa84faeb.png)

잘 작동한다.  

블로그 정리 : https://pseong.tistory.com/17  

참고 코드 : https://github.com/WegraLee/deep-learning-from-scratch

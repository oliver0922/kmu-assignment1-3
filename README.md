# kmu-assignment1-3
Run the opensource

## 3)  2)의 정리한 코드 중 하나 실행해서 결과 확인
- 구현 환경 (시스템, lib, docker 등 실행 환경) 및 실행에 관련된 코드 설명 등

2)에서 조사한 object detection 실행을 Google의 Colab Pro 환경에서 하였다. Google Colab Pro는 기존 Colab 환경에 비해 T4 또는 P100 GPU를 사용할 수 있다. 이 GPU는 임의로 주어지는 것이며 이번 실습에서는 ()가 할당되었다. 또한 python은 3.8 version, numpy library, pytorch version, matplotlib 라이브러리, tensorboard version, pillow라이브러리 ,tqdm 등이 사용되었다.

아래 캡처본은 구글 Colab 환경에서 실행했을 때의 모습을 캡처한 것이다. 

**①우선 google drive의 파일을 사용하고 google drive/mydrive에 실습 결과를 저장해야하므로 gdrive를 mount 해준다. 다음 directory를 /content/gdrive/MyDrive/clean_yolo로 이동한다**

![image](https://user-images.githubusercontent.com/69920975/113879836-e46c0c00-97f5-11eb-9e10-6b5eac60a0dd.png)

**②다음 github에 올려놓은 repository를 clone 해 gdrive 내부에 저장한다. 다음 경로를 이동한다.**

![image](https://user-images.githubusercontent.com/69920975/113879861-eafa8380-97f5-11eb-8310-5de92eefcf99.png)

**③ requirements.txt 파일안의 있는 library들을 다운로드한다.**

![image](https://user-images.githubusercontent.com/69920975/113879901-f352be80-97f5-11eb-9c5b-d7dfcae0fb0f.png)

**④directory를 이동한 후 download_weigths.sh를 실행해 파일 내부 안의 있는 명령어들을 실행한다.**

![image](https://user-images.githubusercontent.com/69920975/113879940-f9e13600-97f5-11eb-9151-340f936e6316.png)

**⑤ dataset을 customize 하기위해 class.names. 파일안의 내용들을 각각 라벨된 class 이름에 맞게 수정해준다. 다음 images와 labels folder에 들어가 압축을 풀어준다.**

![image](https://user-images.githubusercontent.com/69920975/113879967-01084400-97f6-11eb-84d6-4e1c192f58bf.png)

**⑥ glob library에서 glob함수를 불러와 imagse 폴더안에 있는 jpg 확장자 파일들 ,그리고 labels 폴더 안에 있는 txt 확장자 파일들을 각각 img_list, txt_list 라는 리스트에 경로를 넣어준다.**

![image](https://user-images.githubusercontent.com/69920975/113880009-0cf40600-97f6-11eb-8275-4c3e3ff9854c.png)

**⑦ dataset을 0.8과 0.2의 비율로 trainset과 validation set으로 나누기 위해서 sklearn에서 train_test_split 함수를 불러와 dataset을 trainset와 validation set으로 각각 0.8;0.2의 비율로 나누고 잘 나뉘었는지 확인하기 위해 사진의 개수를 출력한다.**

![image](https://user-images.githubusercontent.com/69920975/113880112-27c67a80-97f6-11eb-8a3c-173cb9803fb4.png)

**⑧ train.txt 파일과 valid.txt 파일 안에 위에서 정의한 train_img_list와 val_img_list 안의 있는 각 image, label 파일들의 경로들을 열어 입력한다.**

![image](https://user-images.githubusercontent.com/69920975/113880148-30b74c00-97f6-11eb-913f-1be073bd817c.png)

**⑨ train.py 파일에 여러 가지 option을 주어 실행시킨다 = 학습을 진행한다.
epoch을 10으로 설정하고 config파일을 yolov3-cutom.cfg로 지정하고 pretrained_weigths를 
darknet으로 설정하였다.**

![image](https://user-images.githubusercontent.com/69920975/113880191-390f8700-97f6-11eb-9089-143a19aae1a1.png)

train 동안의 log를 살펴보면 model을 train 할 때와 evaluate 할 때를 각각 기록하는 것을 알 수 있다. 평균적으로 한 epoch 당 약 20분의 시간이 걸렸으며 , model을 evaluate 하는데는 시간이 얼마 걸리지 않았다.

**⑩ tensorboard에 log값을 전달해서 결과를 확인한다.**
![image](https://user-images.githubusercontent.com/69920975/113880235-46c50c80-97f6-11eb-8729-e26221306e94.png)
![image](https://user-images.githubusercontent.com/69920975/113880242-49276680-97f6-11eb-8e1a-bdb41be0a5a5.png)

 위의 그래프를 분석해보면 차례대로  train 되는 동안의 1) epch에 따른 class loss, 즉 객체를 얼마나 잘 판단을 하는지 나타내는 그래프와 2) epoch에 따른 iou loss, 즉 시간이 지나면서 iou 값의 손실함수 3) epoch에 따른 learning rage의 변화 4)epoch에 따른 전체 loss 값 5) 마지막으로 epoch에 따른 obj_loss bounding box loss를 의미한다. loss 값이 계속 감소하고 0에 가까워지는 것을 보아 학습이 굉장히 잘 되고 있는 것을 알 수 있고 learning rate는 증가하다 수렴하는데 최적값에 도달하게 되어 더 이상 크게 증가하지 않는 것을 알 수 있다.
 
 ![image](https://user-images.githubusercontent.com/69920975/113880266-4fb5de00-97f6-11eb-833e-2bb2f29cd46f.png)

이제 valiadation test의 precision recall 함수 그리고 mAP를 살펴보자. 파란색 그래프는 빨간색 그래프가 나타내는 학습시간 동안의 얻은 weight를 갖고 계속 학습한 모습을 나타낸다. 우선 precision은 증가하다가 감소하는 경향이 있으며 recall은 계속 증가하는 경향을 보인다. mAP도 증가하다 감소하는 경향이 있는데, 사실 GPU의 성능이 그리 좋지 않아 학습 횟수를 10번밖에 하지 못했기 때문에 이렇게 성능이 별로 좋지 않게 나온 것이라 생각한다. 또한 hyperparameter tunning도 별로 하지 못한 이유도 성능에 영향을 미쳤을 것이다. 

**⑪ 마지막으로 detect.py를 실행해 object detection이 되는지 확인한다.**

![image](https://user-images.githubusercontent.com/69920975/113880325-5ba1a000-97f6-11eb-914b-7f491f289589.png)
![image](https://user-images.githubusercontent.com/69920975/113880335-5e03fa00-97f6-11eb-919d-bae9a09d0d27.png)

 내 dataset으로는 학습이 완전히 잘되지는 않아서 원래의 pretrained 된 weights를 가지고 한 결과 어느 정도 detection이 잘 되는 것을 확인할 수 있었다.









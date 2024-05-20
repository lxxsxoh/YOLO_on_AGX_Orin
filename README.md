YOLO_on_AGX_Orin
Start without Docker
====================
#### This repository provides software and hardware setup, and object detection using Yolov8 on NVIDIA Jetson AGX Orin.

YOLO(You Only Look Once)
------------------------
- YOLO는 이미지를 한번만 보고 바로 Object Detection을 수행. 이미지에 대해 빠른 속도로 Object Detection을 수행할 수 있음.
- Classification + Localization -> One-stage detection
- region proposal, feature extraction, classification, bbox regression
- CNN 딥러닝 모델을 기반으로 특징을 추출한 뒤 물체의 종류와 위치를 예측함.

#### YOLOv8 model 설치를 위해 Ultralytics Package를 설치해준다.

Install Ultralytics Package
---------------------------
1. Update pakages list, install pip and upgrade to latest
<pre>
<code>
   sudo apt update
   sudo apt install python3-pip -y
   pip install -U pip
</code>
</pre>
2. Install ultralystics pip package with optional dependencies
<pre>
<code>
  pip install ultralystics
</code>
</pre>
- error
  <pre>
  <code>
     ERROR: scipy 1.10.1 has requirement numpy<1.27.0, >=1.19.5, but you'll have numpy 1.17.4 which is incompatible
     ERROR: ...
  </code>
  </pre>
2.1 Reinstall numpy
<pre>
<code>
   pip install numpy==1.24.4
</code>
</pre>
2.2 Reinstall ultralystics pip package with optional dependencies
<pre>
<code>
  pip install ultralystics
</code>
</pre>
3. Reboot the device
<pre>
<code>
  sudo reboot
</code>
</pre>
Install PyTorch and Torchvision
-------------------------------
Python 3.8.10

PyTorch = 2.1.0a0+41361538.nv23.06

Torchvision = 0.16.2+c6f3977

https://elinux.org/Jetson_Zoo#ONNX_Runtime 

Jetpack 5.1.2, Python3.8.10 -> onnxruntime 1.17.0

![KakaoTalk_20240512_220418759](https://github.com/lxxsxoh/YOLO_on_AGX_Orin/assets/136955006/d560278a-6793-4556-916e-68eda98bc39a)


Install onnxruntime 1.17.0
---------------------------
1.
<pre>
   <code>
      pip install onnxruntime-gpu
   </code>
</pre>

2.
<pre>
   <code>
      # Download pip wheel from location above for your version of JetPack
      $ wget https://nvidia.box.com/shared/static/iizg3ggrtdkqawkmebbfixo7sce6j365.whl -O onnxruntime_gpu-1.17.0-cp38-cp38-linux_aarch64.whl

      # Install pip wheel
      $ pip3 install onnxruntime_gpu-1.17.0-cp38-cp38-linux_aarch64.whl
   </code>
</pre>

![KakaoTalk_20240512_220227124](https://github.com/lxxsxoh/YOLO_on_AGX_Orin/assets/136955006/6192f888-6808-42ef-92cb-bedab85eb9e1)

onnxruntime-gpu will automatically revert back the numpy version to latest. So we need to reinstall numpy to 1.23.5 to fix an issue by executing:
------------------------
<pre>
   <code>
      pip install numpy==1.23.5
   </code>
</pre>

Install onnx 1.9.0
------------------
1.
<pre>
   <code>
      sudo apt-get install protobuf-compiler libprotobuf-dev
   </code>
</pre>

2.
<pre>
   <code>
      python3 -m pip install --upgrade pip
   </code>
</pre>

3.
<pre>
   <code>
      pip install onnx==1.9.0
   </code>
</pre>

![KakaoTalk_20240520_202457953](https://github.com/lxxsxoh/YOLO_on_AGX_Orin/assets/136955006/812eb694-d218-4595-8073-b1d785479848)

YOLOv8
------
- YOLOv8은 YOLO 최신 버전으로, Detection/Segment/Classification/Pose/OBB의 다섯가지 Task를 처리한다.
   - Detection: task that involves identifying the location and class of objects in an image or video stream.
      - Detection의 output은 사물을 가리키는 사각형과 그 사물이 뭔지, 어느정도 일치하는지까지를 나타낸다.
     ### Train
        <pre>
         <code>
            from ultralytics import YOLO

            # Load a model
            model = YOLO('yolov8n.yaml')  # build a new model from YAML
            model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)
            model = YOLO('yolov8n.yaml').load('yolov8n.pt')  # build from YAML and transfer weights

            # Train the model
            results = model.train(data='coco8.yaml', epochs=100, imgsz=640)
         </code>
        </pre>
        ![KakaoTalk_20240520_214246835](https://github.com/lxxsxoh/YOLO_on_AGX_Orin/assets/136955006/008acbfb-6fae-4126-971b-648e356d3d9a)
   - Segment: a step further than detection and involves identifying individual objects in an image and segmenting them from he rest of image.
      - Segment는 detection보다 더 세밀한 task로, object의 outline을 따서 물체를 구분한다.
     ### Train
        <pre>
         <code>
            from ultralytics import YOLO

            # Load a model
            model = YOLO('yolov8n-seg.yaml')  # build a new model from YAML
            model = YOLO('yolov8n-seg.pt')  # load a pretrained model (recommended for training)
            model = YOLO('yolov8n-seg.yaml').load('yolov8n.pt')  # build from YAML and transfer weights

            # Train the model
            results = model.train(data='coco8-seg.yaml', epochs=100, imgsz=640)
         </code>
         </pre>
   - Classification: the simplest of the three tasks and involves classifying an entire image into one of a set of predefined classes.
     - Image에 뭐가 있는지 알고 싶고, 위치는 따로 알 필요 없을 때 사용하는 task이다.
     ### Train
        <pre>
         <code>
            from ultralytics import YOLO

            # Load a model
            model = YOLO('yolov8n-cls.yaml')  # build a new model from YAML
            model = YOLO('yolov8n-cls.pt')  # load a pretrained model (recommended for training)
            model = YOLO('yolov8n-cls.yaml').load('yolov8n-cls.pt')  # build from YAML and transfer weights

            # Train the model
            results = model.train(data='mnist160', epochs=100, imgsz=64)
        </code>
      </pre>
   - Pose: a task that involves identifying the location of specific points in an image, usually referred to as keypoints.
     - pose를 알고 싶을 때, keypoint와 선으로 output이 표시된다.
     ### Train
        <pre>
         <code>
            from ultralytics import YOLO

            # Load a model
            model = YOLO('yolov8n-pose.yaml')  # build a new model from YAML
            model = YOLO('yolov8n-pose.pt')  # load a pretrained model (recommended for training)
            model = YOLO('yolov8n-pose.yaml').load('yolov8n-pose.pt')  # build from YAML and transfer weights

            # Train the model
            results = model.train(data='coco8-pose.yaml', epochs=100, imgsz=640)
        </code>
       </pre>
   - OBB(Oriented Bounding Boxes Object Detection): a step further than object detection and introduce an extra angle to locate object more accurate in an image.
     ### Train
        <pre>
         <code>
            from ultralytics import YOLO

            # Load a model
            model = YOLO('yolov8n-obb.yaml')  # build a new model from YAML
            model = YOLO('yolov8n-obb.pt')  # load a pretrained model (recommended for training)
            model = YOLO('yolov8n-obb.yaml').load('yolov8n.pt')  # build from YAML and transfer weights

            # Train the model
            results = model.train(data='dota8.yaml', epochs=100, imgsz=640)
         </code>
         </pre>
         

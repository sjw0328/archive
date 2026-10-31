# hinge-demo

Estimates a laptop's lid angle from the webcam (optical flow) and plays a frosted-glass transition overlay on the screen as the lid closes and opens. Windows, Python.

노트북 웹캠 영상(광학 흐름)으로 덮개 각도를 추정하고, 덮개를 닫고 열 때 화면에 서리 유리 전환 연출을 띄우는 데모입니다.

```
pip install opencv-python numpy PySide6
python hinge_demo.py          # webcam
python hinge_demo.py --sim    # slider only, no webcam
```

- Uses the webcam and captures the screen image locally to draw the overlay. Nothing is sent over the network.
- Angle estimates depend on lighting and camera; they may drift.
- 웹캠을 사용하고, 연출을 그리기 위해 화면을 로컬에서 캡처합니다. 네트워크 전송 코드는 포함되어 있지 않습니다.
- 각도 추정은 조명·카메라에 따라 오차가 쌓일 수 있습니다.

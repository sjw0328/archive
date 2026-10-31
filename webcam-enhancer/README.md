# webcam-enhancer

Real-time webcam image correction (denoise, exposure, white balance, local tone, motion deblur, background blur, etc.) sent to a virtual camera, with a GUI. Windows, Python, single file.

웹캠 화질을 실시간으로 보정(노이즈 제거, 노출, 화이트밸런스, 로컬 톤, 모션 블러 완화, 배경 흐림 등)해 가상 카메라로 내보내는 프로그램입니다. 단일 파일.

```
pip install opencv-python numpy pyvirtualcam customtkinter pillow
pip install mediapipe        # optional: background blur
python webcam_enhancer.py            # GUI
python webcam_enhancer.py --bench    # benchmark
python webcam_enhancer.py --list     # list cameras
```

Virtual camera output needs a virtual camera driver (for example the one installed with OBS Studio). Without it, preview only.
가상 카메라 출력에는 가상 카메라 드라이버(예: OBS Studio 설치 시 함께 설치되는 것)가 필요합니다. 없으면 미리보기만 됩니다.

## Notes / 참고

- Some features download model files on first use into `~/.webcam_enhancer`: YuNet face detector (opencv/opencv_zoo) and MediaPipe selfie segmenter (Google). Those models are covered by their own licenses, not by this repository's license.
- Image quality and frame rate depend on the camera and PC.

- 일부 기능은 처음 쓸 때 모델 파일을 `~/.webcam_enhancer` 로 내려받습니다: YuNet 얼굴 검출기(opencv/opencv_zoo), MediaPipe 셀피 분할 모델(Google). 해당 모델은 이 저장소 라이선스가 아니라 각자의 라이선스를 따릅니다.
- 화질과 프레임은 카메라·PC 사양에 따라 달라집니다.

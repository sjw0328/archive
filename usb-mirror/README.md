# usb-mirror

Shows an Android phone's screen on a Windows PC over a USB cable, without a capture card, using adb and scrcpy-server. Includes keyboard-to-touch mapping (WASD joystick, F9 key placement).

USB 케이블로 안드로이드 폰 화면을 PC에 띄우는 프로그램입니다(캡처보드 불필요, adb + scrcpy-server 사용). 키보드→터치 매핑 포함.

```
pip install av opencv-python
python mirror.py
```

Requires USB debugging enabled on the phone. / 폰에서 USB 디버깅을 켜야 합니다.

## Notes / 참고

- On first run the script downloads adb (Google Android SDK Platform-Tools) and scrcpy-server (Genymobile/scrcpy) from their official URLs into `%LOCALAPPDATA%\usb_mirror`. Those tools are covered by their own licenses and terms, not by this repository's license.
- The script may try to install missing Python packages with pip.
- USB debugging gives the PC broad access to the phone. Use only with devices and PCs you trust.

- 첫 실행 때 adb(Google Android SDK Platform-Tools)와 scrcpy-server(Genymobile/scrcpy)를 공식 주소에서 `%LOCALAPPDATA%\usb_mirror` 로 내려받습니다. 해당 도구는 이 저장소 라이선스가 아니라 각자의 라이선스·약관을 따릅니다.
- 빠진 파이썬 패키지를 pip로 설치하려 할 수 있습니다.
- USB 디버깅은 PC에 폰에 대한 넓은 접근 권한을 줍니다. 신뢰하는 기기·PC에서만 사용하십시오.

# omega-spatial

Real-time spatial audio engine in a single Python file (HRTF, distance attenuation, Doppler, room reverb, stereo widening, etc.) with a simple GUI. Version 2.8.

단일 파이썬 파일로 된 실시간 공간음향 엔진(HRTF, 거리 감쇠, 도플러, 룸 리버브, 스테레오 와이드닝 등)과 간단한 GUI입니다. 버전 2.8.

```
pip install numpy scipy sounddevice   # numba optional
python OMEGA_SPATIAL.py                               # GUI
python OMEGA_SPATIAL.py --spatialize in.wav out.wav   # file
python OMEGA_SPATIAL.py --live                        # system audio (loopback)
```

## Notes / 참고

- Live mode depends on the audio devices and drivers on the PC; it may produce no sound, dropouts or distortion on some setups.
- Start at low volume. Processing bugs can produce loud or harsh output.

- 실시간 모드는 PC의 오디오 장치·드라이버에 따라 소리가 안 나거나 끊기거나 왜곡될 수 있습니다.
- 낮은 볼륨에서 시작하십시오. 처리 오류로 큰 소리나 거친 소리가 날 수 있습니다.

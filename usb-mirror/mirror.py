"""
안드로이드 USB 화면 미러링 뼈대 (캡처보드 없음)

흐름:
  1) adb 로 scrcpy-server 를 폰에 올리고 실행
  2) adb forward 로 USB 터널을 열어 H.264 스트림 수신
  3) PyAV 로 디코딩 -> OpenCV 창에 표시
  4) 폰 소리는 무압축 PCM 으로 받아 PC 스피커로 재생 (sounddevice, 없으면 자동 설치)

조작 (미러링 창에서):
  왼쪽 클릭·드래그 = 터치·스와이프      휠 = 스크롤
  오른쪽 클릭 / ESC = 뒤로             가운데 클릭 / F1 = 홈      F2 = 최근 앱
  키보드 = 영문·숫자·기호 입력, Enter, Backspace, 방향키
  Ctrl+V = PC 클립보드 내용을 폰에 붙여넣기 (한글은 이 방법으로)
  F3 = 폰 키보드 한/영 전환 (Shift+Space 를 보낸다)
  F4 = 소리 켬/끔
  F7 = 녹화 시작/정지 (실행하면 자동으로 녹화가 시작된다. 저장 위치: 동영상\\USB Mirror)
  F12 = 가상 카메라 켬/끔: 폰 영상을 PC 의 웹캠 장치(OBS Virtual Camera)로 내보낸다.
        OBS 에서는 소스 추가 -> 비디오 캡처 장치 -> "OBS Virtual Camera" 로 잡힌다. 줌·디스코드에서도 웹캠으로 보인다.
        OBS Studio 가 설치돼 있어야 한다(가상 카메라 드라이버가 OBS 에 들어 있다). pyvirtualcam 은 자동 설치.
  F6 = 소리 캡처 방식 전환 (output <-> playback). 소리가 안 나는 폰에서 바꿔 본다. 선택은 저장된다.
  종료 = 창 닫기(X)

카메라 모드 (F5 로 전환: 폰 화면 -> 후면 카메라 -> 전면 카메라 -> 폰 화면):
  폰 화면을 켜지 않고 카메라 영상만 받아 온다 (Android 12 이상). 소리는 폰 마이크가 잡힌다.
  F11 = 보이는 방향 90도 회전.  카메라 모드에서는 터치·키보드 조작과 게임 모드가 꺼진다.
  PC 스피커로는 마이크 소리를 내지 않는다(하울링 방지). F4 로 켤 수 있고, 녹화에는 항상 들어간다.

게임 모드 (F8 로 켜고 끔, 윈도우 전용):
  W A S D = 가상 조이스틱 (여러 키 동시 입력, 대각선 이동 가능)
  배치한 키 = 그 위치를 누르고 있는 동안 터치 유지 (조이스틱과 동시에 눌린다)
  왼쪽 클릭 = 그대로 터치.  키보드 글자 입력·우클릭·휠은 오작동 방지를 위해 꺼진다.
  F9 = 키 배치: 키를 한 번 누르고 -> 화면의 버튼을 클릭하면 그 키에 묶인다.
       W/A/S/D 중 하나를 누르고 클릭하면 조이스틱 중심이 옮겨진다. 우클릭 = 그 키 배치 삭제.
       다시 F9 를 누르면 저장하고 끝낸다. (저장 위치: %LOCALAPPDATA%\\usb_mirror\\keymap.json)

준비물: pip install av opencv-python  (이것만 있으면 됨)
  adb 와 scrcpy-server 는 없으면 첫 실행 때 자동으로 내려받는다.
  저장 위치: %LOCALAPPDATA%\\usb_mirror  (한글 경로 문제를 피하려고 스크립트 폴더와 분리)
"""
import glob
import importlib
import io
import json
import math
import os
import re
import shutil
import socket
import struct
import subprocess
import sys
import threading
import time
import urllib.request
import zipfile
from fractions import Fraction

import av
import cv2
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
TOOLS = os.path.join(os.environ.get("LOCALAPPDATA") or os.path.expanduser("~"), "usb_mirror")
ADB = "adb"  # main() 에서 실제 경로로 바뀐다
PORT = 27183
DEVICE_JAR = "/data/local/tmp/scrcpy-server.jar"

SCRCPY_API = "https://api.github.com/repos/Genymobile/scrcpy/releases/latest"
ADB_ZIP = "https://dl.google.com/android/repository/platform-tools-latest-windows.zip"
ADB_FILES = ("adb.exe", "AdbWinApi.dll", "AdbWinUsbApi.dll")

# 화질은 실행할 때 자동으로 정한다 (pick_settings). 아래는 그 계산의 상한·기준값.
MAX_FPS = 60
BITS_PER_PIXEL = 0.2     # 픽셀당 비트. H.264 화면 영상이 원본과 거의 구분 안 되는 수준
LINK_SHARE = 0.3         # 측정한 케이블 속도 중 영상에 쓸 비율 (나머지는 여유분)
BITRATE_CAP = 64_000_000 # 폰 인코더가 안정적으로 내는 한계
BITRATE_FLOOR = 4_000_000
START_MODE = "screen"    # 시작할 때의 모드: "screen"(폰 화면), "back"(후면 카메라), "front"(전면 카메라)
MODES = ("screen", "back", "front")
CAMERA_SIZE = 1920       # 카메라 영상의 긴 변 (16:9 로 받으므로 1920x1080)
CAMERA_FPS = 30
CAMERA_BITRATE = 16_000_000
VCAM_FPS = 60            # 가상 카메라로 내보내는 프레임 수 (PC 가 버거우면 30)
VCAM_BACKEND = None      # None = 자동 선택 (윈도우: OBS Virtual Camera, 없으면 Unity Capture)
AUTO_RECORD = True       # True 면 미러링을 시작할 때 녹화도 자동으로 시작한다 (F7 로 켜고 끔)
RECORD_DIR = os.path.join(os.path.expanduser("~"), "Videos", "USB Mirror")
RECORD_MIN_FREE = 3 * 1024 ** 3   # 디스크 여유가 이보다 적어지면 녹화를 멈춘다 (3GB)
VOLUME = 1.5             # PC 로 내보내는 소리 배율 (1.0 = 폰 원음 그대로)
PHONE_SOUND_TOO = False  # True 면 폰 스피커에서도 계속 소리가 난다 (Android 13 이상, playback 방식 전용)
# 폰 소리를 잡는 방식. output = 폰의 최종 출력 전체, playback = 앱들이 재생하는 소리(캡처를 막은 앱은 제외).
# 폰에 따라 output 이 무음만 내는 경우가 있어, 6초간 무음이면 한 번 playback 으로 자동 전환한다.
AUDIO_SOURCES = ("output", "playback")
SETTINGS_FILE = os.path.join(TOOLS, "settings.json")
AUDIO_RATE = 48000 * 4   # 초당 바이트: 48kHz, 스테레오, 16비트
AUDIO_START = AUDIO_RATE * 40 // 1000    # 이만큼(40ms) 모이면 재생 시작: 끊김 방지용 최소 여유
AUDIO_LIMIT = AUDIO_RATE * 120 // 1000   # 이보다 많이 쌓이면 소리가 화면보다 늦어진 것
AUDIO_KEEP = AUDIO_RATE * 50 // 1000     # 그때 최근 50ms 만 남기고 버려서 따라잡는다
OVERLOAD = 0.8           # 디코딩 스레드 사용률이 이 값을 3초 연속 넘으면 한 단계 낮춘다
WINDOW = "USB Mirror"    # OpenCV 창 제목은 영문만 (한글 깨짐). 창이 앞에 있는지 판별할 때도 쓴다
WINDOW_WIDTH = 1500      # 처음 띄울 창의 최대 크기 (윈도우에서는 실행 시 화면의 85%로 바뀐다)
WINDOW_HEIGHT = 900

KEYMAP_FILE = os.path.join(TOOLS, "keymap.json")
JOY_RADIUS = 0.13        # 조이스틱을 미는 거리 (화면 짧은 변 대비)
JOY_ID = 1               # 조이스틱 터치의 손가락 번호. 배치한 키는 자기 키코드를 번호로 쓴다
VK_W, VK_A, VK_S, VK_D = 0x57, 0x41, 0x53, 0x44
# 배치할 수 있는 키: 숫자, 영문, Space, Shift, Ctrl, Tab
BINDABLE = list(range(0x30, 0x3A)) + list(range(0x41, 0x5B)) + [0x20, 0x10, 0x11, 0x09]
KEY_NAMES = {0x20: "SPC", 0x10: "SHIFT", 0x11: "CTRL", 0x09: "TAB"}

if os.name == "nt":
    import ctypes
    from ctypes import wintypes
    _user32 = ctypes.windll.user32
    _user32.GetAsyncKeyState.argtypes = [ctypes.c_int]
    _user32.GetAsyncKeyState.restype = ctypes.c_short
    _user32.GetForegroundWindow.restype = wintypes.HWND
    _user32.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]


def key_down(vk):
    """그 키가 지금 눌려 있는지 (윈도우 전용). 창 이벤트와 달리 여러 키의 누름·뗌을 동시에 알 수 있다."""
    return bool(_user32.GetAsyncKeyState(vk) & 0x8000)


def window_focused():
    """미러링 창이 맨 앞에 있는지. 다른 창에서 타자를 칠 때 폰이 눌리지 않게 한다."""
    buf = ctypes.create_unicode_buffer(64)
    _user32.GetWindowTextW(_user32.GetForegroundWindow(), buf, 64)
    return buf.value.startswith(WINDOW)


def key_label(vk):
    return KEY_NAMES.get(vk) or chr(vk)


def load_keymap():
    """키 배치. 좌표는 화면 크기에 대한 비율(0~1)이라 해상도가 바뀌어도 유지된다."""
    keymap = {"joystick": [0.17, 0.72], "keys": {}}
    try:
        with open(KEYMAP_FILE, encoding="utf-8") as f:
            saved = json.load(f)
        keymap["joystick"] = [float(v) for v in saved["joystick"]]
        keymap["keys"] = {str(int(k)): [float(v[0]), float(v[1])] for k, v in saved["keys"].items()}
    except (OSError, ValueError, KeyError, TypeError, IndexError):
        pass
    return keymap


def save_keymap(keymap):
    try:
        os.makedirs(TOOLS, exist_ok=True)
        with open(KEYMAP_FILE, "w", encoding="utf-8") as f:
            json.dump(keymap, f, indent=1)
    except OSError as e:
        print(f"키 배치 저장 실패: {e}")


def load_settings():
    try:
        with open(SETTINGS_FILE, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def save_settings(data):
    try:
        os.makedirs(TOOLS, exist_ok=True)
        with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=1)
    except OSError as e:
        print(f"설정 저장 실패: {e}")


def die(msg):
    print(f"[오류] {msg}")
    sys.exit(1)


def fetch(url):
    req = urllib.request.Request(url, headers={"User-Agent": "usb-mirror"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.read()
    except OSError as e:
        die(f"다운로드 실패: {url}\n        {e}")


def ensure_adb():
    """adb 경로를 돌려준다. 스크립트 폴더 -> 도구 폴더 -> PATH 순으로 찾고, 없으면 내려받는다."""
    for folder in (BASE, TOOLS):
        path = os.path.join(folder, "adb.exe")
        if os.path.exists(path):
            return path
    found = shutil.which("adb")
    if found:
        return found
    if os.name != "nt":
        die("adb 가 없습니다. platform-tools 를 설치하세요.")

    print("adb 내려받는 중...")
    os.makedirs(TOOLS, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(fetch(ADB_ZIP))) as z:
        for name in ADB_FILES:
            with open(os.path.join(TOOLS, name), "wb") as f:
                f.write(z.read(f"platform-tools/{name}"))
    return os.path.join(TOOLS, "adb.exe")


def ensure_server():
    """scrcpy-server-vX.Y.Z 파일과 버전을 돌려준다. 없으면 최신 릴리스에서 내려받는다.
    서버는 실행 인자의 버전이 자기 버전과 다르면 바로 종료하므로 파일명의 버전을 그대로 쓴다."""
    pattern = r"scrcpy-server-v(\d+(?:\.\d+)+)$"
    for folder in (BASE, TOOLS):
        for path in sorted(glob.glob(os.path.join(folder, "scrcpy-server-v*")), reverse=True):
            m = re.search(pattern, os.path.basename(path))
            if m:
                return path, m.group(1)

    print("scrcpy-server 내려받는 중...")
    release = json.loads(fetch(SCRCPY_API))
    for asset in release.get("assets", []):
        m = re.search(pattern, asset["name"])
        if m:
            os.makedirs(TOOLS, exist_ok=True)
            path = os.path.join(TOOLS, asset["name"])
            data = fetch(asset["browser_download_url"])
            with open(path, "wb") as f:
                f.write(data)
            return path, m.group(1)
    die("릴리스에서 scrcpy-server 파일을 찾지 못했습니다.")


def adb(*args, serial=None):
    cmd = [ADB] + (["-s", serial] if serial else []) + list(args)
    return subprocess.run(cmd, capture_output=True, text=True, errors="replace")


def find_device():
    """기기가 잡히고 디버깅이 허용될 때까지 최대 60초 기다린다."""
    last = None
    for _ in range(60):
        out = adb("devices").stdout
        rows = [line.split("\t") for line in out.splitlines()[1:] if "\t" in line]
        if rows and rows[0][1] == "device":
            return rows[0][0]
        if not rows:
            msg = "폰을 기다리는 중... 케이블을 꽂고 USB 디버깅을 켜세요."
        elif rows[0][1] == "unauthorized":
            msg = "폰 화면의 'USB 디버깅을 허용하시겠습니까?' 창에서 허용을 누르세요."
        else:
            msg = f"기기 상태: {rows[0][1]} (대기 중)"
        if msg != last:
            print(msg)
            last = msg
        time.sleep(1)
    die("기기를 찾지 못했습니다. 데이터 전송이 되는 케이블인지 확인하세요.")


def measure_link(serial):
    """폰 -> PC 로 24MB 를 흘려 보내 케이블의 실제 전송 속도(bps)를 잰다."""
    size = 24 * 1024 * 1024
    cmd = [ADB, "-s", serial, "exec-out", "dd if=/dev/zero bs=1048576 count=24 2>/dev/null"]
    try:
        t = time.perf_counter()
        got = len(subprocess.run(cmd, capture_output=True, timeout=30).stdout)
        dt = time.perf_counter() - t
    except (OSError, subprocess.TimeoutExpired):
        got = 0
    if got < size // 2:
        return 160_000_000  # 측정 실패 시 USB 2.0 수준으로 가정
    return got * 8 / dt


def screen_size(serial):
    """폰 화면 해상도. 사용자가 해상도를 낮춰 쓰는 경우(Override)가 있으면 그 값을 쓴다."""
    found = re.findall(r"(\d+)x(\d+)", adb("shell", "wm", "size", serial=serial).stdout)
    if not found:
        return 1080, 2340
    w, h = found[-1]
    return int(w), int(h)


def make_settings(w, h, link_bps, scale=1.0):
    """해상도 배율(scale)에 맞는 설정. 비트레이트는 화질 기준·케이블 여유·인코더 한계 중 가장 작은 값."""
    pixels = w * h * scale * scale
    bitrate = min(pixels * MAX_FPS * BITS_PER_PIXEL, link_bps * LINK_SHARE, BITRATE_CAP)
    return {
        "w": w, "h": h, "link": link_bps, "scale": scale,
        # 0 = 원본 해상도 그대로. 낮출 때는 인코더가 안전하게 처리하는 16의 배수로 맞춘다
        "max_size": 0 if scale >= 1 else int(max(w, h) * scale) // 16 * 16,
        "bitrate": int(max(bitrate, BITRATE_FLOOR)),
    }


def start_server(serial, server_path, version, cfg):
    r = adb("push", server_path, DEVICE_JAR, serial=serial)
    if r.returncode != 0:
        die(f"서버 파일 전송 실패: {r.stderr.strip()}")
    r = adb("forward", f"tcp:{PORT}", "localabstract:scrcpy", serial=serial)
    if r.returncode != 0:
        die(f"포트 포워딩 실패: {r.stderr.strip()}")

    cmd = [
        ADB, "-s", serial, "shell",
        f"CLASSPATH={DEVICE_JAR}",
        "app_process", "/", "com.genymobile.scrcpy.Server", version,
        "tunnel_forward=true",
        "audio=true",             # 소리용 소켓 (Android 11 이상. 안 되는 폰은 화면만 나온다)
        "audio_codec=raw",        # 무압축 PCM: 디코딩이 필요 없고 지연이 가장 짧다
        "send_device_meta=false",
        "send_stream_meta=false", # 스트림 앞의 코덱·화면크기 헤더를 끈다 (최신 서버의 옵션 이름)
        "send_codec_meta=false",  # 같은 옵션의 옛 이름. 모르는 쪽은 서버가 경고만 하고 무시한다
        "send_frame_meta=false",  # 헤더 없이 순수 H.264 스트림만 받는다
        "video_codec_options=i-frame-interval=2",  # 2초마다 키프레임: 녹화를 켜면 2초 안에 시작된다
    ]
    if cfg["mode"] == "screen":
        cmd += [
            f"audio_source={cfg['audio_source']}",
            f"audio_dup={'true' if PHONE_SOUND_TOO and cfg['audio_source'] == 'playback' else 'false'}",
            "control=true",           # 마우스·키보드 조작용 소켓을 하나 더 연다
            "clipboard_autosync=false",
            f"max_size={cfg['max_size']}",
            f"video_bit_rate={cfg['bitrate']}",
            f"max_fps={MAX_FPS}",
        ]
    else:
        cmd += [
            "video_source=camera",    # 화면 대신 카메라를 직접 받는다 (Android 12 이상)
            f"camera_facing={cfg['mode']}",
            f"max_size={CAMERA_SIZE}",
            "camera_ar=16:9",
            f"camera_fps={CAMERA_FPS}",
            f"video_bit_rate={CAMERA_BITRATE}",
            "audio_source=mic",
            "control=false",          # 카메라에는 터치할 화면이 없다
            "power_on=false",         # 폰 화면을 켜지 않는다
        ]
    # 이 프로세스가 살아 있는 동안만 서버가 돈다. 로그는 콘솔로 그대로 출력.
    return subprocess.Popen(cmd)


def connect(proc, control=True):
    """서버가 뜰 때까지 재시도하고 (영상 소켓, 소리 소켓, 조작 소켓) 을 돌려준다.
    서버는 영상 -> 소리 -> 조작 순서로 연결을 받고, 전부 연결돼야 보내기 시작한다.
    첫 연결(영상)에만 서버가 확인용 1바이트를 먼저 보낸다. 카메라 모드는 조작 소켓이 없다(None)."""
    for _ in range(100):
        if proc.poll() is not None:
            raise RuntimeError("서버가 바로 종료됐습니다. 위 로그를 확인하세요.")
        try:
            s = socket.create_connection(("127.0.0.1", PORT), timeout=2)
            s.settimeout(2)
            if s.recv(1):
                s.settimeout(None)
                s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                a = socket.create_connection(("127.0.0.1", PORT), timeout=5)
                a.settimeout(None)
                if not control:
                    return s, a, None
                c = socket.create_connection(("127.0.0.1", PORT), timeout=5)
                c.settimeout(None)
                c.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                return s, a, c
            s.close()
        except OSError:
            pass
        time.sleep(0.1)
    raise RuntimeError("서버 연결 시간 초과.")


def scan_nals(data):
    """H.264 패킷의 앞부분을 훑어 (키프레임 여부, 설정 정보 SPS+PPS 바이트) 를 돌려준다.
    설정 정보는 해상도 등 디코딩에 필요한 값으로, 폰은 인코더를 시작할 때 한 번만 보낸다."""
    nals, pos = [], 0
    while len(nals) < 16:
        i = data.find(b"\x00\x00\x01", pos)
        if i < 0 or i + 3 >= len(data):
            break
        kind = data[i + 3] & 0x1F
        nals.append((i, kind))
        if kind in (1, 5):       # 화면 데이터가 시작되면 그 뒤는 볼 필요 없다
            break
        pos = i + 3
    config = b""
    for n, (i, kind) in enumerate(nals):
        if kind in (7, 8):       # 7 = SPS, 8 = PPS
            end = nals[n + 1][0] if n + 1 < len(nals) else len(data)
            config += b"\x00\x00\x00\x01" + data[i + 3:end].rstrip(b"\x00")
    return any(kind == 5 for _, kind in nals), config


def to_mp4(mkv):
    """녹화가 끝난 MKV 를 MP4 로 바꾼다. 영상은 그대로 복사하고 소리만 AAC 로 압축한다."""
    mp4 = mkv[:-4] + ".mp4"
    try:
        with av.open(mkv) as src, av.open(mp4, "w", options={"movflags": "faststart"}) as dst:
            v_in = src.streams.video[0]
            v_out = dst.add_stream_from_template(v_in)
            a_in = src.streams.audio[0] if src.streams.audio else None
            if a_in is not None:
                a_out = dst.add_stream("aac", rate=48000)
                a_out.layout = "stereo"
                a_out.bit_rate = 192_000
                fit = av.AudioResampler(format="fltp", layout="stereo", rate=48000, frame_size=1024)
            written = 0   # 지금까지 넣은 소리 샘플 수

            def encode(frame):
                for part in fit.resample(frame):
                    dst.mux(a_out.encode(part))

            for packet in src.demux():
                if packet.dts is None:
                    continue
                if packet.stream is v_in:
                    packet.stream = v_out
                    dst.mux(packet)
                elif packet.stream is a_in:
                    pcm = bytes(packet)
                    start = int(packet.pts * a_in.time_base * 48000)
                    if start - written > 480:   # 소리가 비어 있던 구간은 무음으로 채워 화면과 어긋나지 않게 한다
                        pcm = bytes((start - written) * 4) + pcm
                    frame = av.AudioFrame.from_ndarray(np.frombuffer(pcm, "<i2").reshape(1, -1),
                                                       format="s16", layout="stereo")
                    frame.sample_rate = 48000
                    frame.pts = written
                    written += len(pcm) // 4
                    encode(frame)
            if a_in is not None:
                encode(None)
                dst.mux(a_out.encode(None))
        os.remove(mkv)
        print(f"녹화 저장됨: {mp4} ({os.path.getsize(mp4) / 1e6:.0f}MB)")
    except Exception as e:
        print(f"MP4 변환 실패. MKV 원본을 남깁니다: {mkv} ({e!r})")
        try:
            os.remove(mp4)
        except OSError:
            pass


class Recorder:
    """받은 영상을 다시 압축하지 않고 그대로 파일에 담는다 (화질 손실·CPU 부담 없음).
    녹화 중에는 프로그램이 갑자기 꺼져도 파일이 살아남는 MKV 로 쓰고, 끝나면 MP4 로 바꾼다.
    화면 회전 등으로 해상도가 바뀌면 한 파일에 담을 수 없어 새 파일로 나눈다."""

    def __init__(self):
        self.lock = threading.RLock()
        self.on = False        # 녹화를 원하는 상태인지
        self.out = None        # 열려 있는 MKV
        self.path = None
        self.config = b""      # 스트림에서 가장 최근에 본 설정 정보 (SPS+PPS)
        self.file_config = b"" # 지금 파일의 설정 정보
        self.t0 = 0.0
        self.last_v = -1
        self.a_pts = 0
        self.count = 0
        self.checked = 0.0
        self.jobs = []         # MP4 변환 스레드

    def toggle(self):
        with self.lock:
            if self.on:
                self.stop()
            else:
                self.on = True
                print("녹화 켜짐 (다음 키프레임부터 기록, 최대 2초)")

    def stop(self):
        with self.lock:
            if self.on:
                self.on = False
                self._close()
                print("녹화 꺼짐")

    def status(self):
        """창 제목에 붙일 표시."""
        if self.out is not None:
            sec = int(time.perf_counter() - self.t0)
            return f"[REC {sec // 60:02d}:{sec % 60:02d}]"
        return "[REC ...]" if self.on else ""

    def wait(self):
        """진행 중인 MP4 변환이 끝날 때까지 기다린다 (프로그램 종료 직전에 호출)."""
        jobs = [j for j in self.jobs if j.is_alive()]
        if jobs:
            print("녹화 파일을 MP4 로 변환하는 중... 창을 닫지 말고 기다리세요.")
        for job in jobs:
            job.join()

    def _open(self, data, ts):
        try:
            os.makedirs(RECORD_DIR, exist_ok=True)
            base = os.path.join(RECORD_DIR, time.strftime("mirror_%Y%m%d_%H%M%S"))
            path, n = base + ".mkv", 1
            while os.path.exists(path) or os.path.exists(path[:-4] + ".mp4"):
                n += 1
                path = f"{base}_{n}.mkv"
            with av.open(io.BytesIO(data), format="h264") as tmpl:   # 첫 키프레임에서 해상도 등 스트림 정보를 읽는다
                out = av.open(path, "w")
                self.vstream = out.add_stream_from_template(tmpl.streams.video[0])
            self.astream = out.add_stream("pcm_s16le", rate=48000)
            self.astream.layout = "stereo"
        except Exception as e:
            print(f"녹화를 시작할 수 없습니다: {e!r}")
            self.on = False
            return False
        self.out, self.path, self.file_config = out, path, self.config
        self.t0, self.last_v, self.a_pts, self.count = ts, -1, 0, 0
        print(f"녹화 시작: {path[:-4]}.mp4")
        return True

    def _close(self):
        out, path, count = self.out, self.path, self.count
        self.out = None
        if out is None:
            return
        try:
            out.close()
        except Exception as e:
            print(f"녹화 파일 닫기 오류: {e!r}")
        if count == 0:
            try:
                os.remove(path)
            except OSError:
                pass
            return
        job = threading.Thread(target=to_mp4, args=(path,))
        job.start()
        self.jobs.append(job)

    def _mux(self, packet):
        try:
            self.out.mux(packet)
            return True
        except Exception as e:
            print(f"녹화 쓰기 오류로 녹화를 멈춥니다: {e!r}")
            self.on = False
            self._close()
            return False

    def video(self, data, ts):
        """영상 패킷 하나 (수신 스레드가 부른다). 녹화 중이 아니어도 설정 정보는 기억해 둔다."""
        key, config = scan_nals(data)
        with self.lock:
            if config:
                self.config = config
            if not self.on:
                return
            if self.out is not None and config and config != self.file_config:
                self._close()                    # 해상도·회전이 바뀜: 새 파일로
            if self.out is None:
                if not key or not self.config:   # 파일은 키프레임으로 시작해야 재생된다
                    return
                if not config:
                    data = self.config + data
                if not self._open(data, ts):
                    return
            packet = av.Packet(data)
            packet.stream = self.vstream
            packet.time_base = Fraction(1, 1000)
            self.last_v = max(int((ts - self.t0) * 1000), self.last_v + 1)
            packet.pts = packet.dts = self.last_v
            packet.is_keyframe = key
            if not self._mux(packet):
                return
            self.count += 1
            if ts - self.checked > 5:            # 5초마다 디스크 여유 확인
                self.checked = ts
                if shutil.disk_usage(RECORD_DIR).free < RECORD_MIN_FREE:
                    print("디스크 여유 공간이 3GB 아래로 내려가 녹화를 멈춥니다.")
                    self.stop()

    def audio(self, data, ts):
        """소리 조각 (4바이트 경계로 맞춰진 PCM, 소리 스레드가 부른다)."""
        with self.lock:
            if self.out is None or not data:
                return
            samples = len(data) // 4
            expected = int((ts - self.t0) * 48000) - samples
            if expected - self.a_pts > 4800:     # 소리가 0.1초 넘게 비었으면 그만큼 건너뛴다
                self.a_pts = expected
            packet = av.Packet(data)
            packet.stream = self.astream
            packet.time_base = Fraction(1, 48000)
            packet.pts = packet.dts = self.a_pts
            self.a_pts += samples
            self._mux(packet)


RECORDER = Recorder()


class Receiver(threading.Thread):
    """수신·디코딩 스레드. 디코딩만 하고 색 변환·축소는 화면에 그릴 프레임에만 한다.
    (모든 프레임을 변환하면 처리가 밀려 소켓에 데이터가 쌓이고 지연이 계속 늘어난다)"""

    def __init__(self, sock):
        super().__init__(daemon=True)
        self.sock = sock
        self.latest = None   # 가장 최근에 디코딩된 av.VideoFrame
        self.seq = 0
        self.alive = True
        self.mbps = 0.0      # 최근 1초 수신량
        self.fps = 0.0
        self.hot = 0         # 디코딩 과부하가 연속된 초 수
        self.errors = 0      # 디코딩 실패한 패킷 수

    def run(self):
        codec = av.CodecContext.create("h264", "r")
        codec.options = {"flags": "low_delay"}
        mark, busy, nbytes, nframes = time.perf_counter(), 0.0, 0, 0
        stamp = None   # 다음에 나올 패킷의 시각: 그 패킷의 첫 바이트가 도착한 때
        try:
            while self.alive:
                data = self.sock.recv(1 << 18)
                if not data:
                    break
                t = time.perf_counter()
                if stamp is None:
                    stamp = t
                # 파서가 바이트 흐름에서 프레임 경계를 찾아 준다 (폰 인코더가 어떻게 쪼개 보내든 안전)
                for packet in codec.parse(data):
                    r = time.perf_counter()
                    RECORDER.video(bytes(packet), stamp)
                    stamp = t
                    busy -= time.perf_counter() - r   # 녹화에 쓴 시간은 디코딩 부하로 치지 않는다
                    try:
                        frames = codec.decode(packet)
                    except av.error.InvalidDataError:
                        self.errors += 1
                        continue
                    for frame in frames:
                        self.latest = frame
                        self.seq += 1
                    nframes += len(frames)
                now = time.perf_counter()
                busy += now - t
                nbytes += len(data)
                if now - mark >= 1.0:   # 1초마다 통계 갱신
                    span = now - mark
                    self.mbps, self.fps = nbytes * 8 / span / 1e6, nframes / span
                    self.hot = self.hot + 1 if busy / span > OVERLOAD else 0
                    mark, busy, nbytes, nframes = now, 0.0, 0, 0
        except OSError:
            pass
        self.alive = False


def load_optional(module, purpose):
    """선택 기능용 라이브러리를 불러온다. 없으면 pip 로 설치를 시도하고, 그래도 안 되면 None."""
    for attempt in range(2):
        try:
            return importlib.import_module(module)
        except ImportError:
            if attempt == 0:
                print(f"{purpose}용 {module} 설치 중...")
                r = subprocess.run([sys.executable, "-m", "pip", "install", module],
                                   capture_output=True, text=True, errors="replace")
                if r.returncode != 0:
                    print(r.stderr.strip()[-500:])
                importlib.invalidate_caches()   # 방금 설치한 패키지를 실행 중인 파이썬이 찾게 한다
        except Exception as e:   # 설치는 됐지만 필요한 시스템 라이브러리를 못 여는 경우
            print(f"{purpose} 기능을 쓸 수 없습니다: {e}")
            return None
    print(f"{module} 설치 실패. {purpose} 없이 진행합니다. (pip install {module})")
    return None


def load_sounddevice():
    return load_optional("sounddevice", "소리 출력")


def rotate_i420(arr, w, h, rot):
    """I420 영상(Y, U, V 평면이 차례로 이어진 배열)을 시계 방향으로 rot x 90도 돌린다."""
    flat = arr.reshape(-1)
    y = flat[:w * h].reshape(h, w)
    u = flat[w * h:w * h * 5 // 4].reshape(h // 2, w // 2)
    v = flat[w * h * 5 // 4:].reshape(h // 2, w // 2)
    return np.concatenate([np.rot90(plane, -rot).ravel() for plane in (y, u, v)])


class VirtualCam(threading.Thread):
    """폰 영상을 PC 의 가상 웹캠 장치로 내보낸다 (pyvirtualcam). OBS·줌·디스코드가 웹캠으로 인식한다.
    디코딩된 프레임(YUV)을 색 변환 없이 그대로 넘겨서 부담이 작다. 화면이 멈춰 있어도 같은 프레임을
    계속 보내 받는 쪽 영상이 끊기지 않게 한다. 해상도가 바뀌면(회전 등) 장치를 새 크기로 다시 연다."""

    def __init__(self):
        super().__init__(daemon=True)
        self.on = False
        self.rx = None      # 지금 영상을 받고 있는 Receiver (세션마다 바뀐다)
        self.state = {"mode": "screen", "rotate": 0}
        self.lib = None
        self.cam = None
        self.size = None

    def toggle(self):
        self.on = not self.on
        self._remember()
        print("가상 카메라 켜는 중..." if self.on else "가상 카메라 꺼짐")

    def status(self):
        if self.cam is not None:
            return "[VCAM]"
        return "[VCAM ...]" if self.on else ""

    def _remember(self):
        data = load_settings()
        data["vcam"] = self.on
        save_settings(data)

    def _close(self):
        cam, self.cam = self.cam, None
        if cam is not None:
            try:
                cam.close()
            except Exception:
                pass

    def run(self):
        sent, arr, size = None, None, None
        while True:
            if not self.on:
                self._close()
                time.sleep(0.1)
                continue
            try:
                if self.lib is None:
                    self.lib = load_optional("pyvirtualcam", "가상 카메라")
                    if self.lib is None:
                        self.on = False
                        self._remember()
                        continue
                rx = self.rx
                seq, frame = (rx.seq, rx.latest) if rx is not None else (0, None)
                rot = self.state["rotate"] if self.state["mode"] != "screen" else 0
                if frame is not None and (id(rx), seq, rot) != sent:
                    sent = (id(rx), seq, rot)
                    w, h = frame.width & ~1, frame.height & ~1   # YUV 4:2:0 은 짝수 크기만 된다
                    if (w, h) != (frame.width, frame.height):
                        frame = frame.reformat(w, h, "yuv420p")
                    arr = frame.to_ndarray(format="yuv420p")
                    if rot:
                        arr = rotate_i420(arr, w, h, rot)
                    size = (h, w) if rot % 2 else (w, h)
                if arr is None:
                    time.sleep(0.05)
                    continue
                if self.cam is None or self.size != size:
                    self._close()
                    self.cam = self.lib.Camera(size[0], size[1], VCAM_FPS, fmt=self.lib.PixelFormat.I420,
                                               backend=VCAM_BACKEND)
                    self.size = size
                    print(f"가상 카메라 켜짐: {self.cam.device} ({size[0]}x{size[1]}, {VCAM_FPS}fps)")
                self.cam.send(arr)
                self.cam.sleep_until_next_frame()
            except Exception as e:
                print(f"가상 카메라를 쓸 수 없습니다: {e}")
                print("OBS Studio 가 설치돼 있는지, OBS 자체의 '가상 카메라 시작'이 꺼져 있는지 확인하세요.")
                self.on = False
                self._remember()
                self._close()


VCAM = VirtualCam()


class Audio(threading.Thread):
    """폰 소리 수신·재생. 받은 PCM 을 버퍼에 쌓고, 사운드카드가 요청할 때마다 꺼내 준다.
    버퍼가 길어지면 그만큼 소리가 화면보다 늦으므로 오래된 부분을 버려 지연을 일정하게 유지한다."""

    def __init__(self, sock, sd):
        super().__init__(daemon=True)
        self.sock = sock
        self.sd = sd
        self.buf = bytearray()   # 항상 4바이트(스테레오 한 샘플) 경계에서 시작한다
        self.lock = threading.Lock()
        self.primed = False      # 재생 중인지 (버퍼가 비면 다시 모일 때까지 잠깐 무음)
        self.muted = False
        self.stream = None
        self.total = 0           # 폰에서 받은 누적 바이트
        self.need = 0            # 사운드카드가 한 번에 요청하는 바이트 (장치마다 다르다)
        self.calls = 0           # 사운드카드 콜백이 불린 횟수
        self.level = 0           # 최근 1초 동안 받은 소리의 최대 크기 (0~100)
        self.peak = 0            # 이번 1초 구간의 최대 샘플값
        self.silent = 0          # 데이터는 오는데 무음인 상태가 이어진 초 수
        self.heard = False       # 무음이 아닌 소리를 받은 적이 있는지
        self.ticks = 0           # state() 가 불린 횟수 (1초에 한 번)
        self.seen = 0            # state() 가 마지막으로 본 누적 바이트
        self.error = None        # 재생 중 난 오류

    def state(self):
        """창 제목에 붙일 소리 상태. 1초에 한 번 불린다."""
        got, self.seen = self.total - self.seen, self.total
        peak, self.peak = self.peak, 0
        self.level = peak * 100 // 32768
        self.silent = self.silent + 1 if got > AUDIO_RATE // 2 and peak <= 30 else 0
        self.ticks += 1
        if self.ticks % 3 == 0 and self.ticks <= 30:   # 처음 30초 동안 3초마다 콘솔에도 상태를 남긴다
            print(f"소리 상태: 폰->PC {got * 8 // 1000}kbps (정상은 약 1536) / 소리 크기 {self.level}/100 / "
                  f"버퍼 {len(self.buf) * 1000 // AUDIO_RATE}ms / 재생 콜백 {self.calls}회")
        if self.sd is None:
            return "snd:no-lib"        # sounddevice 없음
        if self.stream is None:
            return "snd:no-device"     # PC 출력 장치를 못 열었음
        if self.error:
            return "snd:error"
        if self.muted:
            return "snd:muted"
        if got == 0:
            return "snd:no-data"       # 폰에서 소리 데이터가 오지 않음
        return f"snd:{len(self.buf) * 1000 // AUDIO_RATE}ms lv{self.level}"   # 정상: 버퍼 길이, 소리 크기

    def fill(self, outdata, frames, _time, _status):
        """사운드카드 콜백: frames 개 샘플을 채운다."""
        try:
            self._fill(outdata, frames)
        except Exception as e:
            if not self.error:
                self.error = e
                print(f"소리 재생 오류: {e!r}")

    def _fill(self, outdata, frames):
        need = frames * 4
        self.need = need
        if self.calls == 0:
            print(f"PC 재생 시작 (장치가 한 번에 {frames * 1000 // 48000}ms 씩 요청)")
        self.calls += 1
        chunk = b""
        with self.lock:
            have = len(self.buf)
            if not self.primed:
                self.primed = have >= max(AUDIO_START, need)
            if self.primed and have >= need:
                chunk = bytes(self.buf[:need])
                del self.buf[:need]
            else:
                self.primed = False
        if self.muted or not chunk:
            chunk = bytes(need)
        elif VOLUME != 1.0:   # 키운 값이 16비트 범위를 넘으면 잘라서 찢어지는 잡음(값 넘침)을 막는다
            pcm = np.frombuffer(chunk, "<i2").astype(np.float32) * VOLUME
            chunk = np.clip(pcm, -32768, 32767).astype("<i2").tobytes()
        outdata[:] = chunk

    def run(self):
        if self.sd is not None:
            try:
                self.stream = self.sd.RawOutputStream(samplerate=48000, channels=2, dtype="int16",
                                                      latency="low", callback=self.fill)
                self.stream.start()
                try:
                    print(f"PC 소리 출력 장치: {self.sd.query_devices(kind='output')['name']}")
                except Exception:
                    pass
            except Exception as e:
                print(f"PC 소리 장치를 열 수 없습니다: {e}")
                self.stream = None
        carry = bytearray()
        try:
            while True:
                data = self.sock.recv(1 << 14)
                if not data:
                    break
                if self.total == 0:
                    print("폰 소리 수신 시작")
                self.total += len(data)
                carry += data                # 녹화에는 4바이트(스테레오 한 샘플) 단위로 끊어 넘긴다
                whole = len(carry) // 4 * 4
                RECORDER.audio(bytes(carry[:whole]), time.perf_counter())
                del carry[:whole]
                if self.stream is None:
                    continue             # 재생은 못 해도 계속 읽어서 버린다
                with self.lock:
                    self.buf += data
                    # 한도는 장치가 한 번에 요청하는 양보다 넉넉해야 한다 (작으면 영영 재생을 못 한다)
                    if len(self.buf) > max(AUDIO_LIMIT, self.need * 3):
                        del self.buf[:(len(self.buf) - max(AUDIO_KEEP, self.need * 2)) // 4 * 4]
                    end = len(self.buf) // 4 * 4
                    tail = bytes(self.buf[max(0, end - 1024):end])
                if tail:   # 받은 소리의 크기: 폰이 무음을 보내는지, 실제 소리를 보내는지 구분용
                    peak = int(np.abs(np.frombuffer(tail, "<i2").astype(np.int32)).max())
                    self.peak = max(self.peak, peak)
                    if peak > 300 and not self.heard:
                        self.heard = True
                        print("폰에서 실제 소리가 들어오고 있습니다.")
        except OSError:
            pass
        if self.total == 0:
            print("폰이 소리 데이터를 하나도 보내지 않았습니다. 위의 [server] 줄에 이유가 있습니다.")

    def close(self):
        try:
            self.sock.close()
            if self.stream is not None:
                self.stream.stop()
                self.stream.close()
        except Exception:
            pass


def layout(rw, rh, fw, fh):
    """창 안쪽 크기(rw x rh)에 프레임(fw x fh)을 비율 유지로 넣을 위치와 크기 (ox, oy, w, h)."""
    scale = min(rw / fw, rh / fh)
    w, h = max(2, min(rw, round(fw * scale))), max(2, min(rh, round(fh * scale)))
    return (rw - w) // 2, (rh - h) // 2, w, h


SOUND = None   # main() 에서 불러온 sounddevice 모듈 (없으면 None)
_tk = None


def pc_clipboard():
    """PC 클립보드의 글자. 없거나 읽기 실패면 빈 문자열."""
    global _tk
    try:
        import tkinter
        if _tk is None:
            _tk = tkinter.Tk()
            _tk.withdraw()
        return _tk.clipboard_get()
    except Exception:
        return ""


META_SHIFT = 0x41    # 안드로이드 메타 상태: Shift 눌림
META_CTRL = 0x3000   # Ctrl 눌림
# 창이 주는 일반 문자 코드 -> 안드로이드 키코드 (ESC=뒤로, Enter, Backspace, Tab, Space)
CHAR_KEYS = {27: 4, 13: 66, 8: 67, 9: 61, 32: 62}
# 윈도우 가상 키코드 -> 안드로이드 키코드 (방향키, Delete, Home, End, PgUp, PgDn, F1=홈, F2=최근 앱)
SPECIAL_KEYS = {0x25: 21, 0x26: 19, 0x27: 22, 0x28: 20, 0x2E: 112,
                0x24: 122, 0x23: 123, 0x21: 92, 0x22: 93, 0x70: 3, 0x71: 187}


class Control:
    """마우스·키보드 입력을 scrcpy 조작 메시지(빅엔디언 바이너리)로 바꿔 폰에 보낸다."""

    def __init__(self, sock):
        self.sock = sock
        self.view = None    # (ox, oy, w, h, fw, fh): 창 안에서 화면이 그려진 영역과 원본 프레임 크기
        self.down = False   # 왼쪽 버튼으로 터치 중인지
        self.audio = None   # F4 로 켜고 끌 Audio
        self.switch_audio = False   # F6 을 눌렀는지
        self.switch_mode = False    # F5 를 눌렀는지
        self.state = {"mode": "screen", "rotate": 0}   # main() 이 넘겨 주는 모드·회전 상태
        self.game = False   # 게임 모드
        self.bind = False   # 키 배치 중
        self.pending = None # 배치 중 방금 누른 키 (클릭을 기다리는 중)
        self.prev = {}      # 배치 중 키들의 직전 눌림 상태
        self.keymap = load_keymap()
        self.held = set()   # 게임 모드에서 지금 터치로 눌려 있는 키
        self.joy = (0, 0)   # 조이스틱 방향
        self.joy_pos = (0, 0)
        self.overlay = 0    # 배치 표시가 바뀔 때마다 증가 (화면을 다시 그리게 한다)
        if sock is not None:   # 카메라 모드에서는 조작 소켓이 없다: 폰으로 가는 입력은 전부 무시된다
            threading.Thread(target=self.drain, daemon=True).start()

    def drain(self):
        """폰이 보내는 알림 메시지는 쓰지 않으므로 읽어서 버린다 (안 읽으면 버퍼가 찬다)."""
        try:
            while self.sock.recv(4096):
                pass
        except OSError:
            pass

    def send(self, data):
        if self.sock is None:
            return
        try:
            self.sock.sendall(data)
        except OSError:
            pass

    def to_frame(self, x, y):
        ox, oy, w, h, fw, fh = self.view
        fx = min(max(int((x - ox) * fw / w), 0), fw - 1)
        fy = min(max(int((y - oy) * fh / h), 0), fh - 1)
        return fx, fy, fw, fh

    def inside(self, x, y):
        ox, oy, w, h, _, _ = self.view
        return ox <= x < ox + w and oy <= y < oy + h

    def touch_id(self, action, pid, fx, fy):
        """프레임 좌표에 터치. action: 0 누름, 1 뗌, 2 이동. pid 가 다르면 서로 다른 손가락(멀티터치)."""
        fw, fh = self.view[4], self.view[5]
        fx, fy = min(max(int(fx), 0), fw - 1), min(max(int(fy), 0), fh - 1)
        pressure, buttons = (0, 0) if action == 1 else (0xFFFF, 1)
        self.send(struct.pack(">BBqiiHHHII", 2, action, pid, fx, fy, fw, fh,
                              pressure, 0 if action == 2 else 1, buttons))

    def touch(self, action, x, y):
        """창 좌표에 마우스 터치 (손가락 번호 -2)."""
        fx, fy, _, _ = self.to_frame(x, y)
        self.touch_id(action, -2, fx, fy)

    # ---- 게임 모드 ----

    def status(self):
        """창 제목에 붙일 상태 표시."""
        if self.bind:
            if self.pending:
                return f"[BIND] click the button for {key_label(self.pending)}"
            return "[BIND] press a key  (F9 = done)"
        return "[GAME] F8 = off, F9 = bind keys" if self.game else ""

    def toggle_game(self):
        if self.sock is None:
            print("카메라 모드에서는 게임 모드를 쓸 수 없습니다.")
            return
        if os.name != "nt":
            print("게임 모드는 윈도우에서만 됩니다.")
            return
        self.game = not self.game
        if not self.game and self.bind:
            self.toggle_bind()
        self.release_all()
        print("게임 모드 켜짐: WASD = 조이스틱, F9 = 키 배치" if self.game else "게임 모드 꺼짐")

    def toggle_bind(self):
        if self.sock is None:
            return
        if os.name != "nt":
            print("게임 모드는 윈도우에서만 됩니다.")
            return
        self.bind = not self.bind
        self.pending, self.prev = None, {}
        self.overlay += 1
        self.release_all()
        if self.bind:
            self.game = True
            print("키 배치: 키를 누르고 -> 화면의 버튼을 클릭. 우클릭 = 삭제. F9 = 저장하고 끝")
        else:
            save_keymap(self.keymap)
            print(f"키 배치 저장됨: {len(self.keymap['keys'])}개 + 조이스틱")

    def release_all(self):
        """눌려 있는 조이스틱과 키 터치를 모두 뗀다."""
        if self.view is None:
            return
        fw, fh = self.view[4], self.view[5]
        if self.joy != (0, 0):
            self.touch_id(1, JOY_ID, *self.joy_pos)
            self.joy = (0, 0)
        for vk in self.held:
            nx, ny = self.keymap["keys"].get(str(vk), (0, 0))
            self.touch_id(1, vk, nx * fw, ny * fh)
        self.held.clear()

    def joystick(self, dx, dy):
        if (dx, dy) == self.joy:
            return
        fw, fh = self.view[4], self.view[5]
        cx, cy = self.keymap["joystick"][0] * fw, self.keymap["joystick"][1] * fh
        if (dx, dy) == (0, 0):
            self.touch_id(1, JOY_ID, *self.joy_pos)
        else:
            reach = JOY_RADIUS * min(fw, fh) / math.hypot(dx, dy)
            tx, ty = cx + dx * reach, cy + dy * reach
            if self.joy == (0, 0):   # 중심을 누른 뒤 밀어야 게임이 조이스틱 드래그로 인식한다
                self.touch_id(0, JOY_ID, cx, cy)
                self.touch_id(2, JOY_ID, (cx + tx) / 2, (cy + ty) / 2)
            self.touch_id(2, JOY_ID, tx, ty)
            self.joy_pos = (tx, ty)
        self.joy = (dx, dy)

    def poll(self):
        """매 프레임 호출. 게임 모드에서 키보드 상태를 읽어 조이스틱·키 터치로 바꾼다."""
        if not self.game or self.view is None:
            return
        if not window_focused():
            self.release_all()
            return
        if self.bind:
            for vk in BINDABLE:
                now = key_down(vk)
                if now and not self.prev.get(vk):
                    self.pending = vk
                    self.overlay += 1
                self.prev[vk] = now
            return
        self.joystick(key_down(VK_D) - key_down(VK_A), key_down(VK_S) - key_down(VK_W))
        fw, fh = self.view[4], self.view[5]
        for key, (nx, ny) in self.keymap["keys"].items():
            vk = int(key)
            now = key_down(vk)
            if now != (vk in self.held):
                self.touch_id(0 if now else 1, vk, nx * fw, ny * fh)
                (self.held.add if now else self.held.discard)(vk)

    def bind_click(self, x, y):
        """배치 중 클릭: 방금 누른 키를 이 위치에 묶는다."""
        fx, fy, fw, fh = self.to_frame(x, y)
        pos = [round(fx / fw, 4), round(fy / fh, 4)]
        if self.pending in (VK_W, VK_A, VK_S, VK_D):
            self.keymap["joystick"] = pos
            print(f"조이스틱 중심 -> {pos}")
        else:
            self.keymap["keys"][str(self.pending)] = pos
            print(f"{key_label(self.pending)} -> {pos}")
        self.pending = None
        self.overlay += 1

    def draw_overlay(self, img):
        """배치 중에 조이스틱과 키 위치를 화면 위에 표시한다."""
        ox, oy, w, h = self.view[:4]

        def mark(pos, label, radius, color):
            pt = (int(ox + pos[0] * w), int(oy + pos[1] * h))
            cv2.circle(img, pt, radius, (0, 0, 0), 5, cv2.LINE_AA)
            cv2.circle(img, pt, radius, color, 2, cv2.LINE_AA)
            org = (pt[0] - 7 * len(label), pt[1] + 7)
            cv2.putText(img, label, org, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 5, cv2.LINE_AA)
            cv2.putText(img, label, org, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

        mark(self.keymap["joystick"], "WASD", int(JOY_RADIUS * min(w, h)), (255, 200, 0))
        for key, pos in self.keymap["keys"].items():
            mark(pos, key_label(int(key)), 26, (0, 255, 255))

    def scroll(self, x, y, hnotch, vnotch):
        fx, fy, fw, fh = self.to_frame(x, y)

        def fixed(n):  # 휠 한 칸 = 1/16, 범위 [-1, 1] 을 16비트 고정소수로
            return max(-0x8000, min(0x7FFF, int(max(-1.0, min(1.0, n / 16)) * 0x8000)))

        self.send(struct.pack(">BiiHHhhI", 3, fx, fy, fw, fh, fixed(hnotch), fixed(vnotch), 0))

    def key(self, keycode, meta=0):
        """키를 눌렀다 뗀다."""
        for action in (0, 1):
            self.send(struct.pack(">BBIII", 0, action, keycode, 0, meta))

    def text(self, chars):
        data = chars.encode("utf-8")
        self.send(struct.pack(">BI", 1, len(data)) + data)

    def paste(self):
        """PC 클립보드 글자를 폰 클립보드에 넣고 바로 붙여넣는다. 비어 있으면 폰 쪽 Ctrl+V."""
        chars = pc_clipboard()[:50_000]
        if chars:
            data = chars.encode("utf-8")
            self.send(struct.pack(">BQBI", 9, 0, 1, len(data)) + data)
        else:
            self.key(50, META_CTRL)

    def on_mouse(self, event, x, y, flags, _param):
        if self.view is None or self.sock is None:
            return
        if self.bind:
            if self.pending and event == cv2.EVENT_LBUTTONDOWN and self.inside(x, y):
                self.bind_click(x, y)
            elif self.pending and event == cv2.EVENT_RBUTTONDOWN:
                if self.keymap["keys"].pop(str(self.pending), None):
                    print(f"{key_label(self.pending)} 배치 삭제")
                self.pending = None
                self.overlay += 1
            return
        if event in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_LBUTTONDBLCLK):
            if self.inside(x, y):
                self.down = True
                self.touch(0, x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self.down:
            if flags & cv2.EVENT_FLAG_LBUTTON:
                self.touch(2, x, y)
            else:               # 창 밖에서 버튼을 뗀 경우: 터치가 눌린 채 남지 않게 뗀다
                self.down = False
                self.touch(1, x, y)
        elif event == cv2.EVENT_LBUTTONUP and self.down:
            self.down = False
            self.touch(1, x, y)
        elif self.game:
            pass                                  # 게임 중에는 우클릭·가운데 클릭·휠을 막는다
        elif event in (cv2.EVENT_RBUTTONDOWN, cv2.EVENT_RBUTTONDBLCLK):
            self.send(struct.pack(">BB", 4, 0))   # 뒤로 (화면이 꺼져 있으면 켠다)
        elif event == cv2.EVENT_RBUTTONUP:
            self.send(struct.pack(">BB", 4, 1))
        elif event in (cv2.EVENT_MBUTTONDOWN, cv2.EVENT_MBUTTONDBLCLK):
            self.key(3)                           # 홈
        elif event == cv2.EVENT_MOUSEWHEEL:
            self.scroll(x, y, 0, cv2.getMouseWheelDelta(flags) / 120)
        elif event == cv2.EVENT_MOUSEHWHEEL:
            self.scroll(x, y, cv2.getMouseWheelDelta(flags) / 120, 0)

    def on_key(self, k):
        """cv2.waitKeyEx 값 처리. 글자·숫자는 키 입력으로 보내 폰 키보드가 한글 조합까지 처리하게 한다."""
        if k < 0:
            return
        if k == 0x730000:                         # F4: 소리 켬/끔
            if self.audio is not None:
                self.audio.muted = not self.audio.muted
                print("소리 꺼짐" if self.audio.muted else "소리 켜짐")
        elif k == 0x760000:                       # F7: 녹화 시작/정지
            RECORDER.toggle()
        elif k == 0x7B0000:                       # F12: 가상 카메라 켬/끔
            VCAM.toggle()
        elif k == 0x750000:                       # F6: 소리 캡처 방식 전환 (다시 연결된다)
            self.switch_audio = self.sock is not None
        elif k == 0x740000:                       # F5: 폰 화면 / 후면 카메라 / 전면 카메라 전환
            self.switch_mode = True
        elif k == 0x7A0000:                       # F11: 카메라 영상을 보이는 방향만 90도 회전
            if self.sock is None:
                self.state["rotate"] = (self.state["rotate"] + 1) % 4
                self.overlay += 1
        elif k == 0x770000:                       # F8: 게임 모드
            self.toggle_game()
        elif k == 0x780000:                       # F9: 키 배치
            self.toggle_bind()
        elif self.game or self.sock is None:
            pass                                  # 게임 중·카메라 모드에서는 글자 입력을 하지 않는다
        elif k > 0xFFFF and not k & 0xFFFF:       # 특수키: 가상 키코드가 상위 16비트에 온다
            vk = k >> 16
            if vk == 0x72:                        # F3: 한/영 전환 (Shift+Space)
                self.key(62, META_SHIFT)
            elif vk in SPECIAL_KEYS:
                self.key(SPECIAL_KEYS[vk])
        elif k in CHAR_KEYS:
            self.key(CHAR_KEYS[k])
        elif k == 22:                             # Ctrl+V
            self.paste()
        elif 1 <= k <= 26:                        # Ctrl+A ~ Ctrl+Z
            self.key(28 + k, META_CTRL)
        elif ord("a") <= k <= ord("z"):
            self.key(29 + k - ord("a"))
        elif ord("A") <= k <= ord("Z"):
            self.key(29 + k - ord("A"), META_SHIFT)
        elif ord("0") <= k <= ord("9"):
            self.key(7 + k - ord("0"))
        elif 33 <= k <= 126:                      # 나머지 기호는 글자로 직접 입력
            self.text(chr(k))


def run_session(serial, server_path, version, cfg, state):
    """한 번의 미러링. 'quit'(사용자 종료) / 'overload'(PC 가 못 따라감) / 'lost'(연결 끊김) /
    'audio'·'audio-auto'(소리 캡처 방식 전환) / 'mode'(화면·카메라 전환) 을 돌려준다."""
    camera = cfg["mode"] != "screen"
    proc = start_server(serial, server_path, version, cfg)
    try:
        sock, asock, csock = connect(proc, control=not camera)
    except RuntimeError as e:
        proc.terminate()
        adb("forward", "--remove", f"tcp:{PORT}", serial=serial)
        if not camera:
            die(str(e))
        print(f"카메라를 시작하지 못했습니다: {e}")   # 화면 모드로 돌아가도록 끊김으로 처리
        return "lost"
    rx = Receiver(sock)
    rx.start()
    audio = Audio(asock, SOUND)
    audio.muted = camera   # 카메라 모드의 소리는 폰 마이크: 스피커로 내면 하울링이 나므로 기본은 끔
    audio.start()
    ctl = Control(csock)
    ctl.audio = audio
    ctl.state = state
    VCAM.rx, VCAM.state = rx, state
    cv2.setMouseCallback(WINDOW, ctl.on_mouse)

    label = f"[CAM {cfg['mode']}]  " if camera else ""
    drawn = None   # 마지막으로 그린 (프레임 번호, 창 너비, 창 높이, 배치 표시 번호)
    shape = None
    titled = 0.0
    status = None
    result = "lost"
    try:
        while rx.alive:
            if rx.hot >= 3 and not camera:
                result = "overload"
                break
            if ctl.switch_mode:
                result = "mode"
                break
            if ctl.switch_audio:
                result = "audio"
                break
            if cfg["auto_audio"] and not camera and audio.silent >= 6:
                result = "audio-auto"
                break
            seq, frame = rx.seq, rx.latest
            if frame is not None:
                rot = state["rotate"] if camera else 0   # 카메라 영상을 시계 방향으로 rot x 90도 돌려 보여 준다
                fw, fh = (frame.height, frame.width) if rot % 2 else (frame.width, frame.height)
                if (fw, fh) != shape:  # 첫 프레임 또는 화면 회전
                    shape = (fw, fh)
                    fit = min(WINDOW_WIDTH / fw, WINDOW_HEIGHT / fh)   # 세로·가로 화면 모두 모니터 안에 들어가게
                    cv2.resizeWindow(WINDOW, int(fw * fit), int(fh * fit))
                    ctl.release_all()
                # 창 안쪽 크기와 똑같은 그림을 직접 만들어 그린다. 창이 그림을 늘리거나 줄이면 화질이 나빠지고,
                # 이렇게 해야 마우스 좌표가 그림 좌표와 1:1 로 맞는다.
                _, _, rw, rh = cv2.getWindowImageRect(WINDOW)
                if rw <= 0 or rh <= 0:
                    rw, rh = int(fw * WINDOW_HEIGHT / fh), WINDOW_HEIGHT
                if (seq, rw, rh, ctl.overlay) != drawn:
                    drawn = (seq, rw, rh, ctl.overlay)
                    ox, oy, w, h = layout(rw, rh, fw, fh)
                    ctl.view = (ox, oy, w, h, fw, fh)
                    interp = "LANCZOS" if w < fw else "BICUBIC"  # 글씨 윤곽이 가장 또렷한 축소 방식
                    if rot % 2:
                        img = frame.reformat(h, w, "bgr24", interpolation=interp).to_ndarray()
                    else:
                        img = frame.reformat(w, h, "bgr24", interpolation=interp).to_ndarray()
                    if rot:
                        img = np.ascontiguousarray(np.rot90(img, -rot))
                    if (w, h) != (rw, rh):   # 비율이 안 맞는 부분은 검은 여백
                        canvas = np.zeros((rh, rw, 3), np.uint8)
                        canvas[oy:oy + h, ox:ox + w] = img
                        img = canvas
                    if ctl.bind:
                        img = np.array(img)   # 쓰기 가능한 복사본에 표시를 그린다
                        ctl.draw_overlay(img)
                    cv2.imshow(WINDOW, img)
                if time.perf_counter() - titled > 1.0 or ctl.status() != status:  # 창 제목에 상태 표시
                    titled, status = time.perf_counter(), ctl.status()
                    cv2.setWindowTitle(WINDOW, f"{WINDOW}  {fw}x{fh}  {rx.fps:.0f}fps  {rx.mbps:.1f}Mbps  "
                                               f"err {rx.errors}  {audio.state()}  {RECORDER.status()}  {VCAM.status()}  {label}{status}")
            ctl.on_key(cv2.waitKeyEx(1))
            ctl.poll()
            if drawn and cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                result = "quit"
                break
    finally:
        VCAM.rx = None
        rx.alive = False
        sock.close()
        if csock is not None:
            csock.close()
        audio.close()
        proc.terminate()
        adb("forward", "--remove", f"tcp:{PORT}", serial=serial)
    return result


def setup_windows_display():
    """윈도우 화면 배율(125%, 150% 등) 대응. 이걸 안 하면 윈도우가 창 전체를 확대해 글씨가 뭉개진다."""
    global WINDOW_WIDTH, WINDOW_HEIGHT
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)   # 모니터별 실제 픽셀 기준으로 그린다
    except (AttributeError, OSError):
        ctypes.windll.user32.SetProcessDPIAware()
    ctypes.windll.winmm.timeBeginPeriod(1)  # 기본 타이머(약 15ms)를 1ms 로 낮춰 화면 갱신 지연을 줄인다
    WINDOW_WIDTH = int(ctypes.windll.user32.GetSystemMetrics(0) * 0.85)
    WINDOW_HEIGHT = int(ctypes.windll.user32.GetSystemMetrics(1) * 0.85)


def main():
    global ADB, SOUND
    if os.name == "nt":
        setup_windows_display()
    SOUND = load_sounddevice()
    ADB = ensure_adb()
    server_path, version = ensure_server()
    serial = find_device()

    w, h = screen_size(serial)
    link = measure_link(serial)
    cfg = make_settings(w, h, link)
    kind = "USB 3.x" if link > 480_000_000 else "USB 2.0"
    print(f"기기: {serial} / 화면 {w}x{h} / 케이블 실측 {link / 1e6:.0f}Mbps ({kind} 급)")

    print("조작: 클릭·드래그=터치 / 휠=스크롤 / 우클릭·ESC=뒤로 / 가운데 클릭·F1=홈 / F2=최근 앱 / "
          "Ctrl+V=PC 클립보드 붙여넣기 / F3=한영 전환 / F4=소리 켬·끔 / F6=소리 캡처 방식 전환 / F7=녹화 / 종료=창 닫기")
    print("게임: F8=게임 모드(WASD 조이스틱) / F9=키 배치(키 누르고 -> 화면 버튼 클릭)")
    print("카메라: F5=폰 화면 -> 후면 카메라 -> 전면 카메라 전환 / F11=카메라 영상 회전")
    print("OBS: F12=가상 카메라 켬·끔 -> OBS 에서 '비디오 캡처 장치' 소스로 OBS Virtual Camera 선택")

    settings = load_settings()
    source = settings.get("audio_source")
    if source not in AUDIO_SOURCES:
        source = AUDIO_SOURCES[0]
    auto_tried = False
    state = {"mode": START_MODE if START_MODE in MODES else "screen", "rotate": 0}

    VCAM.on = bool(settings.get("vcam"))
    VCAM.start()
    if VCAM.on:
        print("가상 카메라 켜는 중... (지난번에 켜 둔 상태, F12 로 끔)")

    if AUTO_RECORD:
        RECORDER.on = True
        print(f"자동 녹화 켜짐 (F7 로 끔). 저장 폴더: {RECORD_DIR}")

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    try:
        while True:
            cfg["audio_source"] = source
            cfg["auto_audio"] = source == "output" and not auto_tried
            cfg["mode"] = state["mode"]
            if state["mode"] == "screen":
                size = f"긴 변 {cfg['max_size']}px" if cfg["max_size"] else "원본 해상도"
                print(f"설정: {size} / {cfg['bitrate'] / 1e6:.0f}Mbps / {MAX_FPS}fps / 소리 캡처 {source}")
            else:
                name = "후면" if state["mode"] == "back" else "전면"
                print(f"카메라 모드({name}): 최대 {CAMERA_SIZE}px 16:9 / {CAMERA_BITRATE / 1e6:.0f}Mbps / "
                      f"{CAMERA_FPS}fps / 소리 = 폰 마이크 (PC 스피커 출력은 꺼짐, F4 로 켬)")
            result = run_session(serial, server_path, version, cfg, state)
            if result == "mode":
                state["mode"] = MODES[(MODES.index(state["mode"]) + 1) % len(MODES)]
                state["rotate"] = 0
                time.sleep(0.5)
                continue
            if result == "lost" and state["mode"] != "screen":
                print("카메라 연결이 끊겼습니다. 위 [server] 줄에 이유가 있습니다. 폰 화면 모드로 돌아갑니다.")
                state["mode"] = "screen"
                time.sleep(0.5)
                continue
            if result in ("audio", "audio-auto"):
                if result == "audio-auto":
                    auto_tried = True
                    print("폰이 6초 동안 무음만 보냈습니다. 소리 캡처 방식을 바꿔 다시 연결합니다.")
                source = AUDIO_SOURCES[(AUDIO_SOURCES.index(source) + 1) % len(AUDIO_SOURCES)]
                settings = load_settings()   # 다른 기능이 저장한 값을 덮어쓰지 않게 다시 읽는다
                settings["audio_source"] = source
                save_settings(settings)
                time.sleep(0.5)
                continue
            if result != "overload" or cfg["scale"] <= 0.4:
                break
            print("PC 디코딩이 밀려 해상도를 한 단계 낮춥니다.")
            cfg = make_settings(w, h, link, cfg["scale"] * 0.75)
            time.sleep(0.5)
    finally:
        cv2.destroyAllWindows()
        RECORDER.stop()
        RECORDER.wait()


if __name__ == "__main__":
    main()

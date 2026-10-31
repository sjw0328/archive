#!/usr/bin/env python3
# webcam_enhancer.py - OMEGA SYSTEMS
# pip install opencv-python numpy pyvirtualcam customtkinter pillow
# optional: pip install mediapipe   (background blur)
#
#   python webcam_enhancer.py            GUI
#   python webcam_enhancer.py --bench    headless benchmark (no webcam needed)
#   python webcam_enhancer.py --bench --camera 0
#   python webcam_enhancer.py --list     enumerate cameras

import argparse
import base64
import json
import math
import os
import queue
import threading
import urllib.request
import time

os.environ.setdefault("OPENCV_LOG_LEVEL", "FATAL")   # probing backends is noisy

import cv2
import numpy as np

try:
    import pyvirtualcam
    from pyvirtualcam import PixelFormat
except Exception:
    pyvirtualcam = None

try:
    import mediapipe as mp
except Exception:
    mp = None

try:
    import onnxruntime as ort
except Exception:
    ort = None

PRESET_PATH = os.path.join(os.path.expanduser("~"), ".webcam_enhancer.json")
# Bumped whenever defaults change meaning. A preset written by an older build
# carries that build's on/off flags, so a stage that is now on out of the box
# would come back off on every launch and look like the app ignoring its own
# defaults. Mismatched presets keep the device settings and drop the rest.
PRESET_V = 5

# Two limits, and they are different things.
#
# LATENCY_MS - end-to-end ceiling, measured per frame from the moment the grab
# thread gets it off the driver to the moment it is handed to the virtual camera.
# Everything the chain does has to fit under this whatever the frame rate.
# Sensor exposure and driver buffering happen before cap.read() returns and are
# not in it - nothing here can change them.
#
# STAGE_FRAC - the share of one frame interval a single pipeline stage may take.
# Output rate is set by the slowest stage, not by the sum, so this is what holds
# the frame rate: 28 ms at 30 fps, 14 ms at 60. The interval comes from the
# camera's measured rate, not from the number in the menu.
#
# The chain is cut into Pipeline.stages threads that run a frame apart, so the
# work one frame may carry is stages x stage budget, capped by LATENCY_MS. More
# stages buy frames, not latency: at 60 fps two stages carry 28 ms of filtering,
# four carry 56. Raising the stage budget on its own does not buy frames - the
# controller spends whatever it is given - which is why the two numbers are kept
# apart. Once the quality knobs bottom out the controller drops whole stages in
# Pipeline.SHED order.
LATENCY_MS = 67.0
STAGE_FRAC = 0.84
BUDGET_MS = 1000.0 / 30.0 * STAGE_FRAC      # per-stage budget at 30 fps (28 ms)
MAX_STAGES = 4
# Two ways to run, picked in the UI.
#   q30   - 30 fps, and every millisecond the latency budget leaves goes into
#           picture quality (the controller climbs until LATENCY_MS is used).
#   fps60 - 60 fps, offered only when the camera really delivers 60 at the
#           chosen size. No drop-to-a-smaller-mode-and-upscale fallback: that
#           trades the resolution away for the rate, which is the wrong way
#           round for this program.
MODES = {"q30": 30, "fps60": 60}
FPS60_MIN = 51.0        # measured rate that counts as "does 60"
RELATIVE_MAP = hasattr(cv2, "WARP_RELATIVE_MAP")   # OpenCV 4.10+
RESOLUTIONS = {"640x480": (640, 480), "1280x720": (1280, 720), "1920x1080": (1920, 1080)}

STR = {
    "en": {
        "title": "Webcam Enhancer", "camera": "Camera", "resolution": "Resolution",
        "fps": "FPS", "start": "Start", "stop": "Stop", "filters": "Filters",
        "save": "Save preset", "load": "Load preset", "idle": "Idle",
        "stopped": "Stopped", "preview_off": "Preview stopped",
        "original": "ORIGINAL", "enhanced": "ENHANCED", "budget": "budget",
        "denoise": "Denoise", "exposure": "Auto exposure (backlight)",
        "wb": "White balance", "sharpen": "Sharpen", "bg": "Background blur",
        "strength": "Strength", "target": "Target", "radius": "Radius",
        "face": "Face priority", "blur": "Blur", "quality": "Quality",
        "no_mediapipe": "requires: pip install mediapipe",
        "detecting": "Detecting cameras...", "no_camera": "No camera",
        "no_camera_hint": "No camera found - close apps using it, check privacy settings",
        "autoq": "Auto quality (use spare CPU)",
        "speckle": "Hot pixel / speckle guard", "threshold": "Threshold",
        "vibrance": "Vibrance", "contrast": "Contrast",
        "neural": "Neural tone (learned curves)", "max_gain": "Max gain",
        "local": "Local affine (learned grid)",
        "orient": "Mirror / level", "mirror": "Mirror", "rotate": "Rotate",
        "warmth": "Warmth / tint", "tint": "Tint",
        "touchup": "Face touch-up", "spotlight": "Subject spotlight",
        "framing": "Auto framing", "zoom": "Zoom", "follow": "Follow",
        "facelight": "Face light (local lift)", "size": "Kernel",
        "auto": "Scene adaptive", "scene": "scene", "latency": "latency",
        "localtone": "Local tone (backlight)", "detail": "Local contrast",
        "key": "Key",
        "sc_normal": "normal", "sc_low": "low light", "sc_night": "night",
        "sc_backlit": "backlit", "sc_bright": "overexposed",
        "gpu": "GPU (OpenCL, experimental)",
        "cam_settings": "Camera driver settings",
        "mode": "Mode", "m_q30": "30 fps max quality", "m_fps60": "60 fps",
        "no60": "this camera does not do 60 fps at {res} (measured {rate:.0f} fps) - "
                "switched to 30 fps max quality",
        "locked": "always 100%",
        "probing": "checking which camera modes run at full frame rate...",
        "mode_up": "capture {cw}x{ch} (the camera does {rw}x{rh} at only {rf:.0f} fps) "
                   "-> output {ow}x{oh}",
        "cam_settings_na": "this backend has no driver settings window (DSHOW only)",
        "diag_bw": "camera sends only {cap:.0f} fps: uncompressed {fmt} at {res} does "
                   "not fit USB bandwidth, it refused MJPG, and no smaller mode ran faster.",
        "diag_dark": "camera itself sends only {cap:.0f} fps (format {fmt}) - almost always "
                     "auto-exposure stretching in low light. More light, or turn off "
                     "'low light compensation' in Camera driver settings.",
        "no_onnx": "requires: pip install onnxruntime",
        "restore": "Neural restore (GPU only)",
        "motion_deblur": "Motion deblur",
        "no_restore_model": "drop an ONNX model at {path}",
        "no_gpu_ep": "GPU only - pip install onnxruntime-directml "
                     "(any DX12 GPU) or onnxruntime-gpu (NVIDIA + CUDA)",
        "vcam_missing": "preview only - pip install pyvirtualcam",
        "vcam_none": "preview only - no virtual camera driver. Install OBS "
                     "Studio once for its virtual camera; OBS need not run.",
        "no_model": "no model at {path} - train one with dce_train.py",
        "v_both": "Both", "v_after": "Enhanced", "v_before": "Original",
    },
    "ko": {
        "title": "웹캠 화질 보정", "camera": "카메라", "resolution": "해상도",
        "fps": "FPS", "start": "시작", "stop": "정지", "filters": "필터",
        "save": "프리셋 저장", "load": "불러오기", "idle": "대기 중",
        "stopped": "정지됨", "preview_off": "미리보기 정지",
        "original": "ORIGINAL", "enhanced": "ENHANCED", "budget": "예산",
        "denoise": "노이즈 제거", "exposure": "자동 노출 (역광)",
        "wb": "화이트밸런스", "sharpen": "샤프닝", "bg": "배경 흐림",
        "strength": "강도", "target": "목표 밝기", "radius": "반경",
        "face": "얼굴 우선", "blur": "흐림", "quality": "품질",
        "no_mediapipe": "필요: pip install mediapipe",
        "detecting": "카메라 검색 중...", "no_camera": "카메라 없음",
        "no_camera_hint": "카메라를 찾지 못함 - 점유 중인 앱 종료, 개인 정보 설정 확인",
        "autoq": "자동 품질 (남는 CPU 사용)",
        "speckle": "핫픽셀 / 흰 점 제거", "threshold": "임계값",
        "vibrance": "색 생동감", "contrast": "대비",
        "neural": "신경망 톤 (학습된 커브)", "max_gain": "최대 게인",
        "local": "로컬 어파인 (학습된 그리드)",
        "orient": "좌우반전 / 수평", "mirror": "좌우반전", "rotate": "회전",
        "warmth": "색온도 / 틴트", "tint": "틴트",
        "touchup": "얼굴 보정", "spotlight": "피사체 스포트라이트",
        "framing": "자동 프레이밍", "zoom": "확대", "follow": "추적",
        "facelight": "얼굴 조명 (국소 보정)", "size": "커널",
        "auto": "장면 자동 판별", "scene": "장면", "latency": "지연",
        "localtone": "로컬 톤 (역광·명암)", "detail": "로컬 대비",
        "key": "기준 밝기",
        "sc_normal": "일반", "sc_low": "저조도", "sc_night": "야간",
        "sc_backlit": "역광", "sc_bright": "과노출",
        "gpu": "GPU 사용 (OpenCL, 실험적)",
        "cam_settings": "카메라 드라이버 설정",
        "mode": "모드", "m_q30": "30fps 최고 화질", "m_fps60": "60fps",
        "no60": "이 카메라는 {res}에서 60fps를 내지 못합니다 (측정 {rate:.0f} fps) - "
                "30fps 최고 화질로 전환했습니다",
        "locked": "항상 100%",
        "probing": "카메라가 제 속도를 내는 모드 확인 중...",
        "mode_up": "캡처 {cw}x{ch} (이 카메라는 {rw}x{rh}에서 {rf:.0f} fps뿐) "
                   "-> 출력 {ow}x{oh}",
        "cam_settings_na": "이 백엔드는 드라이버 설정 창이 없습니다 (DSHOW 전용)",
        "diag_bw": "카메라가 {cap:.0f} fps만 보냄: {res} 비압축({fmt})은 USB 대역폭에 "
                   "안 들어가고, MJPG 전환도 거부됐으며, 더 작은 모드도 빠르지 않았습니다.",
        "diag_dark": "카메라 자체가 {cap:.0f} fps만 보냄 (포맷 {fmt}) - 거의 항상 어두운 "
                     "환경에서 자동노출이 셔터를 늘린 것입니다. 조명을 밝히거나 카메라 "
                     "드라이버 설정에서 '저조도 보정'을 끄세요.",
        "no_onnx": "필요: pip install onnxruntime",
        "restore": "신경망 복원 (GPU 전용)",
        "motion_deblur": "모션 디블러",
        "no_restore_model": "{path} 에 ONNX 모델을 넣으십시오",
        "no_gpu_ep": "GPU 전용 - pip install onnxruntime-directml "
                     "(DX12 GPU 전부) 또는 onnxruntime-gpu (NVIDIA + CUDA)",
        "vcam_missing": "미리보기 전용 - pip install pyvirtualcam",
        "vcam_none": "미리보기 전용 - 가상 카메라 드라이버 없음. OBS Studio를 한 번 "
                     "설치하면 드라이버가 함께 깔립니다 (OBS 실행은 불필요).",
        "no_model": "{path} 에 모델 없음 - dce_train.py로 학습",
        "v_both": "비교", "v_after": "보정", "v_before": "원본",
    },
}
_env = (os.environ.get("LANG", "") + os.environ.get("LANGUAGE", "")).lower()
LANG = "ko" if _env.startswith("ko") else "en"


def T(key):
    return STR[LANG].get(key, STR["en"].get(key, key))


# ---------------- gpu ----------------

GPU = {"on": False, "device": "", "ep": "CPU"}

# Measured luma noise sigma, in 8-bit levels, shared by every stage that has to
# know the difference between detail and grain. Estimated once per few frames
# from the temporal difference (see Denoise) rather than assumed: a webcam at
# 300 lux and the same webcam at 30 lux differ by a factor of four here, and a
# fixed threshold is wrong at both ends. Sharpen gates on it, the detail
# re-injection cores on it, and the temporal gate scales with it.
# "sigma" is what the sensor delivers; "residual" is what is left by the time
# the sharpener sees it. The two are not the same number and using the first
# where the second belongs is how a sharpener ends up permanently gated shut on
# a camera whose noise denoise has already dealt with.
NOISE = {"sigma": 4.0, "residual": 4.0}

# Shared 3x3 structuring element for halo control: any stage that sharpens or
# locally boosts contrast clamps its result to the local min/max of the
# unmodified luma first, so a strong edge can be crisped but a correction can
# never ring past what its own 3x3 neighbourhood actually contained. See
# Sharpen and LocalTone.
_HALO_K3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))


def gpu_init(enable=True):
    """OpenCV's T-API: wrap frames in cv2.UMat and every supported call runs on
    the GPU through OpenCL. Works with the stock pip wheel - no CUDA build
    needed - so it covers NVIDIA, AMD and Intel alike. Falls back to CPU
    transparently when no device is present."""
    GPU["on"] = False
    if not enable:
        cv2.ocl.setUseOpenCL(False)
        return False
    try:
        if not cv2.ocl.haveOpenCL():
            return False
        cv2.ocl.setUseOpenCL(True)
        if not cv2.ocl.useOpenCL():
            return False
        GPU["device"] = cv2.ocl.Device_getDefault().name()
        GPU["on"] = True
    except Exception:
        GPU["on"] = False
    return GPU["on"]


def to_gpu(frame):
    return cv2.UMat(frame) if GPU["on"] and not isinstance(frame, cv2.UMat) else frame


def down_interp(w, h, sw, sh):
    """INTER_AREA has a fast integer path and a slow general one - 0.26 ms
    halving a 720p frame, 6.7 ms taking it to 0.85. At non-integer ratios the
    quality difference does not pay for a 6 ms bill, so this picks AREA only
    where it is cheap."""
    return (cv2.INTER_AREA if w % sw == 0 and h % sh == 0 else cv2.INTER_LINEAR)


def up(small, w, h, code=None):
    """Upload the small map, then upsample on the device.

    Resizing on the CPU first and uploading the result sends a full-resolution
    image across the bus every frame - 2.7 MB at 720p per map, several maps per
    stage. Uploading the thumbnail instead is a ~20 KB transfer and the resize
    lands on the GPU, where it is free. On the CPU path this is identical work in
    a different order."""
    out = cv2.resize(to_gpu(small), (w, h), interpolation=cv2.INTER_LINEAR)
    return cv2.cvtColor(out, code) if code is not None else out


def npy(x):
    return x.get() if isinstance(x, cv2.UMat) else x


def onnx_providers():
    """Prefer CUDA, then DirectML, then CPU. onnxruntime-gpu / -directml supply
    the first two; the plain wheel only ever reports CPU."""
    have = ort.get_available_providers() if ort else []
    for name, tag in (("CUDAExecutionProvider", "CUDA"),
                      ("DmlExecutionProvider", "DirectML"),
                      ("ROCMExecutionProvider", "ROCm")):
        if name in have:
            GPU["ep"] = tag
            return [name, "CPUExecutionProvider"]
    GPU["ep"] = "CPU"
    return ["CPUExecutionProvider"]


# ---------------- filters ----------------

# Size of the frame currently in flight. UMat has no shape in the Python
# binding and the usual workaround, umat.get().shape, is a full device-to-host
# copy - several megabytes per call, and frame_size() is called by most stages.
# With OpenCL on that alone can cost more than the filters. The pipeline records
# the size once per frame instead; nothing in the chain changes it.
#
# Thread-local, and that is a fix, not a style choice. The chain runs in up to
# four stage threads a frame apart, and stage 0 records the size of the frame
# it just picked up. With one shared record, the moment the controller changed
# the processing scale every later stage read the *next* frame's size for the
# frame it was still holding: chroma planes came back at the wrong size, the
# face box landed on the wrong spot, and a stage thread could die on a merge.
# Each stage now sets the size of the frame it is actually working on.
_SHAPE_TLS = threading.local()


def set_shape(frame):
    set_hw(frame.shape[0], frame.shape[1])


def set_hw(h, w):
    _SHAPE_TLS.hw = (int(h), int(w))


def get_hw():
    return getattr(_SHAPE_TLS, "hw", (0, 0))


def frame_size(f):
    if isinstance(f, list):
        f = f[0]
    if isinstance(f, cv2.UMat):
        return get_hw()
    return f.shape[0], f.shape[1]


class BaseFilter:
    name = "base"
    label_key = "filters"
    params = {}          # key: (label_key, min, max, default)
    available = True
    uses_face = False
    locked = False      # always on at full strength; the UI does not offer it
    hint = ""
    # Colour space this filter wants its input in. Pipeline converts once at each
    # boundary and keeps the frame there for the whole run of stages that agree,
    # so a chain of four YCrCb stages costs one conversion pair instead of four.
    # "ycrcb" filters receive [y, cr, cb] as separate planes, not a merged
    # frame. cv2.split is 1.9 ms at 720p and cv2.merge 0.24 - four stages each
    # splitting and merging their own copy was ~8 ms of pure channel shuffling.
    # Carrying the planes means one split and one merge for the whole run, and
    # the stages that only touch luma - speckle, sharpen, face light, spotlight -
    # get to skip two thirds of their work for free.
    space = "bgr"

    def __init__(self):
        self.enabled = False
        self.auto_off = False      # dropped by the load shedder, not by the user
        self.values = {k: v[3] for k, v in self.params.items()}
        self.ms = 0.0
        self.error = ""

    @property
    def label(self):
        return T(self.label_key)

    def process(self, frame):
        return frame


class SpeckleGuard(BaseFilter):
    """Kills isolated bright or dark pixels - sensor hot pixels and shot-noise
    spikes. They are invisible on the raw feed but the sharpener treats a
    single blown pixel as maximum edge energy and blows it up into a white dot,
    so this has to run before it.

    A pixel is replaced only when it stands out from *all eight* of its
    neighbours by more than the threshold. The earlier version compared
    against a median, and a median also "corrects" things that are not
    defects: the rim of a pair of glasses, a strand of hair, a specular
    glint, the corner of a letter - which is the distortion a larger kernel
    made obvious. A line or an edge always has neighbours as bright as
    itself, so this test cannot touch one. No kernel size to set.
    """
    name = "speckle"
    label_key = "speckle"
    space = "ycrcb"
    params = {"threshold": ("threshold", 6, 60, 16)}
    _RING = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], np.uint8)

    def process(self, planes):
        # luma only: a hot pixel is a luma event, and chroma is averaged at
        # half resolution by denoise anyway
        y, cr, cb = planes
        thr = float(self.values["threshold"])
        mx = cv2.dilate(y, self._RING)      # brightest of the 8 neighbours
        mn = cv2.erode(y, self._RING)       # darkest
        hot = cv2.compare(cv2.subtract(y, mx), thr, cv2.CMP_GT)
        dead = cv2.compare(cv2.subtract(mn, y), thr, cv2.CMP_GT)
        # a plain .copy() on a UMat is a full round trip through host memory
        out = (y.copy() if not isinstance(y, cv2.UMat)
               else cv2.addWeighted(y, 1.0, y, 0.0, 0.0))
        out = cv2.copyTo(mx, hot, out)
        out = cv2.copyTo(mn, dead, out)
        return [out, cr, cb]


_LAP_K = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float32)


def spatial_sigma(y):
    """Luma noise sigma from a single frame (Immerkaer), flat areas only.

    The old estimate read sigma off the frame-to-frame difference, and motion
    that slipped under its static threshold counted as noise: a slow pan took it
    from a true 4.7 to 9-12, the bilateral pass scaled with it, and a moving
    picture came out *softer than the raw camera* (measured: 31.7 dB denoised
    against 31.6 raw). A single-frame estimate cannot be fooled by motion. Only
    the flattest 30% of a centre crop is read, so texture does not count either.
    """
    if isinstance(y, cv2.UMat):
        h, w = frame_size(y)
    else:
        h, w = y.shape[:2]
    cw, ch = min(w, 640), min(h, 360)
    x0, y0 = (w - cw) // 2, (h - ch) // 2
    if isinstance(y, cv2.UMat):
        c = cv2.UMat(y, (y0, y0 + ch), (x0, x0 + cw)).get()   # crop on the device
    else:
        c = y[y0:y0 + ch, x0:x0 + cw]
    lap = cv2.absdiff(cv2.filter2D(c, cv2.CV_16S, _LAP_K), 0)[4:-4, 4:-4]
    g = cv2.GaussianBlur(c, (0, 0), 2.0)
    mag = cv2.add(cv2.absdiff(cv2.Sobel(g, cv2.CV_16S, 1, 0), 0),
                  cv2.absdiff(cv2.Sobel(g, cv2.CV_16S, 0, 1), 0))[4:-4, 4:-4]
    sub = mag[::3, ::3].ravel()
    if sub.size < 100:
        return None
    k = sub.size * 3 // 10
    thr = float(np.partition(sub, k)[k])
    m = (mag <= thr).astype(np.uint8)
    if cv2.countNonZero(m) < 500:
        return None
    return math.sqrt(math.pi / 2.0) / 6.0 * float(cv2.mean(lap, mask=m)[0])


class Planes(list):
    """A YCrCb plane list that carries per-frame side data (`meta`).

    The chain runs in several threads a frame apart, so anything one stage
    hands to a later one has to travel *with the frame* - a module-level
    variable would be overwritten by the next frame's denoise while the
    sharpener is still working on this one. Pipeline.run re-attaches meta to
    whatever a filter returns; a filter that moves pixels around (Framing)
    returns a Planes with empty meta to drop maps that no longer line up.
    """
    meta = None


def frame_meta(frame):
    return getattr(frame, "meta", None) or {}


class Denoise(BaseFilter):
    """Motion-compensated recursive averaging with a per-pixel frame count.

    Temporal. Every pixel keeps a count n of how many frames its reference
    stands for. Each new frame is blended in with weight 1/(n+1), where n is
    first cut down by how far the reference disagrees with the frame (the
    gate). A still pixel converges to the plain average of the last NMAX
    frames; a pixel the reference cannot explain restarts at n = 1 and is
    taken from the current frame. The reference is float32: at 8 bits an
    update smaller than half a level rounds away and the average freezes.

    Motion. The previous output is warped onto the current frame with DIS
    optical flow. Where the flow says content moved, the warped reference is
    taken unless it matches clearly worse; where the flow is ~0 the exact,
    unresampled one is.

    Cost. Everything that decides *how much* to average - the difference
    maps, the gate, the frame count - lives at quarter resolution; only the
    blend itself and the warp touch full-resolution pixels. v19 kept all of
    it at full resolution and ran a bilateral filter over the bounding box of
    every pixel that had moved, which with any motion at all was the whole
    frame: 30 ms at 720p, more than a whole 30 fps stage. That is what kept
    pushing the chain down to half resolution and made the picture soft.

    Noise model. The gate threshold follows the noise level, and on a real
    camera that level depends on brightness (shadows under a backlight carry
    several times the grain of a lit wall; a blown-out window carries none).
    One threshold for the whole frame is set by whichever dominates, and
    everything noisier than that never averages. The floor is therefore
    measured per brightness band from the frame's own temporal difference.

    Spatial. Where averaging could not help (n ~ 1: occlusions, a hand, a
    turning head) each pixel may move toward its local mean by at most
    EPS_Y residual sigmas. Bounded on purpose: grain goes down, and no
    setting of it can wash detail out the way a wide bilateral did.

    Publishes a per-pixel residual-noise map in the frame's meta; Sharpen
    sets its gate from it.

    `boost` is set by Pipeline from spare budget:
      0  no flow (still-picture averaging only)
      1  DIS ultrafast, 320 px
      2  DIS fast, 320 px
      3  DIS fast, 480 px
    """
    name = "denoise"
    label_key = "denoise"
    space = "ycrcb"
    locked = True
    params = {"strength": ("strength", 0.0, 1.0, 1.0)}
    MAX_BOOST = 3
    AGE_S = 16.0        # count stored as n * AGE_S in uint8
    NMAX = 16.0         # frames a still pixel may average over
    MS = 4              # difference maps are stored x4 (quarter levels)
    G_X0 = 1.4          # gate fully open up to this many noise floors
    G_THR = 2.0         # ... and closed this many floors further on
    FLOW_T = 0.5        # px/frame above which the flow counts as motion
    STEP = 1.5          # brightness jump (levels) treated as an exposure change
    LOW_N = 2.0         # spatial pass fades out between n = 1 and this
    L_K = 2.0           # luma bilateral range sigma, in noise sigmas
    TILE = 64           # spatial pass works in tiles of this many pixels
    TILE_MAX = 0.6      # ... and gives up above this share of the frame
    EPS_C = 6.0         # chroma: at most this many levels
    C_K = (9, 9)        # chroma blur window (half-resolution pixels)
    LBINS = 16          # brightness bands for the noise floor

    def __init__(self):
        super().__init__()
        self.boost = 1
        self._dis = None
        self._dkey = None
        self._grids = {}
        self._sig_s = None
        self._sig_t = None
        self._fref = None           # global gate floor, quarter levels
        self._floors = None         # per brightness band
        self._nlut = np.ones(256, np.float32)
        self._n8 = np.full(256, 64, np.uint8)       # noise scale x64
        self._inv8 = np.full(256, 64, np.uint8)     # 1 / noise scale, x64
        self._n = 0
        self._glut = None
        self._gkey = None
        n = np.arange(256, dtype=np.float32) / self.AGE_S
        # gate^2: a reference half in agreement counts for a quarter of its frames
        self._g2lut = ((np.arange(256, dtype=np.float32) / 255.0) ** 2 * 255.0).astype(np.uint8)
        self._alut = (n / (n + 1.0) * 255.0).astype(np.uint8)
        self._rlut = (255.0 / np.sqrt(np.maximum(n, 1.0))).astype(np.uint8)
        self._lowlut = (np.clip((self.LOW_N - n) / max(self.LOW_N - 1.0, 0.1), 0, 1)
                        * 255).astype(np.uint8)
        self.reset()

    def reset(self):
        self._prevf = None      # last temporal output, float32 (the reference)
        self._age = None        # frame count, quarter resolution
        self._pshape = None
        self._gprev = None
        self._pq = None
        self._pcr = self._pcb = None

    # ---------------- motion ----------------
    def _flow(self, gcur, gprev, w, fast):
        """Flow from the current thumbnail to the previous one, in full-res
        pixels, or None when nothing moves."""
        key = "fast" if fast else "ultra"
        if self._dis is None or self._dkey != key:
            self._dis = cv2.DISOpticalFlow_create(
                cv2.DISOPTICAL_FLOW_PRESET_FAST if fast
                else cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST)
            self._dkey = key
        f = self._dis.calc(gcur, gprev, None)
        f *= w / float(gcur.shape[1])
        if float(cv2.mean(cv2.absdiff(f, 0))[0]) < 0.15:
            return None                 # still: the exact reference is better
        return f

    def _warp(self, img, fl, interp=cv2.INTER_LINEAR):
        if RELATIVE_MAP:
            return cv2.remap(img, fl, None, interp | cv2.WARP_RELATIVE_MAP,
                             borderMode=cv2.BORDER_REPLICATE)
        h, w = fl.shape[:2]
        g = self._grids.get((w, h))
        if g is None:
            gx, gy = np.meshgrid(np.arange(w, dtype=np.float32),
                                 np.arange(h, dtype=np.float32))
            g = self._grids[(w, h)] = (gx, gy)
        return cv2.remap(img, cv2.add(fl[..., 0], g[0]), cv2.add(fl[..., 1], g[1]),
                         interp, borderMode=cv2.BORDER_REPLICATE)

    def _qmap(self, d, qw, qh):
        """Blurred |difference| at quarter resolution, in quarter levels.

        A 4x4 area average plus a 3x3 blur is the sigma-2 blur the gate needs
        (the raw per-pixel difference is itself noise), at 1/16 of the pixels.
        Scaled x4 first so the average does not round to whole levels."""
        hh, hw = d.shape[0] // 2, d.shape[1] // 2
        if (hw, hh) == (qw * 2, qh * 2):
            # two exact halvings: OpenCV's 2:1 area path is ~8x faster than 4:1
            q = cv2.resize(cv2.convertScaleAbs(
                cv2.resize(d, (hw, hh), interpolation=cv2.INTER_AREA), alpha=self.MS),
                (qw, qh), interpolation=cv2.INTER_AREA)
        else:
            q = cv2.resize(cv2.convertScaleAbs(d, alpha=self.MS), (qw, qh),
                           interpolation=cv2.INTER_AREA)
        return cv2.GaussianBlur(q, (3, 3), 0.6)

    @staticmethod
    def _quarter(img, qw, qh):
        h, w = img.shape[:2]
        if (w, h) == (qw * 4, qh * 4):
            img = cv2.resize(img, (qw * 2, qh * 2), interpolation=cv2.INTER_AREA)
        return cv2.resize(img, (qw, qh), interpolation=cv2.INTER_AREA)

    @staticmethod
    def _offset(gcur, gprev):
        """Whole-frame brightness difference, to a fraction of a level. Camera
        auto exposure and mains flicker move everything at once; left in, half
        a level of it reads as motion everywhere."""
        d = gcur[::2, ::2].astype(np.int16) - gprev[::2, ::2]
        hist = np.bincount((d + 255).ravel(), minlength=511)
        c = np.cumsum(hist)
        med = int(np.searchsorted(c, c[-1] * 0.5))
        lo, hi = max(med - 3, 0), min(med + 4, 511)
        wgt = hist[lo:hi]
        # mean of the differences within 3 levels of the median: fractional,
        # and blind to anything that actually moved
        return float((wgt * np.arange(lo, hi)).sum()) / max(float(wgt.sum()), 1.0) - 255.0

    # ---------------- noise statistics ----------------
    def _stats(self, yq, base, samp):
        """Gate floor per brightness band, and the noise sigma, from this
        frame's own temporal difference.

        The floor is the 35th percentile of the blurred difference - the
        level a still pixel sits at whatever the grain looks like - taken
        separately in LBINS bands of brightness. Sigma is the mean raw
        difference over pixels at or near their band's floor, so motion never
        counts.
        """
        lb = self.LBINS
        hist = cv2.calcHist([yq, base], [0, 1], None, [lb, 256], [0, 256, 0, 256])
        cnt = hist.sum(1)
        tot = hist.sum(0)
        gc = np.cumsum(tot)
        gp35 = float(np.searchsorted(gc, 0.35 * gc[-1]))
        gp60 = float(np.searchsorted(gc, 0.60 * gc[-1]))
        fref = self._fref
        # most of the frame moving in a way the flow could not follow: the
        # percentiles are motion, not noise - keep the last good reading
        if fref is not None and gp60 >= 2.5 * max(fref, self.MS):
            return
        gp35 = max(gp35, 0.5 * self.MS)
        self._fref = fref = gp35 if fref is None else fref + (gp35 - fref) * 0.15
        valid = cnt >= 200
        if valid.any():
            cdf = np.cumsum(hist, 1)
            p35 = (cdf < 0.35 * cnt[:, None]).sum(1).astype(np.float32)
            idx = np.arange(lb)
            new = np.interp(idx, idx[valid], np.maximum(p35[valid], 0.25 * self.MS))
            self._floors = (new if self._floors is None
                            else self._floors + (new - self._floors) * 0.15)
            ratio = np.clip(self._floors / max(fref, 1e-3), 0.35, 3.0)
            centers = (idx + 0.5) * (256.0 / lb)
            self._nlut = np.interp(np.arange(256), centers, ratio).astype(np.float32)
            self._n8 = np.clip(self._nlut * 64.0, 1, 255).astype(np.uint8)
            self._inv8 = np.clip(64.0 / self._nlut, 1, 255).astype(np.uint8)
        # sigma: raw differences, brought to the reference band, near the floor
        inv = cv2.LUT(yq, self._inv8)
        bn = cv2.multiply(base, inv, scale=1 / 64.0)
        sn = cv2.multiply(samp, inv, scale=self.MS / 64.0)
        sel = sn[bn <= 1.5 * fref]
        if sel.size < 1000:
            return
        st = min(max(float(sel.mean()) / self.MS / 0.798, 0.6), 40.0)
        self._sig_t = st if self._sig_t is None else self._sig_t + (st - self._sig_t) * 0.15

    def _spatial(self, y, low, sig, w, h):
        """Edge-preserving pass over the tiles that hold un-averaged pixels.

        Tiles, not a bounding box: the box around everything that moved is
        nearly always the whole frame, and a full-frame bilateral was a third
        of the old cost. A moving head and a hand are a few dozen tiles.
        """
        T = self.TILE
        qh, qw = low.shape[:2]
        tq = T // 4
        gw, gh = max(qw // tq, 1), max(qh // tq, 1)
        occ = cv2.resize(low, (gw, gh), interpolation=cv2.INTER_AREA)
        act = occ > 6                       # >2% of the tile un-averaged
        n_act = int(act.sum())
        if n_act == 0 or n_act > self.TILE_MAX * act.size:
            return y                        # nothing, or everything (a cut)
        s = max(5.0, self.L_K * sig)
        wt = cv2.resize(low, (w, h), interpolation=cv2.INTER_LINEAR)
        out = y.copy()
        pad = 4
        tw, th = w / float(gw), h / float(gh)
        for r in range(gh):
            row = act[r]
            c = 0
            while c < gw:
                if not row[c]:
                    c += 1
                    continue
                c0 = c
                while c < gw and row[c]:
                    c += 1
                x0, x1 = int(c0 * tw), int(c * tw) if c < gw else w
                y0, y1 = int(r * th), int((r + 1) * th) if r < gh - 1 else h
                px0, py0 = max(x0 - pad, 0), max(y0 - pad, 0)
                px1, py1 = min(x1 + pad, w), min(y1 + pad, h)
                f = cv2.bilateralFilter(y[py0:py1, px0:px1], 5, s, s)
                f = f[y0 - py0:y1 - py0, x0 - px0:x1 - px0]
                src = y[y0:y1, x0:x1]
                k = wt[y0:y1, x0:x1]
                out[y0:y1, x0:x1] = cv2.add(
                    cv2.multiply(f, k, scale=1 / 255.0),
                    cv2.multiply(src, cv2.bitwise_not(k), scale=1 / 255.0))
        return out

    @staticmethod
    def _toward(img, blur, eps):
        """img moved toward blur by at most eps (per pixel), uint8."""
        up = cv2.min(cv2.subtract(blur, img), eps)
        dn = cv2.min(cv2.subtract(img, blur), eps)
        return cv2.subtract(cv2.add(img, up), dn)

    # ---------------- main ----------------
    def process(self, frame):
        h, w = frame_size(frame)
        y, cr, cb = (npy(p) for p in frame)
        boost = int(self.boost)
        self._n += 1
        AS, MS = self.AGE_S, self.MS

        if self._sig_t is None and (self._n % 5 == 1 or self._sig_s is None):
            # start-up only: a first guess until the temporal reading exists
            ss = spatial_sigma(y)
            if ss is not None:
                self._sig_s = ss * 2.0
        sig = self._sig_t if self._sig_t is not None else (self._sig_s or 4.0)
        NOISE["sigma"] = sig

        qw, qh = max(w // 4, 8), max(h // 4, 8)
        yq = self._quarter(y, qw, qh)
        fw = 480 if boost >= 3 else 320
        fh = max(int(round(h * fw / float(w) / 2.0)) * 2, 16)
        gcur = yq if (qw, qh) == (fw, fh) else cv2.resize(
            y, (fw, fh), interpolation=cv2.INTER_AREA)
        # chroma at exactly half: an integer-ratio INTER_AREA is ~30x faster
        # than an odd one, and 4:2:0 cameras carry no more than that anyway
        cw, chh = max(w // 2, 16), max(h // 2, 16)
        crs = cv2.resize(cr, (cw, chh), interpolation=cv2.INTER_AREA).astype(np.float32)
        cbs = cv2.resize(cb, (cw, chh), interpolation=cv2.INTER_AREA).astype(np.float32)

        have_prev = (self._prevf is not None and self._pshape == (h, w)
                     and self._gprev is not None)
        if have_prev and self._gprev.shape != gcur.shape:
            self._gprev = cv2.resize(self._gprev, (fw, fh), interpolation=cv2.INTER_LINEAR)

        if have_prev:
            prevf, age = self._prevf, self._age
            pcr, pcb = self._pcr, self._pcb
            # Whole-frame brightness offset between this frame and the
            # reference. A real exposure step moves the reference with it at
            # once. Anything smaller - mains flicker, auto exposure hunting by
            # a fraction of a level - is only taken out of the *comparison*:
            # the gate stays open, and the averaging then removes the flicker
            # instead of faithfully reproducing it.
            off = self._offset(yq, self._pq)
            if abs(off) >= self.STEP:
                prevf = cv2.add(prevf, off)
                off = 0.0
            ds = cv2.absdiff(y, cv2.convertScaleAbs(prevf, beta=off))
            ms = self._qmap(ds, qw, qh)
            motion = base = ms
            samp = cv2.resize(ds, (qw, qh), interpolation=cv2.INTER_NEAREST)
            f = self._flow(gcur, self._gprev, w, boost >= 2) if boost >= 1 else None
            if f is not None:
                warpedf = self._warp(prevf, cv2.resize(f, (w, h), interpolation=cv2.INTER_LINEAR))
                dw = cv2.absdiff(y, cv2.convertScaleAbs(warpedf, beta=off))
                mw = self._qmap(dw, qw, qh)
                floor = self._fref if self._fref is not None else sig * MS
                # warped only if better by a margin: flow bleeds off a moving
                # object into the still background around it, and there the
                # exact reference is strictly the better one
                mwp = cv2.add(mw, 0.5 * MS + 0.5 * floor)
                use_q = cv2.compare(mwp, ms, cv2.CMP_LT)
                # ... but where the flow says it moved and the warp is not
                # worse, warp: a flat patch matches either way, and the stale
                # copy dragged along frame after frame is what smears
                fq = f if (qw, qh) == (fw, fh) else cv2.resize(
                    f, (qw, qh), interpolation=cv2.INTER_LINEAR)
                fm = cv2.compare(cv2.max(cv2.absdiff(fq[..., 0], 0), cv2.absdiff(fq[..., 1], 0)),
                                 self.FLOW_T, cv2.CMP_GT)
                use_q = cv2.bitwise_or(use_q, cv2.bitwise_and(
                    fm, cv2.compare(mw, ms, cv2.CMP_LE)))
                cv2.copyTo(warpedf, cv2.resize(use_q, (w, h), interpolation=cv2.INTER_NEAREST),
                           prevf)
                cv2.copyTo(self._warp(age, fq * 0.25, cv2.INTER_NEAREST), use_q, age)
                fc = cv2.resize(f, (cw, chh), interpolation=cv2.INTER_LINEAR)
                fc *= 0.5
                uc = cv2.resize(use_q, (cw, chh), interpolation=cv2.INTER_NEAREST)
                cv2.copyTo(self._warp(pcr, fc), uc, pcr)
                cv2.copyTo(self._warp(pcb, fc), uc, pcb)
                motion = cv2.min(ms, mwp)
                base = cv2.min(ms, mw)
                samp = cv2.copyTo(cv2.resize(dw, (qw, qh), interpolation=cv2.INTER_NEAREST),
                                  use_q, samp)
            if self._n % 2 == 0 or self._fref is None:
                self._stats(yq, base, samp)
            fref = self._fref if self._fref is not None else sig * MS

            # gate: 255 where the reference agrees to within the noise expected
            # at this brightness, 0 where it does not
            x0 = self.G_X0 * fref
            thr = max(self.G_THR * fref, 3.0 * MS)
            key = (round(x0, 1), round(thr, 1))
            if key != self._gkey:
                x = np.arange(256, dtype=np.float32)
                self._glut = (np.clip(1.0 - (x - x0) / thr, 0, 1) * 255).astype(np.uint8)
                self._gkey = key
            mn = cv2.multiply(motion, cv2.LUT(yq, self._inv8), scale=1 / 64.0)
            g = cv2.LUT(cv2.LUT(mn, self._glut), self._g2lut)
            n_eff = cv2.multiply(age, g, scale=1 / 255.0)
            aq = cv2.LUT(n_eff, self._alut).astype(np.float32)
            aq *= 1 / 255.0
            age = cv2.min(cv2.add(n_eff, AS), self.NMAX * AS)

            af = cv2.resize(aq, (w, h), interpolation=cv2.INTER_LINEAR)
            yf = y.astype(np.float32)
            cv2.subtract(prevf, yf, dst=prevf)
            cv2.multiply(prevf, af, dst=prevf)
            cv2.add(prevf, yf, dst=prevf)
            y = cv2.convertScaleAbs(prevf)
            ac = cv2.resize(aq, (cw, chh), interpolation=cv2.INTER_LINEAR)
            for cur, ref in ((crs, pcr), (cbs, pcb)):
                cv2.subtract(ref, cur, dst=ref)
                cv2.multiply(ref, ac, dst=ref)
                cv2.add(cur, ref, dst=cur)
        else:
            prevf = y.astype(np.float32)
            age = np.full((qh, qw), int(AS), np.uint8)
        self._prevf, self._age = prevf, age
        self._pshape, self._gprev = (h, w), gcur
        self._pq = self._quarter(y, qw, qh)
        self._pcr, self._pcb = crs, cbs
        NOISE["residual"] = sig / math.sqrt(max(float(cv2.mean(age)[0]) / AS, 1.0))

        crs8, cbs8 = cv2.convertScaleAbs(crs), cv2.convertScaleAbs(cbs)
        # ---- spatial, only where averaging could not help ----
        low = cv2.LUT(age, self._lowlut)
        nl8 = cv2.LUT(yq, self._n8)
        if cv2.countNonZero(low):
            y = self._spatial(y, low, sig, w, h)
            epc = cv2.resize(cv2.convertScaleAbs(low, alpha=self.EPS_C / 255.0),
                             (cw, chh), interpolation=cv2.INTER_LINEAR)
            crs8 = self._toward(crs8, cv2.blur(crs8, self.C_K), epc)
            cbs8 = self._toward(cbs8, cv2.blur(cbs8, self.C_K), epc)
        cr = cv2.resize(crs8, (w, h), interpolation=cv2.INTER_LINEAR)
        cb = cv2.resize(cbs8, (w, h), interpolation=cv2.INTER_LINEAR)
        out = Planes([y, cr, cb])
        # residual noise sigma per pixel, x8, quarter resolution
        out.meta = {"resid": cv2.multiply(cv2.LUT(age, self._rlut), nl8,
                                          scale=sig * 8.0 / (255.0 * 64.0))}
        return out


YUNET_URL = ("https://github.com/opencv/opencv_zoo/raw/main/models/"
             "face_detection_yunet/face_detection_yunet_2023mar.onnx")


class FaceTracker:
    """YuNet (OpenCV Zoo, Apache-2.0) with a Haar fallback.

    Measured on one real face at 640x360: YuNet found it in all of normal, dark
    (x0.35), very dark (x0.18), backlit and 22-degree-rotated conditions; the Haar
    cascade found it in none of them, at six times the cost. Face metering that
    silently never fires is the reason a backlit subject stays dark."""
    DET_W = 320

    def __init__(self):
        self._yunet = None
        self._haar = None
        self._size = None
        self._n = 0
        self._miss = 0
        # (x, y, w, h) as fractions of the frame. It used to be pixels of
        # whatever frame the detector saw, and the detector sees the
        # *processing* frame - so when the processing scale changed, the box
        # was blended across two scales and every face stage (touch-up,
        # deblur, face light) worked on the wrong patch for a second or two.
        self.nbox = None
        self.backend = "-"
        self._fetch = None

    def _model(self):
        """Path to the YuNet model, or None while it is not on disk yet.

        The download used to run inline, inside the frame pipeline: on a slow
        connection the whole picture froze for as long as it took, and with no
        connection it failed once and the detector fell back to Haar for good.
        It now runs in the background and is retried on the next start.
        """
        path = os.path.join(MODEL_DIR, "face_detection_yunet.onnx")
        if os.path.exists(path):
            return path
        if self._fetch is None:
            def get():
                try:
                    os.makedirs(MODEL_DIR, exist_ok=True)
                    tmp = path + ".part"
                    urllib.request.urlretrieve(YUNET_URL, tmp)
                    os.replace(tmp, path)
                except Exception:
                    pass
            self._fetch = threading.Thread(target=get, daemon=True)
            self._fetch.start()
        return None

    def _detect(self, small):
        h, w = small.shape[:2]
        if self._yunet is not False:
            path = self._model() if self._yunet is None else True
            if path:
                try:
                    if self._yunet is None or self._size != (w, h):
                        self._yunet = cv2.FaceDetectorYN.create(
                            path if isinstance(path, str) else self._model(), "", (w, h),
                            0.6, 0.3, 5000)
                        self._size = (w, h)
                        self.backend = "yunet"
                    _, faces = self._yunet.detect(small)
                    if faces is None or len(faces) == 0:
                        return None
                    f = max(faces, key=lambda r: r[2] * r[3])
                    return float(f[0]), float(f[1]), float(f[2]), float(f[3])
                except Exception:
                    self._yunet = False
        if self._haar is None:
            # OpenCV 5 moved the cascade classifier out of the main package;
            # with neither detector there is simply no face box
            try:
                self._haar = cv2.CascadeClassifier(
                    cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
                if self._haar.empty():
                    self._haar = False
            except Exception:
                self._haar = False
        if self._haar is False:
            self.backend = "none"
            return None
        self.backend = "haar" if self._yunet is not None else self.backend
        found = self._haar.detectMultiScale(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY),
                                            1.2, 4, minSize=(24, 24))
        if len(found) == 0:
            return None
        x, y, w2, h2 = max(found, key=lambda r: r[2] * r[3])
        return float(x), float(y), float(w2), float(h2)

    def px(self, w, h):
        """Box in pixels of a w x h frame, or None."""
        b = self.nbox
        if b is None:
            return None
        return b[0] * w, b[1] * h, b[2] * w, b[3] * h

    def update(self, frame, every=3):
        self._n += 1
        if self._n % every:
            return self.nbox
        h, w = frame_size(frame)
        sc = self.DET_W / float(w)
        try:
            small = npy(cv2.resize(frame, (self.DET_W, max(int(h * sc), 16)),
                                   interpolation=cv2.INTER_AREA))
            got = self._detect(small)
        except Exception:
            # the tracker runs outside the per-filter guard: an exception here
            # used to drop the frame it was called on - one in three
            got = None
        if got is None:
            self._miss += 1
            if self._miss > 10:
                self.nbox = None
            return self.nbox
        self._miss = 0
        sw, sh = float(small.shape[1]), float(small.shape[0])
        new = (got[0] / sw, got[1] / sh, got[2] / sw, got[3] / sh)
        old = self.nbox
        self.nbox = new if old is None else tuple(
            0.6 * o + 0.4 * n for o, n in zip(old, new))
        return self.nbox


FACES = FaceTracker()


SCENE = {"key": "normal", "name": "-"}

# target, gamma floor, gamma ceiling, contrast mul, extra clipping allowed,
# deadband (fraction of full scale)
SCENE_PROFILE = {
    "normal":  (118.0, 0.72, 1.40, 1.00, 0.020, 0.075),
    "low":     (112.0, 0.50, 1.10, 0.80, 0.030, 0.055),
    "night":   (100.0, 0.42, 1.05, 0.60, 0.045, 0.050),
    "backlit": (126.0, 0.38, 1.10, 0.90, 0.110, 0.045),
    "bright":  (130.0, 0.85, 1.90, 1.10, 0.008, 0.060),
}


class AutoExposure(BaseFilter):
    """Scene-adaptive tone mapping, solved against the frame's own histogram.

    The previous version solved gamma analytically from metered**gamma == target
    and then applied a different curve than the one it solved: levels first,
    highlight protection blended over the gamma, an S-curve on top. The picture
    therefore never landed on the target, and the size and the sign of the miss
    changed with the scene - which is exactly the 'too dark, and then too bright'
    behaviour. It was open loop against a model of itself that was wrong.

    This version closes the loop. It builds the candidate curve, runs it over the
    256-bin luma histogram of the thumbnail, and bisects gamma until the
    predicted output mean of the metered region lands on the target. Seven
    iterations over a 256-entry array cost microseconds, and nothing is assumed
    about the shape of the curve any more.

    Two constraints then sit on top of the solve, and they are what stop the
    blown frame:
      - the predicted fraction of clipped pixels may not rise past the scene's
        budget (wide for a backlit window, near zero for an already bright room)
      - the predicted global mean is capped, so metering a dark face in a dark
        room can no longer wash out everything behind it

    The scene profile is the rest of the answer to 'it should behave differently
    depending on the situation'. A night frame is not graded to the same target
    as a lit one on purpose: pushing a starved, noisy sensor to 118 amplifies the
    noise faster than it reveals anything, so the target moves down and the gain
    ceiling comes with it.
    """
    name = "auto_exposure"
    label_key = "exposure"
    space = "ycrcb"
    # Default lowered from 0.8 to 0.1 - product decision (2026-09): full-strength
    # exposure correction was too aggressive as an out-of-the-box default,
    # stacking with local_tone's own gain. Left available at full range for
    # anyone who wants it back.
    params = {"strength": ("strength", 0.0, 1.0, 0.1),
              "auto": ("auto", 0.0, 1.0, 1.0),
              "target": ("target", 80, 180, 118),
              "contrast": ("contrast", 0.0, 1.0, 0.35),
              "face": ("face", 0.0, 1.0, 1.0)}
    CLIP_LO, CLIP_HI = 0.004, 0.996
    MEAN_CEIL = 196.0
    SW, SH = 160, 90
    uses_face = True

    def __init__(self):
        super().__init__()
        self._key = None
        self._lut = None
        self._gamma = 1.0
        self._lo, self._hi = 0.0, 1.0
        self._scene = "normal"
        self._cand = "normal"
        self._hold = 0
        self._x = np.arange(256, dtype=np.float32) / 255.0
        self._idx = np.arange(256, dtype=np.float32)
        self._recover = self._idx < 235.0     # input levels that still hold detail

    # -- curve ------------------------------------------------------------
    def _curve(self, lo, hi, gamma, contrast):
        x = np.clip((self._x - lo) / max(hi - lo, 0.08), 0.0, 1.0)      # levels
        keep = x ** 3                                                    # highlights
        x = (1.0 - keep) * np.power(x, gamma) + keep * x                 # shadow lift
        x = (1.0 - contrast) * x + contrast * (x * x * (3.0 - 2.0 * x))  # S-curve
        return x * 255.0

    def _build(self, lo, hi, gamma, contrast):
        key = (round(lo, 3), round(hi, 3), round(gamma, 3), round(contrast, 3))
        if key != self._key:
            self._lut = np.clip(self._curve(lo, hi, gamma, contrast),
                                0, 255).astype(np.uint8)
            self._key = key
        return self._lut

    def _predict(self, curve, hist, n):
        """Mean, and the fraction of pixels this curve would *newly* blow out,
        straight off the histogram - no pixels touched.

        Newly is the operative word. A backlit shot already has a window sitting
        at 250, and counting those against the budget makes the solver refuse to
        lift the face - which is the one thing a backlit shot needs. Only input
        below 235 still holds detail worth protecting, so only that is counted.
        """
        return (float(np.dot(curve, hist) / n),
                float(hist[self._recover & (curve >= 249.0)].sum() / n))

    # -- measurement ------------------------------------------------------
    def _levels(self, hist, n):
        cdf = np.cumsum(hist) / n
        lo = float(np.searchsorted(cdf, self.CLIP_LO)) / 255.0
        hi = float(np.searchsorted(cdf, self.CLIP_HI)) / 255.0
        if hi - lo < 0.08:
            lo, hi = 0.0, 1.0
        self._lo += (lo - self._lo) * 0.2
        self._hi += (hi - self._hi) * 0.2
        return self._lo, self._hi

    def _fbox(self, full_w, full_h):
        box = FACES.px(full_w, full_h)
        if box is None:
            return None
        sx, sy = self.SW / float(full_w), self.SH / float(full_h)
        x, y, fw, fh = box
        x0, y0 = max(int(x * sx), 0), max(int(y * sy), 0)
        x1, y1 = min(int((x + fw) * sx), self.SW), min(int((y + fh) * sy), self.SH)
        return (x0, y0, x1, y1) if (x1 - x0 >= 3 and y1 - y0 >= 3) else None

    def _classify(self, small, hist, n, fbox):
        total = float(np.dot(self._idx, hist))
        mean = total / n
        clip_hi = float(hist[250:].sum() / n)
        face_v = bg_v = None
        if fbox is not None:
            x0, y0, x1, y1 = fbox
            roi = small[y0:y1, x0:x1]
            face_v = float(roi.mean())
            fn = float(roi.size)
            if n - fn > 200:
                bg_v = (total - float(roi.sum())) / (n - fn)
        if clip_hi > 0.12 and mean > 140.0:
            k = "bright"
        elif bg_v is not None and bg_v - face_v > 38.0:
            k = "backlit"
        elif mean < 38.0:
            k = "night"
        elif mean < 85.0:
            k = "low"
        else:
            k = "normal"
        # a switch needs a run of agreeing frames: a hand crossing the lens or a
        # head turn should not re-grade the shot, and a grade that changes every
        # few frames is far more objectionable than one that is slightly off
        if k == self._cand:
            self._hold += 1
        else:
            self._cand, self._hold = k, 1
        if self._hold >= 10 and k != self._scene:
            self._scene = k
        SCENE["key"], SCENE["name"] = self._scene, T("sc_" + self._scene)
        return mean

    # -- main -------------------------------------------------------------
    def process(self, frame):
        s = self.values["strength"]
        if s <= 0.01:
            return frame
        fh, fw = frame_size(frame)
        y, cr, cb = frame
        small = npy(cv2.resize(y, (self.SW, self.SH), interpolation=cv2.INTER_AREA))
        hist = cv2.calcHist([small], [0], None, [256], [0, 256]).ravel()
        n = float(small.size)

        fbox = self._fbox(fw, fh)
        self._classify(small, hist, n, fbox)

        # what the exposure is metered on: the face when there is one, otherwise
        # centre-weighted. Both end up as a histogram so the solver sees the same
        # kind of object either way.
        if fbox is not None and self.values["face"] > 0.5:
            x0, y0, x1, y1 = fbox
            mh = cv2.calcHist([np.ascontiguousarray(small[y0:y1, x0:x1])], [0],
                              None, [256], [0, 256]).ravel()
            mn = float((y1 - y0) * (x1 - x0))
        else:
            c = np.ascontiguousarray(small[int(self.SH * 0.15):int(self.SH * 0.85),
                                           int(self.SW * 0.25):int(self.SW * 0.75)])
            mh = cv2.calcHist([c], [0], None, [256], [0, 256]).ravel() * 0.7 + hist * 0.3
            mn = float(c.size) * 0.7 + n * 0.3

        lo, hi = self._levels(hist, n)
        # The black point may never land inside the subject. A backlit face is
        # the darkest thing in its own frame, so the 0.4 percentile sits just
        # under it: the levels stretch crushes the face to nothing and the gamma
        # lift spends itself undoing that, which is why the subject used to come
        # back out of a backlit correction exactly as dark as it went in.
        mcdf = np.cumsum(mh) / mn
        lo = min(lo, float(np.searchsorted(mcdf, 0.05)) / 255.0 * 0.5)
        lo_eff, hi_eff = lo * s, 1.0 - (1.0 - hi) * s

        auto = self.values["auto"] > 0.5
        tgt, gfloor, gceil, cmul, clipbud, tol = SCENE_PROFILE[
            self._scene if auto else "normal"]
        if auto:
            # the manual target still biases the automatic one, so the slider is
            # not dead while scene adaptation is on
            want = tgt + (float(self.values["target"]) - 118.0) * 0.6
        else:
            want = float(self.values["target"])
        contrast = self.values["contrast"] * s * cmul

        base = self._curve(lo_eff, hi_eff, 1.0, contrast)
        # baseline read with the same metric the constraint uses, so the budget
        # means "how much detail this correction costs", not an absolute count
        _, clip_base = self._predict(base, hist, n)
        cur, _ = self._predict(base, mh, mn)
        # asymmetric deadband: lift a dark subject readily, pull a bright one
        # down grudgingly. A slightly over-bright frame still shows a face; the
        # same error toward dark does not, and an unprompted darkening is what
        # reads as the filter being broken.
        if abs(cur - want) <= tol * 255.0 * (1.9 if cur > want else 1.0):
            gamma = 1.0                       # already exposed: leave it alone
        else:
            g_lo, g_hi, gamma = gfloor, gceil, 1.0
            for _ in range(7):                # mean is monotone decreasing in gamma
                gamma = 0.5 * (g_lo + g_hi)
                m, _ = self._predict(self._curve(lo_eff, hi_eff, gamma, contrast), mh, mn)
                if m < want:
                    g_hi = gamma
                else:
                    g_lo = gamma
            # strength scales the correction, not the curve: 0.5 now means "go
            # half way to correct", which is what the slider reads like
            gamma = 1.0 + (gamma - 1.0) * s

        # A subject far below target buys more room to blow the background: at
        # that point the choice is between a visible face and an intact window,
        # and nobody joins a call to look at the window.
        clipbud *= 1.0 + 2.0 * min(max((want - cur) / max(want, 1.0), 0.0), 1.0)

        # Constraints, and only on the lifting side. Walk gamma back toward 1.0
        # until the frame neither blows its highlights nor washes out as a whole.
        # A gamma above 1.0 is already darkening the frame, and clamping that one
        # toward 1.0 would cancel the correction an overexposed scene asked for.
        while gamma < 0.995:
            c = self._curve(lo_eff, hi_eff, gamma, contrast)
            gm, gc = self._predict(c, hist, n)
            if gc - clip_base <= clipbud and gm <= self.MEAN_CEIL:
                break
            gamma = min(gamma + 0.06, 1.0)

        self._gamma += (gamma - self._gamma) * 0.2     # temporal smoothing
        lut = self._build(lo_eff, hi_eff, self._gamma, contrast)
        return [cv2.LUT(y, lut), cr, cb]


class WhiteBalance(BaseFilter):
    """Shades-of-grey (Minkowski p=6) on a thumbnail, with near-saturated pixels
    excluded. Plain grey-world breaks on any scene with one dominant colour - a
    wooden desk or a coloured wall turns skin green."""
    name = "white_balance"
    label_key = "wb"
    params = {"strength": ("strength", 0.0, 1.0, 0.7)}
    P = 6.0

    def __init__(self):
        super().__init__()
        self._scales = np.ones(3, np.float32)

    def process(self, frame):
        s = self.values["strength"]
        if s <= 0.01:
            return frame
        thumb = npy(cv2.resize(frame, (160, 90),
                               interpolation=cv2.INTER_AREA)).astype(np.float32)
        valid = thumb.max(axis=2) < 240
        if valid.sum() < 200:
            valid = np.ones(thumb.shape[:2], bool)
        est = np.array([np.power(np.mean(np.power(thumb[..., i][valid], self.P)), 1.0 / self.P)
                        for i in range(3)], np.float32)
        target = np.clip(1.0 + (float(est.mean()) / np.maximum(est, 1e-3) - 1.0) * s, 0.6, 1.7)
        self._scales += (target - self._scales) * 0.15
        ramp = np.arange(256, dtype=np.float32)
        lut = np.empty((1, 256, 3), np.uint8)
        for i in range(3):
            lut[0, :, i] = np.clip(ramp * self._scales[i], 0, 255).astype(np.uint8)
        return cv2.LUT(frame, lut)


class NeuralTone(BaseFilter):
    """Learned per-pixel tone curves (Zero-DCE style, ~10K parameters).

    The network never touches full-resolution pixels: it estimates curves on a
    256x144 thumbnail, the enhanced thumbnail is divided by the input to get a
    gain map, and only that smooth gain map is upsampled and multiplied into the
    full frame. Inference cost is therefore fixed regardless of capture
    resolution. Train and export with dce_train.py."""
    name = "neural_tone"
    label_key = "neural"
    params = {"amount": ("strength", 0.0, 1.0, 0.8), "max_gain": ("max_gain", 1.5, 5.0, 3.5)}
    IN_W, IN_H = 256, 144
    Q = 48.0

    def __init__(self):
        super().__init__()
        self.path = os.path.join(MODEL_DIR, "dce.onnx")
        self._sess = None
        if ort is None:
            self.available, self.hint = False, T("no_onnx")
        elif not os.path.exists(self.path):
            self.available, self.hint = False, T("no_model").format(path=self.path)
        else:
            self.available = True

    def _ensure(self):
        if self._sess is None:
            so = ort.SessionOptions()
            so.intra_op_num_threads = max((os.cpu_count() or 2) // 2, 1)
            self._sess = ort.InferenceSession(self.path, so,
                                              providers=onnx_providers())
            self._name = self._sess.get_inputs()[0].name
        return self._sess

    def process(self, frame):
        a = self.values["amount"]
        if a <= 0.01:
            return frame
        sess = self._ensure()
        small = npy(cv2.resize(frame, (self.IN_W, self.IN_H), interpolation=cv2.INTER_AREA))
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        out = sess.run(None, {self._name: rgb.transpose(2, 0, 1)[None]})[0][0]
        enh = np.clip(out.transpose(1, 2, 0), 0, 1)
        gain = (enh + 4e-3) / (rgb + 4e-3)
        gain = 1.0 + (gain - 1.0) * a
        gain = np.clip(gain, 0.5, min(self.values["max_gain"], 255.0 / self.Q))
        # quantise to uint8 (step 1/48) before upsampling: resizing and multiplying
        # a full-resolution float32 map cost 5x the inference itself
        q = np.ascontiguousarray(gain[:, :, ::-1] * self.Q).astype(np.uint8)   # to BGR
        h, w = frame_size(frame)
        return cv2.multiply(frame, up(q, w, h), scale=1.0 / self.Q, dtype=cv2.CV_8U)


class NeuralLocal(BaseFilter):
    """Learned local affine grid - a bilateral grid with the depth axis cut down
    to what a CPU can afford. Trained by lan_train.py, ~13K parameters.

    The network sees a 160x96 thumbnail and returns a 20x12 grid of affine colour
    transforms (3 gains + 3 offsets) per luma bin. The grid is upsampled and
    sliced against the frame's own luma, so the correction follows edges instead
    of bleeding across them: a backlit head is lifted without a glow around it.
    Inference cost is fixed at any capture resolution; the full-resolution work is
    a handful of uint8 OpenCV passes.

    This is a replacement, not an addition. It is trained to reproduce
    auto_exposure + face_light + white_balance + vibrance, so Pipeline skips
    those four while it is on - see Pipeline.SUBSUMED. On one core at 720p it
    costs 11.8 ms against the 10.7 ms those four add up to: the swap is roughly
    cost-neutral and buys local, edge-aware, colour-changing correction that a
    global LUT cannot express.

    An offset is the part a Zero-DCE curve cannot do: a multiplicative gain can
    never pull a raised black point back down, and a raised black point is why a
    cheap sensor looks hazy even after it has been brightened.
    """
    name = "neural_local"
    label_key = "local"
    params = {"amount": ("strength", 0.0, 1.0, 1.0),
              "quality": ("quality", 0.25, 1.0, 1.0)}
    IN_W, IN_H = 160, 96
    QG = 48.0            # gain quantiser - max representable gain 255/48 = 5.31
    OFF = 89             # offset bias in pixel units (+-0.35 * 255)

    def __init__(self):
        super().__init__()
        self.path = None
        for cand in ("lan_int8.onnx", "lan.onnx"):     # int8 first when present
            p = os.path.join(MODEL_DIR, cand)
            if os.path.exists(p):
                self.path = p
                break
        self._sess = None
        self._coef = None
        self._wlut = None
        self.bins = 0
        self._n = 0
        if ort is None:
            self.available, self.hint = False, T("no_onnx")
        elif self.path is None:
            self.available = False
            self.hint = T("no_model").format(path=os.path.join(MODEL_DIR, "lan.onnx"))
        else:
            self.available = True

    def _ensure(self):
        if self._sess is None:
            so = ort.SessionOptions()
            # leave cores for OpenCV: the full-resolution passes are the larger
            # half of this stage and they are the ones that scale with threads
            so.intra_op_num_threads = max((os.cpu_count() or 2) // 2, 1)
            self._sess = ort.InferenceSession(self.path, so, providers=onnx_providers())
            self._name = self._sess.get_inputs()[0].name
            self.bins = int(self._sess.get_outputs()[0].shape[1]) // 6
            self._wlut = self._build_wluts(self.bins)
        return self._sess

    @staticmethod
    def _build_wluts(bins):
        """Triangular membership over the luma axis, as 256-entry uint8 LUTs.
        They are forced to sum to exactly 255 at every input level - a rounding
        error here is a global brightness shift, which is very visible."""
        x = np.arange(256, dtype=np.float32) / 255.0
        if bins == 1:
            return [np.full(256, 255, np.uint8)]
        step = 1.0 / (bins - 1.0)
        w = np.stack([np.clip(1.0 - np.abs(x - i * step) / step, 0.0, 1.0)
                      for i in range(bins)])
        w = np.round(w / w.sum(axis=0, keepdims=True) * 255.0).astype(np.int32)
        w[np.argmax(w, axis=0), np.arange(256)] += 255 - w.sum(axis=0)
        return [np.clip(row, 0, 255).astype(np.uint8) for row in w]

    def _infer(self, frame):
        stride = max(1, int(round(1.0 / max(self.values["quality"], 0.25))))
        self._n += 1
        if self._coef is not None and self._n % stride:
            return self._coef
        small = npy(cv2.resize(frame, (self.IN_W, self.IN_H), interpolation=cv2.INTER_AREA))
        x = np.ascontiguousarray(small.transpose(2, 0, 1)[None].astype(np.float32) / 255.0)
        c = self._sess.run(None, {self._name: x})[0][0]
        # temporal EMA on the grid, not on pixels: illumination moves slowly and
        # an unsmoothed grid pumps visibly whenever the subject shifts
        self._coef = c if self._coef is None else self._coef + (c - self._coef) * 0.35
        return self._coef

    def process(self, frame):
        a = self.values["amount"]
        if a <= 0.01:
            return frame
        self._ensure()
        coef = self._infer(frame)
        h, w = frame_size(frame)
        n3 = self.bins * 3

        parts = []
        for d in range(self.bins):
            g = 1.0 + (coef[d * 3:d * 3 + 3] - 1.0) * a
            o = coef[n3 + d * 3:n3 + d * 3 + 3] * a
            gq = np.clip(g.transpose(1, 2, 0) * self.QG, 0, 255).astype(np.uint8)
            oq = np.clip(o.transpose(1, 2, 0) * 255.0 + self.OFF, 0, 255).astype(np.uint8)
            m = cv2.multiply(frame, up(gq, w, h), scale=1.0 / self.QG, dtype=cv2.CV_8U)
            # one pass for "add the offset map and remove its bias" - cv2.subtract
            # against a scalar takes 20 ms at 720p, addWeighted takes 0.8
            parts.append(cv2.addWeighted(m, 1.0, up(oq, w, h), 1.0, -float(self.OFF)))

        if self.bins == 1:
            return parts[0]

        guide = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        out = None
        for d, part in enumerate(parts):
            wm = cv2.cvtColor(cv2.LUT(guide, self._wlut[d]), cv2.COLOR_GRAY2BGR)
            piece = cv2.multiply(part, wm, scale=1.0 / 255.0, dtype=cv2.CV_8U)
            out = piece if out is None else cv2.add(out, piece)
        return out


class LocalTone(BaseFilter):
    """Model-free local tone mapping. This is the stage that makes a backlit
    frame work, and no global curve can replace it.

    A global gamma has one exposure for the whole frame, so lifting a face that
    sits in front of a window means lifting the window too. The clipping
    constraint in auto_exposure then correctly refuses, and the subject stays
    dark - measured on a real capture, the face came out at 95 against a 126
    target while 8% of the frame was already blown. That is not a tuning
    failure, it is the limit of what a single curve can do.

    So: estimate the illumination at 192x108, divide the target key by it, and
    multiply the frame by that gain map. Dark regions get lifted, bright regions
    do not, and the cost is fixed regardless of capture resolution - the same
    trick the learned stages use, without needing a trained model.

    Two details are the whole difference between this looking right and looking
    like a cheap HDR filter:

    - the illumination estimate is bilateral, not Gaussian. A plain blur across
      a head/window boundary averages the two, so the gain map ramps through the
      edge and leaves a grey halo tracing the subject.
    - the exponent is scaled by (1 - L^2), so the correction fades out in the
      highlights. Without it the whole frame is dragged toward mid grey, which
      is the flat, hazy look people describe as washed out.
    """
    name = "local_tone"
    label_key = "localtone"
    space = "ycrcb"
    # Default lowered from 0.6 to 0.1, and turned off by default in
    # build_filters() below - product decision (2026-09). Kept at full range
    # for anyone who wants backlit correction back on.
    params = {"amount": ("strength", 0.0, 1.0, 0.1),
              "detail": ("detail", 0.0, 1.0, 0.45),
              "key": ("key", 90, 165, 124)}
    SW, SH = 192, 108
    Q = 64.0                   # gain quantiser: max representable gain 255/64
    CG_LO, CG_HI = 0.75, 1.60  # chroma gain, bounded so its bias map stays uint8

    def process(self, frame):
        a, d = self.values["amount"], self.values["detail"]
        if a <= 0.02 and d <= 0.02:
            return frame
        h, w = frame_size(frame)
        y, cr, cb = frame
        small = npy(cv2.resize(y, (self.SW, self.SH), interpolation=cv2.INTER_AREA))
        L = cv2.bilateralFilter(small, 9, 45, 18).astype(np.float32) / 255.0
        L = np.clip(L, 0.05, 1.0)

        key = self.values["key"] / 255.0
        expo = (0.8 * a) * (1.0 - L * L)          # no correction in the highlights
        gain = np.clip((key / L) ** expo, 0.5, 2.8)

        gq = up(np.clip(gain * self.Q, 0, 255).astype(np.uint8), w, h)
        out = cv2.multiply(y, gq, scale=1.0 / self.Q, dtype=cv2.CV_8U)

        if d > 0.02:
            # local contrast: subtract the corrected illumination, which is
            # already in hand at thumbnail size. A full-resolution large-radius
            # blur would cost more than the rest of this stage put together.
            base = up(np.clip(L * 255.0 * gain, 0, 255).astype(np.uint8), w, h)
            # local contrast raises whatever is there, grain included, so it
            # steps back as the measured noise rises
            k = 0.55 * d * min(max(1.0 - (NOISE["residual"] - 4.0) / 18.0, 0.35), 1.0)
            boosted = cv2.addWeighted(out, 1.0 + k, base, -k, 0.0)
            # same halo control as Sharpen: this is an unsharp mask against a
            # smoothed illumination map, and on a high-contrast graphic edge
            # (gold braid on a blue backdrop, on-screen text) it rings the same
            # way. Clamp to the exposure-corrected frame's own 3x3 neighbourhood
            # before the boost, not after gain/postscale have moved it further.
            lo = cv2.erode(out, _HALO_K3)
            hi = cv2.dilate(out, _HALO_K3)
            out = cv2.min(cv2.max(boosted, lo), hi)

        # a luma-only gain desaturates: chroma stays put while luma rises, so the
        # correction quietly costs colour. Scale chroma by a blunted version of
        # the same map to keep it.
        if a > 0.02:
            # (c - 128) * g + 128 == c * g + 128 * (1 - g), and the second term
            # is a map too. Written that way it is one multiply and one
            # addWeighted per channel in uint8, instead of subtract/multiply/add
            # through CV_16S - the 16-bit intermediates were doubling the memory
            # traffic of the most expensive part of this stage.
            cg = np.clip(1.0 + (gain - 1.0) * 0.5, self.CG_LO, self.CG_HI)
            cq = up(np.clip(cg * self.Q, 0, 255).astype(np.uint8), w, h)
            bq = up(np.clip(128.0 * (1.0 - cg) + 128.0, 0, 255).astype(np.uint8), w, h)
            cr = cv2.addWeighted(cv2.multiply(cr, cq, scale=1.0 / self.Q,
                                              dtype=cv2.CV_8U), 1.0, bq, 1.0, -128.0)
            cb = cv2.addWeighted(cv2.multiply(cb, cq, scale=1.0 / self.Q,
                                              dtype=cv2.CV_8U), 1.0, bq, 1.0, -128.0)
        return [out, cr, cb]


class NeuralRestore(BaseFilter):
    """Learned restoration - denoise, deblock and upscale in one network.

    GPU only, and that is a hard requirement rather than a preference. A
    SRVGGNetCompact-class model (Real-ESRGAN general-x4v3 and friends) is about
    1.2M parameters running over every output pixel, which is three orders of
    magnitude more work than the 13K-parameter grid in neural_local - that one
    is cheap because it never touches full resolution, and this one is defined by
    touching it. On a CPU it is seconds per frame. On a discrete GPU at a 640x360
    input it is tens of milliseconds, which is a different conversation.

    Note what the input size means: the model is fed a *downscaled* frame and
    produces the full-resolution one. That is the point. A webcam's native 720p
    is soft and MJPG-blocked, INTER_AREA down to 360p suppresses the block
    structure, and a model trained on degraded inputs reconstructs from there.
    Feeding it native 720p and asking for 1440p would cost four times as much to
    produce detail that gets thrown away on the way back down.

    It also hallucinates. That is what these models do - they invent plausible
    texture, and on a face that is a different face. Keep the amount below 1.0
    and check it against the original view before trusting it on a call.

    Drop a model at ~/.webcam_enhancer/restore.onnx (or restore_fp16.onnx).
    Subsumes denoise, the hot pixel guard and sharpen, which it is doing itself.
    """
    name = "restore"
    label_key = "restore"
    params = {"amount": ("strength", 0.0, 1.0, 0.85)}
    IN_W, IN_H = 640, 360

    def __init__(self):
        super().__init__()
        self.path = None
        for cand in ("restore_fp16.onnx", "restore.onnx"):
            cp = os.path.join(MODEL_DIR, cand)
            if os.path.exists(cp):
                self.path = cp
                break
        self._sess = None
        self._fp16 = False
        if ort is None:
            self.available, self.hint = False, T("no_onnx")
        elif self.path is None:
            self.available = False
            self.hint = T("no_restore_model").format(
                path=os.path.join(MODEL_DIR, "restore.onnx"))
        else:
            self.available = True

    def _ensure(self):
        if self._sess is None:
            provs = onnx_providers()
            if provs[0] == "CPUExecutionProvider":
                # refuse rather than run: a CPU session here does not miss the
                # budget, it misses it by two orders of magnitude
                self.available = False
                self.enabled = False
                self.hint = T("no_gpu_ep")
                return None
            self._sess = ort.InferenceSession(self.path, ort.SessionOptions(),
                                              providers=provs)
            inp = self._sess.get_inputs()[0]
            self._name = inp.name
            if isinstance(inp.shape[2], int) and isinstance(inp.shape[3], int):
                self.IN_H, self.IN_W = inp.shape[2], inp.shape[3]
            self._fp16 = inp.type == "tensor(float16)"
        return self._sess

    def process(self, frame):
        a = self.values["amount"]
        if a <= 0.01:
            return frame
        sess = self._ensure()
        if sess is None:
            return frame
        h, w = frame_size(frame)
        small = npy(cv2.resize(frame, (self.IN_W, self.IN_H),
                               interpolation=cv2.INTER_AREA))
        x = cv2.cvtColor(small, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)[None]
        x = np.ascontiguousarray(
            x.astype(np.float16 if self._fp16 else np.float32) / 255.0)
        y = np.clip(sess.run(None, {self._name: x})[0][0]
                    .transpose(1, 2, 0).astype(np.float32), 0.0, 1.0)
        out = cv2.cvtColor((y * 255.0).astype(np.uint8), cv2.COLOR_RGB2BGR)
        interp = cv2.INTER_AREA if out.shape[1] > w else cv2.INTER_LINEAR
        out = to_gpu(cv2.resize(out, (w, h), interpolation=interp))
        return out if a >= 0.99 else cv2.addWeighted(frame, 1.0 - a, out, a, 0.0)


class MotionDeblur(BaseFilter):
    """얼굴 크롭 전용 초경량 모션 디블러.

    모델은 이 파일 안에 base64로 내장되어 있어 exe로 빌드해도 별도 파일 없이
    동작한다. ~/.webcam_enhancer/motion_deblur.onnx (또는 _fp16.onnx)를 직접
    넣어두면 그쪽이 우선 적용된다 - 재학습한 새 모델을 exe 재빌드 없이
    테스트할 때 씀.

    얼굴 크롭만 처리하고, 모션이 감지될 때만 추론을 돌린다(정지 상태 비용 0에
    가까움). GPU 전용인 이유는 NeuralRestore와 같다.
    """
    name = "motion_deblur"
    label_key = "motion_deblur"
    uses_face = True
    params = {
        "amount": ("strength", 0.0, 1.0, 0.0),
        "sensitivity": ("threshold", 0.0, 1.0, 0.35),
    }
    IN_W = IN_H = 128
    _EMBEDDED_B64 = "CAgSB3B5dG9yY2gaBjIuMTQuMDqIwgYKugEKBWlucHV0Cg5vbm54OjpDb252XzExNgoOb25ueDo6Q29udl8xMTcSGi9zdGVtL3N0ZW0uMC9Db252X291dHB1dF8wGhEvc3RlbS9zdGVtLjAvQ29udiIEQ29udioSCglkaWxhdGlvbnNAAUABoAEHKgwKBWdyb3VwGAGgAQIqFQoMa2VybmVsX3NoYXBlQANAA6ABByoRCgRwYWRzQAFAAUABQAGgAQcqEAoHc3RyaWRlc0ABQAGgAQcKUQoaL3N0ZW0vc3RlbS4wL0NvbnZfb3V0cHV0XzASGi9zdGVtL3N0ZW0uMi9SZWx1X291dHB1dF8wGhEvc3RlbS9zdGVtLjIvUmVsdSIEUmVsdQrRAQoaL3N0ZW0vc3RlbS4yL1JlbHVfb3V0cHV0XzAKEmJsb2Nrcy4wLmR3LndlaWdodBIhL2Jsb2Nrcy9ibG9ja3MuMC9kdy9Db252X291dHB1dF8wGhgvYmxvY2tzL2Jsb2Nrcy4wL2R3L0NvbnYiBENvbnYqEgoJZGlsYXRpb25zQAFAAaABByoMCgVncm91cBgwoAECKhUKDGtlcm5lbF9zaGFwZUADQAOgAQcqEQoEcGFkc0ABQAFAAUABoAEHKhAKB3N0cmlkZXNAAUABoAEHCuQBCiEvYmxvY2tzL2Jsb2Nrcy4wL2R3L0NvbnZfb3V0cHV0XzAKDm9ubng6OkNvbnZfMTE5Cg5vbm54OjpDb252XzEyMBIhL2Jsb2Nrcy9ibG9ja3MuMC9wdy9Db252X291dHB1dF8wGhgvYmxvY2tzL2Jsb2Nrcy4wL3B3L0NvbnYiBENvbnYqEgoJZGlsYXRpb25zQAFAAaABByoMCgVncm91cBgBoAECKhUKDGtlcm5lbF9zaGFwZUABQAGgAQcqEQoEcGFkc0AAQABAAEAAoAEHKhAKB3N0cmlkZXNAAUABoAEHCmgKIS9ibG9ja3MvYmxvY2tzLjAvcHcvQ29udl9vdXRwdXRfMBIiL2Jsb2Nrcy9ibG9ja3MuMC9hY3QvUmVsdV9vdXRwdXRfMBoZL2Jsb2Nrcy9ibG9ja3MuMC9hY3QvUmVsdSIEUmVsdQp6Chovc3RlbS9zdGVtLjIvUmVsdV9vdXRwdXRfMAoiL2Jsb2Nrcy9ibG9ja3MuMC9hY3QvUmVsdV9vdXRwdXRfMBIdL2Jsb2Nrcy9ibG9ja3MuMC9BZGRfb3V0cHV0XzAaFC9ibG9ja3MvYmxvY2tzLjAvQWRkIgNBZGQK1AEKHS9ibG9ja3MvYmxvY2tzLjAvQWRkX291dHB1dF8wChJibG9ja3MuMS5kdy53ZWlnaHQSIS9ibG9ja3MvYmxvY2tzLjEvZHcvQ29udl9vdXRwdXRfMBoYL2Jsb2Nrcy9ibG9ja3MuMS9kdy9Db252IgRDb252KhIKCWRpbGF0aW9uc0ABQAGgAQcqDAoFZ3JvdXAYMKABAioVCgxrZXJuZWxfc2hhcGVAA0ADoAEHKhEKBHBhZHNAAUABQAFAAaABByoQCgdzdHJpZGVzQAFAAaABBwrkAQohL2Jsb2Nrcy9ibG9ja3MuMS9kdy9Db252X291dHB1dF8wCg5vbm54OjpDb252XzEyMgoOb25ueDo6Q29udl8xMjMSIS9ibG9ja3MvYmxvY2tzLjEvcHcvQ29udl9vdXRwdXRfMBoYL2Jsb2Nrcy9ibG9ja3MuMS9wdy9Db252IgRDb252KhIKCWRpbGF0aW9uc0ABQAGgAQcqDAoFZ3JvdXAYAaABAioVCgxrZXJuZWxfc2hhcGVAAUABoAEHKhEKBHBhZHNAAEAAQABAAKABByoQCgdzdHJpZGVzQAFAAaABBwpoCiEvYmxvY2tzL2Jsb2Nrcy4xL3B3L0NvbnZfb3V0cHV0XzASIi9ibG9ja3MvYmxvY2tzLjEvYWN0L1JlbHVfb3V0cHV0XzAaGS9ibG9ja3MvYmxvY2tzLjEvYWN0L1JlbHUiBFJlbHUKfQodL2Jsb2Nrcy9ibG9ja3MuMC9BZGRfb3V0cHV0XzAKIi9ibG9ja3MvYmxvY2tzLjEvYWN0L1JlbHVfb3V0cHV0XzASHS9ibG9ja3MvYmxvY2tzLjEvQWRkX291dHB1dF8wGhQvYmxvY2tzL2Jsb2Nrcy4xL0FkZCIDQWRkCtQBCh0vYmxvY2tzL2Jsb2Nrcy4xL0FkZF9vdXRwdXRfMAoSYmxvY2tzLjIuZHcud2VpZ2h0EiEvYmxvY2tzL2Jsb2Nrcy4yL2R3L0NvbnZfb3V0cHV0XzAaGC9ibG9ja3MvYmxvY2tzLjIvZHcvQ29udiIEQ29udioSCglkaWxhdGlvbnNAAUABoAEHKgwKBWdyb3VwGDCgAQIqFQoMa2VybmVsX3NoYXBlQANAA6ABByoRCgRwYWRzQAFAAUABQAGgAQcqEAoHc3RyaWRlc0ABQAGgAQcK5AEKIS9ibG9ja3MvYmxvY2tzLjIvZHcvQ29udl9vdXRwdXRfMAoOb25ueDo6Q29udl8xMjUKDm9ubng6OkNvbnZfMTI2EiEvYmxvY2tzL2Jsb2Nrcy4yL3B3L0NvbnZfb3V0cHV0XzAaGC9ibG9ja3MvYmxvY2tzLjIvcHcvQ29udiIEQ29udioSCglkaWxhdGlvbnNAAUABoAEHKgwKBWdyb3VwGAGgAQIqFQoMa2VybmVsX3NoYXBlQAFAAaABByoRCgRwYWRzQABAAEAAQACgAQcqEAoHc3RyaWRlc0ABQAGgAQcKaAohL2Jsb2Nrcy9ibG9ja3MuMi9wdy9Db252X291dHB1dF8wEiIvYmxvY2tzL2Jsb2Nrcy4yL2FjdC9SZWx1X291dHB1dF8wGhkvYmxvY2tzL2Jsb2Nrcy4yL2FjdC9SZWx1IgRSZWx1Cn0KHS9ibG9ja3MvYmxvY2tzLjEvQWRkX291dHB1dF8wCiIvYmxvY2tzL2Jsb2Nrcy4yL2FjdC9SZWx1X291dHB1dF8wEh0vYmxvY2tzL2Jsb2Nrcy4yL0FkZF9vdXRwdXRfMBoUL2Jsb2Nrcy9ibG9ja3MuMi9BZGQiA0FkZArUAQodL2Jsb2Nrcy9ibG9ja3MuMi9BZGRfb3V0cHV0XzAKEmJsb2Nrcy4zLmR3LndlaWdodBIhL2Jsb2Nrcy9ibG9ja3MuMy9kdy9Db252X291dHB1dF8wGhgvYmxvY2tzL2Jsb2Nrcy4zL2R3L0NvbnYiBENvbnYqEgoJZGlsYXRpb25zQAFAAaABByoMCgVncm91cBgwoAECKhUKDGtlcm5lbF9zaGFwZUADQAOgAQcqEQoEcGFkc0ABQAFAAUABoAEHKhAKB3N0cmlkZXNAAUABoAEHCuQBCiEvYmxvY2tzL2Jsb2Nrcy4zL2R3L0NvbnZfb3V0cHV0XzAKDm9ubng6OkNvbnZfMTI4Cg5vbm54OjpDb252XzEyORIhL2Jsb2Nrcy9ibG9ja3MuMy9wdy9Db252X291dHB1dF8wGhgvYmxvY2tzL2Jsb2Nrcy4zL3B3L0NvbnYiBENvbnYqEgoJZGlsYXRpb25zQAFAAaABByoMCgVncm91cBgBoAECKhUKDGtlcm5lbF9zaGFwZUABQAGgAQcqEQoEcGFkc0AAQABAAEAAoAEHKhAKB3N0cmlkZXNAAUABoAEHCmgKIS9ibG9ja3MvYmxvY2tzLjMvcHcvQ29udl9vdXRwdXRfMBIiL2Jsb2Nrcy9ibG9ja3MuMy9hY3QvUmVsdV9vdXRwdXRfMBoZL2Jsb2Nrcy9ibG9ja3MuMy9hY3QvUmVsdSIEUmVsdQp9Ch0vYmxvY2tzL2Jsb2Nrcy4yL0FkZF9vdXRwdXRfMAoiL2Jsb2Nrcy9ibG9ja3MuMy9hY3QvUmVsdV9vdXRwdXRfMBIdL2Jsb2Nrcy9ibG9ja3MuMy9BZGRfb3V0cHV0XzAaFC9ibG9ja3MvYmxvY2tzLjMvQWRkIgNBZGQK1AEKHS9ibG9ja3MvYmxvY2tzLjMvQWRkX291dHB1dF8wChJibG9ja3MuNC5kdy53ZWlnaHQSIS9ibG9ja3MvYmxvY2tzLjQvZHcvQ29udl9vdXRwdXRfMBoYL2Jsb2Nrcy9ibG9ja3MuNC9kdy9Db252IgRDb252KhIKCWRpbGF0aW9uc0ABQAGgAQcqDAoFZ3JvdXAYMKABAioVCgxrZXJuZWxfc2hhcGVAA0ADoAEHKhEKBHBhZHNAAUABQAFAAaABByoQCgdzdHJpZGVzQAFAAaABBwrkAQohL2Jsb2Nrcy9ibG9ja3MuNC9kdy9Db252X291dHB1dF8wCg5vbm54OjpDb252XzEzMQoOb25ueDo6Q29udl8xMzISIS9ibG9ja3MvYmxvY2tzLjQvcHcvQ29udl9vdXRwdXRfMBoYL2Jsb2Nrcy9ibG9ja3MuNC9wdy9Db252IgRDb252KhIKCWRpbGF0aW9uc0ABQAGgAQcqDAoFZ3JvdXAYAaABAioVCgxrZXJuZWxfc2hhcGVAAUABoAEHKhEKBHBhZHNAAEAAQABAAKABByoQCgdzdHJpZGVzQAFAAaABBwpoCiEvYmxvY2tzL2Jsb2Nrcy40L3B3L0NvbnZfb3V0cHV0XzASIi9ibG9ja3MvYmxvY2tzLjQvYWN0L1JlbHVfb3V0cHV0XzAaGS9ibG9ja3MvYmxvY2tzLjQvYWN0L1JlbHUiBFJlbHUKfQodL2Jsb2Nrcy9ibG9ja3MuMy9BZGRfb3V0cHV0XzAKIi9ibG9ja3MvYmxvY2tzLjQvYWN0L1JlbHVfb3V0cHV0XzASHS9ibG9ja3MvYmxvY2tzLjQvQWRkX291dHB1dF8wGhQvYmxvY2tzL2Jsb2Nrcy40L0FkZCIDQWRkCtQBCh0vYmxvY2tzL2Jsb2Nrcy40L0FkZF9vdXRwdXRfMAoSYmxvY2tzLjUuZHcud2VpZ2h0EiEvYmxvY2tzL2Jsb2Nrcy41L2R3L0NvbnZfb3V0cHV0XzAaGC9ibG9ja3MvYmxvY2tzLjUvZHcvQ29udiIEQ29udioSCglkaWxhdGlvbnNAAUABoAEHKgwKBWdyb3VwGDCgAQIqFQoMa2VybmVsX3NoYXBlQANAA6ABByoRCgRwYWRzQAFAAUABQAGgAQcqEAoHc3RyaWRlc0ABQAGgAQcK5AEKIS9ibG9ja3MvYmxvY2tzLjUvZHcvQ29udl9vdXRwdXRfMAoOb25ueDo6Q29udl8xMzQKDm9ubng6OkNvbnZfMTM1EiEvYmxvY2tzL2Jsb2Nrcy41L3B3L0NvbnZfb3V0cHV0XzAaGC9ibG9ja3MvYmxvY2tzLjUvcHcvQ29udiIEQ29udioSCglkaWxhdGlvbnNAAUABoAEHKgwKBWdyb3VwGAGgAQIqFQoMa2VybmVsX3NoYXBlQAFAAaABByoRCgRwYWRzQABAAEAAQACgAQcqEAoHc3RyaWRlc0ABQAGgAQcKaAohL2Jsb2Nrcy9ibG9ja3MuNS9wdy9Db252X291dHB1dF8wEiIvYmxvY2tzL2Jsb2Nrcy41L2FjdC9SZWx1X291dHB1dF8wGhkvYmxvY2tzL2Jsb2Nrcy41L2FjdC9SZWx1IgRSZWx1Cn0KHS9ibG9ja3MvYmxvY2tzLjQvQWRkX291dHB1dF8wCiIvYmxvY2tzL2Jsb2Nrcy41L2FjdC9SZWx1X291dHB1dF8wEh0vYmxvY2tzL2Jsb2Nrcy41L0FkZF9vdXRwdXRfMBoUL2Jsb2Nrcy9ibG9ja3MuNS9BZGQiA0FkZArUAQodL2Jsb2Nrcy9ibG9ja3MuNS9BZGRfb3V0cHV0XzAKEmJsb2Nrcy42LmR3LndlaWdodBIhL2Jsb2Nrcy9ibG9ja3MuNi9kdy9Db252X291dHB1dF8wGhgvYmxvY2tzL2Jsb2Nrcy42L2R3L0NvbnYiBENvbnYqEgoJZGlsYXRpb25zQAFAAaABByoMCgVncm91cBgwoAECKhUKDGtlcm5lbF9zaGFwZUADQAOgAQcqEQoEcGFkc0ABQAFAAUABoAEHKhAKB3N0cmlkZXNAAUABoAEHCuQBCiEvYmxvY2tzL2Jsb2Nrcy42L2R3L0NvbnZfb3V0cHV0XzAKDm9ubng6OkNvbnZfMTM3Cg5vbm54OjpDb252XzEzOBIhL2Jsb2Nrcy9ibG9ja3MuNi9wdy9Db252X291dHB1dF8wGhgvYmxvY2tzL2Jsb2Nrcy42L3B3L0NvbnYiBENvbnYqEgoJZGlsYXRpb25zQAFAAaABByoMCgVncm91cBgBoAECKhUKDGtlcm5lbF9zaGFwZUABQAGgAQcqEQoEcGFkc0AAQABAAEAAoAEHKhAKB3N0cmlkZXNAAUABoAEHCmgKIS9ibG9ja3MvYmxvY2tzLjYvcHcvQ29udl9vdXRwdXRfMBIiL2Jsb2Nrcy9ibG9ja3MuNi9hY3QvUmVsdV9vdXRwdXRfMBoZL2Jsb2Nrcy9ibG9ja3MuNi9hY3QvUmVsdSIEUmVsdQp9Ch0vYmxvY2tzL2Jsb2Nrcy41L0FkZF9vdXRwdXRfMAoiL2Jsb2Nrcy9ibG9ja3MuNi9hY3QvUmVsdV9vdXRwdXRfMBIdL2Jsb2Nrcy9ibG9ja3MuNi9BZGRfb3V0cHV0XzAaFC9ibG9ja3MvYmxvY2tzLjYvQWRkIgNBZGQK1AEKHS9ibG9ja3MvYmxvY2tzLjYvQWRkX291dHB1dF8wChJibG9ja3MuNy5kdy53ZWlnaHQSIS9ibG9ja3MvYmxvY2tzLjcvZHcvQ29udl9vdXRwdXRfMBoYL2Jsb2Nrcy9ibG9ja3MuNy9kdy9Db252IgRDb252KhIKCWRpbGF0aW9uc0ABQAGgAQcqDAoFZ3JvdXAYMKABAioVCgxrZXJuZWxfc2hhcGVAA0ADoAEHKhEKBHBhZHNAAUABQAFAAaABByoQCgdzdHJpZGVzQAFAAaABBwrkAQohL2Jsb2Nrcy9ibG9ja3MuNy9kdy9Db252X291dHB1dF8wCg5vbm54OjpDb252XzE0MAoOb25ueDo6Q29udl8xNDESIS9ibG9ja3MvYmxvY2tzLjcvcHcvQ29udl9vdXRwdXRfMBoYL2Jsb2Nrcy9ibG9ja3MuNy9wdy9Db252IgRDb252KhIKCWRpbGF0aW9uc0ABQAGgAQcqDAoFZ3JvdXAYAaABAioVCgxrZXJuZWxfc2hhcGVAAUABoAEHKhEKBHBhZHNAAEAAQABAAKABByoQCgdzdHJpZGVzQAFAAaABBwpoCiEvYmxvY2tzL2Jsb2Nrcy43L3B3L0NvbnZfb3V0cHV0XzASIi9ibG9ja3MvYmxvY2tzLjcvYWN0L1JlbHVfb3V0cHV0XzAaGS9ibG9ja3MvYmxvY2tzLjcvYWN0L1JlbHUiBFJlbHUKfQodL2Jsb2Nrcy9ibG9ja3MuNi9BZGRfb3V0cHV0XzAKIi9ibG9ja3MvYmxvY2tzLjcvYWN0L1JlbHVfb3V0cHV0XzASHS9ibG9ja3MvYmxvY2tzLjcvQWRkX291dHB1dF8wGhQvYmxvY2tzL2Jsb2Nrcy43L0FkZCIDQWRkCrwBCh0vYmxvY2tzL2Jsb2Nrcy43L0FkZF9vdXRwdXRfMAoLaGVhZC53ZWlnaHQKCWhlYWQuYmlhcxITL2hlYWQvQ29udl9vdXRwdXRfMBoKL2hlYWQvQ29udiIEQ29udioSCglkaWxhdGlvbnNAAUABoAEHKgwKBWdyb3VwGAGgAQIqFQoMa2VybmVsX3NoYXBlQANAA6ABByoRCgRwYWRzQAFAAUABQAGgAQcqEAoHc3RyaWRlc0ABQAGgAQcKNgoFaW5wdXQKEy9oZWFkL0NvbnZfb3V0cHV0XzASDS9BZGRfb3V0cHV0XzAaBC9BZGQiA0FkZAo/EhIvQ29uc3RhbnRfb3V0cHV0XzAaCS9Db25zdGFudCIIQ29uc3RhbnQqFAoFdmFsdWUqCBABSgQAAAAAoAEECkMSFC9Db25zdGFudF8xX291dHB1dF8wGgsvQ29uc3RhbnRfMSIIQ29uc3RhbnQqFAoFdmFsdWUqCBABSgQAAIA/oAEECk4KDS9BZGRfb3V0cHV0XzAKEi9Db25zdGFudF9vdXRwdXRfMAoUL0NvbnN0YW50XzFfb3V0cHV0XzASBm91dHB1dBoFL0NsaXAiBENsaXASCm1haW5fZ3JhcGgq4Q0IMAgBCAMIAxABQhJibG9ja3MuMC5kdy53ZWlnaHRKwA12BiE+dkzrPCyFZj6Jjyo+n3t7vkuLgT3BC5Q+t0v5PBvebj6AAmC+ZiyHPji0iL6HFxS+A1KTPfZkDT5HVIc9FhVhPmOmOL70dtA9wsnEvYc6mLy4ZYq+4PIdvo3RRT4GoGa+yY79PR34rr6k1++9ytKZPVYF173pVMA91HJBPvHYXj7SZaq+/idiPfsbXT6dE6U+lW5jPkqWH77o7pU+f2d8vstbFT51yNQ9+rgTvsxlKL7EsO+9d4cgvJ01nD4sM/S9SsmCvhoao75iJ6A+LjWOPqIukb0ESFY+ojwbvVJYlj7yUTO+ncUnPstvJ7zVtgY+OpaVPiqspj7GGWc9p32Wvg93Lz0Hm5c+Z12OvmdPjL6wXay9QsTBPXvE/b2LRgM9gT6TPvfGhr7T6b49cUquPAlsnz0HBf891qBOPl+nnr4nLIi+SUUevMkEMb6zlp4+A1kfPugcjz3nlqG+aX6sPVUeHD4YPSg96P1Wvs/U770FoSa+i6mJvh9C+z1BXN89ROSFPqHlv701c4o+lRxJvpVGjL7PVHq+ja7lPa8QL74Arn8+GW1HPmt2oz70zLo9yw+BvvcJq7yQCfa9GFYNvY9Lf74WNp4+sFZ8Ph3tZL5kGzI+BXAFvkd0Mz7EFaE9LEAuPtaSQb4TqmY+qgOMvkXYfrzlcKY+ds8hvi/y9L3FSYA+4+UpPoG8LL7UwhM+0k4svgc/Xr2vNny+tfUivkHAJL6LF6G+BlGYvuGFTr7XQA2+aEO6PURed73qBQA+wykyPYzAbr7A/l8+JCeiveObED6xYxO+0AO/O6BXXb5jz5E+UZprPYDQNL483uU9GCCEvmFWy72eXFw+zggpPuOJoj6ConE+vrJ4vl7jp7xYfPg7viIivQ7KA76+F0C9WcIdvpiOab6FSyU+KLc0vjnBj74CAaK++K+APtqtOb4rD5g+OKFGPmuPkD5VmW+9FQYFvpk9Trxblv09nzFAPotI/LxjnLe9gXyKPt9BQj4Jtam9kteyvIeCoL4tdoy+keaTvt/Fkb7ArIw+B2ievkSDQb4JPbS9wz1YvoAz4L3S/yw9+J2GvbL3pr5b84w+TGL6PRhAmz4bYgA9tHRsPk7HnL7aMEy+TiyQPsMowb0TpxE+fnIKPQknvr04Ma69oO/6PEdHLz5SdBc8yDYCvFYPc758+rc8bZ0GPhvVbr7nm/28aFGmvYx+2D2WrOU9rsVhvQN1Cr7RcQc94cQAPThGQT6L8N+9ZIKqPcO2m7zrACy8k/FRvkWZk77apx09mThSvWdaoT4z3Ju90IuXvT01fz3lo5q+qOq3u1JyEb7/NAW+Knghvl46lr5b2Ls986F9venGXD7wxpM9TcSrPt8LkD4Pae+9yUdVPr63kL5VrEC+xYZBvK+Gez7XvHm+QJo8PhRYVb70ZiI+2ITSPVAm2L2k5828fc2jPZ+fM745moE+mdGVPj29fr5OMAG+NKmSveVDFz4J8JA+zOqpvVU2CT51caI+oMt5vpe/cT6Iw4U+2+mvvriV/T2rxIs+3OMnvYVkHD52Y04+9jSQPhGspj5LXg49JJM8vqgL8j2/6Zc+BU6Rvm7zdT323Iw+ph5bPg5WaD5TvQE+8+U8vmRVZD7GE8w9xOUFvnzPiL3IXEO+KVOePame3r1po649BGFQPm7TpL0Oeo8+XgUnPjaa8z18U5A9R1CmvrYMAL77lQc+2+yNvkJrNb5tvDe+dD0LPY/6kb7+xhK+0otSPpGoFD6oLls+jeJgvv5+HL2raD6+bCWoPdWskL6uswa+OVlsvucTpzsorTy+AkbiPQjgr74dKJQ+U3k5vq0zWD7MYTk+rgrCPajRZz6x+FY9C8u2vQSapr7awPC8F70OPipY371Eb5U+VxYjPviOsTz1IRQ+YAw0PYIVJryrS4u8GQGNPuGDmj6ZfWY+FDyHPN+a8TqvBo6+iI8jvrKCkT1rFpU+HrU5PgPkoj5Q4U2+SxolvkI4tDyzRqy+XGFXvbqBfr7WV0w+PXQjPgGsjL6XaGu+V0WXPnUMj75rZYk+d6hkPulcYT5bLJI+w+UNvhTyP73v2T8+4opjPQsKlT6h1Ss+cQhoPjAXir4xYts9w01HPlhqjr1E65U+dW9uvTtmIbxiK0q+3ORqPhTxSr5Sv5c+hUQ+Pl/LSb2sgmQ+LIK1vcu1bb4Raha+pPxkvp5NFj1g/2E+X72evdvTYj16f1I+dcvpvEKdnb1AdXy+7KmXvuMBJb7MFn+988KXPkzd4D0GsIG+L14CPcGiLz69qrM9HQFVvhp5Ib4q4Q0IMAgBCAMIAxABQhJibG9ja3MuMS5kdy53ZWlnaHRKwA05s9s9iduQPuBdfb6BfCQ+knesviMOgT7/T6O+sMVfvpmNCb54ncu9xEUovu+LMbx8by++VK6LPvS2Ab4a2kM+lSlLvvK1CT4BYRe++SVfvvLji777/eM64g2MPoBiVr5ExiK+GjGgPo5rMT1qwAM+MzdlPtmDg74wz3w+VwyIvubKgr7KOA4+fX5TPux2J74B4FW+aqWmvhBfer1JJo8+gASOvrcnVj4Cbom+y2GLvk9dr706mSa+1i+qPDgRi76LEHc+u7KBPjdAEj5oKAG+B16wPQqt8L1mEZk+NKVOviw5o73NOCa9Hld+vn2hnz0L+IS9qoOBPnWYbb5UDMU7W77bvNviZL1cEp2+P0wEPbEi9Dx2FZs9ntIHPaU1fz3wVXE+NhsBvsF9ur2vyQ2+/bp0Psasi7oBSUo8vFmcPj7zjL7FMXY+vy0cPrrM3b1SEmw7UfM7PlNjab7pwJS+LyA3vkjPKb71GAI+jAUdvUrKJD6MSb698XEiPGWVmj593A2+C69vPvXnnL65pKu+KMoFvhyYrr5zQ+s95/DwPR1YWT5zMBy+v+1jvjjzMb6fzUu+dKr1PeG4ej6TfYO+8h97vgFxY75ZKEU+RRN1PUJLpr5N7om8aA3CPYPk0D0MoEg+SEJcvT515T1eNYi+GZZ5vm/8R70XaoI8nzWuPllKhj4NqIu+Hx90Prs/K7x/H6q9T6OUvbxGiz5j3pG+nVgePpoi47zcdIc++/JYvr7per4bVFA8wTubPiVGib7MW0o+S/B1vdZvqb7UXdM9NtCrvrI8JD73hgY+1CRXvlW+or5QtpQ+dxPCvRKPZD5SCaQ+ulikvvAm1z3SnRc+NNYnvlgluD0fr6I+uXnCvXHfTr7DbT49WqiyvLJFXT5TJ5K+AN+CPgTViD7b5oo+6P2Cvt+0PT3ysj2+Nl6tvSyFqboe422+LUSaPrs8oT1Ybh4+zKROPTJfEb6xcW69MdGIPuTglTz2lqK9RQI7vojBdL1Hmjy+xdznPfppSj6keZO+hO+jPlGPkr49Vtg7ZSaTvg2Zkj4uDpQ+LKMMPpY0O75B9l++vbRYPqAsAL6kIQK+J7x1vBNoqjsgHEe9axmjPcwyQL4zzTA++ttYvrx9Vj2QvpY9TU+RPtiReD15g568/BG1PGdXnr0QIBS9Ni5cPoeru710Tuw9Rf0OPuqxK70O55I+4nxmvu0aY771+mu+rsE+vvcnkj6FiZc+D3Y7vphAlr1JZXg9RTJWPeitS76NYda96byLvnwIo75ZMMy90RRkvkgIE73+l9G9on6OvuQRiT4eJA2+cUUmPu3FEDycoyQ+DwsUvRQYyb2r92e+wRXIPdomnz6/0lq+LHQ7vjjPD77LgKE+dXfLPeA4p77Papy+XsKvPE98NT58kUo+dMGSvuIfkL7ntxC9X09BvXKuIz0Ap7I8mwKwvVgng77n6pQ+biQIPjIPoj4JG+Q8J1mjPfAXmL65yPi9vbE3PtlQt712IJC94W9TPgP6iT6i4T+9rWCjPq+Cjb4dp628CjCdPljrub2cIqY908FKPnqFgb1pOoM+uZDivRwRZz1l/qE+/QNevJOCdT7B7UC+ymwKPlMlor22Il2+yMdOPjcU7jxyhkQ+sYIlvuRUnj3gbhA+p6FvvmyRP7od2H++GVOWvStYZj4Zx8A9m+CjvlLsCr2Cciy+es6CPr4ubL4Vxs89YK9vvgJagT7wLkK+j+FFvgVdP74AiHi+HMdPPMXjML47ZBi+JNT/PPW8Mz4WS6Y+9T7WPBPhLz5XqF6+MIqVPNdFCT4lQei9aq2IvW2SYb2uVQE9gdADvnooGb4uS1A+4ckKvhio9jtIwxM+YoodPtaRzz0RPsQ9rwpLPkDdpD500QC+glM+vprsj74W7pu9VUNyvoIfEj4JQI4+FLZCPsoegD0sh389BoQ9PajhjL6cpY6+MMGcPcyGZz7EspA+iYN/POgOBD5mZ869ui6NvYkH7r2BoEk+wLXBu9ObpL4/5Me9yrTbvZPfgz5PtSQ+r8SDPj8eDD46U4q9xVpOvp+oo77uk/K9U4WBvvRAFDxI1dM9ybQgPi1mUz5MUJa70xChPFXZBD1f2Sy+J37yPfV+2j01ZMi9QDvSvf8uUzvZeNO9LWB9vAAXVL4l53o+zqtFvqshpz7kRGu+vT8yPTO7lL00LZ2+1MdxPvmZ0b08P5c7CY2YvlnPDj6Xwlo+uOpwPlu/0r0cLxC+RSHQvR8abL6+15U+3lBcPrElfr3Jiu48dELivdu3ZL510QY8qjU3vkTZqLsq4Q0IMAgBCAMIAxABQhJibG9ja3MuMi5kdy53ZWlnaHRKwA32ta29z4iIPiUsvj0mq/E9477bPS6+8b18NYU+VEGqPIpgbz5mYGU+04YkvhfIl730hkA+fHozPnUzMD1WeaU+9FFWvpZaCD1c/I++wd/bvWscO75S/4C9JEG9vYW42j08KpA+P5A/vkLtRr6lYGo+8ONWvhvZD74BNYa+o2OMPcglaj4Abxc+g8i0vcA1iL4ZXao8+O+LvisXuj05jmQ9mg5tvQb1oL4Cnzm+0qCGviTFoj0tz58+NiGjPAc+Wj4loEI9c7IdPGPzDr7Ma4E7g727vaDKjT6Xke07Pd6MPsz4jr72VpM+4+NIvjQlgL79E5A+irIEvjhmSz69Wj8+at11Pqvlob7imjI+GngXviwIkb61u5k+0jN9PoaSB76sepU+b5bzvYISg74ZaK2+EowSvsr9nr5i516+JNuKvraicb7Z7ke+szMKPYDUHT4Wgkq58XrxvUMslj5l+4++wOEDvl0pDz46Ag8+4HQ7vpllVb7k3Ua+S8VZPlIHRj5LvVe+5/KsPmbqaT2P+aE+0LilvbmD/T2aG6C+/yynviuFWL4KhYc9f/+MPu0CtTwbUbk8lILoubC5PL7KP/E9XxuQvmIWmr5SlTs+9vGTvsr0jb4oXiS+0/KxPdJvrz3g7cw9Va6duw9xQz4ep2i80m4HPk4U1r2/DIa+qKjLPckCbb3oAXM8Gq+WPT/20b13ueY9AmCFPul1ST4UNJC+Lp5ePhrbQLtJMas9bNZIvpnbCj6gwym+axafvcvNGr5ZiKW9zVCPPg190b07/GQ+HySevvT7ibw5HVu9IHQQPl7rjT6udC+8LtzrvHH7MT4e4pi+SWqBPle0Wj5QtDC+cNF8Pm74oT74U6I9UTZnvorY7L2R4qO71v+mPuhrWz7pP2k+E6+Xvpo/F76BWns9v2B7vqm5qT01Q0w+FcgKPrerkz3jZ6g+swV7PbSLDT64m6u+gI9qvjVKib6IWPE9E9Z1Pvlmp70MGSc+sdICPIGEmz19c5y+/LqkvMJ40b0Iw4s+5j0BPtPidL1HiX4+h4+VPn38aL66kJG6qxabPgsOrz6gTXC+nrVXvjU+Jz3NXQM+zGRCvlBSCL6dsg0+/hSAvlzIzb1TnIk+/pM3vsC9cL5GWGm826wrPaXHkLxBN4e+zAy/vT51Mj5VrF+91f2yPRFxmb64ucY9yP5mPgNVfb5jmIq+Tm6ePk/lPT3d5KM+qbikvtEtKj5KOjU9g+aWPl9dbz5gr6Y+kD6Dvp7UUb47UiO84lgNPdA/rD6yAgy+0OmHPvHZLT4LRI4+cLsQPl8For7nm7U9uVrtvVBflb7Xaoy9Y1XpvbjyxL18Uoa7QD2uvQeG2ryHQR6+ptaGvkTHOr73oYE+JIyYPlfGxT1eaRu+ZT5vvuBDqj5A7Im+VlODPQqSsb2mL6W+cnA9vjIIh778Jdg9btmLvhWOOz7Xcps+DHkTviyVBr01kG68I42iPkM5GD7GaYs9wqURvoR9az6Ij7G9qWU1PpN0pb4U8aq9ki9uvl1Uhb3wtfI9QySFPn48Sb4KQ6o+m+UWPiMCOb4AQ0C+7VOVvnwyEz6ID56+eHkcPuP/Sj7YfV69RW5hPCL8zTtJvpw+CEyHvu5a3r0IUck9SVw9PupYsLyUpXs+DR9XPghml7697gu+Y62OvqODCb7xCZm+b7Z4voSePr5ACI0+q9eoPq1pV74tGdI9oyASvPdlqz5RjDS9os6SPrfDrT10W5u+LFfvPeckmr5ziWw+s+9sPqcsgz7xAQS7qDecPhko9D3+bRM+ANzPvX2GkT0ci6S92ooXvs8r4j1QXry9ZvDbvWOAoL4IerI93nzfvdD6kz4vEQI9V/uzPYibkj7yYOQ7DBpYvqCd7D1rbKu84rSiPmVppz2mbCS+8HClviWFnz4+K+k8dFOMPoUyAT6UwoM+UTdXPgxqpr7ZlaG9v8+LPt3Peb2mMGo8HHBLvsDXpr7CYFo+b3WJvsOfMr3Q/bQ9OnB0vjavmL0Drag+jqidvXlWBL4eswa9fcOgPqI6KD7HIg29xJR9vlI9NL0VlKM+caygPty9lD1TlXQ+FN5yPsNfoD3jwp0+IZiLviOldz5CUJM+/X+dPjYk6b2X6Iq+qFOKPo9RH77D33O+mzDxvSBvcr6SNQ0+fY1Bvjg0e74WDjS+3sN3vtzDZD4uPo4+nrizvdwelz6TFX8+ds1XPtEQaz6szDo+u/eGPpBGjr0MkXQ+NwmUvgPyLD7bxpc+Rx1+vnwEnT1EMBO+CIxJPiL7mT7bZ4s+wxRZPpFfnL4q4Q0IMAgBCAMIAxABQhJibG9ja3MuMy5kdy53ZWlnaHRKwA38cqQ+8PKBPouvVz40M3a9OdNOPkyg2T0er5e+dLuEPojtkr5Lfaa7rbCGPk1fo77ZNIe9v4+QPpvSi76mBcE9yGCoPkBF071pR/q9HJpBPo+Ll773brM9m1qxPj1LgL0onvq9M8YcPe0MUD6Oal0+c0VFvo7/jr3+isq9SWJRPjxgWL6oCZu+02NMPU6KAzwMPgo+sOpsvvcKcL7BAYa+ntp3Pityqb2/9z++MMIUPEQOdD6Lt4I+6HZTvmlGDj2eXD6+YE5bvVnBDD5A0pM+9mOZvTSunz5iIQ++VutaPQTEFT47YdU9HcBcvseB2T1dedO9Sfc4vtSCY763rqq9CnMxPjw40Tz0MGi+rWBgPrNz272KAaA7sy1avjoSjr7duBI90TeJvoMZaT3ZT5K+XWiBvqAH7b07ZgO+FVnZvbJuAb4dPZ++4hCRvqtwgj5cc5s9aelLvtwlcT7Dmd88k4E8vl1jjz3j66Y+gNM0vD/4dT6famw+9P+KPUwjor09yQa96RuCvppL4b19JlU97nx9vkwIpL4mkpw+VCSwPPLVFL5+5si9MaCQvoRmsb0fefK8DCJyPSt6XD5lVpq+7Uw8PhEubj6jthE+O5dovmsFgb4BFJ8+FlUPPOvHEz5vwJS+U2YIvSlHVj66EFs+ntFmPnwGH759ABO8kWc5ve7kej7bmTA+APOCPnZlGrw1K8i9GommvgmRhr7G5De+vvRBvQdPUL25BL48NkqAvtdKyT0si8c9AeaKvsr6uj2tS1G+jIwHPjMVjz6u85+9hZJoPXVltb2XjqO+5HdvPfpVpj0hpie+n6ucPpMXcz7XcS0+8a/jPDTRBz4ivW8+rDnovIy3MD5W10y+jsx9vC7g3T2SxMS8ebJRPYy6ir5YiLI9V37rvcu/6z14vUy+GIVHPtaYUr5mjIm++ESePvlBRj5df0c+LbARPB0OpL4EUTk+PjVVPqTBIr0hGq08M7V9PqKbVj7BJWc+hIOnvTN4Iz4csWM+CDlLvmStPj7DIZU+kxYNPRXXxT2G2o4+2fmKPkvmlD4XUS4+dwKGPpYmHj5kzka+grU3vqRBgj5BD1++m/BUvpPXHjuAfJc+tHYzPlTWGr6lImS8fFCRPaBt/70mmWa9FuwUvni+pL6N4JI935B4PtV1PD71liy+VUg8vfeOkz6aF0O+yrJ9PgZL1b3J85++gcmAPW4jGT5YxvW9ZMxsPnSFp71uQXw+XqHmvaIKhj4lruc8zYJBva+MCD1bHmK+Qo+fPosGCz6sVE0+mdKpPPtrkLwOFCi+IlijPjIwXz4/rg0+XPymvhl78b3ICKg+Vg6mPnqhbj7ZVn6+hylEPoFnkz5LNKg+OkyjPSyVXz4ypJk+fL0lPJGfND4NMHC+yyUHvoKAfDzcSYY+tJ5vveKojb34l647LPzlPB+4TT6pzik+4ZriPegkHL6uT2k+HO+Bvm607r09/wU+aOE3PuHEfr6sU3g9yn0bvkvEhL2LYi66hzWbvn05Hz5xfxo+dI2hPoASZ77OZWI+tQNRvuxCez2jRGs+muL8PbaToT7oNYY+3fUSPs5IB76ou/q8EPhvPseDbz4hCF6+PrG1O/XjAb5urz2+QBMqveFDUL7mMSs9JzMwPoy/Sr7zs/O9h+TevT+LVL7IQYK8kqlHPoLql77pD1g+CMmjvldjqz7aL4i+S7j9vXk8e72o/Yi+T+80vmZTJL5Fsx2+ji6pPuVFsj3jxYC90RVWPmBrqT3jf4e+1+sEvnglUj3U15++D+OUvo7Irb25vp68BOLFPatKKr47/j69G/SaPmYVTz68OJM9j0DMPXugZT4EBUY+OTQLveRSmT4HvBW++uDUPc1sq71+94Y9dTCSvafm7runo0s+EpgzvhYIBL68KKY98KqbPcDbxT0wiPC99tySPpbEk75hJyK+qPDQPG5TPj73O4e96fuFvuoGO70AVZM+0Y9Qvs0Sv73FbnW+wmOSvk62Kj6lWQS9FwUEvu92h70i9UE+WDOsvsIEib3NSku8VrWZvemI5r1jSYQ+2WGnvpx/q745nZY+/CMuPj3+nD4Dbqi+fZaMPr3Wgj12ciw+jTkjPb9uLD1A27O9PqmUvjISQb5fpjy+RESBvijASb72ItE9I26jPrLiyT1k9Im+TRIhvnJphr7E44i+SqE9vkxow73CCR09J/gkPg8aQr7mB56+u0f5vIQpdL6uR728Ofp9viZ/XT7o2Xc+Wps4PgM/pL6i0Ak83muSPhKdlD7iDnc+WgikvsnuAb1mG32+laAPvcBY4j0q4Q0IMAgBCAMIAxABQhJibG9ja3MuNC5kdy53ZWlnaHRKwA0+n4y+6a2TPu2/jr7R6HS9o7U+vRiRGD50pFo+4PQxvdxEBD4Drti9U0g+veHoWL7Q8yk7JxopPtCfcD1ZMS2+hASMvQ8fiD48U1Y+IEGSPqv3yzw8hHQ+H/P1PTH6IT77NhK+BW91vs84iD5zphG+SSBQvPq7MT63v9M90TE2vvH+FT5BSpG981wjPtk0gT5P15S+0smhvFcBnL6OGJO+q9sOvukMQ74ArJI98Rb5PVzChjzuWqS+onADvdbIsD2e5Tw+M7/2vbPiWz4R60W+nvpjvppqiT2vODE+c82kvqlo5L3/xY4+fJ82vht2pr18mFy+kDSQvaC+D74iDTW+ecqNPsMbnz6FkDI+OsMpvncbij6UKoG+E1Kdvrc4cb590Qq+5AJGvdn/Hj2vW1W+SaGEPmUDZD7ILZu9zGA/PtJnDL47nH6+7Kd1PnHQJz4a6J09rjX5vTidXT2lF1I96w9avkXujL75Pyq97M0SvuI4Wb5htx4+EL1fPRrGG724XCG9ibEdPlpzYz4KXJE93YIfvmLFwryDnZE+rLWMPmg1EL6i7hi8f4otvfLXEr728em9ovySPYvkQb3/d5G9DhpLPsknGz5MIp6+CAtBPtsuZb5pOJy+9wxAvv4yAz7iZTi+8F6bPlkJv7rlaZ0+3AlyPoyaXr0vIRI+9gV1Pl7Mfz1EcaY+zmChvg9snD7mXEm+bNsSvsbqJD4LT66+rqaiPsTdiD2lCZo94EuFvtpkQb1uRKY9CfWhPmLWkD6Cgmm+5AeJPkMUqz4QcKu+aB8svuZCXb4adWO+ewSWPuOrk76+Doo+tgbNPS0Tob6BZns+xRSWPhEtoz5x0gO+rAKHvi+Peb6zv/Y9hoqOPmZrPb5gZSu+xAoMvPXUlz7G/Vs+jOu5vPeupTxeKgq+66K+PE1mZ77hEAc+R0SAPs+5MT5M3Yc+WIaSPjCzl73muYC+HmRaPt/uA74sA4w9qveiPvYvDj1DllU+7hQVPq89773g/R49x4f0PaBOjj7SyBk+k5uYPlZ+nr3t/zW+xfySvkYyZL5ciwo+3oTvvdh2r70anlw+J32OPSc1Kj7Vram9B8fYvX4mXD4Pbpc+sdOOPItFKD7BEGS+HLtNvOPcJD4V2Iy9WVOUPmNOxLxcdkS9eL+avk2GX71NK0y+jcSNPp6Aoz73Q5I9VEZgvpKLpD5xPyW+M58dvuZWhb2jZpo+XeJPvI4dJr6A0ow+Fr2TPY6mr7zvdJC97wGOPm6QiD7cCUA+I3tnvdegKT2lYJE+1DTjPKfXaL6Trh+9L7bBvR+bab4xk3e9O92Ivoik3ryG/oc+WqN4PrfPoL4HH4u+DhODvjQkPT54kSG+xpOqPWOYAz4UVwo8QqEgPvkSjj1iUZ+7Fs6cvoErBz3etu+8I72RPrSLlj4dTiY+ul51vumZ0j1IF/u98iF5Pisdyj2WWAQ+kR80PjDRhD1t9J6+8f2lvkQ10T2pTYQ+m/AcPnofQ728u/y9299pPfsD8Dysjo6+u1KdvnYYhz7Gq6a+mUB0voz0qz3elFk+ilvDvZ3+NT4Q+cA9tgmXPjgZVr48u9g8101bPpUwHb4ybLy8psi8O6yiSL4s4Fs+Ns5kvrKZVL75zxi+3TAJPpCA07yqrZ++j0pLvv2MQb5k5jK+KtRUPkZj0T0C6Ju+3j2WvqTiHT6Rw3++loffvURHhD5fcSm+gE3BveRmT75GxyU+10XHPXubQ76DwCW+CxVTPgO7kT1lQS898iJ4vWUkJz6SwkA++TBWPaQQoz49vZ8+EJV+PcIqej435wo+VkBEPPhjo75eyvM8/7QJPVZnWL1zzse9ZLPZPVWFLj6YjQc+pgfQvd7cTr08OxY+PpdvPgUDqr05xZI+5iqmPl8sDr3xDXC+QnpMPmd1ArybHxS+FB0cPnnA+L2vaHe+a9yYO5hoI74ZICg+MuJ0PojAPL7TN0K9kEPqPZylDT6Wihi9Vylnvp3vmD4/OEg+wiqBPacWhD4ks4W+foCdvmmjeD6Zb4K9Vg6avnmvfT6BmSu9d3tGPjDXOb4AaKU9twyIPo5HKD609+G9BPWUvnozgb2uZG6+ant2vnWlSj7i1DM+l3p3vga9kr1PI7S9GNaXPrTxRz4kcD++NVaNvT7WUr6peVu8EjXcPEA23T1dUpY+RoTePRjeir032F2+RJplOkv2Bz4geaa+2gFnvRwAGT5Jxeq9IxSJPnAikT73qra9Sc6Tvj0JAb5UXay+kkBiPWZWd74pJKa+rMlBvmaclL6XOD8+y0yVPqRuiz4q4Q0IMAgBCAMIAxABQhJibG9ja3MuNS5kdy53ZWlnaHRKwA3toXo91rcZPmNsZz6Fc4A+iYJfPqrDPz2F6cY8FqgPPs2W0j3WXwy+/4OZvinLn74fZYc+fWSYvbKATr7d1qc+e7YSvmOnYD1sAJA+Wg0LPr9NKL4faZ0+AjLVPICnpD5aa5K9uu5sPit9qDwdea29wXqKvifOWL5Beqk8vCocvW5LZz4TWIS+W+z5PYVHoT5JLSk+RuaOPqlDXT0NPOG8MPGPPuXdEj4dMYC+EKKjvsDSHL5zP2W+Gfl0O9S3P70X5qE9LTCcPhVkaD1cmaA+RKQjvcjmiD6/Bx6+DPh2vo/3ib5yRm8+Kj6ZvQx/KL4isKw+FHt6vkhndj5Lbne+gvGBvvIh2j3egmW9mpSRPipmAT4adE6+Yk+GvFMMf76EKym9efNcvmti4D0sNKy9unaAvvrrMb6+Qme+KR6fvgBMlr1PfUm+CY+Jvj1Ymb07lGQ+xVhkPmrpXb697YW8JLlNvm3ZV767yXQ+hdsmvqFCOD763Y6+VaofvXJzIL6y7VW+PCYtPl0fFD4/hLo81fwovnxPBD6TazE+zrFMvi1lPT60/5m+Qoofvn6Z4D0A14E+TU5EPLDLUD0RGWS+KpkGvr2rCL6YoX0+xniEPsK/ib483ZQ+iY1bPkraaz5Psfm9NmvoPZsgLL7DOGU+Z6UtvlKsy73rKo++X357Pl4emD3ul5w96MtivhXsXb37I4a+fRqnPpAVkb3o/zQ9a42KvqslgT1rJe+8qMwVPJQYWb5fOmy9lIuIvoXmmL6KgmK7MyNLPqVkqj612HQ+jl5mvQ6aLb7NBZM+HhFavj2Shz6t9vg9yLievhgBDr7pe5Q+4aBmPWQIgT5sDSg+E4KTvQ7CKL6s8Qs8GGMevt+MXD0GKFY+2crnPQQcEjyGG4k+eHb+PaeQD75IriW+fDxOvXqHqLs2a4i9WK4Ivmg7jD5NJW4+Zkk2vXN1hzwlLmG+AgjTPe8sbj6kRTW+1XOEvoyupD4annI+pwKMvp+0hz2ONEY+zN45ProFRD4MDaG+ZCQcPsog+73o40i+g5V5Pdj5Mj4SX6Q+Z+w7vHJkbL5obY6+g4NQvvE1Ob6o+Jy+2HGbvlDhlb0P6Zc851bTPWYMU76A96c8hL6Cvc9Vnj6e2Iq+N/devhXXqT7kTUM9xqwgvt7jq75GtjW+CEanPnmqBj7bp48+HdEdPr9xNj62Xle+Nk+PvnarN75fkJY96yUyvsTMJL4hiks+sp5tvjdqQD0L820+neybvvHFib1zVJg+Vk+gvjyon774JJO9yTybPKAofD7kXzg9ltyaPjVWcr4gRSy+n1WkPtCquz3NltE9Io2GvvBOo70b4i8+9C9ePELUfj5OyzQ+m5xgPp5obb3hOYY+AegqvhzZkr6dgB4+di9GPjP+BrxU+Hw+W6pjPnYyoj3rXls+guUwPorXHz6T9hA+UM5hPn+/6j3dBT4+tZScvohGLL7RtEq+MkAnvvjCpT3mEqC+lTc0PqyDCT4qn6G99dr4PWtrEz2Znpo+HHPPvRlcc75f/CM+KWOwvTIajD7lk1U+H5mYPqhA2j1gXpy+/SDiPUuSsz2W5Gi9oMtcviI1dLwUfEI9rbVwveJnYz7HPRa+Z8QXvjnKA74Stwk+6LQNPpr6oLz2URq+NEsQvsD6IL2dx3s9JF5WPD5bDr7UurU9xXlHvkGXtb3R3fY9oAxIvs0/ZT4i456+qu4svQ35Gj6jCpq6bvc0PuhMh76Fq+k9kxkpPX1mg72HA5Y9WjVYvjO9gL6/Vok9DGiRPtkKMr5cVT2+EeaavgVdeb7+Uyc+loXqPGF1iD6y9Ge+0SKsPm9moD7zZ6w9aGTnPCRnHD6ouYI+VvcevlypMj6rMI+9PrQCvl5pSD1GnmE+ZeKLPsJRSz7RBaa+PsQGve6/Tb0xYou9BnITvqAnSD1oVJC+vk99vkU1wj0A6Bm+xDUzPjuhAj4GxYM+UcN3Pf9ls7xg7ZM+y1VQPUCpFb6PcT+96qxAPSRCRL5+45m+QO4TPsQnTT51X6i81OZ9va/Smb5XHga+Nz8UPHb4Iz7v00e+/13+PdSIkz6wGK69V286PpONIr5P+jQ+JMbMvbIenb6VBCM9a8xLvYOVmz4fbTE+fGerPvC7jL6jlnq933AZPijLKL61ABg9DNiOvUdUAz2Zraa+piuLvv41Mr69F7C9LeWkPho5wTxPTsA9G6ciPkrJtb3UoVK+sVJJvrmaRj4DfoU+Qox+PlObEj5pkfw6R6CJPovjib5GAWm+X06ePrSCwr0sYYU+ku6TvmqsXT4q4Q0IMAgBCAMIAxABQhJibG9ja3MuNi5kdy53ZWlnaHRKwA2wMRm+qfOQPUngAD72jks+7amKPWjM9bxYzzi9m/GHPUm6MT4cfhC+a3yZPq6u+7wPBZ++NDs1PksTN76D7yU+J3I2vvNQPj1tPmE+paiPvoSxDj7RRls+dVwAPnKbpj5i9yc+aAK+vVM+gD0ngV6+S2wFvsUBhr57usK949ZmPTuvXLzzD6i+AKY8PUtfBjs0BCW+OZWJPFCiQj6rDnc+x64PPd7Zhb4Bkna+U0iZvk0ma7w/xqo+8pEZvS/6ZT7GIUS+ADHDPIoUcD5qiTQ96XUZPv12WL3YFFQ8ng/iPaIVYb44HgY+auldvaXrgr6iRVU9AW2EPgfsBr5YHpq+gCtJvn3cgj6oilk+KuluvTueVD7Ym2a+8nUGvgt2Pj5fdAM+MsOCPij+Sr5wFIM70sA2vmKLi75rlro9xXSXvtF5472F8ri81D2OPidKnz3FmaY+KiXOPSVtUb4C8p4+/70dvZFeRT1H3GM8yWWXPl2z6L2no6U+CRIhuTA57D0goES+J3qxvXPkNj4yU+89v/lpvHPUlr6p66E+iHBtPnhMRD63XNY9L38xPtrhRj44mRC+wHKBPvxpCT47G4W95lDdPbkqn75GlOy9kCGavsVj/rzj5ym+dUuXPvljFj6ruXM9q5J1vRvuML77FE29zniLPuK5Ar6vwAu+5s9Evl7ciz47ZYw9rM5zvlW9ur3d7lW+u7pwvFzFZT0bHCA+wHT6Padebb6g4QQ8rIhIvoKzcLw5MqU+80ZkvcAQVT6os5g+xECnPW64pD4ZvA8+lymOPopKzz3XGB6+bkd7PZ6yX76UbKM+8ewrvtIaMz4AK+s8iSVsvvA8Lz4q16e+aY+ovqosaD4LVxG9+HzOPIWnJb7VbZC9vidzPvTNlr3ZVWK723OHvlyZZLrC6Fa+DNh2vdKM9z2cxDs+/iA8PkrZB74O6xK9bz5bPicINL59QoE9wAxKvlsKrT3CRDw9ONwZvp/eh70l1L69e+MAPh+mnD5kVTO+7AeVPsuAVT6BbWU+T3ePvnBOCz7RpKE++aICPkwQOL6acoc+a5NXvm0StjwLVRS+5IaGPY7EcL7CmTk+UH1CviwHUL577i0+NMZDvoNb3z2aTGK+8AcQPcZj2L3k+G2+KtZzvOubuz0VIqc9WayavpjWF75LQaO+O3qfvj7isb2qAGq+NJWRvrcYVT6+QJU+zpVzPst8ID6FkG2+XxOdPq79AT6ql3S+++2ovq8rij592xY++9dpPgh7S75p/WS+JbWDvhgUAj7PGgO+n39XPsiiMr7K1BG+0hs9vp857b3E1Hg+996uPqnusr2iEpK+AUShPStfNr5fnHK92jjPPQD8pT56rI6+7o4+Pg0sjr4Zy0Q+Vo0UPndCpb4ZzBE9BuQCPkaMG74rltY7JFKgvcKQWz5jIoa+s2TzvXK+Wr6Tzhw9n/DjOWYoJL6oagI+2A+RvoLrpL2jCBG+EMqXvSaPlz7xhOK9qq0CPv9dar7Xrlu+W++gPROUHz6/bp++tPWWPlFXoT4L5fq9nvkLPoDpfD5q5xO9hRyhPk6AiD7lQqm9owTBvTPkVr6uDKM+9MpdvjpJ1j0SHoE9YrLdPapaUL0LcF6+UVovPhvFKz63+g4+wvarvYZd0r1yHqW9b5+evW4ERj46lRG9dVxQvRm1SD7Jzxm+rGlWPj4inT1TbKG+nXjRPXbKrD3ciD0+3jESPHcGwz11xn++2E9/vuxUTDyJTjW+nN49Pm7hNL6iyqW9oiKZvk5R+j1XnH0+jN8uPK63hT4cNJm+WL0dPrjGob4JG6K+ih6gPnQBXT5fQ0o+nzuIPlimn74aQZm+ZHCJPh0Spz4EvRE+fPUrvsc5Xb7k13a+pIGcPn/rXj4bASg9/+JGvrNBqL0LWYM9pJl8vuJ7xr0454o+9ViGvlc+572ya6Q+GJX1O0VTjz0ljj2+rxnrPOq0kb0kV5O9GdqCPihsmD6tZXw9NTJvvjfq97xpFH+9nT2vPQXMoz35zTM+VfWVvuQzn71NOqS9kDwuPcj0hD6Op0q+Nl4AvSyal746qE28cJpoPsWbwT10xHy+ApmVvr9wqL7RuQe9YAGOPYNgL74QbbI8m+dDPtLEhD2Zyoe+xy+XviaIkD6SPn0+iwiUPij0lL6dKUi+eXl1vn6XfT4xfOU9ZBCWvhyLZ75PrXE+K2GkviueJz7qv8I9UPp9PhKDRT7OdO89/WGdPpUsp76BxOG8u3loPrlVMD43QHa+4196PnIe572xuki+FW/XPZd6Jzwydnc98OSePm7XOT4q4Q0IMAgBCAMIAxABQhJibG9ja3MuNy5kdy53ZWlnaHRKwA1wi9e8Q/+HvVJ/Br7e+UA+59y1PXoBhz73p1y+ndjBPfceGL6NNZ4+ihMbPZ9eWj6YQ809+scdPiPsSz3unyI+HOEjvoU5U76lxqi+gy6UvqjNdr5OGWy9TuRuvi7SOz3jw8m8TfkAvqwNI77AmE8++nr4vQ49yr1MbFi+H6lGvlRe5z1So+k7WkQtPb34d70KsHG+aRghvehe/rxCwnY+qAiBuzURiT410Z2+GITbPDEGej10wx8+Bnw0vhLnWT5AYKW+FddMvocJFb6v3nc+88SuPmNHir4JcCk+5uJbvYm4Aj7IohA+plInvjXeor5uYv29oj6UvL4tKD5fsxi+uQtiPuwEpL4QbJk9mJ9Vvup2Jb6/95M+bnGZPuHEcD5OjZC+X7d4PcR9O75y6oC9DCqjPrD6Jr5V9BO+Y+hAPsiqCLyYyIU9AJzjvaQ4ab6dAhS+3DI4PdJUazxwbKI9qOBOPvqtZ766FU89dRCKvinEq75dyKU+5K6ZvhFrjz57CZQ+ql+KPlxbHb6Nu3W+hb73PNOQq73+yCa9qZapvcVPPj5iUhM901iUPu4e+71kypY9bosBPj9XeL7ZWWm8saBxPY4UQ76bPAq+ZfB3vq5Ojb1Zm0A+g7GXOhisVbzz3QG+GymSPnTW8bzT/FI9W4ugvudAhT6c/cO9phpjvkmz1b0GwUC+ofGcvoryCz4S0Xo9nzpJPY+EVb5aEY0+zqtpvv8vlz391IM+vUhSPfGdjD4UEEk9xFtQvis8HL5uwey9W/qZPTH+Mr5X3Pu8QgQ8PXnw1z2DX26+EAlJvUbAPz5RzaK9gsiaPZgOdL7lBKC+eZyaPtrInD6cXnM+jvSXPnLrlD6hVEc+vlYqvnHvPb6JEiW+JMZDvhgEKr4Gv049ylAjPP/fg76hjce8+a8PuwL4Rr6i3pK+qEWovUnM5b0qSr09xyePPpV0Iz7Uk+q9GgoXvq0LC77mG5A+yOIlPZX0I703cZy+o7mMPkIssD3dvCW+2Pabvh7FOL7X256+GVyZPoreHD41npk+dyOHPU2vCr5KSkk+NFnfvd4t7j336By+d/TTvP1MIr4KLgo+nYM1vpmhob5GQcI9H0jrvW5l8junA5s7eAaCvvqYk77EZ6c+l7cpPiO/kT6hHqs+2Ye7PTs3Hb5GtLQ9RHtZvtx2Wz4nJeu839snO7eWCj4BGQK+NvtNPqnJAzy/Xfe9duhbPcdvl76PEbK9q1ZYPQmqXr7/ZI4+sZhbPoW4VL6YoIM+QR1/Pl94cD3S5NW9qVJ7vtspcL5/GJ0+G/ZKvjEtKr6hBz69FUmSPQWfhL5NZeG9d/wcPb1gJr7Glvo7F2QSvlJPmT6pZqI+yg0cvmO6Cr4Dko29Fvz+PXR0LT1vxSU8pK4TPhU7IT5n2YG7ZWpXPleUF77PgMg9vZuMPdEe1byVJWK+69yZPc4cdL7RWxM+7/aMPpDqoj6Gnsy9zzs/PljPmj5GVKK+IgpWvjUbw7s/6YU+GkA+PvLj2j0pQma+D0ADPIajTr5Dqo6+GxcNvjQom71kahk+Qu9vPvIYkD62QCW+/MSTvo8QtD1Ca3s8g5jKvQV9G774sdS6yzIXPlR6KT6o7Rs+zKx+PuNVAj4eFVy+5/WbviMvSz5vEjo8jFrivWNoXz3cHVM+5EM7vtPtc77hxqA9FqPXvNYFbL4/OEC+iUqHPoM6jL4udX8+9QVTvjFHzj0p8mc+lFthvms/LLz3zrq90Kc7PvCLkD5yXXQ+fpRSPoRFUj7c4RQ9tbvVPbTweT5R94w++84oPlpfkL22FqG+swpyPQlW3z2Y3XC+uBWPPa05Jr6GF5y9zLn5vTiWRbwddqk+E6kivma1S758Sqg+uoCAvAvIvD0d1Xs+V8MLPjZVCz6IFys9SeV+vsT8+z0T2iy+1q8rPUnUsD0VvWu9rkAUPUHFfT6vZRu+wuyAPt58kr4UlHW+FhmYPs4oxb2z/KG9zVCEPoV6qT5mO7M9SYMOPhkm5D1eHiC+8U01vpLNjD4pCTO+OhdyvbNMKb4D3zy+RBl/vlLUJb7WWKa+tZA5vldMmj4dycu829FNvZhMnz4Ryng+SB0UPpMgAT7H5V8+11eIPndhpT628GU+F6fWOhyAiT4JODi+tOIgvepDlb2Fgaw+6eGUPnBODL4bMpK9BkSjPl3Vh74J4xe8j9WIuynmZL0AwJM+A8M+viCkTTwCyEa+/RYLvpOJHj77p4e+wVKPPj8QAz6RMF89l7qTvvU+Ob6W8dw9+Wi0PHbamL1pPU27eWWCvj+soz4q2igIAwgwCAMIAxABQgtoZWFkLndlaWdodErAKI1xszuOcbM7jXGzO41xszuNcbM7kHGzO5BxszuQcbM7jnGzO4xxszuLcbM7jXGzO4xxszuNcbM7jHGzO4xxszuMcbM7jXGzO49xszuMcbM7jXGzO4xxszuNcbM7jXGzO4xxszuPcbM7kHGzO45xszuNcbM7j3GzO4xxszuNcbM7jHGzO45xszuOcbM7jXGzO45xszuPcbM7j3GzO5BxszuNcbM7jnGzO41xszuOcbM7j3GzO49xszuMcbM7jXGzO5BxszuPcbM7jXGzO5BxszuOcbM7jXGzO4xxszuMcbM7jXGzO41xszuOcbM7kHGzO49xszuNcbM7jXGzO41xszuMcbM7jXGzO41xszuMcbM7jHGzO4xxszuMcbM7jXGzO41xszuPcbM7j3GzO45xszuNcbM7j3GzO49xszuNcbM7kHGzO49xszuPcbM7jnGzO41xszuNcbM7jXGzO41xszuPcbM7jXGzO41xszuLcbM7jHGzO41xszuNcbM7i3GzO41xszuMcbM7i3GzO4xxszuNcbM7jHGzO4xxszuMcbM7jHGzO4xxszuMcbM7jHGzO45xszuNcbM7jXGzO5BxszuPcbM7j3GzO49xszuNcbM7jnGzO4xxszuPcbM7jXGzO49xszuPcbM7j3GzO49xszuOcbM7jXGzO4txszuLcbM7jHGzO4txszuLcbM7inGzO4txszuMcbM7i3GzO41xszuNcbM7jHGzO4xxszuNcbM7jXGzO41xszuNcbM7jXGzO41xszuNcbM7i3GzO41xszuMcbM7jXGzO4xxszuNcbM7jXGzO41xszuPcbM7j3GzO41xszuOcbM7jXGzO5BxszuOcbM7j3GzO41xszuNcbM7jXGzO41xszuMcbM7jHGzO4xxszuMcbM7jHGzO49xszuPcbM7kHGzO41xszuMcbM7jHGzO4xxszuOcbM7jXGzO41xszuPcbM7jnGzO41xszuPcbM7jXGzO41xszuNcbM7j3GzO4txszuNcbM7jnGzO4txszuNcbM7jnGzO41xszuNcbM7j3GzO41xszuNcbM7jXGzO41xszuNcbM7jXGzO41xszuMcbM7jXGzO41xszuNcbM7jnGzO41xszuPcbM7j3GzO41xszuNcbM7jXGzO45xszuPcbM7jHGzO41xszuNcbM7j3GzO41xszuMcbM7jXGzO41xszuNcbM7jXGzO49xszuNcbM7j3GzO49xszuNcbM7j3GzO49xszuMcbM7jXGzO45xszuNcbM7jXGzO41xszuOcbM7jnGzO41xszuMcbM7j3GzO5BxszuNcbM7j3GzO5BxszuNcbM7jnGzO4xxszuPcbM7jHGzO41xszuPcbM7j3GzO41xszuOcbM7j3GzO41xszuOcbM7jHGzO45xszuMcbM7jnGzO5BxszuNcbM7j3GzO41xszuNcbM7j3GzO41xszuPcbM7j3GzO49xszuOcbM7jXGzO41xszuNcbM7kHGzO45xszuPcbM7j3GzO45xszuQcbM7jXGzO41xszuOcbM7jXGzO41xszuNcbM7jXGzO45xszuNcbM7j3GzO49xszuNcbM7jnGzO45xszuPcbM7jXGzO49xszuMcbM7jnGzO4xxszuNcbM7jnGzO41xszuNcbM7jXGzO4xxszuMcbM7jXGzO49xszuM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"

    def __init__(self):
        super().__init__()
        self.path = None
        for cand in ("motion_deblur_fp16.onnx", "motion_deblur.onnx"):
            cp = os.path.join(MODEL_DIR, cand)
            if os.path.exists(cp):
                self.path = cp
                break
        self._sess = None
        self._fp16 = False
        self._prev_luma = None
        self.available = ort is not None
        if ort is None:
            self.hint = T("no_onnx")

    def _ensure(self):
        if self._sess is None:
            provs = onnx_providers()
            if provs[0] == "CPUExecutionProvider":
                self.available = False
                self.enabled = False
                self.hint = T("no_gpu_ep")
                return None
            model_src = self.path if self.path else base64.b64decode(self._EMBEDDED_B64)
            self._sess = ort.InferenceSession(model_src, ort.SessionOptions(), providers=provs)
            inp = self._sess.get_inputs()[0]
            self._name = inp.name
            if isinstance(inp.shape[2], int) and isinstance(inp.shape[3], int):
                self.IN_H, self.IN_W = inp.shape[2], inp.shape[3]
            self._fp16 = inp.type == "tensor(float16)"
        return self._sess

    def _ellipse(self, rw, rh):
        m = np.zeros((rh, rw), np.uint8)
        cv2.ellipse(m, (rw // 2, rh // 2), (int(rw * 0.46), int(rh * 0.49)),
                    0, 0, 360, 255, -1)
        return cv2.GaussianBlur(m, (0, 0), max(rw, rh) * 0.07)

    def _motion_score(self, roi_luma):
        if self._prev_luma is None or self._prev_luma.shape != roi_luma.shape:
            self._prev_luma = roi_luma
            return 0.0
        diff = cv2.absdiff(roi_luma, self._prev_luma)
        self._prev_luma = roi_luma
        return float(min(diff.mean() / 25.0, 1.0))

    def process(self, frame):
        a = self.values["amount"]
        if a <= 0.01 or FACES.nbox is None:
            return frame
        img = npy(frame)
        h, w = img.shape[:2]
        box = FACES.px(w, h)
        if box is None:
            return frame
        x, y, fw, fh = box
        cx, cy = x + fw / 2.0, y + fh / 2.0
        rw, rh = fw * 1.35, fh * 1.45
        x0, y0 = int(max(cx - rw / 2, 0)), int(max(cy - rh / 2, 0))
        x1, y1 = int(min(cx + rw / 2, w)), int(min(cy + rh / 2, h))
        if x1 - x0 < 32 or y1 - y0 < 32:
            return frame
        roi = img[y0:y1, x0:x1]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        small_gray = cv2.resize(gray, (48, 48), interpolation=cv2.INTER_AREA)
        motion = self._motion_score(small_gray)
        thresh = self.values["sensitivity"]
        if motion < thresh:
            return frame

        sess = self._ensure()
        if sess is None:
            return frame

        rh_px, rw_px = roi.shape[:2]
        x_in = cv2.resize(roi, (self.IN_W, self.IN_H), interpolation=cv2.INTER_AREA)
        x_in = cv2.cvtColor(x_in, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)[None]
        x_in = np.ascontiguousarray(
            x_in.astype(np.float16 if self._fp16 else np.float32) / 255.0)
        y_out = np.clip(sess.run(None, {self._name: x_in})[0][0]
                        .transpose(1, 2, 0).astype(np.float32), 0.0, 1.0)
        out_roi = cv2.cvtColor((y_out * 255.0).astype(np.uint8), cv2.COLOR_RGB2BGR)
        interp = cv2.INTER_AREA if self.IN_W > rw_px else cv2.INTER_LINEAR
        out_roi = cv2.resize(out_roi, (rw_px, rh_px), interpolation=interp)

        eff = a * min((motion - thresh) / max(1.0 - thresh, 1e-3) + 0.3, 1.0)
        m = self._ellipse(rw_px, rh_px)
        am = cv2.cvtColor((m.astype(np.float32) * eff).astype(np.uint8),
                          cv2.COLOR_GRAY2BGR)
        inv = cv2.bitwise_not(am)
        out = img.copy()
        out[y0:y1, x0:x1] = cv2.add(
            cv2.multiply(roi, inv, scale=1 / 255.0, dtype=cv2.CV_8U),
            cv2.multiply(out_roi, am, scale=1 / 255.0, dtype=cv2.CV_8U))
        return out


class FaceLight(BaseFilter):
    """Local lift confined to a soft ellipse over the face. Global tone mapping
    has to trade the face against the background; this does not - it raises the
    subject and leaves a bright window behind them exactly where it was."""
    name = "face_light"
    label_key = "facelight"
    # Off by default: auto_exposure already meters on the face, so a second
    # face-targeted lift on top of it is half of why the subject came out blown.
    params = {"amount": ("strength", 0.0, 1.0, 0.0), "target": ("target", 100, 190, 128)}
    uses_face = True
    space = "ycrcb"
    Q = 100.0
    TOLERANCE = 10.0

    def process(self, frame):
        a = self.values["amount"]
        if a <= 0.01 or FACES.nbox is None:
            return frame
        y, cr, cb = frame
        h, w = frame_size(frame)
        box = FACES.px(w, h)
        if box is None:
            return frame
        x, y0f, fw, fh = box
        # the box's y used to be unpacked into `y`, overwriting the luma plane
        # this stage then multiplies - it could never have worked when enabled
        cx, cy = int(x + fw / 2), int(y0f + fh / 2)
        thumb = npy(cv2.resize(y, (160, 90), interpolation=cv2.INTER_AREA))
        sx, sy = 160.0 / w, 90.0 / h
        x0, y0 = max(int(x * sx), 0), max(int(y0f * sy), 0)
        x1, y1 = min(int((x + fw) * sx), 160), min(int((y0f + fh) * sy), 90)
        if x1 - x0 < 2 or y1 - y0 < 2:
            return frame
        mean = max(float(thumb[y0:y1, x0:x1].mean()), 4.0)
        target = self.values["target"]
        if mean >= target - self.TOLERANCE:          # face is already bright enough
            return frame
        gain = min(max(1.0 + ((target - self.TOLERANCE) / mean - 1.0) * a, 1.0), 1.9)

        sw, sh = max(w // 8, 32), max(h // 8, 24)
        mask = np.zeros((sh, sw), np.float32)
        cv2.ellipse(mask, (int(cx * sw / w), int(cy * sh / h)),
                    (max(int(fw * 0.85 * sw / w), 2), max(int(fh * 1.0 * sh / h), 2)),
                    0, 0, 360, 1.0, -1)
        mask = cv2.GaussianBlur(mask, (0, 0), max(sw * 0.02, 1.5))
        gmap = ((1.0 + (gain - 1.0) * mask) * self.Q).astype(np.uint8)
        return [cv2.multiply(y, up(gmap, w, h), scale=1.0 / self.Q, dtype=cv2.CV_8U),
                cr, cb]


class Vibrance(BaseFilter):
    """Webcams ship flat, desaturated colour. Scaling chroma away from neutral is
    the cheapest change in the whole pipeline that a person actually notices."""
    name = "vibrance"
    label_key = "vibrance"
    space = "ycrcb"
    params = {"amount": ("strength", 0.0, 1.0, 0.5)}

    def __init__(self):
        super().__init__()
        self._k = None
        self._lut = None

    def process(self, frame):
        a = self.values["amount"]
        if a <= 0.01:
            return frame
        k = round(a, 3)
        if k != self._k:
            # Vibrance, not saturation: the boost falls off as a channel moves
            # away from neutral, so a washed-out frame gains colour while
            # anything already saturated - lips, a red shirt - is left alone.
            # Scaling every chroma value by the same factor is what turns skin
            # orange the moment the slider does anything visible.
            #
            # Measured on real skin in YCrCb: |cr-128| and |cb-128| sit around
            # 15-30 even for a neutrally-lit face, not near zero. A 56-wide
            # falloff still gives that a ~50-65% share of the full boost, which
            # is the orange-skin bug in practice, not just in theory - a
            # default-settings frame measured a +40pt jump in mean face
            # saturation. Falloff radius 24 with a 1.5 exponent instead of 1
            # pushes the curve down fast enough that skin sits near the floor
            # (~10% of full boost at |d|=17) while flat, near-neutral colour
            # (|d|<10) still gets most of the lift the slider promises.
            x = np.arange(256, dtype=np.float32)
            dist = np.clip(1.0 - np.abs(x - 128.0) / 24.0, 0.0, 1.0) ** 1.5
            self._lut = np.clip(128.0 + (x - 128.0) * (1.0 + 1.15 * a * dist),
                                0, 255).astype(np.uint8)
            self._k = k
        y, cr, cb = frame
        return [y, cv2.LUT(cr, self._lut), cv2.LUT(cb, self._lut)]


class Sharpen(BaseFilter):
    """Detail restoration, gated pixel by pixel on the noise actually left there.

    An unsharp mask restores what webcam optics and MJPG soften, and it boosts
    grain exactly as hard as it boosts detail. So the gate threshold is set per
    pixel from the residual-noise map Denoise publishes: where many frames
    were averaged the grain is a fraction of the raw level and the gate opens
    almost fully; where nothing could be averaged (a moving hand, a turning
    head) or the picture is simply noisier (shadows) it stays shut until the
    detail clearly stands above the grain.

    Halo control: the result is clamped to the 3x3 min/max of the source,
    widened by 30% of the local range and at most 12 levels - a strong edge
    can overshoot by 12, fine texture keeps its contrast.
    """
    name = "sharpen"
    label_key = "sharpen"
    space = "ycrcb"
    params = {"amount": ("strength", 0.0, 2.0, 1.0), "radius": ("radius", 0.5, 4.0, 1.2),
              "quality": ("quality", 0.5, 1.0, 1.0)}
    TK = 1.6            # gate threshold, in residual-noise sigmas
    RAMP = 2.0          # gate ramp width, in residual sigmas
    HF = 0.55           # fine-band share of the noise sigma
    MARGIN_F = 0.3      # halo margin, fraction of the local 3x3 range
    MARGIN_C = 12       # ... capped at this many levels

    def __init__(self):
        super().__init__()
        self._gl = None
        self._gkey = None
        x = np.arange(256, dtype=np.float32)
        self._ml = np.minimum(x * self.MARGIN_F, self.MARGIN_C).astype(np.uint8)

    def process(self, frame):
        amt = self.values["amount"]
        if amt <= 0.01:
            return frame
        # Luma only: colour detail is the first thing vision throws away and
        # what 4:2:0 already discarded; sharpening it only lifts chroma noise.
        meta = frame_meta(frame)
        y, cr, cb = frame
        y = npy(y)
        h, w = y.shape[:2]
        blur = cv2.GaussianBlur(y, (0, 0), self.values["radius"])
        if self.values["quality"] < 0.6:
            # cheap fallback under load: ungated, pulled back
            a = amt * 0.6
            return [cv2.addWeighted(y, 1.0 + a, blur, -a, 0.0), cr, cb]
        # all uint8: the 16-bit signed version of this cost three times as much
        pos = cv2.subtract(y, blur)             # detail above the local mean
        neg = cv2.subtract(blur, y)             # ... and below (one of the two is 0)
        rq = meta.get("resid")
        if rq is not None:
            t = cv2.resize(cv2.convertScaleAbs(rq, alpha=self.TK * self.HF / 8.0), (w, h),
                           interpolation=cv2.INTER_LINEAR)
        else:
            t = max(1.0, self.TK * self.HF * NOISE["residual"])
        ramp = round(max(self.RAMP * self.HF * NOISE["residual"], 1.0), 1)
        if self._gkey != ramp:
            x = np.arange(256, dtype=np.float32)
            self._gl = (np.clip(x / ramp, 0, 1) * 255).astype(np.uint8)
            self._gkey = ramp
        gate = cv2.LUT(cv2.subtract(cv2.max(pos, neg), t), self._gl)
        sharp = cv2.subtract(cv2.add(y, cv2.multiply(pos, gate, scale=amt / 255.0)),
                             cv2.multiply(neg, gate, scale=amt / 255.0))
        lo = cv2.erode(y, _HALO_K3)
        hi = cv2.dilate(y, _HALO_K3)
        m = cv2.LUT(cv2.subtract(hi, lo), self._ml)
        sharp = cv2.min(cv2.max(sharp, cv2.subtract(lo, m)), cv2.add(hi, m))
        return [sharp, cr, cb]


SEG_MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/image_segmenter/"
                 "selfie_segmenter/float16/latest/selfie_segmenter.tflite")
MODEL_DIR = os.path.join(os.path.expanduser("~"), ".webcam_enhancer")


def seg_model_path():
    """~250 KB, fetched once on first use and cached."""
    path = os.path.join(MODEL_DIR, "selfie_segmenter.tflite")
    if not os.path.exists(path):
        os.makedirs(MODEL_DIR, exist_ok=True)
        tmp = path + ".part"
        urllib.request.urlretrieve(SEG_MODEL_URL, tmp)
        os.replace(tmp, path)
    return path


class BackgroundBlur(BaseFilter):
    name = "background"
    label_key = "bg"
    params = {"blur": ("blur", 0.0, 1.0, 0.7), "quality": ("quality", 0.3, 1.0, 0.5)}

    def __init__(self):
        super().__init__()
        self.available = mp is not None
        self.hint = "" if mp is not None else T("no_mediapipe")
        self._seg = None
        self._mask = None
        self._ts = 0

    def _ensure(self):
        if self._seg is not None:
            return True
        try:
            from mediapipe.tasks import python as mpp
            from mediapipe.tasks.python import vision
            self._seg = vision.ImageSegmenter.create_from_options(
                vision.ImageSegmenterOptions(
                    base_options=mpp.BaseOptions(model_asset_path=seg_model_path()),
                    running_mode=vision.RunningMode.VIDEO,
                    output_confidence_masks=True, output_category_mask=False))
            return True
        except Exception as e:
            self.available = False
            self.enabled = False
            self.hint = f"segmenter unavailable: {e}"
            return False

    def process(self, frame):
        """Everything except inference runs at reduced resolution: the mask is
        soft and the background is blurred anyway, so full-res work is wasted."""
        if not self._ensure():
            return frame
        h, w = frame_size(frame)
        q = self.values["quality"]
        small = npy(cv2.resize(frame, (max(int(w * q), 64), max(int(h * q), 64)),
                               interpolation=cv2.INTER_AREA))
        self._ts += 33
        res = self._seg.segment_for_video(
            mp.Image(image_format=mp.ImageFormat.SRGB,
                     data=cv2.cvtColor(small, cv2.COLOR_BGR2RGB)), self._ts)
        m = cv2.GaussianBlur(np.squeeze(res.confidence_masks[0].numpy_view()), (0, 0), 2)
        if self._mask is not None and self._mask.shape == m.shape:
            m = self._mask * 0.5 + m * 0.5
        self._mask = m
        a3 = up((m * 255).astype(np.uint8), w, h, cv2.COLOR_GRAY2BGR)
        d = max(int(2 + 8 * self.values["blur"]), 2)
        bg = cv2.resize(cv2.GaussianBlur(
            cv2.resize(frame, (max(w // d, 16), max(h // d, 16)), interpolation=cv2.INTER_AREA),
            (0, 0), 1.5), (w, h), interpolation=cv2.INTER_LINEAR)
        return cv2.add(cv2.multiply(frame, a3, scale=1 / 255.0, dtype=cv2.CV_8U),
                       cv2.multiply(bg, cv2.bitwise_not(a3), scale=1 / 255.0,
                                    dtype=cv2.CV_8U))


def build_filters():
    """Everything except background blur is on out of the box - a first run that
    looks identical to the raw camera reads as a broken app."""
    # Order is chosen so the YCrCb stages form one unbroken run: the pipeline
    # then splits once on the way in and merges once on the way out instead of
    # every stage shuffling channels for itself.
    fs = [Orientation(), NeuralRestore(), MotionDeblur(), NeuralTone(), NeuralLocal(),
          WhiteBalance(), Warmth(),
          # -- one YCrCb run --
          SpeckleGuard(), Denoise(), AutoExposure(), LocalTone(), FaceLight(),
          Vibrance(), Spotlight(), Framing(), Sharpen(),
          # -- back to BGR --
          TouchUp(), BackgroundBlur()]
    # Everything that costs nothing at its default value is left on, so the
    # sliders are discoverable; the ones that change the picture the moment they
    # are enabled start off.
    #
    # local_tone added here 2026-09: product decision to ship it off by
    # default (strength dropped to 0.1 too, above) - auto_exposure alone at
    # 10% is now the whole default tone correction. Backlit-specific lift is
    # opt-in, not automatic.
    for f in fs:
        f.enabled = f.available and f.name not in ("background", "neural_tone",
                                                   "neural_local", "restore",
                                                   "local_tone")
        if f.locked:
            f.enabled = True
    return fs


class Orientation(BaseFilter):
    """Mirror and level. Runs first so everything downstream, face detection
    included, sees the geometry the viewer will see.

    Mirror defaults off on purpose: a call app already mirrors your self-view
    locally and sends the unmirrored image, so mirroring here is what makes the
    text on your shirt come out backwards for everyone else.
    """
    name = "orientation"
    label_key = "orient"
    params = {"mirror": ("mirror", 0.0, 1.0, 0.0), "rotate": ("rotate", -8.0, 8.0, 0.0)}

    def process(self, frame):
        if self.values["mirror"] >= 0.5:
            frame = cv2.flip(frame, 1)
        a = self.values["rotate"]
        if abs(a) < 0.05:
            return frame
        h, w = frame_size(frame)
        # zoom just enough to keep the rotated corners inside the frame
        m = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), a, 1.0 + abs(a) * 0.009)
        return cv2.warpAffine(frame, m, (w, h), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REPLICATE)


class Warmth(BaseFilter):
    """Manual colour trim on top of the automatic white balance.

    Auto white balance is correct and frequently unflattering - it neutralises
    the warmth that makes skin look alive. This is the knob people reach for
    first, so it is worth the one LUT it costs.
    """
    name = "warmth"
    label_key = "warmth"
    params = {"warmth": ("warmth", -1.0, 1.0, 0.0), "tint": ("tint", -1.0, 1.0, 0.0)}

    def __init__(self):
        super().__init__()
        self._key = None
        self._lut = None

    def process(self, frame):
        wm, ti = self.values["warmth"], self.values["tint"]
        if abs(wm) < 0.02 and abs(ti) < 0.02:
            return frame
        key = (round(wm, 2), round(ti, 2))
        if key != self._key:
            x = np.arange(256, dtype=np.float32)
            g = (1.0 - 0.22 * wm - 0.06 * ti,       # B
                 1.0 + 0.12 * ti,                   # G
                 1.0 + 0.22 * wm - 0.06 * ti)       # R
            lut = np.empty((1, 256, 3), np.uint8)
            for i in range(3):
                lut[0, :, i] = np.clip(x * g[i], 0, 255).astype(np.uint8)
            self._lut, self._key = lut, key
        return cv2.LUT(frame, self._lut)


class TouchUp(BaseFilter):
    """Skin evening confined to the face box, blended through a soft ellipse.

    Frequency separation, the way a retoucher does it: the face is split into
    a soft base (tone, blotches, shine) and a fine layer (pores, lashes, the
    rim of a pair of glasses). Only the base is smoothed; the fine layer is
    put back untouched. Smoothing the picture itself - what this did before -
    takes the fine layer with it, and at 50% that read as a haze over the
    face rather than as better skin.
    """
    name = "touchup"
    label_key = "touchup"
    uses_face = True
    params = {"amount": ("strength", 0.0, 1.0, 0.5)}

    def __init__(self):
        super().__init__()
        self._mask = None
        self._mkey = None

    def _ellipse(self, rw, rh):
        if self._mkey != (rw, rh):
            m = np.zeros((rh, rw), np.uint8)
            cv2.ellipse(m, (rw // 2, rh // 2), (int(rw * 0.44), int(rh * 0.47)),
                        0, 0, 360, 255, -1)
            self._mask = cv2.GaussianBlur(m, (0, 0), max(rw, rh) * 0.07)
            self._mkey = (rw, rh)
        return self._mask

    def process(self, frame):
        a = self.values["amount"]
        if a <= 0.02 or FACES.nbox is None:
            return frame
        # numpy ROI: this stage opts out of the OpenCL path rather than pay a
        # full-frame pass to stay on it
        img = npy(frame)
        h, w = img.shape[:2]
        box = FACES.px(w, h)
        if box is None:
            return frame
        x, y, fw, fh = box
        cx, cy = x + fw / 2.0, y + fh / 2.0
        rw, rh = fw * 1.35, fh * 1.45
        x0, y0 = int(max(cx - rw / 2, 0)), int(max(cy - rh / 2, 0))
        x1, y1 = int(min(cx + rw / 2, w)), int(min(cy + rh / 2, h))
        if x1 - x0 < 24 or y1 - y0 < 24:
            return frame
        roi = img[y0:y1, x0:x1]
        base = cv2.GaussianBlur(roi, (0, 0), 1.6)
        fine = cv2.subtract(roi, base, dtype=cv2.CV_16S)
        sm = cv2.add(cv2.bilateralFilter(base, 9, 14 + 30 * a, 5), fine, dtype=cv2.CV_8U)
        m = self._ellipse(x1 - x0, y1 - y0)
        am = cv2.cvtColor((m.astype(np.float32) * min(1.3 * a, 1.0)).astype(np.uint8),
                          cv2.COLOR_GRAY2BGR)
        inv = cv2.bitwise_not(am)
        out = img.copy()
        out[y0:y1, x0:x1] = cv2.add(cv2.multiply(roi, inv, scale=1 / 255.0, dtype=cv2.CV_8U),
                                    cv2.multiply(sm, am, scale=1 / 255.0, dtype=cv2.CV_8U))
        return out


class Spotlight(BaseFilter):
    """Darkens away from the subject through a soft ellipse on the face.

    The cheap stand-in for background blur. Segmentation costs a fixed 18 ms at
    720p whatever its quality slider says, which is most of a 30 fps budget on
    its own; this costs one upsample and one multiply, needs no model, and gets
    a good part of the same effect - a cluttered room stops competing with your
    face for attention.
    """
    name = "spotlight"
    label_key = "spotlight"
    uses_face = True
    space = "ycrcb"
    params = {"amount": ("strength", 0.0, 1.0, 0.0), "size": ("size", 0.6, 2.2, 1.2)}
    MW, MH = 160, 90

    def process(self, frame):
        a = self.values["amount"]
        if a <= 0.02:
            return frame
        y, cr, cb = frame
        h, w = frame_size(frame)
        sz = self.values["size"]
        nb = FACES.nbox
        if nb is not None:
            bx, by, bw, bh = nb
            cx, cy = bx + bw / 2.0, by + bh / 2.0
            rx, ry = bw * 1.9 * sz, bh * 2.1 * sz
        else:
            cx, cy, rx, ry = 0.5, 0.55, 0.30 * sz, 0.42 * sz
        m = np.zeros((self.MH, self.MW), np.uint8)
        cv2.ellipse(m, (int(cx * self.MW), int(cy * self.MH)),
                    (max(int(rx * self.MW), 4), max(int(ry * self.MH), 4)),
                    0, 0, 360, 255, -1)
        m = cv2.GaussianBlur(m, (0, 0), 9.0)
        # map the mask into a gain: 1.0 on the subject, 1-a at the edges
        m = (255.0 * (1.0 - a) + m.astype(np.float32) * a).astype(np.uint8)
        # luma only. Darkening the chroma too would desaturate the background as
        # well as dim it, and leaving it alone is closer to what a real light does.
        return [cv2.multiply(y, up(m, w, h), scale=1 / 255.0, dtype=cv2.CV_8U),
                cr, cb]


class Framing(BaseFilter):
    """Crop in on the subject and follow them, like Center Stage.

    Placed after everything metered on the face and immediately *before* the
    sharpener, which is not cosmetic ordering. The crop is upsampled back to
    output size, and upsampling after sharpening throws the sharpening away:
    measured at a 1.18 crop with framing last, Laplacian variance fell from 332
    to 44. Sharpening the frame the viewer actually receives costs nothing extra
    and keeps it.

    The crop box is heavily smoothed and has a dead zone - a framing that tracks
    every head movement is far worse to watch than one that lags.
    """
    name = "framing"
    label_key = "framing"
    space = "ycrcb"
    uses_face = True
    # Default off, and the reason is measured rather than conservative: a 1.18
    # crop upsampled back to output size takes Laplacian variance from 332 to 64,
    # and no interpolator changes that (linear 45, cubic 64, Lanczos 68). The
    # sharpener's boost lives in the top octave, which is precisely what any
    # resample destroys. Framing and maximum sharpness cannot both be had at a
    # fixed output size - so it is offered, not imposed.
    params = {"zoom": ("zoom", 1.0, 2.2, 1.15), "follow": ("follow", 0.0, 1.0, 0.5)}
    DEAD = 0.015

    def __init__(self):
        super().__init__()
        self._c = None

    def process(self, frame):
        z = self.values["zoom"]
        # Never crop blind. With no face the centre is a guess, and a guess that
        # throws away 30% of the frame is worse than not cropping at all.
        nb = FACES.nbox
        if z <= 1.01 or nb is None:
            return frame
        h, w = frame_size(frame)
        tx, ty = nb[0] + nb[2] / 2.0, nb[1] + nb[3] / 2.0 - 0.05
        k = 0.02 + 0.13 * self.values["follow"]
        if self._c is None:
            self._c = (tx, ty)
        else:
            # dead zone: below this the crop does not move at all. Tracking a
            # head that is only breathing produces a slow drift that is far more
            # distracting than being slightly off centre.
            dx, dy = tx - self._c[0], ty - self._c[1]
            if abs(dx) < self.DEAD:
                dx = 0.0
            if abs(dy) < self.DEAD:
                dy = 0.0
            self._c = (self._c[0] + dx * k, self._c[1] + dy * k)
        cw, ch = w / z, h / z
        x0 = min(max(self._c[0] * w - cw / 2, 0), w - cw)
        y0 = min(max(self._c[1] * h - ch / 2, 0), h - ch)
        # cubic, not linear: bilinear upsampling by a small factor averages every
        # output sample from two neighbours, which is a heavy low-pass
        y0i, x0i = int(y0), int(x0)
        out = Planes([cv2.resize(npy(c)[y0i:y0i + int(ch), x0i:x0i + int(cw)], (w, h),
                                 interpolation=cv2.INTER_CUBIC) for c in frame])
        out.meta = {}       # the frame-count map no longer lines up with the pixels
        return out


class Pipeline:
    """Filters that expose a 'quality' param process at reduced resolution. The
    controller pushes that scale up until the frame budget is ~70% used, so a fast
    CPU ends up running them at full resolution instead of sitting idle."""

    # stages a learned filter is trained to reproduce - running both
    # double-corrects. Keyed by the stage that replaces them.
    SUBSUMES = {
        "neural_local": ("auto_exposure", "local_tone", "face_light",
                         "white_balance", "vibrance"),
        # denoise is not on this list: it is pinned on whatever else runs
        "restore": ("speckle", "sharpen"),
    }
    SUBSUMED = SUBSUMES["neural_local"]      # kept for the benchmark summary

    # Dropped in this order once the quality knobs bottom out, cheapest loss
    # first. Measured at 720p on one core: background blur is a hard 18.4 ms even
    # at its lowest quality setting - the segmentation inference does not scale
    # with the knob - so it is the one stage that cannot coexist with 30 fps on a
    # weak machine. Denoise is last because a noisy frame is the failure this
    # program exists to fix.
    SHED = ("background", "motion_deblur")
    # Dropped only after the processing scale has already come down, and
    # before denoise ever would - denoise is never dropped.
    SHED_LATE = ("sharpen", "touchup")
    # Boost is the headroom spend: extra denoise work (motion compensation,
    # then a better flow) raised one tier at a time while load stays low. A
    # tier that pushed the load over is barred for BOOST_COOL seconds, so the
    # controller does not sit flapping on the edge.
    BOOST_COOL = 30.0

    # Below this the filters stop being cheaper and start being visibly wrong:
    # denoise drops its chroma plane to a thumbnail and colour bleeds across
    # edges, sharpen falls back to the ungated path. Shedding one stage costs
    # less picture than starving all of them.
    MIN_SCALE = 0.55

    # Below this the tone stages meter a frame that no longer has the scene's
    # fine structure in it, and the re-injected detail is carrying more of the
    # picture than the pipeline is.
    MIN_WORK = 0.5
    # Lowest denoise tier the controller may fall to. Tier 0 switches motion
    # compensation off, and then anything that moves - a head, a hand - loses
    # its frame average and shows raw grain, which the sharpener then lifts:
    # the "crawling dots" look. Tier 1 costs ~1 ms of flow on a 320 px
    # thumbnail plus one warp, far less than any other step the controller
    # takes, so on a machine with 8+ logical cores it is never given up.
    # Weak machines keep tier 0 as a last resort.
    MIN_BOOST = 1 if (os.cpu_count() or 1) >= 8 else 0
    # Exactly one step, and it is a half. INTER_AREA has a fast integer path -
    # 0.26 ms halving a 720p frame against 5-7 ms at any other ratio - and it is
    # also the only ratio that does not alias, which matters because the detail
    # re-injection reads the downscaled frame back: an aliased downscale puts
    # the noise it folded straight back into the output. 0.8 and 0.65 were
    # measured and dropped; between the resampling bill and the extra grain they
    # gave back more than they saved.
    WORK_STEPS = (1.0, 0.5)

    def __init__(self, filters):
        self.filters = filters
        self.total_ms = 0.0
        self.loop_ms = 0.0
        self.budget_ms = BUDGET_MS
        self.frame_ms = 1000.0 / 30.0      # set by Worker from the capture rate
        self.latency_ms = 0.0              # measured, grab -> send
        self.mode = "q30"
        self.boost = 1
        self._boost_bar = {}               # tier -> time it may be retried
        self._boost_t = 0.0                # when the last raise happened
        self._work_bar = {}                # processing scale -> time it may be retried
        self._work_back = {}               # ... and the back-off it is on
        self._work_t = 0.0                 # when the last raise happened
        self.head_ms = 0.0                 # stage 0 work outside the filters
        self.tail_ms = 0.0                 # last stage work outside the filters
        self.auto_quality = True
        self.scale = 1.0
        self.shed = []
        self._shed_cost = {}
        self.conv_ms = 0.0
        # Frame-level pipelining. The filter chain is cut into up to MAX_STAGES
        # pieces that run in separate threads, one frame apart, so the output
        # rate is set by the slowest piece instead of by the sum. 60 ms of work becomes 60 ms of
        # added latency at ~30 fps rather than 16 fps - which is the distinction
        # the frame budget is about. OpenCV releases the GIL inside every op, so
        # this is real parallelism in Python.
        #
        # Cutting by frames instead of by stages would have been simpler and
        # wrong: denoise averages against the previous frame and the tone stages
        # all carry EMAs, so frames have to stay in order. Each thread here sees
        # every frame in sequence, so all of that state stays valid.
        # Processing resolution. Every stage costs pixels, so this is the one
        # knob whose effect is quadratic: at 0.7 the whole chain does half the
        # work. What the downscale removes is put back from the original in
        # postscale(), so what is actually given up is tone and colour precision
        # at fine scales - far less than losing denoise or sharpen outright.
        self.work = 1.0
        self.detail_keep = 1.0
        self._clut = None
        self._ckey = -1.0
        # one stage per two logical cores: each stage thread gets a share of
        # OpenCV's thread team, and a stage with fewer than two has nothing to
        # parallelise its own ops with
        self.stages = max(1, min(MAX_STAGES, (os.cpu_count() or 1) // 2))
        self.stage_ms = [0.0] * self.stages
        self.cuts = self.balance()
        self._n = 0
        self._hot = 0
        self._cool = 0
        self._apply()

    def _cost(self, ms, lo, hi, j, k):
        c = sum(ms[lo:hi])
        if j == 0:
            c += self.head_ms
        if j == k - 1:
            c += self.tail_ms
        return c

    def stage_cost(self, cuts):
        """Predicted wall time of each stage for a given set of cut points."""
        ms = [f.ms for f in self.filters]
        k = len(cuts) - 1
        return [self._cost(ms, cuts[j], cuts[j + 1], j, k) for j in range(k)]

    def balance(self):
        """Cut points [0, c1, ..., n] that minimise the slowest stage.

        Linear partition over measured cost - with up to four stages a greedy
        halving leaves one piece carrying most of the chain. Stage 0 also pays
        for prescale and face detection, the last one for the merge, upscale and
        send, so those are counted where they actually land. Empty stages are
        allowed; a pass-through costs a queue hop and nothing else."""
        n, k = len(self.filters), self.stages
        if k <= 1:
            return [0, n]
        ms = [f.ms for f in self.filters]
        if sum(ms) <= 0.5:
            step = n / k
            return [0] + [int(round(step * j)) for j in range(1, k)] + [n]
        inf = float("inf")
        best = [[inf] * (n + 1) for _ in range(k + 1)]
        back = [[0] * (n + 1) for _ in range(k + 1)]
        best[0][0] = 0.0
        for j in range(1, k + 1):
            for i in range(n + 1):
                for p in range(i + 1):
                    if best[j - 1][p] == inf:
                        continue
                    v = max(best[j - 1][p], self._cost(ms, p, i, j - 1, k))
                    if v < best[j][i] - 1e-9:
                        best[j][i], back[j][i] = v, p
        cuts, i = [n], n
        for j in range(k, 0, -1):
            i = back[j][i]
            cuts.append(i)
        return cuts[::-1]

    @property
    def geometry_changed(self):
        """True when a stage moves pixels, which invalidates the detail map: it
        is taken from the input frame and would land misaligned."""
        for f in self.filters:
            if not (f.enabled and f.available and not f.auto_off):
                continue
            if f.name == "orientation" and (f.values["mirror"] >= 0.5
                                            or abs(f.values["rotate"]) >= 0.05):
                return True
            if f.name == "framing" and f.values["zoom"] > 1.01:
                return True
        return False

    def prescale(self, frame):
        """Shrink before the chain. Returns (work frame, context for postscale)."""
        set_shape(frame)
        if self.work >= 0.999:
            return frame, None
        h, w = frame.shape[:2]
        sw = max(int(w * self.work) // 2 * 2, 240)
        sh = max(int(h * self.work) // 2 * 2, 180)
        if sw >= w:
            return frame, None
        small = cv2.resize(frame, (sw, sh), interpolation=down_interp(w, h, sw, sh))
        set_shape(small)
        return small, (frame, small)

    def postscale(self, out, ctx):
        """Back to capture size, with the original's high frequency put back.

            out = upsample(processed) + k * (original - upsample(downsample(original)))

        The second term is exactly what the downscale removed, so the
        corrections are computed cheaply at low resolution while the sharpness
        stays at capture resolution. One Laplacian level, and the reason running
        at 0.7 does not look like running at 0.7.
        """
        if ctx is None:
            return out
        orig, small = ctx
        h, w = orig.shape[:2]
        out = cv2.resize(out, (w, h), interpolation=cv2.INTER_LINEAR)
        if self.detail_keep <= 0.01 or self.geometry_changed:
            return out
        # the downscaled original is already in hand from prescale; recomputing
        # it here cost more than every other line in this method combined
        base = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
        # Centred in uint8 so the difference can be cored with a LUT. Coring is
        # not optional here: the band being put back is the top of the spectrum,
        # which is where the sensor noise lives, so re-injecting it raw would
        # hand back exactly the grain denoise had just removed. Below the noise
        # floor the term is zeroed; above it, real detail passes.
        d = cv2.LUT(cv2.addWeighted(orig, 0.5, base, -0.5, 128.0), self._core_lut())
        k = self.detail_keep
        return cv2.addWeighted(out, 1.0, d, 2.0 * k, -256.0 * k)

    def _core_lut(self):
        # sigma, not residual: this band is taken from the *original* frame, which
        # denoise never touched, so the grain in it is at full sensor amplitude
        t = 1.1 * NOISE["sigma"]           # half-difference units
        if self._clut is None or abs(t - self._ckey) > 0.4:
            x = np.arange(256, dtype=np.float32) - 128.0
            soft = np.sign(x) * np.clip(np.abs(x) - t, 0.0, None)
            self._clut = np.clip(soft + 128.0, 0, 255).astype(np.uint8)
            self._ckey = t
        return self._clut

    def _apply(self):
        for f in self.filters:
            if "quality" in f.params:
                lo, hi = f.params["quality"][1], f.params["quality"][2]
                f.values["quality"] = round(lo + (hi - lo) * self.scale, 3)
            if f.name == "denoise":
                f.boost = self.boost

    def set_mode(self, mode):
        """Called on every (re)start: begin from full quality, not from wherever
        the last session's controller happened to leave the knobs."""
        self.mode = mode if mode in MODES else "q30"
        self._boost_bar.clear()
        self._work_bar.clear()
        self._work_back.clear()
        for f in self.filters:
            f.auto_off = False
        self.shed = []
        self.work, self.scale, self.boost = 1.0, 1.0, max(1, self.MIN_BOOST)
        self._n = self._hot = self._cool = 0
        self._apply()

    def _live(self, f):
        return f.enabled and f.available and not f.auto_off

    def _shed_one(self, names, by_name):
        for name in names:
            f = by_name.get(name)
            if f is not None and self._live(f):
                # its timer reads 0 from the next frame on, so what it cost has
                # to be kept now, normalised to full processing scale
                self._shed_cost[name] = f.ms / max(self.work, self.MIN_WORK) ** 2
                f.auto_off = True
                self.shed.append(name)
                return True
        return False

    def _tune(self):
        """Hold the frame rate and LATENCY_MS; spend whatever is left on quality.

        Two constraints, whichever is closer to its limit drives the knobs: the
        slowest stage against the frame interval, and the measured grab-to-send
        time against LATENCY_MS.

        Going down, the order is cheapest picture loss first:
          boost tier -> quality knobs -> background / deblur ->
          processing scale (only after sustained overload) -> sharpen / touch-up
        Going up, exactly the reverse, each step only when predicted to fit.

        The processing-scale step used to come second. It halves the frame the
        whole chain works on, so a two-second CPU spike (a browser tab, an
        antivirus scan) showed up as the whole picture suddenly going soft.
        It is now the last thing given up before the face stages, and only
        after the overload has lasted.
        """
        self._n += 1
        if not self.auto_quality or self._n % (5 if self._n < 120 else 15):
            return
        now = time.perf_counter()
        self.budget_ms = self.frame_ms * STAGE_FRAC
        stage = max(max(self.stage_ms), self.loop_ms)
        load = max(stage / max(self.budget_ms, 1e-3), self.latency_ms / LATENCY_MS)
        by_name = {f.name: f for f in self.filters}
        # 30 fps max quality climbs further into the budget than 60 fps does
        climb = 0.68 if self.mode == "q30" else 0.60
        # The first seconds are not representative: the camera is still
        # settling, models load, OpenCV builds its kernels, the face model may
        # be downloading. Only the cheap knobs move then - the heavy steps
        # (dropping stages, halving the processing scale) wait for real load.
        warm = self._n < 120

        if load > 0.90:
            self._hot = min(self._hot + 1, 20)
            self._cool = 0
            if self.boost > self.MIN_BOOST:
                # a raise that tipped it straight over is barred for long; a
                # tier lost to a later spike may be tried again sooner
                recent = now - self._boost_t < 6.0
                self._boost_bar[self.boost] = now + (self.BOOST_COOL if recent else 10.0)
                self.boost -= 1
            elif self.scale > self.MIN_SCALE:
                self.scale = max(self.scale - 0.10, self.MIN_SCALE)
            elif warm:
                return
            elif self._hot >= 3 and self._shed_one(self.SHED, by_name):
                pass
            elif self._hot >= 8 and self.work > self.MIN_WORK:
                # a raise that tipped it straight back over is retried later
                # and later (30 s, 60 s, ... 4 min); one lost to a passing
                # spike is retried soon
                cur = self.work
                if now - self._work_t < 8.0:
                    back = min(self._work_back.get(cur, 15.0) * 2.0, 240.0)
                else:
                    back = 10.0
                self._work_back[cur] = back
                self._work_bar[cur] = now + back
                self.work = next((v for v in self.WORK_STEPS if v < self.work - 1e-3),
                                 self.MIN_WORK)
                return
            elif self._hot >= 8:
                self._shed_one(self.SHED_LATE, by_name)
                return
            else:
                return
        elif load < climb:
            self._hot = 0
            self._cool += 1
            if self._cool < 2:
                return
            late = bool(self.shed) and self.shed[-1] in self.SHED_LATE
            if self.shed and (late or self.work >= 0.999):
                # bring a stage back only if it is predicted to fit, or the
                # chain cycles shed -> restore -> overload with a spike each time
                cost = self._shed_cost.get(self.shed[-1], 0.0) * self.work ** 2 * 1.2
                if (stage + cost < 0.75 * self.budget_ms
                        and self.latency_ms + cost < 0.75 * LATENCY_MS):
                    by_name[self.shed.pop()].auto_off = False
                    self._cool = 0
                return
            if self.work < 0.999:
                # Back up by trying, not by prediction. The old rule assumed
                # every millisecond grows 4x with the pixel count, so once the
                # scale had dropped - often during start-up - it could never
                # come back: a 7 ms stage "predicted" 29 ms and the chain sat
                # at half resolution with three quarters of its budget idle.
                # Flow, face detection, thread hand-offs and per-call overhead
                # do not scale with pixels. Now: after a few calm seconds it
                # simply tries, and a try that fails is barred with back-off.
                nxt = next((v for v in reversed(self.WORK_STEPS)
                            if v > self.work + 1e-3), 1.0)
                barred = now < self._work_bar.get(nxt, 0.0)
                if (not barred and self._cool >= 4
                        and stage * 2.0 < self.budget_ms
                        and self.latency_ms * 1.5 < LATENCY_MS):
                    self.work = nxt
                    self._work_t = now
                    self._cool = 0
                    self._apply()
                    return
                if not barred:
                    return
                # full scale proved too heavy for now: meanwhile spend the
                # headroom there is at this scale instead of sitting idle

            if self.scale < 0.999:
                self.scale = min(self.scale + 0.06, 1.0)
            else:
                nxt = self.boost + 1
                dn = by_name.get("denoise")
                if (dn is None or nxt > dn.MAX_BOOST or self._cool < 3
                        or now < self._boost_bar.get(nxt, 0.0)):
                    return
                self.boost = nxt
                self._boost_t = now
                self._cool = 0
        else:
            return
        self._apply()

    @staticmethod
    def convert(frame, frm, to):
        """BGR frame <-> [y, cr, cb] planes."""
        if frm == to:
            return frame
        if to == "ycrcb":
            return list(cv2.split(cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)))
        return cv2.cvtColor(cv2.merge(frame), cv2.COLOR_YCrCb2BGR)

    def run(self, frame, lo, hi, space="bgr", detect=True, hw=None):
        """Run filters[lo:hi] and hand back (frame, space, filter ms, convert ms).

        The colour space travels with the frame across a stage boundary, so
        splitting the chain never costs an extra conversion. hw is the size of
        this frame; every stage thread records it for itself.

        NOISE["residual"] is no longer reset here per frame. With the chain in
        several threads, stage 0 resetting it for the next frame landed between
        denoise and sharpen of the previous one, so the sharpener's gate read a
        different noise level from frame to frame - visible as sharpness
        flickering. Denoise is always on now and writes it every frame.
        """
        if hw is not None:
            set_hw(*hw)
        total = 0.0
        conv = 0.0
        # detect on the frame the way it will be seen: Orientation runs first and
        # a mirrored frame puts the face box on the wrong side otherwise
        need_face = detect and any(f.enabled and f.available and not f.auto_off
                                   and f.uses_face for f in self.filters)
        # only let the learned stage take over the tone chain if it is actually
        # running - shed or erroring, it would otherwise leave the frame with no
        # exposure correction at all
        skip = set()
        for f in self.filters:
            if (f.name in self.SUBSUMES and f.enabled and f.available
                    and not f.auto_off and not f.error):
                skip.update(self.SUBSUMES[f.name])
        for f in self.filters[lo:hi]:
            if f.locked:
                f.enabled, f.auto_off = True, False
            if f.name in skip or f.auto_off or not (f.enabled and f.available):
                f.ms = 0.0
                continue
            want = f.space
            if want != space:
                # face detection only ever sees BGR: run it on this side of the
                # boundary rather than handing YuNet a YCrCb frame
                if need_face and space == "bgr":
                    FACES.update(frame)
                    need_face = False
                tc = time.perf_counter()
                frame = self.convert(frame, space, want)
                conv += (time.perf_counter() - tc) * 1000.0
                space = want
            if need_face and f.uses_face:
                FACES.update(frame)
                need_face = False
            t = time.perf_counter()
            meta = getattr(frame, "meta", None)
            try:
                frame = f.process(frame)
                f.error = ""
                if meta is not None and isinstance(frame, list) and not isinstance(frame, Planes):
                    frame = Planes(frame)
                    frame.meta = meta
            except Exception as e:
                f.error = f"{type(e).__name__}: {e}"[:60]
            f.ms = f.ms * 0.8 + (time.perf_counter() - t) * 1000.0 * 0.2
            total += f.ms
        # Stage 0 is the only one allowed to detect. If its piece of the chain
        # holds no face stage and no conversion, nothing above triggered the
        # update - and with the cut moving at runtime that left the tracker
        # frozen for as long as the cut stayed there.
        if need_face and space == "bgr":
            FACES.update(frame)
        return frame, space, total, conv

    def to_bgr(self, frame, space):
        if space == "bgr":
            return frame, 0.0
        tc = time.perf_counter()
        frame = self.convert(frame, space, "bgr")
        return frame, (time.perf_counter() - tc) * 1000.0

    def settle(self, stage_ms, conv_ms=0.0):
        # conv_ms is per frame. It used to be a running sum that nothing reset
        # outside process(), so the latency readout crept up for the whole
        # session.
        self.stage_ms = stage_ms
        self.conv_ms = self.conv_ms * 0.8 + conv_ms * 0.2
        self.total_ms = sum(stage_ms) + self.conv_ms
        self._tune()

    def process(self, frame):
        """Whole chain in one call - benchmarks, tests, and single-core runs."""
        t = time.perf_counter()
        work, ctx = self.prescale(frame)
        out, space, ms, cv = self.run(to_gpu(work), 0, len(self.filters),
                                      hw=work.shape[:2])
        out, c2 = self.to_bgr(out, space)
        out = self.postscale(npy(out), ctx)
        loop = (time.perf_counter() - t) * 1000.0
        self.loop_ms = self.loop_ms * 0.8 + loop * 0.2
        self.latency_ms = self.latency_ms * 0.8 + loop * 0.2
        self.settle([ms] + [0.0] * (self.stages - 1), cv + c2)
        return out


# ---------------- camera discovery ----------------

VIRTUAL_HINTS = ("obs virtual", "virtual camera", "vcam", "unity video", "webcam enhancer",
                 "manycam", "xsplit", "snap camera", "droidcam source", "v4l2loopback", "dummy")
IR_HINTS = ("ir camera", "infrared", " ir ", "depth", "3d camera")


def backends():
    if os.name == "nt":
        return [(cv2.CAP_DSHOW, "DSHOW"), (cv2.CAP_MSMF, "MSMF"), (cv2.CAP_ANY, "ANY")]
    return [(cv2.CAP_V4L2, "V4L2"), (cv2.CAP_ANY, "ANY")]


def device_names():
    """Index-aligned device names when the platform can give them, else []."""
    if os.name == "nt":
        try:
            from pygrabber.dshow_graph import FilterGraph
            return list(FilterGraph().get_input_devices())
        except Exception:
            return []
    names = []
    for i in range(10):
        path = f"/sys/class/video4linux/video{i}/name"
        if not os.path.exists(path):
            break
        try:
            names.append(open(path).read().strip())
        except Exception:
            names.append("")
    return names


UNCOMPRESSED = ("YUY2", "YUYV", "NV12", "I420", "RGB3", "BGR3", "RGB", "BGR")


def fourcc_of(cap):
    cc = int(cap.get(cv2.CAP_PROP_FOURCC))
    return "".join(chr((cc >> (8 * i)) & 0xFF) for i in range(4)).strip("\x00 ") or "?"


# How the properties are written decides what gets negotiated, and the two
# Windows backends need opposite things (OpenCV 4.x cap_dshow.cpp /
# cap_msmf.cpp):
#
# DirectShow rebuilds the stream on every size or FOURCC change, and any
# rebuild without an explicit FOURCC walks a fixed subtype list in which YUY2
# comes before MJPG - so a plain size change lands a USB 2 camera on YUY2,
# which carries 720p at 10 fps. A FOURCC change rebuilds at the *current* size
# and at the *requested* frame time, and the frame time is only ever requested
# by CAP_PROP_FPS; left unset, the rebuild inherits the 10 fps interval of the
# YUY2 mode it is replacing. And CAP_PROP_FPS itself rebuilds without a
# FOURCC. The one order that ends on MJPG at full rate is therefore: rate
# first, then size, then FOURCC - and nothing after it. Setting the rate last,
# as this program did, is exactly what put the camera back on YUY2.
#
# Media Foundation picks the native type itself - closest size, then closest
# rate - so size and rate are all it needs. CAP_PROP_FOURCC there sets the
# *output* format, and asking it for MJPG output breaks decoding outright.
#
# The Windows Camera app negotiates the same way Media Foundation does, which
# is why it gets 720p30 from a camera OpenCV was opening at 10.
FORMAT_ORDERS = {
    "rate_size_fourcc": ("fps", "size", "fourcc"),
    "legacy":           ("fourcc", "size", "fourcc", "fps"),
    "plain":            ("size", "fps"),
}


def orders_for(api):
    if os.name == "nt" and api == cv2.CAP_MSMF:
        return ("plain",)
    if os.name == "nt" and api == cv2.CAP_DSHOW:
        return ("rate_size_fourcc", "legacy")
    return ("rate_size_fourcc", "plain")


def set_format(cap, size, fps, order="rate_size_fourcc"):
    """Ask for MJPG at the given size and rate, writing properties in `order`."""
    mjpg = cv2.VideoWriter_fourcc(*"MJPG")
    for step in FORMAT_ORDERS.get(order, FORMAT_ORDERS["rate_size_fourcc"]):
        if step == "fourcc":
            cap.set(cv2.CAP_PROP_FOURCC, mjpg)
        elif step == "size":
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, size[0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, size[1])
        elif step == "fps" and fps:
            cap.set(cv2.CAP_PROP_FPS, fps)
    return fourcc_of(cap)


def open_camera(index, size=None, fps=None, prefer=None):
    """Some laptop cameras (MIPI / Windows Hello modules) are invisible to
    DirectShow and only open through Media Foundation, and vice versa."""
    order = backends()
    if prefer is not None:
        order = sorted(order, key=lambda x: x[0] != prefer)
    tried = []
    for api, name in order:
        cap = cv2.VideoCapture(index, api)
        if not cap.isOpened():
            tried.append(f"{name}:not-opened")
            cap.release()
            continue
        if size:
            set_format(cap, size, fps, orders_for(api)[0])
        elif fps:
            cap.set(cv2.CAP_PROP_FPS, fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        ok, frame = False, None
        for _ in range(5):
            ok, frame = cap.read()
            if ok:
                break
            time.sleep(0.05)
        if ok:
            return cap, api, name, frame, tried
        tried.append(f"{name}:no-frame")
        cap.release()
    return None, None, None, None, tried


MODE_PATH = os.path.join(os.path.expanduser("~"), ".webcam_enhancer_modes.json")
# Sizes tried below the requested one, largest first. UVC cameras on USB 2
# commonly list 960x540 / 848x480 / 800x600 uncompressed at 30 fps even when
# 1280x720 is 10 fps; the driver snaps any request to its nearest listed mode,
# so what matters is the size that actually comes back.
MODE_LADDER = ((1280, 720), (1024, 576), (960, 540), (848, 480), (800, 600),
               (640, 480), (640, 360))


def measure_fps(cap, seconds=0.7):
    """Frames per second the camera really delivers. The first reads after a
    mode change come from the old mode or from exposure still settling."""
    frame = None
    for _ in range(3):
        ok, f = cap.read()
        if ok:
            frame = f
    n, t0 = 0, time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        ok, f = cap.read()
        if ok:
            n, frame = n + 1, f
    return n / max(time.perf_counter() - t0, 1e-3), frame


def _mode_cache():
    try:
        with open(MODE_PATH, encoding="utf-8") as fp:
            return json.load(fp)
    except Exception:
        return {}


def _mode_store(key, value):
    data = _mode_cache()
    data[key] = value
    try:
        with open(MODE_PATH, "w", encoding="utf-8") as fp:
            json.dump(data, fp, indent=2)
    except Exception:
        pass


def _try_mode(index, api, size, fps, order, skip=()):
    """Open, configure, measure. None if the camera does not open, hands back
    something that is not a BGR frame, or lands on a mode already measured."""
    cap = cv2.VideoCapture(index, api)
    if not cap.isOpened():
        cap.release()
        return None
    try:
        fcc = set_format(cap, size, fps, order)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        ok, frame = cap.read()
    except Exception:
        cap.release()
        return None
    if not ok or frame is None or frame.ndim != 3 or frame.shape[2] != 3:
        cap.release()      # e.g. a backend passing compressed bytes through
        return None
    key = (api, (frame.shape[1], frame.shape[0]), fcc)
    if key in skip:
        cap.release()      # the driver snapped to a mode already measured
        return None
    rate, frame = measure_fps(cap)
    return cap, rate, frame, key


def find_mode(index, size, fps, prefer=None, name="", log=None, cache=True):
    """Open the camera in the largest mode that runs at the requested rate.

    Returns (cap, api id, api name, frame, tried, info); info carries the
    measured rate, the property order that worked, and - when the size had to
    come down - what the requested size delivered, so the UI can say why.

    The requested size is tried on every backend with every property order
    first: a camera the Windows Camera app runs at 720p30 is almost always
    being left on YUY2 by the negotiation, not limited by the sensor. Only then
    smaller sizes, and one is taken only if it reaches full rate where the
    larger one did not. If every size is equally slow the cause is exposure,
    not bandwidth, and the requested size is kept. The winner is remembered per
    camera so this runs once."""
    order = backends()
    if prefer is not None:
        order = sorted(order, key=lambda x: x[0] != prefer)
    names = dict(order)
    want = 0.8 * float(fps or 30)
    key = f"v2|{name or index}|{size[0]}x{size[1]}|{fps}"
    say = log or (lambda *a: None)

    hit = _mode_cache().get(key) if cache else None
    if hit and hit.get("api") in names:
        got = _try_mode(index, hit["api"], tuple(hit["size"]), fps,
                        hit.get("order", orders_for(hit["api"])[0]))
        if got and got[1] >= want:
            cap, rate, frame, _ = got
            info = dict(hit.get("info", {}), rate=rate, order=hit.get("order", orders_for(hit["api"])[0]))
            return cap, hit["api"], names[hit["api"]], frame, [], info
        if got:
            got[0].release()

    area = size[0] * size[1]
    ladder = [size] + [m for m in MODE_LADDER if m[0] * m[1] < area]
    seen = set()
    top = None          # (actual size, best rate) at the largest size reached
    dark = False
    best = None         # full-rate result, kept while a larger one may follow
    for req in ladder:
        if best is not None and req[0] * req[1] <= best[5][0] * best[5][1]:
            break       # nothing left to try is bigger than what already works
        for api, api_name in order:
            if api == cv2.CAP_ANY:
                continue
            tries = orders_for(api) if req == size else orders_for(api)[:1]
            for how in tries:
                got = _try_mode(index, api, req, fps, how, seen)
                if got is None:
                    continue
                cap, rate, frame, k = got
                seen.add(k)
                actual, fcc = k[1], k[2]
                say(f"  {api_name:<6} {how:<13} asked {req[0]}x{req[1]} -> "
                    f"{actual[0]}x{actual[1]} {fcc:<5} {rate:5.1f} fps")
                if rate >= want:
                    info = {"rate": rate, "order": how}
                    if top and actual[0] * actual[1] < top[0][0] * top[0][1]:
                        info.update(req_w=top[0][0], req_h=top[0][1], req_rate=top[1])
                    if actual == tuple(size):
                        best = None
                        if cache:
                            _mode_store(key, {"api": api, "size": list(req),
                                              "order": how, "info": info})
                        return cap, api, api_name, frame, [], info
                    # a driver snapping a request down can skip past a larger
                    # mode further along the ladder - keep looking above this
                    if best is None or actual[0] * actual[1] > best[5][0] * best[5][1]:
                        if best is not None:
                            best[0].release()
                        best = (cap, api, api_name, frame, info, actual, req, how)
                    else:
                        cap.release()
                    break
                cap.release()
                if top is None or actual[0] * actual[1] > top[0][0] * top[0][1]:
                    top = (actual, rate)
                elif actual == top[0]:
                    top = (actual, max(top[1], rate))
                elif rate <= top[1] * 1.2:
                    # A smaller mode that is no faster: the rate is being set
                    # by exposure, and every smaller mode will say the same.
                    dark = True
                    break
                if fcc == "MJPG":
                    break       # compressed and still slow: order is not it
            if dark or (best is not None and best[6] == req):
                break
        if dark:
            break
    if best is not None:
        cap, api, api_name, frame, info, actual, req, how = best
        if cache:
            _mode_store(key, {"api": api, "size": list(req), "order": how, "info": info})
        return cap, api, api_name, frame, [], info
    cap, api, api_name, frame, tried = open_camera(index, size, fps, prefer)
    return cap, api, api_name, frame, tried, {
        "rate": top[1] if top else 0.0, "order": orders_for(api)[0] if cap else "",
        "dark": dark}


def open_60(index, size, prefer=None, name="", cache=True):
    """The camera at exactly `size` and a measured 60 fps, or nothing.

    Returns the find_mode tuple on success, (None, ..., {"rate": best}) when no
    backend and no property order gets there. Only a success is remembered: a
    camera measured slow in a dark room is not written off for good.
    """
    size = tuple(size)
    order = backends()
    if prefer is not None:
        order = sorted(order, key=lambda x: x[0] != prefer)
    names = dict(order)
    key = f"fps60|{name or index}|{size[0]}x{size[1]}"
    hit = _mode_cache().get(key) if cache else None
    best = 0.0
    if hit and hit.get("api") in names:
        got = _try_mode(index, hit["api"], size, 60, hit.get("order", "rate_size_fourcc"))
        if got:
            cap, rate, frame, k = got
            if k[1] == size and rate >= FPS60_MIN:
                return cap, hit["api"], names[hit["api"]], frame, [], {
                    "rate": rate, "order": hit.get("order")}
            best = max(best, rate)
            cap.release()
    seen = set()
    for api, api_name in order:
        if api == cv2.CAP_ANY:
            continue
        for how in orders_for(api):
            got = _try_mode(index, api, size, 60, how, seen)
            if got is None:
                continue
            cap, rate, frame, k = got
            seen.add(k)
            if k[1] == size and rate >= FPS60_MIN:
                if cache:
                    _mode_store(key, {"api": api, "order": how})
                return cap, api, api_name, frame, [], {"rate": rate, "order": how}
            if k[1] == size:
                best = max(best, rate)
            cap.release()
    return None, None, None, None, [], {"rate": best}


def output_size(cap_wh, req_wh):
    """Where a smaller capture is scaled back up to: the requested height, the
    capture's own aspect. Cropping 4:3 to 16:9 would cost a quarter of the
    picture for a shape every call app accepts either way."""
    cw, ch = cap_wh
    if ch >= req_wh[1]:
        return None
    oh = req_wh[1]
    ow = int(round(cw * oh / ch / 2.0)) * 2
    if abs(ow - req_wh[0]) <= 0.02 * req_wh[0]:
        ow = req_wh[0]      # 848x480 is 16:9 give or take a pixel
    return ow, oh


def upscale(img, size):
    """Bicubic up plus a light unsharp mask. No detail is recovered - the point
    is that the call app gets a full-size frame to scale instead of doing its own
    cheaper upscale of a smaller one. The chain has already denoised, so the
    mask sharpens edges rather than grain; it backs off when residual noise is
    still high."""
    up = cv2.resize(img, size, interpolation=cv2.INTER_CUBIC)
    amt = 0.45 if NOISE["residual"] < 3.0 else 0.25
    blur = cv2.GaussianBlur(up, (0, 0), 1.1)
    return cv2.addWeighted(up, 1.0 + amt, blur, -amt, 0)


def probe_cameras(max_index=8):
    """Walk the indices once and keep whatever actually delivers a frame."""
    names = device_names()
    limit = min(max_index, len(names)) if names else max_index
    found = []
    for i in range(limit):
        cap, api, api_name, frame, _ = open_camera(i)
        if cap is None:
            continue
        cap.release()
        raw = names[i] if i < len(names) else f"camera {i}"
        low = f" {raw.lower()} "
        found.append({
            "index": i, "api": api, "api_name": api_name, "name": raw,
            "w": frame.shape[1], "h": frame.shape[0],
            "virtual": any(k in low for k in VIRTUAL_HINTS),
            "ir": any(k in low for k in IR_HINTS),
        })
    return found


def pick_camera(cams):
    """Never auto-pick our own virtual output (feedback loop) or an IR sensor."""
    real = [c for c in cams if not c["virtual"]] or cams
    if not real:
        return None
    return sorted(real, key=lambda c: (c["ir"], -c["w"] * c["h"], c["index"]))[0]


def cam_label(c):
    name = c["name"] if len(c["name"]) <= 24 else c["name"][:23] + "\u2026"
    tag = "  [virtual]" if c["virtual"] else ("  [IR]" if c["ir"] else "")
    return f"{c['index']}: {name}{tag}"


# ---------------- virtual camera ----------------

# Windows has no way to publish a camera from user code alone. A virtual webcam
# is a registered DirectShow filter (or, on Windows 11 22H2+, a signed MSIX
# Media Foundation package), and a Python process cannot install one. So the
# driver has to come from somewhere: in practice the OBS installer, which
# registers "OBS Virtual Camera" as a system-wide DirectShow source. OBS itself
# never has to run - only its filter has to be registered - and Unity Capture is
# the same deal from a different installer. This is a limitation of the platform,
# not of pyvirtualcam.
#
# Without one of those the program still works; it just cannot hand the result
# to Zoom or Discord. The preview is live and everything else is unchanged.
VCAM_BACKENDS = ("obs", "unitycapture") if os.name == "nt" else ("v4l2loopback", "obs")


def vcam_probe(w=640, h=480, fps=30):
    """Which backends can actually be opened right now, and why the others cannot."""
    if pyvirtualcam is None:
        return [], [("pyvirtualcam", "not installed - pip install pyvirtualcam")]
    ok, bad = [], []
    for name in VCAM_BACKENDS:
        try:
            cam = pyvirtualcam.Camera(width=w, height=h, fps=fps,
                                      fmt=PixelFormat.BGR, backend=name)
            cam.close()
            ok.append(name)
        except Exception as e:
            bad.append((name, str(e).strip().splitlines()[0][:90]))
    return ok, bad


def open_vcam(w, h, fps):
    """Try each backend by name rather than letting the library pick.

    Returns (camera, note). A failure is never fatal - it costs the virtual
    camera output and nothing else."""
    if pyvirtualcam is None:
        return None, T("vcam_missing")
    for name in VCAM_BACKENDS:
        try:
            return pyvirtualcam.Camera(width=w, height=h, fps=fps,
                                       fmt=PixelFormat.BGR, backend=name), ""
        except Exception:
            continue
    try:                                    # let the library choose, as a last try
        return pyvirtualcam.Camera(width=w, height=h, fps=fps, fmt=PixelFormat.BGR), ""
    except Exception:
        return None, T("vcam_none")


# ---------------- capture worker ----------------

class Worker(threading.Thread):
    """Capture thread + Pipeline.stages processing threads.

    Stage 0 runs here: pick the newest frame, prescale, first piece of the chain.
    Stages 1..n-1 each run their piece in their own thread, one frame behind the
    one before, linked by one-slot queues. The last stage merges, upscales and
    sends. Every frame is stamped when the grab thread gets it and checked again
    at send, which is the latency the controller holds under LATENCY_MS.
    """

    def __init__(self, index, size, mode, pipeline, api=None, name=""):
        super().__init__(daemon=True)
        self.mode = mode if mode in MODES else "q30"
        self.index, self.size, self.prefer = index, size, api
        self.fps = MODES[self.mode]
        self.fallback = ""          # set when 60 was asked for and not available
        self.pipeline_error = ""
        self.present_lat = 0.0
        self._last_seq = -1
        self._entered = 0
        self._base = 0.0            # grab -> send, seconds
        self.cam_name = name
        self._out = None
        self._dark = False
        self.mode_note = ""
        self.pipeline = pipeline
        pipeline.budget_ms = BUDGET_MS
        pipeline.latency_ms = 0.0
        pipeline.loop_ms = 0.0
        pipeline.set_mode(self.mode)
        # not _stop: that name is Thread's own method, and join() calls it - an
        # Event there made every join() on a finished worker raise TypeError
        self._halt = threading.Event()
        self._lock = threading.Lock()
        # the grab thread signals new frames instead of stage 0 polling for
        # them: a 2 ms sleep is a 15.6 ms sleep on any Windows Python older
        # than 3.11, and that was latency added before the first filter ran
        self._gcond = threading.Condition()
        n = pipeline.stages
        # one-slot handoffs, depth 1 on purpose, so an upstream stage waits for
        # the one below it instead of building a backlog
        self._qs = [queue.Queue(maxsize=1) for _ in range(n - 1)]
        self._loop = [0.0] * n
        # frames currently between stage 0 and send; cut points only move when
        # this is zero, so no filter ever sees two frames at once or out of order
        self._flight = 0
        self._fcond = threading.Condition()
        self._cam = None
        self._raw = None
        self._seq = 0
        self._t_raw = 0.0
        self._cap_dt = 0.0
        self.latest = None
        self.grabbed = 0
        self.fps_actual = 0.0
        self.fps_capture = 0.0
        self.status = "..."
        self.vcam_note = ""
        self.api = ""
        self.fmt = "?"
        self.cam_fps = 0.0
        self.error = None
        self.diag = ""
        self._slow = 0
        self._renego = False
        self._renego_done = False
        self._settings = False
        self._wh = (0, 0)

    def request_settings(self):
        """Open the driver's own property page. Done from the grab thread: it is
        the only thread allowed to touch the capture object."""
        self._settings = True

    def stop(self):
        self._halt.set()
        with self._gcond:
            self._gcond.notify_all()
        with self._fcond:
            self._fcond.notify_all()

    def get_frames(self):
        with self._lock:
            return self.latest

    def _head(self):
        w, h = self._wh
        # CAP_PROP_FPS is what the driver agreed to, not what it delivers - the
        # status shows the measured capture rate next to it for that reason
        head = f"{self.api}/{self.fmt} {w}x{h}@{self.cam_fps:.0f}"
        if self._out:
            head += f"->{self._out[0]}x{self._out[1]}"
        self.status = (f"{head} -> {self._dev}" if self._dev
                       else f"{head} | {self.vcam_note}")

    def _diagnose(self):
        """The camera, not the chain, is short of frames. Say which way."""
        want = min(float(self.fps), self.cam_fps) if self.cam_fps > 1 else float(self.fps)
        if self.fps_capture <= 0 or self.fps_capture >= 0.8 * want:
            self._slow = 0
            self.diag = ""
            return
        self._slow += 1
        if self._slow < 4:          # 2 s: ignore start-up and a hand over the lens
            return
        w, h = self._wh
        raw = self.fmt.upper() in UNCOMPRESSED
        if raw and not self._renego_done:
            self._renego = True     # one more try at MJPG, from the grab thread
            return
        key = ("diag_bw" if raw and w * h > 640 * 480 and not self._dark
               else "diag_dark")
        self.diag = T(key).format(cap=self.fps_capture, fmt=self.fmt, res=f"{w}x{h}")

    def _target_fps(self):
        """Frame rate the chain is sized for: the menu value, or what the camera
        actually delivers if that is less. Cameras routinely accept 60 and hand
        back 30, and many halve their rate in the dark."""
        want = float(self.fps)
        if self._cap_dt > 1e-4:
            return min(want, 1.0 / self._cap_dt)
        return min(want, self.cam_fps) if self.cam_fps > 1.0 else want

    def run(self):
        self.status = T("probing")
        cap = None
        if self.mode == "fps60":
            cap, _, api, frame, tried, info = open_60(self.index, self.size,
                                                      self.prefer, self.cam_name)
            if cap is None:
                self.fallback = T("no60").format(res=f"{self.size[0]}x{self.size[1]}",
                                                 rate=info.get("rate", 0.0))
                self.mode, self.fps = "q30", MODES["q30"]
                self.pipeline.set_mode(self.mode)
        if cap is None:
            cap, _, api, frame, tried, info = find_mode(self.index, self.size, self.fps,
                                                        self.prefer, self.cam_name)
        if cap is None:
            self.error = (f"camera {self.index}: no frame  [tried {', '.join(tried)}]  "
                          f"- in use by another app, or blocked by privacy settings?")
            return
        h, w = frame.shape[:2]
        self.api = api
        self._wh = (w, h)
        self.fmt = fourcc_of(cap)
        self.cam_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0

        self._dark = info.get("dark", False)
        self._order = info.get("order") or "rate_size_fourcc"
        self._out = output_size((w, h), self.size)
        ow, oh = self._out or (w, h)
        if self._out and "req_w" in info:
            self.mode_note = T("mode_up").format(
                cw=w, ch=h, rw=info["req_w"], rh=info["req_h"], rf=info["req_rate"],
                ow=ow, oh=oh)
        cam, note = open_vcam(ow, oh, self.fps)
        self._dev = cam.device if cam is not None else ""
        self.vcam_note = note
        self._head()

        self._cam = cam
        pl = self.pipeline
        n = pl.stages
        if n > 1:
            # OpenCV forks and joins a thread team inside every op. Stage threads
            # each claiming the whole machine spend their time fighting over it,
            # so the team is split between them.
            cv2.setNumThreads(max((os.cpu_count() or 2) // n, 1))
        pl.cuts = pl.balance()
        self._raw, self._seq, self._t_raw = frame, 0, time.perf_counter()
        grab = threading.Thread(target=self._grab, args=(cap,), daemon=True)
        grab.start()
        stages = [threading.Thread(target=self._stage, args=(k,), daemon=True)
                  for k in range(1, n)]
        for t in stages:
            t.start()

        seen = -1
        due = 0.0
        nb = 0
        while not self._halt.is_set():
            with self._gcond:
                while self._seq == seen and not self._halt.is_set():
                    self._gcond.wait(0.1)
                if self._halt.is_set():
                    break
                frame, seen, t_cap = self._raw, self._seq, self._t_raw

            # Rate gate. The camera paces the input, so nothing downstream
            # sleeps any more; this only thins the stream when the camera runs
            # faster than the menu asks for. It used to be sleep_until_next_frame
            # after every send, which parked a finished frame in the queue for up
            # to a whole interval whenever the camera and the pacer were out of
            # phase - latency that no readout showed.
            interval = 1.0 / max(float(self.fps), 1.0)
            slack = 0.5 * (self._cap_dt if self._cap_dt > 1e-4 else interval)
            if t_cap + slack < due:
                continue
            due = max(due + interval, t_cap + interval - slack)

            nb += 1
            if n > 1 and nb % 30 == 0:
                self._rebalance()
            cuts = tuple(pl.cuts)

            self._entered += 1
            t = time.perf_counter()
            try:
                work, ctx = pl.prescale(frame)
                hw = work.shape[:2]
                mid, space, ms, cv = pl.run(to_gpu(work), cuts[0], cuts[1], "bgr", True, hw)
            except Exception as e:
                self._drop(0, e)
                continue
            item = [frame, mid, space, [ms], cv, cuts, ctx, t_cap, self._entered, hw]
            loop = (time.perf_counter() - t) * 1000.0
            if n == 1:
                self._finish(0, item, t)
                continue
            self._loop[0] = self._loop[0] * 0.8 + loop * 0.2
            pl.head_ms = pl.head_ms * 0.8 + max(loop - ms, 0.0) * 0.2
            with self._fcond:
                self._flight += 1
            self._put(0, item)

        self._halt.set()
        grab.join(timeout=1.0)
        for t in stages:
            t.join(timeout=1.0)
        if self._cam is not None:
            self._cam.close()
        cap.release()

    def _rebalance(self):
        """Move the cut points only when it pays, and only on an empty pipeline.

        A frame carries the cuts it started with, so every frame still runs every
        filter exactly once. But a filter moved to an earlier stage would see the
        new frame before the old one has reached it: two threads inside the same
        filter, and denoise averaging against the wrong previous frame. Draining
        first costs about one frame of latency, once, and the 15% hysteresis
        keeps that rare."""
        pl = self.pipeline
        new = pl.balance()
        if list(new) == list(pl.cuts):
            return
        if max(pl.stage_cost(new)) > max(pl.stage_cost(pl.cuts)) * 0.85 - 0.3:
            return
        with self._fcond:
            if self._fcond.wait_for(lambda: self._flight == 0 or self._halt.is_set(),
                                    timeout=0.2) and not self._halt.is_set():
                pl.cuts = new

    def _put(self, k, item):
        while not self._halt.is_set():
            try:
                self._qs[k].put(item, timeout=0.1)
                return
            except queue.Full:
                continue
        with self._fcond:
            self._flight = max(self._flight - 1, 0)

    def _stage(self, k):
        """Stage k of the chain. Every frame passes through in order, which is
        why the cut is by stage and not by frame: denoise averages against the
        previous frame and every tone stage carries an EMA."""
        pl = self.pipeline
        last = k == pl.stages - 1
        while not self._halt.is_set():
            try:
                item = self._qs[k - 1].get(timeout=0.1)
            except queue.Empty:
                continue
            t = time.perf_counter()
            raw, mid, space, mss, cv, cuts, ctx, t_cap, seq, hw = item
            try:
                mid, space, ms, c = pl.run(mid, cuts[k], cuts[k + 1], space, False, hw)
            except Exception as e:
                # A conversion between filters is outside the per-filter guard.
                # An exception here used to end this thread for good and freeze
                # the picture; the frame is dropped instead.
                self._drop(k, e)
                continue
            mss.append(ms)
            item[1], item[2], item[4] = mid, space, cv + c
            if last:
                self._finish(k, item, t)
                continue
            loop = (time.perf_counter() - t) * 1000.0
            self._loop[k] = self._loop[k] * 0.8 + loop * 0.2
            self._put(k, item)

    def _drop(self, k, e):
        self.pipeline_error = f"{type(e).__name__}: {e}"[:80]
        if k > 0:
            with self._fcond:
                self._flight = max(self._flight - 1, 0)
                self._fcond.notify_all()

    def _finish(self, k, item, t):
        pl = self.pipeline
        raw, mid, space, mss, cv, cuts, ctx, t_cap, seq, hw = item
        try:
            out, c = pl.to_bgr(mid, space)
            out = pl.postscale(npy(out), ctx)
            if self._out is not None:
                out = upscale(out, self._out)
        except Exception as e:
            self._drop(k, e)
            return
        self._present(out, raw, t_cap, seq)
        now = time.perf_counter()
        loop = (now - t) * 1000.0
        self._loop[k] = self._loop[k] * 0.8 + loop * 0.2
        if k > 0:
            pl.tail_ms = pl.tail_ms * 0.8 + max(loop - mss[-1], 0.0) * 0.2
            with self._fcond:
                self._flight = max(self._flight - 1, 0)
                self._fcond.notify_all()
        # the output rate is set by the slowest stage, not by the sum
        pl.loop_ms = max(self._loop)
        self._base = self._base * 0.8 + (now - t_cap) * 0.2
        pl.latency_ms = self._base * 1000.0
        pl.frame_ms = 1000.0 / max(self._target_fps(), 1.0)
        pl.settle(mss + [0.0] * (pl.stages - len(mss)), cv + c)

    def _present(self, out, raw, t_rep, seq):
        """Hand one frame to the outputs; t_rep is its capture time."""
        if seq <= self._last_seq:
            return
        self._last_seq = seq
        self._send(out, raw)
        lat = (time.perf_counter() - t_rep) * 1000.0
        self.present_lat = lat if self.present_lat <= 0 else self.present_lat * 0.8 + lat * 0.2
        self._count = getattr(self, "_count", 0) + 1
        t0 = getattr(self, "_t0", None)
        if t0 is None:
            self._t0 = time.time()
        elif time.time() - t0 >= 0.5:
            dt = time.time() - t0
            self.fps_actual = self._count / dt
            self.fps_capture = self.grabbed / dt
            self._count, self.grabbed, self._t0 = 0, 0, time.time()
            self._diagnose()

    def _send(self, out, raw):
        with self._lock:
            self.latest = (raw, out)
        if self._cam is not None:
            try:
                self._cam.send(out)
            except Exception:
                self._cam = None
                self.status = "virtual cam disconnected - preview only"

    def _grab(self, cap):
        """Capture in its own thread, keeping only the newest frame.

        The driver keeps filling its buffer at the camera's rate whatever we do.
        With capture and processing in one loop, cap.read() hands back the oldest
        buffered frame, so once processing is slower than the camera the delay
        grows until the buffer saturates and then frames arrive in bursts.
        Dropping stale frames here trades throughput we never had for latency.

        CAP_PROP_BUFFERSIZE would be the tidy fix but DSHOW and MSMF ignore it.
        """
        last = 0.0
        while not self._halt.is_set():
            if self._settings:
                self._settings = False
                try:
                    ok = cap.set(cv2.CAP_PROP_SETTINGS, 1)
                except Exception:
                    ok = False
                if not ok:
                    self.diag = T("cam_settings_na")
            if self._renego:
                self._renego, self._renego_done = False, True
                try:
                    self.fmt = set_format(cap, self._wh, self.fps,
                                          getattr(self, "_order", "rate_size_fourcc"))
                    self.cam_fps = cap.get(cv2.CAP_PROP_FPS) or self.cam_fps
                except Exception:
                    pass
                self._slow = 0
                self._head()
                last = 0.0
            ok, f = cap.read()
            if not ok:
                time.sleep(0.005)
                continue
            now = time.perf_counter()
            if last:
                dt = min(now - last, 0.5)
                self._cap_dt = dt if self._cap_dt <= 0 else self._cap_dt * 0.9 + dt * 0.1
            last = now
            with self._gcond:
                self._raw = f
                self._t_raw = now
                self._seq += 1
                self.grabbed += 1
                self._gcond.notify_all()


# ---------------- benchmark ----------------

def synth_frame(w, h):
    rng = np.random.default_rng(7)
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    img = np.dstack([120 + 100 * np.sin(x / 90.0),
                     90 + 90 * np.cos(y / 70.0),
                     60 + 120 * np.sin((x + y) / 130.0)])
    cv2.ellipse(img, (w // 2, int(h * 0.55)), (w // 8, h // 4), 0, 0, 360, (95, 120, 165), -1)
    img[:, int(w * 0.75):] += 90                  # backlit window
    img += rng.normal(0, 9, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def bench(camera=None, seconds=1.0):
    print(f"opencv {cv2.__version__} | cv threads {cv2.getNumThreads()} | "
          f"mediapipe {'yes' if mp else 'no'} | pyvirtualcam {'yes' if pyvirtualcam else 'no'}")
    if ort:
        onnx_providers()
    print(f"opencl {'on: ' + GPU['device'] if GPU['on'] else 'off'} | onnx ep {GPU['ep']}")
    targets = list(RESOLUTIONS.items())
    for label, (w, h) in targets:
        if camera is not None:
            cap = cv2.VideoCapture(camera, cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
            ok = False
            for _ in range(5):
                ok, frame = cap.read()
            cap.release()
            if not ok:
                continue
            h, w = frame.shape[:2]
            label = f"{w}x{h}"
        else:
            frame = synth_frame(w, h)

        print(f"\n  {label}")
        total = 0.0
        per = {}
        for f in build_filters():
            if not f.available:
                print(f"    {f.name:<14}  unavailable ({f.hint})")
                continue
            f.enabled = True
            src = Pipeline.convert(frame, "bgr", f.space)
            dup = ((lambda: [c.copy() for c in src]) if isinstance(src, list)
                   else (lambda: src.copy()))
            try:
                f.process(dup())                  # warm up
                n, t0 = 0, time.perf_counter()
                while time.perf_counter() - t0 < seconds:
                    f.process(dup())
                    n += 1
            except Exception as e:
                print(f"    {f.name:<14}  error: {e}")
                continue
            if not f.available:
                print(f"    {f.name:<14}  unavailable ({f.hint})")
                continue
            ms = (time.perf_counter() - t0) / n * 1000.0
            total += ms
            per[f.name] = ms
            print(f"    {f.name:<14}{ms:7.2f} ms   {1000.0 / ms:6.0f} fps solo")

        # neural_local stands in for four classical stages, so the two are never
        # both live - summing every filter would price a configuration that
        # cannot happen
        classical = sum(v for k, v in per.items() if k not in ("neural_local", "neural_tone"))
        neural = sum(v for k, v in per.items()
                     if k not in Pipeline.SUBSUMED and k != "neural_tone")
        for tag, val in (("classical", classical),
                         ("+ neural_local", neural if "neural_local" in per else None)):
            if val is None:
                continue
            n = Pipeline([]).stages
            cap = min(BUDGET_MS * n, LATENCY_MS)
            print(f"    {tag:<14}{val:7.2f} ms   {1000.0 * n / val:6.0f} fps max "
                  f"over {n} stage(s)   -> {'OK' if val < cap else 'OVER'} "
                  f"({val / cap * 100:.0f}% of {cap:.0f} ms at 30 fps)")


def gpu_check(seconds=1.5):
    """Is the GPU actually helping? Runs the same chain both ways and says so.

    OpenCV's OpenCL path is not free: every stage that falls back to numpy pulls
    the frame across the bus and pushes it back. On a laptop with a discrete GPU
    that is a PCIe round trip per readback, and a chain with three of them can
    land slower than plain CPU. This is a per-machine question, so measure it
    rather than assume."""
    print(f"opencl device : {cv2.ocl.Device_getDefault().name() if cv2.ocl.haveOpenCL() else 'none'}")
    if ort is None:
        print("onnxruntime   : not installed")
    else:
        have = ort.get_available_providers()
        print(f"onnx providers: {', '.join(have)}")
        if not any(k in have for k in ("CUDAExecutionProvider",
                                       "DmlExecutionProvider",
                                       "ROCMExecutionProvider")):
            print("                CPU only - the installed wheel is plain "
                  "'onnxruntime'.")
            print("                pip install onnxruntime-directml   "
                  "(any DX12 GPU, ~150 MB)")
            print("                pip install onnxruntime-gpu        "
                  "(NVIDIA, needs CUDA + cuDNN runtime)")
    frame = synth_frame(1280, 720)
    for use in (False, True):
        if use and not cv2.ocl.haveOpenCL():
            continue
        gpu_init(use)
        fs = build_filters()
        pl = Pipeline(fs)
        pl.auto_quality = False
        for _ in range(8):
            pl.process(frame.copy())
        n, t0 = 0, time.perf_counter()
        while time.perf_counter() - t0 < seconds:
            pl.process(frame.copy())
            n += 1
        ms = (time.perf_counter() - t0) / n * 1000.0
        worst = sorted((f for f in fs if f.ms > 0.3), key=lambda f: -f.ms)[:4]
        print(f"\n  {'OpenCL' if use else 'CPU':<7}{ms:7.2f} ms  {1000.0/ms:5.1f} fps   "
              + "  ".join(f"{f.name} {f.ms:.1f}" for f in worst))
    n = Pipeline([]).stages
    print(f"\n  {n} stage(s) divide this for throughput; latency stays at the full "
          f"number and is capped at {LATENCY_MS:.0f} ms.")


def list_modes(index, size="1280x720", fps=30):
    """What the driver says it offers, then what each way of asking for it
    actually delivers. Ignores the saved result so it always measures."""
    w, h = (int(v) for v in size.lower().split("x"))
    if os.name == "nt":
        try:
            from pygrabber.dshow_graph import FilterGraph
            g = FilterGraph()
            g.add_video_input_device(index)
            fmts = g.get_input_device().get_formats()
            print(f"driver formats (DirectShow), camera {index}:")
            for f in fmts:
                print("  ", {k: f[k] for k in f if k != "index"})
        except Exception as e:
            print(f"driver formats: unavailable ({type(e).__name__}: {e}) "
                  f"- pip install pygrabber")
    print(f"\nmeasured, asking for {w}x{h} @ {fps}:")
    cap, api, api_name, frame, _, info = find_mode(index, (w, h), fps, log=print,
                                                   cache=False)
    if cap is None:
        print("no frame")
        return
    cap.release()
    print(f"\nchosen: {api_name} {frame.shape[1]}x{frame.shape[0]} "
          f"{info.get('rate', 0):.1f} fps, order {info.get('order')}"
          + ("  (every size equally slow - low light)" if info.get("dark") else ""))


def list_cameras():
    print(f"opencv {cv2.__version__} | names {'yes' if device_names() else 'no (pip install pygrabber)'}")
    cams = probe_cameras()
    for c in cams:
        print(f"  [{c['index']}] {c['api_name']:<5} {c['w']}x{c['h']}  {c['name']}"
              f"{'  [virtual]' if c['virtual'] else ''}{'  [IR]' if c['ir'] else ''}")
    best = pick_camera(cams)
    print(f"  -> auto pick: {cam_label(best)}" if best else "  no usable camera found")
    if not cams:
        print("  - close OBS / Zoom / Teams / Camera app and retry")
        print("  - Windows: Settings > Privacy & security > Camera > "
              "'Let desktop apps access your camera' must be ON")
        print("  - check the hardware camera toggle / Fn key on the laptop")


# ---------------- gui ----------------

def run_gui():
    import tkinter as tk
    import customtkinter as ctk
    from PIL import Image, ImageTk

    class FilterCard(ctk.CTkFrame):
        def __init__(self, master, flt):
            super().__init__(master)
            self.flt = flt
            self.rows = {}
            head = ctk.CTkFrame(self, fg_color="transparent")
            head.pack(fill="x", padx=10, pady=(8, 2))
            self.var = ctk.BooleanVar(value=flt.enabled)
            sw = ctk.CTkSwitch(head, text=flt.label, variable=self.var, command=self._toggle)
            sw.pack(side="left")
            self.ms = ctk.CTkLabel(head, text="", width=64, anchor="e",
                                   font=ctk.CTkFont(size=11), text_color="#8a8a8a")
            self.ms.pack(side="right")
            if not flt.available:
                sw.configure(state="disabled")
                ctk.CTkLabel(self, text=flt.hint, font=ctk.CTkFont(size=11),
                             text_color="#c08040").pack(anchor="w", padx=12, pady=(0, 6))
                return
            if flt.locked:
                sw.configure(state="disabled")
                ctk.CTkLabel(head, text=T("locked"), font=ctk.CTkFont(size=11),
                             text_color="#60b080").pack(side="left", padx=8)
            for key, (lk, lo, hi, _) in flt.params.items():
                if flt.locked and key == "strength":
                    continue
                row = ctk.CTkFrame(self, fg_color="transparent")
                row.pack(fill="x", padx=10, pady=(0, 6))
                ctk.CTkLabel(row, text=T(lk), width=84, anchor="w").pack(side="left")
                val = ctk.CTkLabel(row, text="", width=44, anchor="e")
                sl = ctk.CTkSlider(row, from_=lo, to=hi, command=lambda v, k=key: self._set(k, v))
                sl.set(flt.values[key])
                sl.pack(side="left", fill="x", expand=True, padx=6)
                val.pack(side="right")
                val.configure(text=f"{flt.values[key]:.2f}")
                self.rows[key] = (sl, val)

        def _toggle(self):
            self.flt.enabled = self.var.get()

        def _set(self, key, v):
            self.flt.values[key] = float(v)
            self.rows[key][1].configure(text=f"{v:.2f}")

        def tick(self):
            if self.flt.error:
                self.ms.configure(text=self.flt.error, text_color="#e06060")
            else:
                self.ms.configure(text=f"{self.flt.ms:.1f} ms" if self.flt.enabled else "",
                                  text_color="#8a8a8a")

        def sync_quality(self):
            row = self.rows.get("quality")
            if row and abs(row[0].get() - self.flt.values["quality"]) > 1e-3:
                row[0].set(self.flt.values["quality"])
                row[1].configure(text=f"{self.flt.values['quality']:.2f}")

        def refresh(self):
            if self.flt.locked:
                self.flt.enabled = True
            self.var.set(self.flt.enabled)
            for key, (sl, val) in self.rows.items():
                sl.set(self.flt.values[key])
                val.configure(text=f"{self.flt.values[key]:.2f}")

    class App(ctk.CTk):
        def __init__(self):
            super().__init__()
            self.title(T("title"))
            self.geometry("1320x780")
            self.minsize(900, 600)
            ctk.set_appearance_mode("dark")
            self.filters = build_filters()
            self.pipeline = Pipeline(self.filters)
            self.worker = None
            self._photo = None
            self.cams = []
            self._pref_index = None
            self._pv = 0

            left = ctk.CTkFrame(self)
            self.preview = tk.Label(left, bg="#161616", text=T("preview_off"), fg="#777777")
            self.preview.pack(fill="both", expand=True, padx=8, pady=8)
            bottom = ctk.CTkFrame(left, fg_color="transparent")
            bottom.pack(fill="x", padx=10, pady=(0, 8))
            self.status = ctk.CTkLabel(bottom, text=T("idle"), anchor="w", justify="left",
                                       wraplength=700)
            self.status.pack(side="left", fill="x", expand=True)
            self.view = ctk.CTkSegmentedButton(
                bottom, values=[T("v_both"), T("v_after"), T("v_before")])
            self.view.set(T("v_both"))
            self.view.pack(side="right")

            right = ctk.CTkFrame(self, width=340)
            # packed before the preview side: the packer serves widgets in pack
            # order, and a long status line used to squeeze this panel to a sliver
            right.pack(side="right", fill="y", padx=(0, 10), pady=10)
            right.pack_propagate(False)
            left.pack(side="left", fill="both", expand=True, padx=10, pady=10)

            top = ctk.CTkFrame(right)
            top.pack(fill="x", padx=8, pady=8)
            top.grid_columnconfigure(1, weight=1)
            ctk.CTkLabel(top, text=T("camera")).grid(row=0, column=0, padx=6, pady=4, sticky="w")
            self.dev = ctk.CTkOptionMenu(top, values=[T("detecting")], width=150,
                                         command=lambda _: self.on_pick())
            self.dev.grid(row=0, column=1, padx=6, pady=4, sticky="ew")
            ctk.CTkButton(top, text="\u21bb", width=28,
                          command=self.rescan).grid(row=0, column=2, padx=(0, 6))
            ctk.CTkLabel(top, text=T("resolution")).grid(row=1, column=0, padx=6, pady=4,
                                                         sticky="w")
            self.res = ctk.CTkOptionMenu(top, values=list(RESOLUTIONS), width=120,
                                         command=lambda _: self.on_pick())
            self.res.set("1280x720")
            self.res.grid(row=1, column=1, padx=6, pady=4)
            ctk.CTkLabel(top, text=T("mode")).grid(row=2, column=0, padx=6, pady=4, sticky="w")
            self._mode_label = {"q30": T("m_q30"), "fps60": T("m_fps60")}
            self.mode = ctk.CTkSegmentedButton(
                top, values=[self._mode_label["q30"], self._mode_label["fps60"]],
                command=lambda _: self.on_pick())
            self.mode.set(self._mode_label["q30"])
            self.mode.grid(row=2, column=1, columnspan=2, padx=6, pady=4, sticky="ew")
            self._note = ""

            self.btn = ctk.CTkButton(right, text=T("start"), command=self.toggle)
            self.btn.pack(fill="x", padx=8, pady=(0, 6))
            # Off by default. Denoise and sharpen - most of the work - run on
            # the CPU, so with OpenCL on every frame went up to the GPU and
            # came back down mid-chain; on laptops OpenCL also tends to pick
            # the weak integrated GPU, not the discrete one. Measured on a
            # Ryzen 9 7945HX laptop: with it on, full processing resolution
            # blew the latency budget and the chain stayed at half scale.
            gpu_init(False)
            self.gpu = ctk.BooleanVar(value=False)
            ctk.CTkSwitch(right, text=T("gpu"), variable=self.gpu,
                          command=self.on_gpu).pack(anchor="w", padx=14, pady=(0, 2))
            ctk.CTkButton(right, text=T("cam_settings"), height=24,
                          command=lambda: self.worker and self.worker.request_settings()
                          ).pack(fill="x", padx=8, pady=(2, 4))
            self.autoq = ctk.BooleanVar(value=True)
            ctk.CTkSwitch(right, text=T("autoq"), variable=self.autoq,
                          command=lambda: setattr(self.pipeline, "auto_quality",
                                                  self.autoq.get())).pack(
                anchor="w", padx=14, pady=(0, 8))

            scroll = ctk.CTkScrollableFrame(right, label_text=T("filters"))
            scroll.pack(fill="both", expand=True, padx=8, pady=(0, 8))
            self.cards = []
            for f in self.filters:
                c = FilterCard(scroll, f)
                c.pack(fill="x", pady=4)
                self.cards.append(c)

            bar = ctk.CTkFrame(right, fg_color="transparent")
            bar.pack(fill="x", padx=8, pady=(0, 8))
            ctk.CTkButton(bar, text=T("save"), command=self.save).pack(
                side="left", expand=True, fill="x", padx=(0, 4))
            ctk.CTkButton(bar, text=T("load"), command=self.load).pack(
                side="left", expand=True, fill="x", padx=(4, 0))

            self.load(silent=True)
            self.protocol("WM_DELETE_WINDOW", self.close)
            self.after(33, self.tick)
            self.rescan(autostart=True)

        # -- camera detection --
        def rescan(self, autostart=False):
            if self.worker:
                self.toggle()
            self.dev.configure(values=[T("detecting")])
            self.dev.set(T("detecting"))
            self.status.configure(text=T("detecting"))
            threading.Thread(target=self._probe, args=(autostart,), daemon=True).start()

        def _probe(self, autostart):
            cams = probe_cameras()
            self.after(0, self._probe_done, cams, autostart)

        def _probe_done(self, cams, autostart):
            self.cams = cams
            if not cams:
                self.dev.configure(values=[T("no_camera")])
                self.dev.set(T("no_camera"))
                self.status.configure(text=T("no_camera_hint"))
                return
            self.dev.configure(values=[cam_label(c) for c in cams])
            want = next((c for c in cams if c["index"] == self._pref_index), None)
            self.dev.set(cam_label(want or pick_camera(cams)))
            self.status.configure(text=T("idle"))
            if autostart:
                self.toggle()

        def current_cam(self):
            label = self.dev.get()
            return next((c for c in self.cams if cam_label(c) == label), None)

        def on_gpu(self):
            gpu_init(self.gpu.get())
            # costs just changed: let the controller try full scale again now
            # instead of waiting out a back-off earned under the old setting
            self.pipeline._work_bar.clear()
            self.pipeline._work_back.clear()

        def mode_key(self):
            label = self.mode.get()
            return next((k for k, v in self._mode_label.items() if v == label), "q30")

        def on_pick(self):
            if self.worker:
                self.toggle()
                self.toggle()

        def toggle(self):
            if self.worker and self.worker.is_alive():
                self.worker.stop()
                self.worker.join(timeout=2)
                self.worker = None
                self.btn.configure(text=T("start"))
                self.status.configure(text=T("stopped"))
                self.preview.configure(image="", text=T("preview_off"))
                return
            cam = self.current_cam()
            if cam is None:
                self.status.configure(text=T("no_camera_hint"))
                return
            self._pref_index = cam["index"]
            self._note = ""
            self.worker = Worker(cam["index"], RESOLUTIONS[self.res.get()],
                                 self.mode_key(), self.pipeline, api=cam["api"],
                                 name=cam.get("name", ""))
            self.worker.start()
            self.btn.configure(text=T("stop"))

        def tick(self):
            t_tick = time.perf_counter()
            w = self.worker
            if w and w.is_alive():
                pl = self.pipeline
                stage = max(max(pl.stage_ms), pl.loop_ms)
                lat = w.present_lat
                mark = ("  !" if stage > pl.budget_ms or lat > LATENCY_MS else "")
                if w.fallback:
                    # 60 was not available: show 30 as what is actually running
                    self._note = w.fallback
                    if self.mode.get() != self._mode_label[w.mode]:
                        self.mode.set(self._mode_label[w.mode])
                gen = (f"\n!! {w.pipeline_error}" if w.pipeline_error else "")
                self.status.configure(
                    text=(f"!! {self._note}\n" if self._note else "")
                         + (f"!! {w.diag}\n" if w.diag else "")
                         + (f"{w.mode_note}\n" if w.mode_note and not w.diag else "")
                         + f"{w.status}   |   {w.fps_actual:.1f}/{w.fps_capture:.0f} fps   |   "
                         f"{T('budget')} {stage:.1f}/{pl.budget_ms:.1f} ms"
                         f" x{pl.stages}   |   "
                         f"{T('latency')} {lat:.0f}/{LATENCY_MS:.0f} ms" + gen
                         + "   |   "
                         f"{T('scene')} {SCENE['name']}   |   "
                         f"q {self.pipeline.scale:.2f} t{self.pipeline.boost}"
                         + (f" w{self.pipeline.work:.2f}"
                            if self.pipeline.work < 0.999 else "")
                         + (f" -{','.join(self.pipeline.shed)}" if self.pipeline.shed else "")
                         + "   |   "
                         f"{'GPU' if GPU['on'] else 'CPU'}/{GPU['ep']}{mark}")
                # Card widgets and the status text were being rebuilt on every
                # preview frame. With a dozen cards that is the most expensive
                # thing in the callback, and after(33) only starts counting once
                # the callback returns - so the preview was landing near 10 fps
                # while the pipeline was fine. They run at 5 Hz now.
                self._pv += 1
                if self._pv % 6 == 0:
                    for c in self.cards:
                        c.tick()
                        c.sync_quality()
                frames = w.get_frames()
                if frames:
                    raw, out = frames
                    both = self.view.get() == T("v_both")
                    imgs = [raw, out] if both else \
                        [out if self.view.get() == T("v_after") else raw]
                    pw = min(max(self.preview.winfo_width() - 4, 320), 1200)
                    ph = min(max(self.preview.winfo_height() - 4, 240), 700)
                    h0, w0 = raw.shape[:2]
                    sc = min((pw / len(imgs)) / w0, ph / h0)
                    size = (max(int(w0 * sc), 64), max(int(h0 * sc), 48))
                    # INTER_LINEAR downscaling a 1280 px frame to ~600 px
                    # aliases every fine edge - the preview looked far worse than
                    # the frame actually being sent
                    ip = cv2.INTER_AREA if size[0] < w0 else cv2.INTER_LINEAR
                    shown = np.hstack([cv2.resize(i, size, interpolation=ip)
                                       for i in imgs])
                    if both:
                        cv2.putText(shown, T("original"), (10, 26), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.6, (255, 255, 255), 1, cv2.LINE_AA)
                        cv2.putText(shown, T("enhanced"), (size[0] + 10, 26),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 180), 1, cv2.LINE_AA)
                    self._photo = ImageTk.PhotoImage(
                        Image.fromarray(cv2.cvtColor(shown, cv2.COLOR_BGR2RGB)))
                    self.preview.configure(image=self._photo, text="")
                    self._shown_id = id(frames)
            elif w and w.error:
                self.status.configure(text=w.error)
                self.btn.configure(text=T("start"))
                self.worker = None
            # schedule against elapsed time, not on top of it
            spent = (time.perf_counter() - t_tick) * 1000.0
            self.after(max(int(33 - spent), 1), self.tick)

        def save(self):
            data = {f.name: {"enabled": f.enabled, "values": f.values} for f in self.filters}
            data["_device"] = {"index": self._pref_index, "res": self.res.get(),
                               "mode": self.mode_key(), "gpu": bool(self.gpu.get())}
            data["_v"] = PRESET_V
            with open(PRESET_PATH, "w", encoding="utf-8") as fp:
                json.dump(data, fp, ensure_ascii=False, indent=2)
            self.status.configure(text=PRESET_PATH)

        def load(self, silent=False):
            if not os.path.exists(PRESET_PATH):
                return
            try:
                with open(PRESET_PATH, encoding="utf-8") as fp:
                    data = json.load(fp)
            except Exception:
                return
            if data.get("_v") != PRESET_V:
                data = {"_device": data.get("_device", {})}
            for f in self.filters:
                item = data.get(f.name)
                if not item:
                    continue
                f.enabled = (bool(item.get("enabled")) and f.available) or f.locked
                for k, v in item.get("values", {}).items():
                    if k in f.values and not (f.locked and k == "strength"):
                        f.values[k] = float(v)
            dev = data.get("_device", {})
            self._pref_index = dev.get("index")
            self.res.set(dev.get("res", "1280x720"))
            self.mode.set(self._mode_label.get(dev.get("mode"), self._mode_label["q30"]))
            if bool(dev.get("gpu", False)) != bool(self.gpu.get()):
                self.gpu.set(bool(dev.get("gpu", False)))
                self.on_gpu()
            for c in self.cards:
                c.refresh()

        def close(self):
            if self.worker:
                self.worker.stop()
                self.worker.join(timeout=2)
            self.destroy()

    App().mainloop()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--camera", type=int, default=None)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--modes", action="store_true",
                    help="measure which backend/property order gives full rate")
    ap.add_argument("--size", default="1280x720")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--gpu-check", action="store_true")
    ap.add_argument("--vcam-check", action="store_true")
    ap.add_argument("--lang", choices=["en", "ko"], default=None)
    ap.add_argument("--gpu", action="store_true", help="use OpenCL (off by default)")
    ap.add_argument("--no-gpu", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.lang:
        LANG = args.lang
    gpu_init(args.gpu and not args.no_gpu)
    print(f"gpu: {'OpenCL ' + GPU['device'] if GPU['on'] else 'off (CPU)'}")
    if args.vcam_check:
        ok, bad = vcam_probe()
        print("usable backends:", ", ".join(ok) if ok else "none")
        for name, why in bad:
            print(f"  {name:<14} {why}")
        if not ok:
            print("\n" + T("vcam_none"))
            print("  https://obsproject.com/  - install once, it registers the "
                  "virtual camera system-wide")
    elif args.gpu_check:
        gpu_check()
    elif args.list:
        list_cameras()
    elif args.modes:
        list_modes(args.camera or 0, args.size, args.fps)
    elif args.bench:
        bench(args.camera)
    else:
        run_gui()

"""
hinge_demo.py - 웹캠 기반 힌지 각도 추정 + 서리 유리 전환 오버레이 (데모)

아이폰 듀오의 펼침 전환처럼, 덮개를 닫으면 화면이 서리 낀 유리처럼 초점을 잃고
가장 흐린 지점에서 '닫힘 화면(시계)'으로 넘어간 뒤 다시 초점이 맞는다. 여는 동작은 역순.
전환의 모든 프레임이 실제 힌지 각도에 대응한다.

설치:  pip install opencv-python numpy PySide6
실행:  python hinge_demo.py            # 웹캠 모드
       python hinge_demo.py --sim      # 웹캠 없이 슬라이더로 연출만 확인
옵션:  --cam 0  --screen 0  --open 110  --vfov 43  --res 0.8  --no-hud  --log hinge.csv

사용법:
  1) 노트북을 평소 쓰는 각도로 펼친 뒤 [현재 각도를 '완전 펼침'으로] 클릭
  2) 상판을 천천히 닫았다 열어 본다
  3) 닫는데 각도가 올라가면 [방향 반전] 클릭
  * 끝까지 닫는 연출을 보려면 전원 옵션에서 덮개 닫기 동작을 '아무 것도 안 함'으로 변경
"""
import argparse
import collections
import math
import sys
import threading
import time

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets


def _clamp(x, a=0.0, b=1.0):
    return max(a, min(b, x))


def _smooth(x):
    x = _clamp(x)
    return x * x * (3 - 2 * x)


# ───────────────────────── 필터 ─────────────────────────
class OneEuro:
    """느릴 땐 강하게, 빠를 땐 약하게 걸리는 저역 필터 (지터 제거용). dx = 각속도(도/초)."""

    def __init__(self, min_cutoff=1.5, beta=0.08, d_cutoff=2.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.x = None
        self.dx = 0.0

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def reset(self, x=None):
        self.x, self.dx = x, 0.0

    def __call__(self, x, dt):
        if self.x is None:
            self.x = x
            return x
        dx = (x - self.x) / dt
        self.dx += self._alpha(self.d_cutoff, dt) * (dx - self.dx)
        cutoff = self.min_cutoff + self.beta * abs(self.dx)
        self.x += self._alpha(cutoff, dt) * (x - self.x)
        return self.x


# ───────────────────── 힌지 각도 추정 ─────────────────────
class HingeEstimator:
    """
    상판이 힌지를 축으로 돌면 카메라는 피치 회전만 한다. 이때 영상 속 각 점의
    '시선 각도' atan((y - cy) / f) 는 모든 점에서 똑같은 양만큼 변한다.
    점마다 그 변화량을 구해 이상치를 걸러 평균하면 프레임당 회전량이 1/100도 단위로 나온다.
    카메라가 키보드에 가려 어두워지면 직전 각속도로 외삽한다.
    """

    MAX_ANGLE = 140.0
    NOISE_DEG = 0.006        # 이보다 작은 프레임당 변화는 잡음으로 보고 누적하지 않는다

    def __init__(self, open_angle=110.0, vfov_deg=43.0, dark_thresh=25.0):
        import cv2  # sim 모드에서는 필요 없으므로 지연 import

        self.cv2 = cv2
        self.open_angle = open_angle
        self.vfov = math.radians(vfov_deg)
        self.dark_thresh = dark_thresh
        self.angle = open_angle
        self.vel = 0.0          # 도/초
        self.dtheta = 0.0       # 마지막 프레임의 회전량(도)
        self.npts = 0           # 계산에 쓰인 점 개수
        self.sign = 1.0
        self.state = "init"
        self.prev = None
        self.pts = None

    def calibrate(self):
        self.angle = self.open_angle
        self.vel = 0.0

    def flip(self):
        self.sign = -self.sign

    def _detect(self, gray):
        return self.cv2.goodFeaturesToTrack(gray, maxCorners=300, qualityLevel=0.01, minDistance=10)

    def step(self, gray, dt):
        cv2 = self.cv2
        h = gray.shape[0]
        f = (h / 2) / math.tan(self.vfov / 2)       # 초점거리(px)
        cy0 = h / 2
        self.dtheta = 0.0

        # 1) 가려짐: 닫히는 중이었다면 각속도로 외삽
        if float(gray.mean()) < self.dark_thresh:
            if self.vel < 0:
                self.angle = max(0.0, self.angle + self.vel * dt)
            self.prev, self.pts, self.npts = None, None, 0
            self.state = "blocked"
            return self.angle

        # 2) 특징점 확보
        if self.prev is None or self.pts is None or len(self.pts) < 40:
            self.prev = gray
            self.pts = self._detect(gray)
            self.npts = 0 if self.pts is None else len(self.pts)
            self.state = "seek"
            return self.angle

        # 3) LK 광학 흐름 (서브픽셀)
        nxt, st, _ = cv2.calcOpticalFlowPyrLK(
            self.prev, gray, self.pts, None, winSize=(21, 21), maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.003),
        )
        good = st.ravel() == 1 if st is not None else np.zeros(0, bool)
        if good.sum() < 15:
            self.prev, self.pts = gray, self._detect(gray)
            self.vel *= 0.8
            self.npts = int(good.sum())
            self.state = "lost"
            return self.angle

        p0 = self.pts[good].reshape(-1, 2)
        p1 = nxt[good].reshape(-1, 2)
        d = p1 - p0
        dx, dy = float(np.median(d[:, 0])), float(np.median(d[:, 1]))

        # 4) 점별 회전량 → 이상치 제거 → 평균
        a = np.degrees(np.arctan((p1[:, 1] - cy0) / f) - np.arctan((p0[:, 1] - cy0) / f))
        med = float(np.median(a))
        mad = float(np.median(np.abs(a - med)))
        inl = np.abs(a - med) <= max(0.03, 2.5 * mad)
        dth = float(a[inl].mean())
        self.npts = int(inl.sum())

        # 5) 점들이 한 방향으로, 세로 위주로 움직였을 때만 각도에 반영
        vertical = abs(dy) >= abs(dx) or abs(dx) < 0.3
        if abs(dth) > self.NOISE_DEG and vertical and inl.mean() > 0.5:
            dth *= self.sign
            self.dtheta = dth
            self.angle = min(self.MAX_ANGLE, max(0.0, self.angle + dth))
            self.vel += 0.5 * (dth / dt - self.vel)
            self.state = "track"
        else:
            self.vel *= 0.8
            self.state = "idle"

        self.prev = gray
        self.pts = nxt[good].reshape(-1, 1, 2)
        if len(self.pts) < 80:                      # 점이 화면 밖으로 빠지면 즉시 보충 (프레임 손실 방지)
            self.pts = self._detect(gray)
        return self.angle


class CameraThread(threading.Thread):
    PROC_W = 640     # 처리 해상도 폭. 클수록 각도 분해능이 좋아진다

    def __init__(self, est, cam_index=0):
        super().__init__(daemon=True)
        self.est, self.cam_index = est, cam_index
        self.running = True
        self.error = None
        self.lock = threading.Lock()
        self.fps = 0.0
        self.frame = 0

    def run(self):
        cv2 = self.est.cv2
        cap = cv2.VideoCapture(self.cam_index, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))   # 60fps 를 받으려면 대개 MJPG 필요
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 60)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)                              # 밀린 프레임 없이 최신 것만
        if not cap.isOpened():
            self.error = "웹캠을 열 수 없음 (다른 앱이 점유 중인지 확인)"
            return
        last = time.perf_counter()
        while self.running:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.02)
                continue
            now = time.perf_counter()
            dt, last = max(1e-3, now - last), now
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if gray.shape[1] > self.PROC_W:
                s = self.PROC_W / gray.shape[1]
                gray = cv2.resize(gray, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
            with self.lock:
                self.est.step(gray, dt)
                self.fps += 0.1 * (1.0 / dt - self.fps)
                self.frame += 1
        cap.release()

    def read(self):
        with self.lock:
            e = self.est
            return dict(raw=e.angle, state=e.state, pts=e.npts, dtheta=e.dtheta, fps=self.fps, frame=self.frame)


# ───────────────────────── 오버레이 ─────────────────────────
class Overlay(QtWidgets.QWidget):
    """
    아이폰 듀오식 '서리 낀 유리' 전환을 힌지 각도에 물린 연출. 3D 모델 없이 화면 자체가 변한다.
      - 덮개를 닫으면 빛나는 유리 경계선이 화면 위에서 아래로 내려온다 (위치 = 힌지 각도)
      - 경계선 아래는 지금 화면, 위는 푸른 서리 유리. 경계에서는 화면이 렌즈처럼 굴절되고 색이 갈라진다
      - 서리 유리는 닫힐수록 걷히며 '닫힘 화면(시계)'의 초점이 맞는다
      - 빨리 움직일수록 경계선이 굵고 밝아지고 불꽃 입자가 뒤로 흩날린다
    여는 동작은 정확히 역순.
    """

    STIFF, DAMP = 300.0, 30.0       # 스프링 (거의 임계감쇠: 각도를 바짝 따라가되 떨림만 걸러낸다)
    SWEEP_T = 0.45
    SIGMAS = (0, 1.2, 2.5, 5, 10, 20, 40, 70)    # 흐림 단계(버퍼 폭 1600px 기준 시그마)

    def __init__(self, screen, megapixels=0.8, hud=True):
        super().__init__()
        # QScreen 객체를 붙잡아 두면 안 된다: 덮개를 닫아 내장 화면이 꺼지거나 모니터 구성이 바뀌면
        # 윈도우가 그 화면을 없애고 Qt 가 객체를 삭제한다. 이름만 기억해 두고 쓸 때마다 다시 찾는다.
        self.screen_name = screen.name()
        self.hud = hud
        self.q, self.vel, self.target = 1.0, 0.0, 1.0
        self.sweep = None
        self.clock = 0.0
        self.parts = []                 # 불꽃 입자 [x, y, vx, vy, 남은수명, 전체수명, 크기]
        self._yc = None                 # 경계선의 화면 내 높이(0~1), 가로 위치별
        self.info = {}
        self.hist = None                # (시각, 필터 각도, 원시 각도) deque - Panel 과 공유
        self.budget = max(0.2, megapixels) * 1e6     # 움직이는 동안의 내부 렌더 해상도(픽셀 수)
        self.snap = self.A = self.B = self.tint = self.noise = self._up = None
        self.minute = -1
        self.setWindowFlags(
            QtCore.Qt.FramelessWindowHint
            | QtCore.Qt.WindowStaysOnTopHint
            | QtCore.Qt.Tool
            | QtCore.Qt.WindowTransparentForInput
        )
        self.setAttribute(QtCore.Qt.WA_ShowWithoutActivating)
        self.setAttribute(QtCore.Qt.WA_OpaquePaintEvent)
        self.setGeometry(screen.geometry())
        self.timer = QtCore.QTimer(self)
        self.timer.setTimerType(QtCore.Qt.PreciseTimer)
        self.timer.timeout.connect(self._tick)
        self.last = time.perf_counter()

    # ── 상태 ──
    def set_target(self, p, info=None):
        self.target = p
        if info is not None:
            self.info = info
        if p < 1.0:
            self.sweep = None
            if not self.isVisible():
                scr = self._screen()
                if scr is None:                      # 대상 화면이 지금 꺼져 있음(덮개 닫힘 등) → 돌아오면 다시 시도
                    return
                try:
                    snap = scr.grabWindow(0)         # 오버레이가 뜨기 전 화면을 캡처
                    geo = scr.geometry()
                except RuntimeError:                 # 찾는 사이에 화면이 사라진 경우
                    return
                if snap.isNull():
                    return
                self.setGeometry(geo)
                self.prepare(snap)
                self.q, self.vel = 1.0, 0.0
                self.last = time.perf_counter()
                self.show()
                self.timer.start(8)

    def _screen(self):
        """연출을 띄울 화면을 이름으로 다시 찾는다. 없으면 None."""
        for sc in QtGui.QGuiApplication.screens():
            try:
                if sc.name() == self.screen_name:
                    return sc
            except RuntimeError:
                continue
        return None

    def prepare(self, snap):
        self.snap = snap
        self.A = self.B = None               # 다음 그리기 때 흐림 단계를 다시 만든다

    def _tick(self):
        now = time.perf_counter()
        dt, self.last = min(1 / 30, max(1e-3, now - self.last)), now
        self.clock += dt
        self._particles(dt)
        for _ in range(4):
            h = dt / 4
            acc = self.STIFF * (min(self.target, 1.0) - self.q) - self.DAMP * self.vel
            self.vel += acc * h
            self.q += self.vel * h
        if self.target >= 1.0:
            if self.sweep is None and self.q > 0.97:
                self.sweep = 0.0
            elif self.sweep is not None:
                self.sweep += dt
            settled = abs(self.q - 1) < 0.003 and abs(self.vel) < 0.02
            if settled and self.sweep is not None and self.sweep >= self.SWEEP_T:
                self.timer.stop()
                self.hide()
                self.snap = self.A = self.B = self._up = None
                self.parts.clear()
                return
        self.update()

    # ── 미리 만들어 두는 흐림 단계 ──
    def _stack(self, base):
        """한 장의 그림을 흐림 단계별로 미리 만들어 둔다. 프레임마다 흐리면 느리므로 이웃 단계를 섞어 쓴다."""
        import cv2

        bh, bw = base.shape[:2]
        k = bw / 1600.0
        out = [base]
        for s in self.SIGMAS[1:]:
            sg = s * k
            f = max(1, int(sg / 2))          # 많이 흐릴수록 작게 줄여서 흐린 뒤 다시 키운다(훨씬 빠름)
            if f > 1:
                small = cv2.resize(base, (max(2, bw // f), max(2, bh // f)), interpolation=cv2.INTER_AREA)
                small = cv2.GaussianBlur(small, (0, 0), sg / f)
                out.append(cv2.resize(small, (bw, bh), interpolation=cv2.INTER_CUBIC))
            else:
                out.append(cv2.GaussianBlur(base, (0, 0), max(0.3, sg)))
        return out

    def _ensure(self, bw, bh):
        import cv2

        minute = time.localtime().tm_min
        if self.A is None or self.A[0].shape[:2] != (bh, bw):
            img = self.snap.toImage().convertToFormat(QtGui.QImage.Format_RGB32)
            w, h = img.width(), img.height()
            a = np.frombuffer(img.constBits(), np.uint8).reshape(h, img.bytesPerLine())[:, : w * 4].reshape(h, w, 4)
            inter = cv2.INTER_AREA if bw < w else cv2.INTER_LINEAR
            self.A = self._stack(cv2.resize(a, (bw, bh), interpolation=inter))
            self.tint = np.empty((bh, bw, 4), np.uint8)
            self.tint[:] = (255, 176, 96, 255)                       # BGRA: 푸른 서리 색
            rng = np.random.default_rng(7)
            self.noise = rng.integers(96, 160, (bh, bw, 1), dtype=np.uint8).repeat(4, 2)
            self.noise[..., 3] = 128                                  # 알파에는 영향 없게
            self._u = np.linspace(0, 1, bw, dtype=np.float32)
            self._xs = np.arange(bw, dtype=np.float32)[None, :]
            self.B = None
        if self.B is None or minute != self.minute:
            self.minute = minute
            dark = cv2.addWeighted(self.A[-1], 0.40, self.tint, 0.08, 0)
            dark[..., 3] = 255
            self.B = self._stack(self._closed_card(dark))

    @staticmethod
    def _closed_card(bg):
        """닫힘 화면: 어둡게 가라앉힌 바탕 위에 시계."""
        bh, bw = bg.shape[:2]
        img = QtGui.QImage(bg.data, bw, bh, bw * 4, QtGui.QImage.Format_RGB32).copy()
        p = QtGui.QPainter(img)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        p.setRenderHint(QtGui.QPainter.TextAntialiasing)
        now = time.localtime()
        font = QtGui.QFont("Segoe UI")
        font.setWeight(QtGui.QFont.DemiBold)
        font.setPixelSize(max(10, int(bh * 0.20)))
        p.setFont(font)
        p.setPen(QtGui.QColor(255, 255, 255, 240))
        p.drawText(QtCore.QRectF(0, 0, bw, bh * 0.56), QtCore.Qt.AlignHCenter | QtCore.Qt.AlignBottom,
                   time.strftime("%H:%M", now))
        font.setWeight(QtGui.QFont.Normal)
        font.setPixelSize(max(8, int(bh * 0.036)))
        p.setFont(font)
        p.setPen(QtGui.QColor(255, 255, 255, 170))
        date = f"{now.tm_mon}월 {now.tm_mday}일 {'월화수목금토일'[now.tm_wday]}요일"
        p.drawText(QtCore.QRectF(0, bh * 0.585, bw, bh * 0.2), QtCore.Qt.AlignHCenter | QtCore.Qt.AlignTop, date)
        p.end()
        return np.frombuffer(img.constBits(), np.uint8).reshape(bh, img.bytesPerLine())[:, : bw * 4].reshape(bh, bw, 4).copy()

    def _layer(self, stack, lam, scale):
        """흐림 정도 lam(0~단계수-1, 실수)과 확대율로 한 장을 만든다."""
        import cv2

        n = len(stack)
        lam = _clamp(lam, 0.0, n - 1.0)
        i = min(int(lam), n - 2)
        f = lam - i
        if f < 0.004:
            img = stack[i]
        elif f > 0.996:
            img = stack[i + 1]
        else:
            img = cv2.addWeighted(stack[i], 1 - f, stack[i + 1], f, 0)
        if abs(scale - 1) > 1e-4:
            bh, bw = img.shape[:2]
            M = np.float32([[scale, 0, (1 - scale) * bw / 2], [0, scale, (1 - scale) * bh / 2]])
            img = cv2.warpAffine(img, M, (bw, bh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        return img

    def _compose(self, bw, bh):
        import cv2

        self._ensure(bw, bh)
        m = 1 - _clamp(self.q)                       # 0 = 펼침, 1 = 닫힘
        top = len(self.SIGMAS) - 1
        speed = _clamp(abs(self.vel) / 2.5)          # 움직임 세기 (0~1)
        fa = _smooth(m * 1.3) * 0.5                  # 지금 화면: 닫힐수록 조금씩 흐려지며 뒤로 물러난다
        fb = _smooth((1 - m) / 0.6)                  # 서리 유리: 닫힐수록 걷힌다
        A = self._layer(self.A, fa * top, 1 + 0.07 * m)

        # 유리 경계선: 가운데가 앞서가는 완만한 곡선 + 물결
        hb = max(6, int(bh * (0.075 + 0.05 * speed)))            # 경계 띠의 반폭
        cur = bh * 0.04
        pad = hb + cur
        y0 = -pad + m * (bh + 2 * pad)
        u = self._u
        yc = y0 + cur * (np.cos(np.pi * (u - 0.5)) - 0.6) + bh * 0.005 * np.sin(u * 18.8 + self.clock * 3.0)
        self._yc = yc / bh
        ya = int(max(0, math.floor(float(yc.min()) - hb)))
        yb = int(min(bh, math.ceil(float(yc.max()) + hb)))
        if yb <= 0:                                  # 경계선이 아직 화면 위
            return A

        B = self._layer(self.B, fb * top, 1 + 0.08 * fb)
        t = 0.24 * fb                                # 푸른 기
        if t > 0.003:
            B = cv2.addWeighted(B, 1 - t, self.tint, t, 0)
        k = 0.08 * fb                                # 서리 입자
        if k > 0.003:
            B = cv2.addWeighted(B, 1.0, self.noise, k, -128 * k)
        if ya >= bh:                                 # 경계선이 화면 아래로 지나감
            return B

        out = A.copy()
        out[:ya] = B[:ya]
        if yb > ya:
            n = yb - ya
            rows = np.arange(ya, yb, dtype=np.float32)[:, None]
            d = (rows - yc[None, :]) / hb                                # 경계선까지의 거리(띠 반폭 단위)
            mask = np.clip(0.5 - d / 0.7, 0, 1)
            mask = (mask * mask * (3 - 2 * mask)).astype(np.float32)     # 1 = 유리 쪽
            S = cv2.blendLinear(A[ya:yb], B[ya:yb], 1 - mask, mask)     # 8비트 그대로 섞는다(빠름)
            # 굴절: 경계 근처가 렌즈처럼 확대되고 물결친다. R/B 채널을 조금 다르게 꺾어 색을 가른다
            p = np.clip(1 - d * d, 0, 1) ** 2
            disp = -0.55 * hb * p * d
            wav = (bw * 0.004) * p * np.sin(rows * 0.08 + self.clock * 5.0)
            mx = np.ascontiguousarray(np.broadcast_to(self._xs, (n, bw)) + wav, dtype=np.float32)
            my = np.ascontiguousarray((rows - ya) + disp, dtype=np.float32)
            ca = 0.25 + 0.6 * speed
            sx = (ca * bw * 0.004) * p
            L, RP = cv2.INTER_LINEAR, cv2.BORDER_REPLICATE
            G = cv2.remap(S, mx, my, L, borderMode=RP)
            G[..., 2] = cv2.remap(np.ascontiguousarray(S[..., 2]), mx + sx, my + disp * ca, L, borderMode=RP)
            G[..., 0] = cv2.remap(np.ascontiguousarray(S[..., 0]), mx - sx, my - disp * ca, L, borderMode=RP)
            # 빛: 가는 흰 심선 + 넓은 푸른 번짐 (포화 덧셈)
            inten = 0.55 + 0.45 * speed
            core = np.exp(-((d / 0.045) ** 2)) * (235 * inten)
            halo = np.exp(-((d / 0.5) ** 2)) * inten
            glow = np.empty((n, bw, 4), np.uint8)
            glow[..., 0] = np.minimum(core + halo * 120, 255)
            glow[..., 1] = np.minimum(core + halo * 85, 255)
            glow[..., 2] = np.minimum(core + halo * 45, 255)
            glow[..., 3] = 0
            out[ya:yb] = cv2.add(G, glow)
        out[..., 3] = 255
        return out

    def _particles(self, dt):
        """경계선에서 튀어 뒤로 흩날리는 불꽃. 움직임이 빠를수록 많이 나온다."""
        W, H = self.width(), self.height()
        speed = _clamp(abs(self.vel) / 2.5)
        if self._yc is not None and speed > 0.03 and len(self.parts) < 400:
            back = 1.0 if self.vel > 0 else -1.0             # 닫는 중이면 위로, 여는 중이면 아래로 남는다
            for _ in range(np.random.poisson(speed * 280 * dt)):
                u = np.random.random()
                y = float(self._yc[int(u * (len(self._yc) - 1))]) * H
                if 0 <= y <= H:
                    life = 0.35 + 0.6 * np.random.random()
                    self.parts.append([u * W, y, (np.random.random() - 0.5) * 90,
                                       back * (40 + 260 * np.random.random()) * (0.4 + speed),
                                       life, life, 1.0 + 2.2 * np.random.random()])
        alive = []
        for p in self.parts:
            p[0] += p[2] * dt
            p[1] += p[3] * dt
            p[3] *= 0.97
            p[4] -= dt
            if p[4] > 0:
                alive.append(p)
        self.parts = alive

    # ── 그리기 ──
    def paintEvent(self, _):
        if self.snap is None:
            return
        qp = QtGui.QPainter(self)
        qp.setRenderHint(QtGui.QPainter.SmoothPixmapTransform)
        Wl, Hl = float(self.width()), float(self.height())
        if self.target >= 1.0 and abs(1 - self.q) < 0.004:
            # 다 펴진 뒤에는 원본 해상도 그대로
            qp.drawPixmap(QtCore.QRectF(0, 0, Wl, Hl), self.snap, QtCore.QRectF(self.snap.rect()))
        else:
            import cv2

            dpr = self.devicePixelRatioF()
            Wd, Hd = int(round(Wl * dpr)), int(round(Hl * dpr))
            k = min(1.0, math.sqrt(self.budget / (Wd * Hd)))
            bw, bh = max(16, int(Wd * k)), max(16, int(Hd * k))
            f = self._compose(bw, bh)
            if (bw, bh) != (Wd, Hd):
                f = cv2.resize(f, (Wd, Hd), interpolation=cv2.INTER_LINEAR)
            self._up = np.ascontiguousarray(f)       # QImage 가 참조하는 동안 살아 있어야 한다
            img = QtGui.QImage(self._up.data, Wd, Hd, Wd * 4, QtGui.QImage.Format_RGB32)
            img.setDevicePixelRatio(dpr)
            qp.drawImage(0, 0, img)
            if self.parts:                           # 불꽃 입자 (빛을 더하는 방식으로 그린다)
                qp.save()
                qp.setRenderHint(QtGui.QPainter.Antialiasing)
                qp.setCompositionMode(QtGui.QPainter.CompositionMode_Plus)
                qp.setPen(QtCore.Qt.NoPen)
                us = max(0.8, Hl / 1000.0)
                for x, y, _, _, life, full, size in self.parts:
                    a = life / full
                    qp.setBrush(QtGui.QColor(150, 200, 255, int(230 * a)))
                    r = (size * a + 0.4) * us
                    qp.drawEllipse(QtCore.QPointF(x, y), r, r)
                qp.restore()
            if self.hud:
                self._draw_hud(qp, Wl, Hl)
        self._draw_sweep(qp, Wl, Hl)
        qp.end()

    def _draw_sweep(self, qp, W, H):
        """다 펴졌을 때 빛이 한 번 훑는다."""
        if self.sweep is None or self.sweep >= self.SWEEP_T:
            return
        u = self.sweep / self.SWEEP_T
        x = (-0.4 + 1.8 * u) * W
        g = QtGui.QLinearGradient(x - 0.22 * W, 0, x + 0.22 * W, 0.2 * H)
        g.setColorAt(0, QtGui.QColor(255, 255, 255, 0))
        g.setColorAt(0.5, QtGui.QColor(255, 255, 255, int(60 * math.sin(math.pi * u))))
        g.setColorAt(1, QtGui.QColor(255, 255, 255, 0))
        qp.fillRect(QtCore.QRectF(x - 0.3 * W, 0, 0.6 * W, H), g)

    def _draw_hud(self, qp, Wl, Hl):
        """실시간 힌지 값 (좌상단). 글자는 원본 해상도로 그려야 선명하다."""
        info = self.info
        fade = _clamp((1 - _clamp(self.q)) * 10)
        if fade <= 0 or not info:
            return
        qp.save()
        qp.setRenderHint(QtGui.QPainter.Antialiasing)
        qp.setOpacity(fade)
        u = max(0.8, Hl / 1000.0)
        bx, by, bwid, bhei = 24 * u, 24 * u, 380 * u, 168 * u
        qp.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255, 50), 1))
        qp.setBrush(QtGui.QColor(8, 9, 14, 170))
        qp.drawRoundedRect(QtCore.QRectF(bx, by, bwid, bhei), 14 * u, 14 * u)
        mono = QtGui.QFont("Consolas")
        mono.setStyleHint(QtGui.QFont.Monospace)
        mono.setPixelSize(int(11 * u))
        qp.setFont(mono)
        qp.setPen(QtGui.QColor(255, 255, 255, 140))
        qp.drawText(QtCore.QPointF(bx + 16 * u, by + 24 * u), "HINGE ANGLE")
        mono.setPixelSize(int(46 * u))
        mono.setBold(True)
        qp.setFont(mono)
        qp.setPen(QtGui.QColor(255, 255, 255))
        qp.drawText(QtCore.QPointF(bx + 14 * u, by + 72 * u), f"{info.get('angle', 0.0):7.2f}°")
        mono.setBold(False)
        mono.setPixelSize(int(12 * u))
        qp.setFont(mono)
        qp.setPen(QtGui.QColor(255, 255, 255, 190))
        qp.drawText(
            QtCore.QPointF(bx + 16 * u, by + 94 * u),
            f"ω {info.get('vel', 0.0):+8.2f}°/s   raw {info.get('raw', 0.0):7.2f}°   Δ {info.get('dtheta', 0.0):+.3f}°",
        )
        qp.drawText(
            QtCore.QPointF(bx + 16 * u, by + 112 * u),
            f"{info.get('fps', 0.0):5.1f} fps   pts {info.get('pts', 0):3d}   {info.get('state', '')}",
        )
        if self.hist:                                    # 최근 6초 그래프
            gx, gy, gw, gh = bx + 16 * u, by + 122 * u, bwid - 32 * u, 34 * u
            t1 = self.hist[-1][0]
            top = max(a for _, a, _ in self.hist) + 10
            pts = [QtCore.QPointF(gx + gw * (1 - (t1 - t) / 6.0), gy + gh * (1 - _clamp(a / top)))
                   for t, a, _ in self.hist if t1 - t <= 6.0]
            qp.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255, 40), 1))
            qp.drawLine(QtCore.QPointF(gx, gy + gh), QtCore.QPointF(gx + gw, gy + gh))
            if len(pts) > 1:
                qp.setPen(QtGui.QPen(QtGui.QColor(140, 200, 255), 2 * u))
                qp.drawPolyline(QtGui.QPolygonF(pts))
        qp.restore()


# ───────────────────────── 컨트롤 패널 ─────────────────────────
class Graph(QtWidgets.QWidget):
    """최근 10초 각도 그래프: 굵은 선 = 필터 값, 가는 선 = 원시 값."""

    SPAN = 10.0

    def __init__(self, hist, top):
        super().__init__()
        self.hist, self.top = hist, top
        self.setMinimumSize(360, 130)

    def paintEvent(self, _):
        qp = QtGui.QPainter(self)
        qp.setRenderHint(QtGui.QPainter.Antialiasing)
        w, h = self.width(), self.height()
        qp.fillRect(self.rect(), QtGui.QColor(16, 17, 22))
        qp.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255, 28), 1))
        f = qp.font()
        f.setPixelSize(10)
        qp.setFont(f)
        for deg in range(0, int(self.top) + 1, 30):
            y = h - 4 - (h - 8) * deg / self.top
            qp.drawLine(0, int(y), w, int(y))
            qp.drawText(4, int(y) - 2, f"{deg}°")
        if len(self.hist) < 2:
            return
        t1 = self.hist[-1][0]
        raw, flt = [], []
        for t, a, r in self.hist:
            if t1 - t > self.SPAN:
                continue
            x = w * (1 - (t1 - t) / self.SPAN)
            flt.append(QtCore.QPointF(x, h - 4 - (h - 8) * _clamp(a / self.top)))
            raw.append(QtCore.QPointF(x, h - 4 - (h - 8) * _clamp(r / self.top)))
        qp.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255, 90), 1))
        qp.drawPolyline(QtGui.QPolygonF(raw))
        qp.setPen(QtGui.QPen(QtGui.QColor(90, 170, 255), 2))
        qp.drawPolyline(QtGui.QPolygonF(flt))


class Panel(QtWidgets.QWidget):
    CLOSED_ANGLE = 15.0     # 이 각도 이하 = 완전히 닫힌 것으로 취급
    DEADZONE = 6.0          # 완전 펼침에서 이만큼은 흔들려도 연출 없음

    def __init__(self, args, screen):
        super().__init__()
        self.args = args
        self.sim = args.sim
        self.hist = collections.deque(maxlen=2400)
        self.overlay = Overlay(screen, args.res, not args.no_hud)
        self.overlay.hist = self.hist
        self.filt = OneEuro()
        self.cam = None
        self.auto_t = None
        self.last = time.perf_counter()
        self.t0 = self.last
        self.last_frame = -1
        self.log = None
        if args.log:
            self.log = open(args.log, "w", encoding="utf-8", newline="")
            self.log.write("t_sec,raw_deg,filtered_deg,vel_deg_s,dtheta_deg,points,fps,state\n")

        self.setWindowTitle("Hinge Demo")
        self.setWindowFlag(QtCore.Qt.WindowStaysOnTopHint)
        lay = QtWidgets.QVBoxLayout(self)
        self.label = QtWidgets.QLabel("-")
        self.label.setStyleSheet("font: 700 34px 'Consolas';")
        self.sub = QtWidgets.QLabel("")
        self.sub.setStyleSheet("font: 12px 'Consolas';")
        lay.addWidget(self.label)
        lay.addWidget(self.sub)
        self.graph = Graph(self.hist, args.open + 10)
        lay.addWidget(self.graph, 1)

        if self.sim:
            self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
            self.slider.setRange(0, int(args.open * 100))
            self.slider.setValue(int(args.open * 100))
            lay.addWidget(self.slider)
            ba = QtWidgets.QPushButton("자동 재생 (닫았다 열기)")
            ba.clicked.connect(lambda: setattr(self, "auto_t", 0.0))
            lay.addWidget(ba)
        else:
            est = HingeEstimator(open_angle=args.open, vfov_deg=args.vfov)
            self.cam = CameraThread(est, args.cam)
            self.cam.start()
            b1 = QtWidgets.QPushButton("현재 각도를 '완전 펼침'으로")
            b1.clicked.connect(self.calibrate)
            b2 = QtWidgets.QPushButton("방향 반전")
            b2.clicked.connect(self.flip)
            lay.addWidget(b1)
            lay.addWidget(b2)
        bq = QtWidgets.QPushButton("종료")
        bq.clicked.connect(QtWidgets.QApplication.quit)
        lay.addWidget(bq)

        self.timer = QtCore.QTimer(self)
        self.timer.setTimerType(QtCore.Qt.PreciseTimer)
        self.timer.timeout.connect(self.tick)
        self.timer.start(8)

    def calibrate(self):
        with self.cam.lock:
            self.cam.est.calibrate()
        self.filt.reset(self.args.open)

    def flip(self):
        with self.cam.lock:
            self.cam.est.flip()

    def tick(self):
        now = time.perf_counter()
        dt, self.last = max(1e-3, now - self.last), now
        if self.sim:
            if self.auto_t is not None:
                self.auto_t += dt
                u = self.auto_t / 5.0
                self.slider.setValue(int(self.args.open * 100 * (0.5 + 0.5 * math.cos(2 * math.pi * min(u, 1.0)))))
                if u >= 1.0:
                    self.auto_t = None
            d = dict(raw=self.slider.value() / 100.0, state="sim", pts=0, dtheta=0.0, fps=1.0 / dt, frame=self.last_frame + 1)
        else:
            if self.cam.error:
                self.label.setText("오류")
                self.sub.setText(self.cam.error)
                return
            d = self.cam.read()
        angle = self.filt(d["raw"], dt)
        full = self.args.open - self.DEADZONE
        p = _clamp((angle - self.CLOSED_ANGLE) / (full - self.CLOSED_ANGLE))
        info = dict(d, angle=angle, vel=self.filt.dx)
        self.overlay.set_target(p, info)

        self.hist.append((now, angle, d["raw"]))
        if self.log and d["frame"] != self.last_frame:          # 카메라 프레임마다 한 줄
            self.log.write(
                f"{now - self.t0:.4f},{d['raw']:.4f},{angle:.4f},{self.filt.dx:.3f},"
                f"{d['dtheta']:.4f},{d['pts']},{d['fps']:.1f},{d['state']}\n"
            )
        self.last_frame = d["frame"]
        self.label.setText(f"{angle:7.2f}°")
        self.sub.setText(
            f"ω {self.filt.dx:+8.2f}°/s   raw {d['raw']:7.2f}°   Δ {d['dtheta']:+.3f}°\n"
            f"{d['fps']:5.1f} fps   pts {d['pts']:3d}   {d['state']:<7}  진행도 {p:.3f}"
        )
        self.graph.update()

    def closeEvent(self, e):
        QtWidgets.QApplication.quit()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", action="store_true", help="웹캠 없이 슬라이더로 테스트")
    ap.add_argument("--cam", type=int, default=0, help="웹캠 인덱스")
    ap.add_argument("--screen", type=int, default=0, help="연출을 띄울 모니터 인덱스 (노트북 화면)")
    ap.add_argument("--open", type=float, default=110.0, help="평소 펼침 각도(도)")
    ap.add_argument("--res", type=float, default=0.8, help="움직이는 동안의 렌더 해상도(메가픽셀). 끊기면 낮추고, 여유 있으면 올린다")
    ap.add_argument("--vfov", type=float, default=43.0, help="웹캠 세로 화각(도). 각도가 덜/더 변하면 조정")
    ap.add_argument("--no-hud", action="store_true", help="연출 화면 위의 실시간 수치 표시 끄기")
    ap.add_argument("--log", default="", help="힌지 값을 프레임마다 기록할 CSV 경로")
    args = ap.parse_args()

    app = QtWidgets.QApplication(sys.argv)
    screens = app.screens()
    screen = screens[min(args.screen, len(screens) - 1)]
    panel = Panel(args, screen)
    panel.show()
    code = app.exec()
    if panel.cam:
        panel.cam.running = False
    if panel.log:
        panel.log.close()
    sys.exit(code)


if __name__ == "__main__":
    main()

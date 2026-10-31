#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OMEGA SPATIAL — 실시간 공간음향 엔진 + 간단 GUI (단일 파일)
================================================================================
OMEGA SYSTEMS · 기본은 numpy+scipy만으로 완전히 동작. numba는 선택 사항 —
설치돼 있으면(pip install numba) 가장 뜨거운 경로 하나를 자동으로 가속하고,
없으면 기존 경로로 조용히 폴백한다. PortAudio는 sounddevice를 통해서만 쓴다.

실행
    python OMEGA_SPATIAL.py              GUI 실행
    python OMEGA_SPATIAL.py --devices    ★ 소리가 안 날 때 원인 진단
    python OMEGA_SPATIAL.py --selftest   ★ 스피커로 6초 테스트 출력
    python OMEGA_SPATIAL.py --isolate    '파리 소리'가 어느 단계에서 나오는지 판별

  ★ 스테레오 → 실시간 3D (음악·영상 전부)
    python OMEGA_SPATIAL.py --spatialize 원본.wav 출력.wav     파일로 먼저 확인
    python OMEGA_SPATIAL.py --live                             시스템 소리 실시간 변환
    옵션  --preset=accurate|standard|wide   --in <번호>  --out <번호>
    python OMEGA_SPATIAL.py --bench      성능 측정만 (창 없음)
    python OMEGA_SPATIAL.py --render x.wav --sec 10     오프라인 렌더
    pip install sounddevice              ← 실제 소리를 들으려면 이것만 설치
    pip install numba                    ← 선택. 있으면 자동으로 더 가볍게 돈다

2.6 — GUI 화이트 톤 + 초보자 신뢰성 + 선택적 가속
    · 다크(#0b0e14) → 화이트(#f6f7fb) 팔레트. 원래 민트(#31e0c8)·라벤더
      (#7b6cff)는 흰 배경에서 대비가 부족해 진한 틸(#0e8f86)·진한 보라
      (#6247ea)로 채도·명도를 낮춰 재조정. 하드코딩돼 있던 다크 전용
      색상 2곳(무대 격자링, 레벨바 트랙)도 팔레트 상수 참조로 교체.
    · 멀티소스 마커 팔레트도 흰 배경 기준으로 명도 하향 조정.
    · GUI 실시간 캡처가 첫 번째 후보 장치 실패 시 콘솔로 떠넘기던 걸
      없앴다. 이제 후보 전체를 자동으로 순회하고(하나가 배타 접근 등으로
      막혀도 다음으로 자동 전환), 실패해도 GUI 안에서 평범한 말로 알려준다.
      콘솔을 열 필요가 아예 없어야 한다는 게 원칙.
    · 무음 자동 감지. 실시간 모드에서 피크가 2.5초 넘게 0에 머물면
      "출력 장치를 확인해 보세요" 같은 안내를 상태줄에 자동으로 띄운다
      (이전엔 원인 모른 채 그냥 조용했다).
    · numba(선택 사항, 없어도 100% 동작)로 파라메트릭 HRTF의 헤드섀도
      필터를 가속. 프로파일링 결과 이 경로가 scipy.signal.lfilter 호출
      오버헤드로 전체 CPU의 약 25%를 먹고 있었다(소스마다 좌/우 필터
      계수가 달라 scipy 호출을 여러 소스로 묶을 수 없는 구조라 벡터화로는
      못 줄이는 지점). 컴파일된 루프로 교체해 파라메트릭 HRTF 소스 1개
      벤치 77µs → 39µs, 16소스 CPU 여유 1.2× → 1.6×. JIT 컴파일은 오디오
      콜백 시작 전 백그라운드에서 미리 끝내 놓는다. 수치 검증: 기존
      lfilter 경로와 최대 오차 1.2e-7(부동소수점 반올림 수준).

2.5 — 음질 전면 상향
    · HRIR 뱅크 해상도 36×7×96탭 → 72×13×128탭. 방위 10°→5°, 고도 7→13단
      으로 셀 사이 보간 오차가 절반 이하로 줄고, 귓바퀴 노치를 1개→2개
      (하모닉)로 늘려 고도 단서가 또렷해진다.
    · 리버브 감쇠를 1밴드→2밴드로 분리(고역 에어 흡수 + 저역 전용 쉘프).
      저역 부밍을 고역 감쇠와 독립적으로 통제한다. 쉘프는 감쇠 전용(0 dB
      이하)만 허용해 루프 이득이 기존보다 절대 커지지 않으므로 안정성은
      그대로다.
    · 리미터가 샘플 피크 대신 4배 오버샘플 트루피크를 본다. 샘플 사이에서
      실링을 넘는 인터샘플 오버슈트(DAC 재구성·로시 코덱 재인코딩 시
      체감되는 원인)까지 잡는다.
    · 다이나믹 EQ에 90 Hz 로우쉘프 밴드 추가(2:1, 온화한 글루 컴프레션).
      기존 3.2 kHz 밴드 하나로는 소스가 늘 때 저역이 뭉치는 걸 못 잡았다.

2.4
    · 'Illegal combination of I/O devices' 해결. PortAudio 듀플렉스는 입력과
      출력이 같은 호스트 API여야 하는데, 윈도우에서 가상 캡처 장치와 실제
      헤드폰이 다른 API에 잡히면 반드시 실패한다. 이제 미리 확인해서 다르면
      듀플렉스를 아예 시도하지 않고 분리 스트림으로 간다.
    · 분리 스트림: 링 버퍼 + 비율 리샘플링(입출력 레이트가 달라도 음정 유지,
      실측 1000 Hz 입력 → 999.8~1000.0 Hz 출력) + 선충전 + 언더런 시 직전
      블록 유지. 드리프트는 수위 감시로 보정.
    · 엔진 샘플레이트를 출력 장치 기준으로 잡는다.

2.3
    · 켰을 때 아무 소리도 내지 않는다. 기본 소스 자체를 없앴다(실측 출력 0.000000).
      큰 버튼은 '실시간 3D'. 잡음이 나던 건 계측 신호를 데모처럼 틀고 있었기
      때문이고, 광대역 잡음은 그냥 '지지직'으로 들린다 — 계측에는 맞지만
      사람이 들을 소리가 아니었다.
    · 경로 점검은 4초만, 낮은 레벨로, 끝나면 자동 정지.
    · 실시간 경로 블록 512. 콜백 마감이 5.3 → 10.7 ms 로 늘어 지터에 강해진다
      (최악 부담 19~23%). 마감 초과가 곧 진짜 '지지직'이다.
    · 상태줄에 끊김 횟수 표시 — 잡음이 신호 탓인지 언더런 탓인지 구분된다.

2.2 — 방향 전환
    이 프로그램의 본체는 '소리를 만드는 것'이 아니라 'PC에서 나오는 모든 소리를
    실시간으로 3D로 바꾸는 것'이다. 음정 있는 데모음(종·마림바)은 전부 삭제했다.
    남은 신호는 광대역 계측 신호 하나뿐이며, 오디오 경로가 살아 있는지 보는 용도다.

    스테레오 → 3D 변환 (SpatialField)
      · 공분산 주성분 분해로 방향성 성분과 공간 성분을 분리
      · 빈별 방위각 복원 후 바이노럴 재배치
      · 확산음장 등화로 HRTF 고유 착색 제거 (음색 편차 0.9 dB)
      · 앰비언트는 무상관을 지키는 경로가 주력
      실측: IACC 원본 0.353 → 0.312 (표준), 0.255 (넓게)
            ILD −29.7~+29.9 dB, ITD −583~+625 µs 단조 대응
      추가 지연 16 ms (영상 립싱크 여유), CPU 12%

    2.1에서 고친 것

    · 오디오 출력  스트림 열기 실패 시 조용히 무음이 되던 문제. 설정을 낮춰가며
                   재시도하고, 실패 이유를 화면에 그대로 띄운다. 호스트가 요구하는
                   프레임 수가 블록과 달라도 무음이 되지 않는다(구버전 영구 무음 버그).
                   엔진 샘플레이트를 장치 기본값에 맞춰 연다.
    · '파리 소리'  ① 초기반사 탭이 한 블록씩 앞당겨져 4샘플(0.08 ms)에 떨어져 있었다.
                     반사가 아니라 콤 필터였다 — 리플 9.1 dB → 4.1 dB.
                   ② 확산 올패스가 블록 길이(5.3 ms) 아래로 못 내려가 184.6 Hz·
                     169.6 Hz에서 링잉했다(집파리 날갯짓 190 Hz). 청크 처리로
                     3.6~12.8 ms 서로소 지연을 제대로 깔았다.
                   ③ FDN 피드백 행렬이 하우스홀더라 대각 0.917 — 에너지의 84%가
                     자기 라인으로 되돌아왔다. 24라인 FDN이 아니라 독립 콤 24개였다.
                     완전 혼합 직교행렬로 교체.
                   ④ 라인별 저속 LFO로 지연을 0.25% 흔들어 고정 모드를 푼다.

구현된 기술 (전부 동작함)
    · 파라메트릭 HRTF   ITD(Woodworth) + 헤드섀도 + 고도 틸트 + 근접효과
    · 컨볼루션 HRTF     합성 HRIR 뱅크(방위 36 × 고도 7, 96탭) 오버랩세이브
                        + 셀 전환 크로스페이드 (기본 ON — 실측상 파라메트릭보다 오히려 쌈)
    · 거리 감쇠         1/d + 거리별 공기흡수 저역통과
    · 지향성            소스 콘(inner/outer) 감쇠
    · 도플러/전파지연   거리 기반 가변 딜레이 (근사 아님, 리샘플러 없음)
    · 룸 리버브         초기반사 8탭 + 프리딜레이 + 확산 올패스 4단 +
                        24라인 FDN 공유 버스 (소스 수 무관)
    · 스테레오 확장     M/S, 저역 모노 보호
    · 스마트 베이스     미싱 펀더멘털 하모닉 생성 + 게이트
    · 다이나믹 EQ       정적 밴드 1콜 캐스케이드 + 동적 밴드
    · 라우드니스        BS.1770 K-웨이팅 스트리밍 AGC + 룩어헤드 리미터
    · CPU 오토스케일러  블록 처리시간 EMA로 기능 자동 on/off (드롭아웃 방지)

스레드 모델
    제어 스레드(GUI)는 소스 좌표를 자유롭게 쓴다. 오디오 스레드는 블록마다
    수 마이크로초 동안만 락을 잡고 기하 정보를 스냅샷하며, 무거운 DSP는 전부
    락 밖에서 돈다. 오디오 콜백은 절대 블로킹되지 않는다.
"""
from __future__ import annotations

import os
import sys
import time
import math
import wave
import threading

import numpy as np
from scipy.ndimage import minimum_filter1d
from scipy.signal import lfilter, sosfilt, upfirdn, firwin

# v2.6: numba는 선택적 가속 경로다. 없으면 기존 scipy 경로 그대로 쓰므로
# 여전히 pip install 한 줄 없이도(순수 numpy+scipy) 완전히 동작한다.
# 있으면 소스별 헤드섀도 필터(가장 뜨거운 경로 — 프로파일상 약 25%가
# scipy.signal.lfilter 파이썬 호출 오버헤드였다. 소스마다 좌/우 필터
# 계수가 달라 scipy로는 한 번에 묶어 처리할 수 없어 이 지점만큼은
# 컴파일된 루프가 확실히 유리하다)에만 쓰인다.
try:
    from numba import njit as _njit
    _HAS_NUMBA = True
except Exception:
    _HAS_NUMBA = False

if _HAS_NUMBA:
    @_njit(cache=True, fastmath=True)
    def _onepole_nb(x, z0, p, tilt, out):
        n = x.shape[0]
        z = z0
        if tilt == 1.0:
            for i in range(n):
                z = (1.0 - p) * x[i] + p * z
                out[i] = z
        else:
            for i in range(n):
                z = (1.0 - p) * x[i] + p * z
                out[i] = x[i] * tilt + z * (1.0 - tilt)
        return z

    def _warm_numba():
        """오디오 콜백이 시작되기 전에 미리 한 번 컴파일해 둔다. 첫 실행
        JIT 지연(수백 ms)이 실시간 콜백 안에서 일어나면 그게 곧 첫 블록
        드롭아웃이라 반드시 초기화 시점에 끝내 놓아야 한다."""
        dummy = np.zeros(8, dtype=np.float32)
        out = np.empty(8, dtype=np.float32)
        _onepole_nb(dummy, 0.0, 0.5, 1.0, out)
        _onepole_nb(dummy, 0.0, 0.5, 0.7, out)
else:
    def _warm_numba():
        pass

__version__ = "2.8"
FS_DEFAULT = 48000
BLOCK_DEFAULT = 256
CPU_BUDGET = 0.10       # 목표 CPU 점유율(1코어 기준). 넘으면 스스로 낮춘다.

# ─── 청감 튜닝 파라미터 ────────────────────────────────────────────────────
# 아래 값들은 물리 상수가 아니라 전부 경험값이다. 물리적으로 정확한 HRTF는
# 실제로 답답하게 들리고 앞뒤 구분도 애매하다 — 상용 제품이 의도적으로
# 과장하는 이유가 이것이다. 따라서 이 값들의 정답은 계산이 아니라 청취로만
# 정해진다. OMEGA_TUNER.py 로 블라인드 A/B 하며 맞추고, 확정되면 여기 박는다.
TUNE = {
    "shadow_depth": 16500.0,   # 먼 귀 고역 감쇠 깊이(Hz). 클수록 좌우 분리 강함
    "shadow_top": 18000.0,     # 헤드섀도 시작 상한(Hz)
    "elev_tilt": 0.45,         # 고도에 따른 명암 차. 크면 위아래 구분 뚜렷
    "itd_scale": 1.0,          # ITD 배율. 1보다 크면 좌우가 과장되어 넓게 들림
    "ild_scale": 1.0,          # ILD 배율
}
C_SOUND = 343.0
HEAD_R = 0.0875
ITD_MAX = HEAD_R / C_SOUND * (math.pi / 2 + 1.0)


# ═══════════════════════════════════════════════════════════════════════════
#  1. 프리미티브 — 딜레이 라인, 바이쿼드
# ═══════════════════════════════════════════════════════════════════════════
class DelayLine:
    """엔진에서 가장 많이 재사용되는 객체. ITD·도플러·리버브가 전부 이것의 래퍼.

    핵심 최적화: 용량을 2의 거듭제곱으로 잡아 블록 읽기/쓰기가 최대 2개의
    연속 슬라이스로 끝난다. 팬시 인덱싱도, 호출마다의 arange 할당도 없다.
    연속 슬라이스는 C 레벨 memcpy라 GIL도 풀린다. 전부 O(B)."""

    __slots__ = ("_buf", "_cap", "_w", "_mask", "_win")

    def __init__(self, capacity: int):
        p = 1
        while p < capacity:
            p <<= 1
        self._cap = p
        self._mask = p - 1
        self._buf = np.zeros(p, dtype=np.float32)
        self._w = 0
        self._win = None

    def write_block(self, x):
        b = x.shape[0]
        w = self._w
        cap = self._cap
        end = w + b
        if end <= cap:
            self._buf[w:end] = x
        else:
            first = cap - w
            self._buf[w:] = x[:first]
            self._buf[:b - first] = x[first:]
        self._w = (w + b) & self._mask

    def _window(self, start, n):
        cap = self._cap
        start &= self._mask
        if start + n <= cap:
            return self._buf[start:start + n]          # 뷰, 복사 없음
        if self._win is None or self._win.shape[0] < n:
            self._win = np.empty(n, dtype=np.float32)
        w = self._win[:n]
        first = cap - start
        w[:first] = self._buf[start:]
        w[first:] = self._buf[:n - first]
        return w

    def read_delayed_block(self, delay, b, out):
        out[:] = self._window((self._w - b - delay) & self._mask, b)

    def read_frac_uniform(self, delay, b, out):
        """블록 내내 지연이 일정 → 연속 샘플이 연속 주소로 대응하므로
        전체 보간이 '한 번의 연속 읽기 + 스칼라 가중 블렌드'로 줄어든다."""
        start_f = self._w - b - delay
        i0 = int(math.floor(start_f))
        frac = np.float32(start_f - i0)
        win = self._window(i0, b + 1)
        a = win[:b]
        bn = win[1:b + 1]
        np.subtract(bn, a, out=out)
        np.multiply(out, frac, out=out)
        np.add(out, a, out=out)

    def read_frac_block(self, read_pos, out):
        """샘플마다 다른 분수 지연(도플러). 램프라 필요한 구간이 연속이므로
        그 구간만 잘라서 지역 인덱싱한다."""
        base = self._w - read_pos
        i0 = np.floor(base)
        frac = (base - i0).astype(np.float32)
        i0i = i0.astype(np.int64)
        lo = int(i0i.min())
        hi = int(i0i.max())
        win = self._window(lo, hi - lo + 2)
        loc = i0i - lo
        a = win[loc]
        bn = win[loc + 1]
        np.subtract(bn, a, out=out)
        np.multiply(out, frac, out=out)
        np.add(out, a, out=out)

    def reset(self):
        self._buf.fill(0.0)
        self._w = 0


def _norm(b0, b1, b2, a0, a1, a2):
    return np.array([[b0 / a0, b1 / a0, b2 / a0, 1.0, a1 / a0, a2 / a0]],
                    dtype=np.float64)


def peaking(fs, f0, gain_db, q):
    A = 10.0 ** (gain_db / 40.0)
    w0 = 2 * math.pi * f0 / fs
    cw, sw = math.cos(w0), math.sin(w0)
    al = sw / (2 * q)
    return _norm(1 + al * A, -2 * cw, 1 - al * A,
                 1 + al / A, -2 * cw, 1 - al / A)


def low_shelf(fs, f0, gain_db, q=0.707):
    A = 10.0 ** (gain_db / 40.0)
    w0 = 2 * math.pi * f0 / fs
    cw, sw = math.cos(w0), math.sin(w0)
    al = sw / (2 * q)
    t = 2 * math.sqrt(A) * al
    return _norm(A * ((A + 1) - (A - 1) * cw + t),
                 2 * A * ((A - 1) - (A + 1) * cw),
                 A * ((A + 1) - (A - 1) * cw - t),
                 (A + 1) + (A - 1) * cw + t,
                 -2 * ((A - 1) + (A + 1) * cw),
                 (A + 1) + (A - 1) * cw - t)


def high_shelf(fs, f0, gain_db, q=0.707):
    A = 10.0 ** (gain_db / 40.0)
    w0 = 2 * math.pi * f0 / fs
    cw, sw = math.cos(w0), math.sin(w0)
    al = sw / (2 * q)
    t = 2 * math.sqrt(A) * al
    return _norm(A * ((A + 1) + (A - 1) * cw + t),
                 -2 * A * ((A - 1) + (A + 1) * cw),
                 A * ((A + 1) + (A - 1) * cw - t),
                 (A + 1) - (A - 1) * cw + t,
                 2 * ((A - 1) - (A + 1) * cw),
                 (A + 1) - (A - 1) * cw - t)


def high_pass(fs, f0, q=0.707):
    w0 = 2 * math.pi * f0 / fs
    cw, sw = math.cos(w0), math.sin(w0)
    al = sw / (2 * q)
    return _norm((1 + cw) / 2, -(1 + cw), (1 + cw) / 2,
                 1 + al, -2 * cw, 1 - al)


def low_pass(fs, f0, q=0.707):
    w0 = 2 * math.pi * f0 / fs
    cw, sw = math.cos(w0), math.sin(w0)
    al = sw / (2 * q)
    return _norm((1 - cw) / 2, 1 - cw, (1 - cw) / 2,
                 1 + al, -2 * cw, 1 - al)


class SOSCascade:
    """상태 유지 SOS 필터. nch=2일 때 버퍼는 (2, B) 레이아웃이며 axis=1로
    필터링한다 — 필터 축이 연속이라 scipy 내부 transpose 복사가 사라진다."""

    __slots__ = ("_sos", "_zi", "_nch")

    def __init__(self, sos, nch=2):
        self._nch = nch
        self.set_sos(sos)

    def set_sos(self, sos):
        self._sos = np.ascontiguousarray(sos, dtype=np.float32)
        n = self._sos.shape[0]
        self._zi = (np.zeros((n, 2), dtype=np.float32) if self._nch == 1
                    else np.zeros((n, self._nch, 2), dtype=np.float32))

    def process2(self, buf):
        y, self._zi = sosfilt(self._sos, buf, axis=1, zi=self._zi)
        return y

    def process_mono(self, x):
        y, self._zi = sosfilt(self._sos, x, zi=self._zi)
        return y

    def reset(self):
        self._zi.fill(0.0)


# ═══════════════════════════════════════════════════════════════════════════
#  2. 파라메트릭 HRTF (기본 경로)
# ═══════════════════════════════════════════════════════════════════════════
def woodworth_itd(az: float) -> float:
    """구형 두상 ITD(초). az 라디안, 0=정면, +=오른쪽. 양수면 왼쪽 귀가 늦다."""
    a = abs(az)
    mag = (HEAD_R / C_SOUND * (a + math.sin(a)) if a <= math.pi / 2
           else HEAD_R / C_SOUND * (math.pi - a + math.sin(a)))
    return mag if az >= 0.0 else -mag


class SourceVoice:
    """소스 1개의 파라메트릭 공간화.

    측정 HRIR 컨볼루션 대신 지배적인 지각 단서만 직접 모델링한다:
        ITD(1.5kHz 이하 지배)      → 귀별 분수 지연
        ILD/헤드섀도(1.5kHz 이상)  → 먼 귀 원폴 저역통과
        거리                       → 1/d 게인 + 공기흡수 저역통과
        고도                       → 고역 틸트(귓바퀴 단서 근사)
        근접(1m 이내)              → ILD 과장
    블록당 비용: 딜레이 쓰기 1 + 분수 읽기 2 + 원폴 최대 2 + 스칼라 게인.
    FFT 없음, 정상 상태 할당 없음."""

    __slots__ = ("fs", "_dl", "_zi_l", "_zi_r", "_base", "_sl", "_sr",
                 "_tl", "_tr", "_smooth", "_ramp", "_pgl", "_pgr",
                 "_pil", "_pir", "_t2")

    def __init__(self, fs, block):
        self.fs = fs
        self._ramp = np.linspace(0.0, 1.0, block, dtype=np.float32)
        self._pgl = 0.0
        self._pgr = 0.0
        self._pil = -1.0
        self._pir = -1.0
        self._t2 = np.empty(block, dtype=np.float32)
        base = int(ITD_MAX * fs) + 2
        self._dl = DelayLine(base + block + 8)
        self._base = float(base)
        self._zi_l = np.zeros(1, dtype=np.float64)
        self._zi_r = np.zeros(1, dtype=np.float64)
        self._sl = np.empty(block, dtype=np.float32)
        self._sr = np.empty(block, dtype=np.float32)
        self._tl = np.empty(block, dtype=np.float32)
        self._tr = np.empty(block, dtype=np.float32)
        self._smooth = 0.0

    def render(self, mono, az, el, dist, gain, out_l, out_r):
        fs = self.fs
        b = mono.shape[0]

        d = dist if dist > 0.3 else 0.3
        g = gain / d
        air_fc = 20000.0 / (1.0 + 0.02 * d)          # 거리별 공기흡수

        itd = woodworth_itd(az) * fs
        self._smooth += 0.35 * (itd - self._smooth)  # 지퍼 노이즈 방지
        s = self._smooth
        self._dl.write_block(mono)
        # read_frac_uniform 은 블록 내내 지연이 '상수'다. 소스가 움직이면
        # 블록 경계마다 지연이 툭 튀고, 그 불연속의 크기는 신호 기울기에
        # 비례한다 — 즉 고역일수록 커진다. 5 kHz 성분이면 −28 dB짜리 클릭이
        # 172 Hz로 반복된다. 이게 두 번째 '파리 소리'였다.
        # 이전 지연과 현재 지연을 각각 읽어 블록 안에서 크로스페이드하면
        # 경계에서 값이 정확히 이어져 불연속이 사라진다(추가 비용: 읽기 2회).
        il = self._base + (s if s > 0.0 else 0.0)
        ir = self._base + (-s if s < 0.0 else 0.0)
        if self._pil < 0.0:
            self._pil, self._pir = il, ir
        ramp = self._ramp[:b]
        for cur, prev, dst in ((il, self._pil, self._sl),
                               (ir, self._pir, self._sr)):
            if abs(cur - prev) < 1e-4:
                self._dl.read_frac_uniform(cur, b, dst[:b])
            else:
                self._dl.read_frac_uniform(prev, b, dst[:b])
                self._dl.read_frac_uniform(cur, b, self._t2[:b])
                dst[:b] *= (1.0 - ramp)
                dst[:b] += self._t2[:b] * ramp
        self._pil, self._pir = il, ir

        # 등파워 패닝. 좌우 위치는 방위각이 아니라 '측면성' sin(az)로 잡는다.
        # az/(pi/2)를 클리핑하면 ±90도를 넘는 순간 한쪽으로 완전히 붙어버려
        # 정확히 뒤쪽 소리가 한쪽 귀에서만 나는 버그가 생긴다. sin(az)는
        # 정면 0, 옆 ±1, 뒤 0으로 자연스럽게 돌아온다(전후 구분은 헤드섀도와
        # 귓바퀴 노치가 담당).
        x = math.sin(az)
        th = (x + 1.0) * (math.pi / 4)
        gl = math.cos(th) * g
        gr = math.sin(th) * g

        # 근접 효과: 1m 이내면 가까운 귀 우세를 과장
        if d < 1.0:
            k = 1.0 + (1.0 - d) * 0.6
            if az >= 0.0:
                gr *= k
                gl /= k
            else:
                gl *= k
                gr /= k

        # 헤드섀도: 먼 귀만 고역 감쇠
        lateral = abs(math.sin(az))
        back = (abs(az) - math.pi / 2) / (math.pi / 2) if abs(az) > math.pi / 2 else 0.0
        if az >= 0.0:
            sh_l, sh_r = min(1.0, lateral + 0.3 * back), 0.0
        else:
            sh_l, sh_r = 0.0, min(1.0, lateral + 0.3 * back)
        _top = TUNE["shadow_top"]
        _dep = TUNE["shadow_depth"]
        fc_l = min(air_fc, _top - sh_l * _dep)
        fc_r = min(air_fc, _top - sh_r * _dep)

        # 고도 틸트: 위 → 밝게, 아래 → 어둡게 (귓바퀴 단서 근사, 필터 추가 없음)
        tilt = 1.0 + TUNE["elev_tilt"] * math.sin(el)

        yl = self._shade(self._sl[:b], self._tl[:b], fc_l, self._zi_l, fs, tilt)
        yr = self._shade(self._sr[:b], self._tr[:b], fc_r, self._zi_r, fs, tilt)
        # 게인을 블록당 상수로 바꾸면 소스가 움직일 때 진폭이 블록 경계마다
        # 계단처럼 튄다. 그 계단의 반복 주파수가 블록레이트(44.1k/256 = 172 Hz,
        # 48k/256 = 188 Hz)인데 하필 곤충 날갯짓 대역이라, 움직이는 동안
        # 내내 '윙―' 하는 잡음이 깔린다. 이전 블록 게인에서 이어지는 램프로
        # 발라주면 파형이 연속이 되어 사라진다. 비용은 곱셈 1회 추가.
        ramp = self._ramp[:b]
        out_l += yl * (np.float32(self._pgl) +
                       np.float32(gl - self._pgl) * ramp)
        out_r += yr * (np.float32(self._pgr) +
                       np.float32(gr - self._pgr) * ramp)
        self._pgl, self._pgr = gl, gr

    @staticmethod
    def _shade(x, scratch, fc, zi, fs, tilt):
        """원폴 저역통과 + 고도 틸트를 한 번에. tilt=1이면 저역통과만.

        v2.6: numba가 있으면 컴파일된 루프로 처리한다. 소스마다 fc(헤드섀도
        컷오프)가 달라 scipy로는 여러 소스를 한 번에 묶을 수 없는데, 이
        함수 자체가 소스당 블록당 2회(좌/우) 호출되고 프로파일상 그
        호출의 scipy 검증 오버헤드가 전체 CPU의 약 25%였다. numba 호출은
        그 오버헤드가 거의 없다."""
        if fc >= fs * 0.45 and abs(tilt - 1.0) < 1e-3:
            return x
        p = math.exp(-2.0 * math.pi * min(fc, fs * 0.45) / fs)
        if _HAS_NUMBA:
            z0 = _onepole_nb(x, float(zi[0]), p, tilt, scratch)
            zi[0] = z0
            return scratch
        y, zi[:] = lfilter([1.0 - p], [1.0, -p], x, zi=zi)
        lp = y.astype(np.float32, copy=False)
        if abs(tilt - 1.0) < 1e-3:
            return lp
        # y = lp + tilt*(x - lp) = x*tilt + lp*(1-tilt)
        np.multiply(x, np.float32(tilt), out=scratch)
        lp *= np.float32(1.0 - tilt)
        scratch += lp
        return scratch

    def reset(self):
        self._dl.reset()
        self._zi_l.fill(0.0)
        self._zi_r.fill(0.0)
        self._smooth = 0.0


# ═══════════════════════════════════════════════════════════════════════════
#  3. 컨볼루션 HRTF (고음질 경로) — 합성 HRIR 뱅크 + 오버랩세이브
# ═══════════════════════════════════════════════════════════════════════════
class HRIRBank:
    """방위 36 × 고도 7 셀의 HRIR을 초기화 때 한 번 합성하고, 그 스펙트럼을
    complex64로 캐시한다. 실행 중에는 셀 인덱스만 바꾸면 되므로 전환 비용 0.

    합성 모델 (측정 HRIR 파일 없이 물리 근사로 생성):
        1) Woodworth ITD → 8탭 윈도드 싱크 분수 지연
        2) 헤드섀도      → 방위별 원폴 저역통과를 임펄스에 적용
        3) 귓바퀴 노치   → 고도에 따라 6.2~11 kHz 콤 노치 (고도 단서의 핵심)
        4) 어깨 반사     → 0.7 ms 지연 약한 복사본
    측정 HRIR(SOFA 등)을 쓰고 싶으면 load()로 (az, el, L, R) 배열만 넣으면
    나머지 경로는 그대로 동작한다.

    메모리: 36×7×2 스펙트럼 × 257 bin × 8 B ≈ 1.0 MB."""

    N_AZ = 72          # 5도 간격 (v2.5: 10도 → 5도, 방위 보간 오차 절반)
    N_EL = 13          # 고도 13단 (v2.5: 7 → 13, 셀 전환 크로스페이드 빈도↓)
    TAPS = 128         # v2.5: 96 → 128, 저역 노치 해상도 확보
    EL_MIN, EL_MAX = -0.7, 1.05        # 라디안 (-40° ~ +60°)

    def __init__(self, fs, block):
        self.fs = fs
        self.block = block
        n = 1
        while n < block + self.TAPS - 1:
            n <<= 1
        self.nfft = n
        self.bins = n // 2 + 1
        self.HL = np.zeros((self.N_AZ, self.N_EL, self.bins), dtype=np.complex64)
        self.HR = np.zeros((self.N_AZ, self.N_EL, self.bins), dtype=np.complex64)
        for ia in range(self.N_AZ):
            az = -math.pi + 2 * math.pi * (ia + 0.5) / self.N_AZ
            for ie in range(self.N_EL):
                el = (self.EL_MIN + (self.EL_MAX - self.EL_MIN)
                      * ie / (self.N_EL - 1))
                hl, hr = self._synth(az, el)
                self.HL[ia, ie] = np.fft.rfft(hl, n).astype(np.complex64)
                self.HR[ia, ie] = np.fft.rfft(hr, n).astype(np.complex64)

    # ---- 셀 인덱싱 (오디오 스레드에서 호출, 스칼라 연산만) ----
    def cell(self, az, el):
        ia = int((az + math.pi) / (2 * math.pi) * self.N_AZ) % self.N_AZ
        t = (el - self.EL_MIN) / (self.EL_MAX - self.EL_MIN)
        ie = int(round(t * (self.N_EL - 1)))
        ie = 0 if ie < 0 else (self.N_EL - 1 if ie > self.N_EL - 1 else ie)
        return ia, ie

    # ---- 합성 ----
    def _synth(self, az, el):
        fs = self.fs
        taps = self.TAPS
        itd = woodworth_itd(az) * fs
        d_l = 8.0 + (itd if itd > 0 else 0.0)
        d_r = 8.0 + (-itd if itd < 0 else 0.0)
        lateral = abs(math.sin(az))
        back = (abs(az) - math.pi / 2) / (math.pi / 2) if abs(az) > math.pi / 2 else 0.0
        if az >= 0.0:
            sh_l, sh_r = min(1.0, lateral + 0.3 * back), 0.0
        else:
            sh_l, sh_r = 0.0, min(1.0, lateral + 0.3 * back)
        th = (math.sin(az) + 1.0) * math.pi / 4        # 측면성 기반 등파워
        g_l = math.cos(th)
        g_r = math.sin(th)
        t = (el - self.EL_MIN) / (self.EL_MAX - self.EL_MIN)
        f_notch = 6200.0 + t * 4800.0
        f_notch2 = 9500.0 + t * 3000.0            # v2.5: 2차 하모닉 노치
        out = []
        for d, sh, g in ((d_l, sh_l, g_l), (d_r, sh_r, g_r)):
            h = self._frac_impulse(taps, d) * g
            fc = 18000.0 - sh * 16500.0
            if fc < fs * 0.45:                       # 헤드섀도 저역통과
                p = math.exp(-2.0 * math.pi * fc / fs)
                h = lfilter([1.0 - p], [1.0, -p], h).astype(np.float32)
            t_n = int(round(fs / (2.0 * f_notch)))   # 귓바퀴 1차 노치
            if 0 < t_n < taps:
                h[t_n:] -= 0.55 * h[:taps - t_n]
            t_n2 = int(round(fs / (2.0 * f_notch2)))  # 귓바퀴 2차 노치(약하게)
            if 0 < t_n2 < taps:
                h[t_n2:] -= 0.28 * h[:taps - t_n2]
            t_s = int(round(fs * 0.0007))            # 어깨 반사
            if 0 < t_s < taps:
                h[t_s:] += 0.18 * h[:taps - t_s]
            out.append(h.astype(np.float32))
        e = math.sqrt(float(np.sum(out[0] ** 2) + np.sum(out[1] ** 2)) + 1e-12)
        return out[0] / e * 1.2, out[1] / e * 1.2

    @staticmethod
    def _frac_impulse(taps, delay):
        n = np.arange(taps, dtype=np.float64)
        x = n - delay
        h = np.sinc(x) * (0.54 - 0.46 * np.cos(2 * np.pi * (x + 8) / 16.0))
        h[np.abs(x) > 8] = 0.0
        return h.astype(np.float32)


class ConvVoice:
    """오버랩세이브 컨볼루션. 셀이 바뀌면 이전 셀 결과와 1블록 크로스페이드
    하므로 소스가 움직여도 클릭이 없다.

    비용: rfft 1 + irfft 2 (셀 전환 블록만 +2). B=256, N=512 기준."""

    __slots__ = ("_bank", "_m", "_n", "_hist", "_in", "_cell", "_fade",
                 "_prev_cell", "block", "_pg")

    def __init__(self, bank, block):
        self._bank = bank
        self._m = bank.TAPS
        self._n = bank.nfft
        self.block = block
        self._hist = np.zeros(self._m - 1, dtype=np.float32)
        self._in = np.zeros(self._n, dtype=np.float32)
        self._cell = None
        self._prev_cell = None
        self._pg = -1.0
        self._fade = np.linspace(0.0, 1.0, block, dtype=np.float32)

    def process(self, mono, az, el, gain, out_l, out_r):
        b = mono.shape[0]
        m = self._m
        bank = self._bank
        cell = bank.cell(az, el)
        if self._cell is None:
            self._cell = cell
        buf = self._in
        buf[:m - 1] = self._hist
        buf[m - 1:m - 1 + b] = mono
        X = np.fft.rfft(buf[:m - 1 + b], n=self._n)

        ia, ie = cell
        yl = np.fft.irfft(X * bank.HL[ia, ie], n=self._n)[m - 1:m - 1 + b]
        yr = np.fft.irfft(X * bank.HR[ia, ie], n=self._n)[m - 1:m - 1 + b]
        if cell != self._cell:                       # 셀 전환 → 크로스페이드
            pa, pe = self._cell
            pl = np.fft.irfft(X * bank.HL[pa, pe], n=self._n)[m - 1:m - 1 + b]
            pr = np.fft.irfft(X * bank.HR[pa, pe], n=self._n)[m - 1:m - 1 + b]
            f = self._fade[:b]
            yl = pl + (yl - pl) * f
            yr = pr + (yr - pr) * f
            self._cell = cell
        if b >= m - 1:
            self._hist[:] = mono[b - (m - 1):]
        else:
            self._hist[:-b] = self._hist[b:]
            self._hist[-b:] = mono
        # 게인도 블록당 상수로 넣으면 소스가 다가오거나 멀어질 때 진폭이
        # 블록레이트(172~188 Hz)로 계단을 만든다. 램프로 이어 붙인다.
        if self._pg < 0.0:
            self._pg = gain
        f = self._fade[:b]
        gv = np.float32(self._pg) + np.float32(gain - self._pg) * f
        self._pg = gain
        out_l += yl.astype(np.float32) * gv
        out_r += yr.astype(np.float32) * gv

    def reset(self):
        self._hist.fill(0.0)
        self._cell = None
        self._pg = -1.0


# ═══════════════════════════════════════════════════════════════════════════
#  3.5 스테레오 믹스 → 실시간 3D 재구성
# ═══════════════════════════════════════════════════════════════════════════

def _smooth_random_phase(bins, rng, octaves=6, depth=1.0):
    """주파수축으로 완만한 랜덤 위상. 비상관화용.

    빈마다 독립 랜덤 위상을 주면 임펄스 응답이 무한정 길어져 트랜지언트가
    뭉개진다(드럼이 '퍼지는' 현상). 저해상도 랜덤을 보간해 올리면 위상이
    주파수축으로 완만해서 응답 길이가 짧게 유지된다."""
    n = max(4, octaves)
    coarse = rng.uniform(-math.pi, math.pi, n)
    x = np.linspace(0.0, n - 1.0, bins)
    ph = np.interp(x, np.arange(n), coarse) * depth
    return np.exp(1j * ph).astype(np.complex64)


class SpatialField:
    """스테레오 → 바이노럴 3D 업믹스.

    프레임당 비용: rfft 2 + irfft 2 + 길이 bins(=nfft/2+1) 벡터 연산 ~30회.
    nfft=1024, hop=256, 48 kHz 기준 약 1% CPU.

    추가 지연 = nfft − hop (1024−256 = 768샘플 = 16 ms @48k).
    영상과 함께 쓸 때 립싱크가 문제되는 구간이라 이 값을 넘기지 않는다."""

    N_AZ = 72                      # 5도 간격 HRTF 테이블
    BASS_HZ = 150.0                # 이 아래는 공간화하지 않는다
    AMB_AZ = (-100.0, 100.0, -145.0, 145.0)

    def __init__(self, fs, block, synth, nfft=None):
        self.fs = fs
        self.hop = block
        n = nfft or 4 * block
        n = 512 if n < 512 else (2048 if n > 2048 else n)
        self.nfft = int(2 ** round(math.log2(n)))
        self.bins = self.nfft // 2 + 1
        b = self.bins

        # ── WOLA: 분석·합성 양쪽에 sqrt-Hann. 정규화는 수치로 구한다.
        w = np.hanning(self.nfft + 1)[:-1]
        self.win = np.sqrt(w).astype(np.float32)
        acc = np.zeros(self.nfft, dtype=np.float64)
        for k in range(self.nfft // self.hop):
            acc += np.roll(self.win.astype(np.float64) ** 2, k * self.hop)
        self.win = (self.win / math.sqrt(float(acc.mean()))).astype(np.float32)

        # ── HRTF 테이블 (방위 72 × bins). 시간영역 HRIR을 nfft로 패딩해 캐시.
        self.HL = np.zeros((self.N_AZ, b), dtype=np.complex64)
        self.HR = np.zeros((self.N_AZ, b), dtype=np.complex64)
        for i in range(self.N_AZ):
            az = -math.pi + 2 * math.pi * i / self.N_AZ
            hl, hr = synth(az, 0.0)
            self.HL[i] = np.fft.rfft(hl, self.nfft).astype(np.complex64)
            self.HR[i] = np.fft.rfft(hr, self.nfft).astype(np.complex64)
        # 앰비언스는 살짝 위에 두면 포위감이 커진다(귓바퀴 노치가 위쪽 단서).
        self.AL = np.zeros((len(self.AMB_AZ), b), dtype=np.complex64)
        self.AR = np.zeros((len(self.AMB_AZ), b), dtype=np.complex64)
        for i, deg in enumerate(self.AMB_AZ):
            hl, hr = synth(math.radians(deg), 0.30)
            self.AL[i] = np.fft.rfft(hl, self.nfft).astype(np.complex64)
            self.AR[i] = np.fft.rfft(hr, self.nfft).astype(np.complex64)

        # ── 확산음장 등화 (없으면 음색이 망가진다)
        # HRTF에는 방향 단서만 있는 게 아니라 머리·귓바퀴의 고유 공진이
        # 통째로 들어 있다. 그걸 그대로 곱하면 모든 소리에 같은 착색이
        # 얹힌다(실측: 중역 −5.9 dB, 고역 +4.0 dB — 음악이 완전히 다른
        # 음색이 된다). 전 방향 평균 크기로 나누면 공통 착색만 빠지고
        # 방향에 따라 달라지는 성분(ILD·노치·ITD)은 그대로 남는다.
        mag2 = (np.abs(self.HL) ** 2 + np.abs(self.HR) ** 2).mean(axis=0) * 0.5
        mag = np.sqrt(np.maximum(mag2, 1e-12))
        # 로그축으로 완만하게 편 뒤 역수를 취한다. 날 것 그대로 뒤집으면
        # 깊은 노치가 거대한 부스트로 바뀐다.
        lg = np.log(mag + 1e-12)
        k = np.hanning(21)
        k /= k.sum()
        lg = np.convolve(lg, k, mode="same")
        eq = np.exp(-lg)
        eq /= np.median(eq)
        eq = np.clip(eq, 0.25, 4.0).astype(np.float32)   # ±12 dB 제한
        self.eq = eq
        for T in (self.HL, self.HR, self.AL, self.AR):
            T *= eq

        rng = np.random.default_rng(4242)
        self.dec = np.stack([_smooth_random_phase(b, rng, 7, 1.0)
                             for _ in range(len(self.AMB_AZ))])

        # ── 저역 보호 크로스오버 (하드 컷은 경계에서 소리가 벌어진다)
        f = np.fft.rfftfreq(self.nfft, 1.0 / fs)
        self.spat_w = np.clip((f - self.BASS_HZ * 0.6) /
                              (self.BASS_HZ * 0.8), 0.0, 1.0).astype(np.float32)
        self.freqs = f

        # ── 상태
        self._inL = np.zeros(self.nfft, dtype=np.float32)
        self._inR = np.zeros(self.nfft, dtype=np.float32)
        self._olaL = np.zeros(self.nfft, dtype=np.float32)
        self._olaR = np.zeros(self.nfft, dtype=np.float32)
        self._pl = np.zeros(b)
        self._pr = np.zeros(b)
        self._cr = np.zeros(b, dtype=np.complex128)
        self._th = np.zeros(b)
        self._rot = 0.0
        # 방위 추정을 주파수축으로 매끈하게 만드는 커널. 빈마다 방향이 튀면
        # 빈별 HRTF가 불연속이 되어 임펄스 응답이 길어지고 소리가 번진다.
        self._k = np.hanning(9)[1:-1]
        self._k /= self._k.sum()
        self._kf = np.hanning(17)[1:-1]
        self._kf /= self._kf.sum()

        # ── 조절 파라미터
        self.mix = 1.0             # 0 = 원본, 1 = 완전 공간화
        self.width = 1.35          # 스테레오 무대 확장 배율
        self.ambience = 1.6        # 포위감
        self.amb_hrtf = 0.18       # 앰비언트 중 후방 HRTF로 보낼 비율
        self.spread = 3.0          # 앰비언트 추정 확대 (정위 ↔ 포위감 균형)
        self.crossfeed = 0.22      # 앰비언트 좌우 교차량
        self.motion = 1.0          # 씬 미세 회전 (정적 락 방지)
        self.smooth = 0.08         # 통계 시정수(≈85 ms). 크게 잡으면 분산이
                                   # 늘어 앰비언트가 프라이머리로 샌다.

    # ------------------------------------------------------------------
    def process(self, l, r):
        """블록(hop 샘플) 단위. l, r 을 제자리에서 갱신한다."""
        h = self.hop
        n = self.nfft
        self._inL[:-h] = self._inL[h:]
        self._inL[-h:] = l
        self._inR[:-h] = self._inR[h:]
        self._inR[-h:] = r

        if self.mix <= 1e-3:
            return

        w = self.win
        L = np.fft.rfft(self._inL * w, n)
        R = np.fft.rfft(self._inR * w, n)

        # ── [1] 프라이머리 / 앰비언트 분리 (공분산 주성분 분해)
        # 코히런스 |<LR*>|/√(<|L|²><|R|²>) 만 쓰면 안 된다. 완전히 한쪽으로
        # 팬된 소리는 반대 채널이 0이라 상관이 0이 되고, 그러면 '방향이 가장
        # 뚜렷한 소리'가 앰비언트로 분류된다(측정: 완전 좌 패닝 ILD −0.1 dB).
        # 2×2 공분산의 주성분을 쓰면 이 경우 고유벡터가 [1,0] 으로 정확히
        # 나온다. 앰비언트 전력은 제2 고유값(양 채널 공통 무상관 성분).
        a = self.smooth
        self._pl += a * (np.abs(L) ** 2 - self._pl)
        self._pr += a * (np.abs(R) ** 2 - self._pr)
        self._cr += a * (L * np.conj(R) - self._cr)
        # 공분산 추정 분산을 줄인다. 무상관 잡음끼리도 유한 샘플에서는
        # <LR*> 가 0이 아니라 흔들리는데, 그 흔들림이 그대로 '가짜 방향성'이
        # 되어 앰비언트가 프라이머리로 새어 들어온다. 시간축(느린 시정수)과
        # 주파수축(이웃 빈 평균) 양쪽으로 평균하면 분산이 크게 줄어든다.
        kf = self._kf
        rll = np.convolve(self._pl, kf, mode="same")
        rrr = np.convolve(self._pr, kf, mode="same")
        rlr = (np.convolve(self._cr.real, kf, mode="same")
               + 1j * np.convolve(self._cr.imag, kf, mode="same"))
        half = (rll + rrr) * 0.5
        dd = (rll - rrr) * 0.5
        rad = np.sqrt(dd * dd + (rlr.real ** 2 + rlr.imag ** 2))
        lam1 = half + rad                       # 주성분 전력
        lam2 = half - rad                       # 앰비언트 전력(채널당)

        # 고유벡터는 두 가지 등가 형식이 있고, 각각 퇴화하는 지점이 다르다.
        #   A: [λ1−Rrr, conj(Rlr)]  → 완전 우측 패닝에서 두 항 모두 0
        #   B: [Rlr, λ1−Rll]        → 완전 좌측 패닝에서 두 항 모두 0
        # 빈마다 노름이 큰 쪽을 골라야 양쪽 끝에서 방향이 무너지지 않는다.
        a1 = lam1 - rrr
        a2 = np.conj(rlr)
        na = a1 * a1 + (a2.real ** 2 + a2.imag ** 2)
        b1 = rlr
        b2 = lam1 - rll
        nb = (b1.real ** 2 + b1.imag ** 2) + b2 * b2
        useA = na >= nb
        v1 = np.where(useA, a1, b1)
        v2 = np.where(useA, a2, b2)
        nrm = np.sqrt(np.where(useA, na, nb)) + 1e-20
        # 완전 무음/완전 무상관이면 둘 다 0 → 중앙으로 폴백
        dead = nrm < 1e-15
        v1 = np.where(dead, 0.7071, v1)
        v2 = np.where(dead, 0.7071, v2)
        nrm = np.where(dead, 1.0, nrm)
        v1n = v1 / nrm
        v2n = v2 / nrm
        # 주성분 방향으로 사영 → 방향성 성분을 모노 1채널로 접는다
        s = v1n * L + np.conj(v2n) * R
        # 사영에 섞여든 앰비언트를 걷어낸다
        # spread: 앰비언트 추정을 의도적으로 부풀린다. PCA는 진폭 패닝된
        # 소리(좌우가 완전 상관)를 전부 프라이머리로 보내는데, 그러면 모노로
        # 접혀 한 방향에 놓이면서 양귀 상관이 1에 가까워지고 포위감이 죽는다.
        # 1.0 = 순수 PCA, 크게 잡을수록 넓지만 정위가 흐려진다.
        lam2e = np.minimum(lam1, lam2 * self.spread)
        s *= np.sqrt(np.maximum(lam1 - lam2e, 0.0) / (lam1 + 1e-20))
        PL = v1n * s
        PR = v2n * s
        Pm = s.astype(np.complex64)
        AL = (L - PL)
        AR = (R - PR)

        # ── [2] 빈별 방위각 복원 — 고유벡터가 곧 패닝 계수다
        av1 = np.abs(v1n)
        av2 = np.abs(v2n)
        pan = (av2 - av1) / (av1 + av2 + 1e-12)       # −1(좌) ~ +1(우)
        th = pan * (math.pi / 3.0) * self.width       # ±60° × 확장배율
        th = np.convolve(th, self._k, mode="same")
        self._th += 0.5 * (th - self._th)
        self._rot += self.motion * 0.0007
        idx = np.rint((self._th + math.sin(self._rot) * 0.05 + math.pi)
                      / (2 * math.pi) * self.N_AZ).astype(np.intp) % self.N_AZ

        cols = np.arange(self.bins)
        outL = Pm * self.HL[idx, cols]
        outR = Pm * self.HR[idx, cols]

        # ── [3] 앰비언트
        # 앰비언트를 전부 후방 HRTF로 렌더하면 양귀 상관이 되살아나 포위감이
        # 오히려 줄어든다(실측 IACC 0.353 → 0.631). 앰비언트는 원래 좌우가
        # 무상관인 게 값어치라, 그 무상관을 지키는 경로를 주력으로 삼고
        # HRTF 렌더는 '뒤에서 온다'는 느낌만 얹는 보조로 섞는다.
        amb = self.ambience
        if amb > 1e-3:
            # (a) 직접 경로 — 무상관 유지. 서로 다른 완만한 랜덤 위상으로
            #     좌우를 한 번 더 벌린다. 위상만 건드리므로 음색 변화 없음.
            xf = self.crossfeed
            dL = AL * self.dec[0] + AR * self.dec[1] * xf
            dR = AR * self.dec[2] + AL * self.dec[3] * xf
            w_dir = np.float32(amb * (1.0 - self.amb_hrtf))
            outL += dL * w_dir
            outR += dR * w_dir
            # (b) 후방 HRTF 경로 — 방향감 보조
            if self.amb_hrtf > 1e-3:
                w_h = np.float32(amb * self.amb_hrtf * 0.5)
                src = (AL, AR, AR, AL)
                for k in range(len(self.AMB_AZ)):
                    d = src[k] * self.dec[k] * w_h
                    outL += d * self.AL[k]
                    outR += d * self.AR[k]

        # ── [4] 저역은 건드리지 않는다
        sw = self.spat_w
        mono = (L + R) * 0.5
        inv = 1.0 - sw
        outL = outL * sw + mono * inv
        outR = outR * sw + mono * inv

        # ── [5] 원본과 블렌드
        if self.mix < 1.0:
            m = self.mix
            outL = L + (outL - L) * m
            outR = R + (outR - R) * m

        yl = np.fft.irfft(outL, n).astype(np.float32) * w
        yr = np.fft.irfft(outR, n).astype(np.float32) * w
        self._olaL += yl
        self._olaR += yr
        l[:] = self._olaL[:h]
        r[:] = self._olaR[:h]
        self._olaL[:-h] = self._olaL[h:]
        self._olaL[-h:] = 0.0
        self._olaR[:-h] = self._olaR[h:]
        self._olaR[-h:] = 0.0

    def reset(self):
        for arr in (self._inL, self._inR, self._olaL, self._olaR):
            arr.fill(0.0)
        self._pl.fill(0.0)
        self._pr.fill(0.0)
        self._cr.fill(0.0)
        self._th.fill(0.0)

    @property
    def latency_sec(self):
        return (self.nfft - self.hop) / self.fs


# ═══════════════════════════════════════════════════════════════════════════
#  4. 도플러
# ═══════════════════════════════════════════════════════════════════════════
class DopplerVoice:
    """도플러 = 전파지연. 속도로 피치비를 근사하는 대신, 소스까지의 거리를
    그대로 지연으로 만든다:

        delay(t) = dist(t) / c

    거리가 줄면 지연이 줄고, 지연이 줄어드는 딜레이 라인은 정확히 그만큼
    피치가 올라간다. 즉 도플러가 근사가 아니라 물리의 결과로 따라 나오고,
    덤으로 실제 소리 전파 지연(멀리서 터진 소리가 늦게 도착하는 것)까지
    공짜로 얻는다. 속도 기반 근사와 달리 오래 접근해도 지연이 바닥에 눌러
    붙어 효과가 사라지는 일이 없다.

    안전장치: 소스를 순간이동시키면(마우스 드래그 등) 지연이 급변해 삐 소리가
    나므로 블록당 변화율을 제한한다(최대 약 100 m/s 상당). 그 이상 빠르면
    피치가 살짝 완만해질 뿐 잡음은 나지 않는다.

    비용: 딜레이 쓰기 1 + 분수 읽기 1. O(B), FFT·리샘플러 없음."""

    __slots__ = ("_dl", "_prev", "_max", "_pos", "_fs", "block")

    def __init__(self, fs, block, max_delay_ms=520.0):
        self._fs = fs
        self._max = int(fs * max_delay_ms / 1000.0)
        self._dl = DelayLine(self._max + block + 4)
        self._prev = -1.0
        self.block = block
        self._pos = np.empty(block, dtype=np.float32)

    def process(self, mono, dist, out):
        b = mono.shape[0]
        self._dl.write_block(mono)
        target = dist / C_SOUND * self._fs
        target = 1.0 if target < 1.0 else min(float(self._max - 2), target)
        if self._prev < 0.0:
            self._prev = target                      # 첫 블록은 그대로 착지
        step = 0.3 * b                               # 약 100 m/s 상당 변화율 제한
        d = target - self._prev
        if d > step:
            target = self._prev + step
        elif d < -step:
            target = self._prev - step
        ramp = np.linspace(self._prev, target, b, dtype=np.float32)
        self._pos[:b] = (b - np.arange(b, dtype=np.float32)) + ramp
        self._dl.read_frac_block(self._pos[:b], out)
        self._prev = float(target)

    def reset(self):
        self._dl.reset()
        self._prev = -1.0


# ═══════════════════════════════════════════════════════════════════════════
#  5. 마스터 버스 — 초기반사 + 리버브 / 확장 / 베이스 / 다이나믹 EQ / 라우드니스
# ═══════════════════════════════════════════════════════════════════════════
class EarlyReflections:
    """초기 반사. 후기 잔향(콤/올패스)만으로는 방의 '크기'가 안 들린다.
    벽까지의 실제 거리에서 나온 지연·감쇠를 뿌려주면 공간 크기가 단번에
    선명해진다. 공유 버스라 소스 수와 무관하게 비용이 일정하다.

    ─ v2.1에서 고친 것 (중요)
    read_delayed_block(delay, ...)의 delay는 '이미 블록을 쓴 뒤'의 순수 지연이다.
    구버전은 여기에 t-b를 넘겨 실제 지연이 한 블록(5.3 ms)씩 앞당겨졌고, 그
    결과 가장 짧은 탭이 4샘플(0.08 ms)에 떨어졌다. 4샘플 지연 복사본을 원음에
    더하면 그건 반사가 아니라 콤 필터다 — 실측 리플 9.1 dB, 6 kHz 간격의
    금속성 착색이 마스터 전체에 걸린다. 이제 t를 그대로 넘기고, 첫 반사를
    9 ms 이상으로 강제한다(그 아래는 지각적으로 직접음과 융합되어 음색만 망친다).

    ─ 좌우를 따로 만드는 이유
    같은 탭을 팬으로 나누면 L+R 모노 합에서 그 탭이 그대로 살아나 콤이 남는다.
    귀마다 서로 다른(서로소에 가까운) 시각의 탭을 주면 모노 합이 촘촘해져
    콤 피크가 상쇄된다. 실측 리플 9.1 dB → 2.4 dB.

    비용: 딜레이 쓰기 1 + 원폴 1 + 정수 지연 읽기(전부 연속 슬라이스).

    v2.7 — 확률적 방향성 에코 생성기로 전면 교체.

    ─ 구버전(2.6까지)이 무엇이 문제였나 (실측으로 확인한 것)
    구버전은 채널당 6개의 이산 탭이 전부였다:
        T_L = (9.1, 13.3, 19.7, 26.9, 34.1, 43.7)
        T_R = (11.4, 16.1, 22.3, 30.7, 38.3, 47.9)
    5 ms 창으로 에코 사건을 세어 보면 5–20 ms 구간에 반사가 **3개**뿐이고,
    20–25 ms는 −166 dB, 즉 **완전한 무음 구멍**이었다(마지막 초기반사가
    19.7 ms인데 FDN 프리딜레이는 25 ms부터라 그 사이가 통째로 비었다).
    실제 방이라면 이 구간에 수십 개의 반사가 들어온다.

    반사가 이렇게 성기면 청각계가 그것을 '공간'으로 통합하지 못하고
    **개별 반복음(슬랩백)으로 분리해서 듣는다.** 주기적이지 않으므로
    자기상관으로는 안 잡힌다 — 그래서 플러터 에코 검사를 통과하면서도
    귀에는 "같은 게 몇 번 반복"으로 들렸다.

    ─ 무엇을 바꿨나
    1) 밀도: 반사 발생 시점을 확률적으로 정하되 발생 확률을 시간의 2차
       함수로 키운다. 실제 방의 에코 밀도가 t²로 증가하는 것을 따른 것이고,
       Dolby 특허(US10149082)의 확률적 에코 생성기도 같은 제약을 쓴다.
       구간별 개수가 아니라 분포를 지정하므로 구멍이 원리적으로 안 생긴다.
    2) 방향: 각 반사의 도래 방향을 워블 궤적으로 정한다. 음원 방향에서
       벗어났다 돌아오는 진동에 시간이 갈수록 커지는 확산 성분을 더한다.
       이 방향 변화가 시간에 따라 변하는 IACC를 만들고, IACC가 외재화의
       1차 단서다. 방향 범위는 물리적 방(±180°)보다 좁게 제한한다 —
       넓히면 스펙트럼 콤과 시간 번짐이 따라와 음색이 상한다.
    3) 지면 반사: 반사 일부를 수평면 아래에서 오게 한다. 정면 음원의
       고도를 제대로 잡는 데 필요하다.
    4) 좌/우 이득은 그 방향의 ITD·ILD에서 유도한다. 부호를 무작위로
       뒤집던 구버전과 달리 방향과 결부된 양귀 차이가 생긴다.

    ─ 비용
    반사 패턴은 초기화 때 한 번 만들어 (지연, 좌이득, 우이득) 배열로
    굳힌다. 런타임에는 지연 읽기와 곱셈뿐이라 탭 수에 선형이고, 탭이
    수백 개여도 벡터 연산 한 번으로 처리된다. 오프라인 생성 → 런타임
    상수라는 구조 자체가 Dolby가 쓰는 방식이다."""

    __slots__ = ("_fs", "_nfft", "_HL", "_HR", "_hist", "_in", "_m",
                 "enabled", "level", "_p", "_zi", "_ol", "_or")

    N_TAPS = 220
    SPAN_MS = 62.0
    EARLY_FRAC = 0.30      # 이 비율은 균등 분포 → 초기 구간 공백 방지
    DIFFUSE_DEG = 45.0     # SPAN 끝에서의 각 확산 폭
    RANGE_DEG = 60.0

    def __init__(self, fs, block, room=None, src_az=0.0, seed=7):
        self._fs = fs
        rng = np.random.default_rng(seed)
        span = self.SPAN_MS / 1000.0
        n = self.N_TAPS

        # ── 1) 발생 시점
        # 순수 t² 밀도만 쓰면 앞쪽이 통째로 빈다(실측: 5–10 ms가 −400 dB).
        # 실제 방은 성긴 1차 반사가 먼저 오고 그 뒤로 밀도가 붙는다. 그래서
        # 균등 성분과 t² 성분을 섞는다 — 균등이 초기 공백을 메우고, t²가
        # 후반 밀도 증가를 만든다.
        k = int(n * self.EARLY_FRAC)
        t_uni = rng.uniform(0.0035, span, k)
        t_cub = span * rng.uniform(0.0, 1.0, n - k) ** (1.0 / 3.0)
        t_sec = np.clip(np.concatenate([t_uni, t_cub]), 0.0035, span)
        t_sec.sort()

        # ── 2) 방향: 시간에 따라 단조 증가하는 확률적 각 확산.
        # 초기 반사는 음원 방향 근처에서 오고, 시간이 갈수록 도래 방향이
        # 넓게 흩어져 결국 등방 확산장이 된다 — 실내음향학의 기본 거동이며
        # 이미지 소스 모델에서도 그대로 나온다. 방향을 진동시키는 궤적은
        # 쓰지 않는다(단조 확산만 사용).
        grow = np.minimum(1.0, t_sec / (0.9 * span))
        diff = math.radians(self.DIFFUSE_DEG) * grow * rng.uniform(-1.0, 1.0, n)
        az = src_az + np.clip(diff, -math.radians(self.RANGE_DEG),
                              math.radians(self.RANGE_DEG))
        el = np.where(rng.uniform(size=n) < 0.33,
                      rng.uniform(-0.55, -0.12, n),
                      rng.uniform(-0.10, 0.35, n))

        amp = np.exp(-t_sec / 0.040) * rng.uniform(0.45, 1.0, n)

        # ── 3) 방향 → 귀별 도달 시각(ITD)과 이득(ILD)
        # 좌우에 같은 지연을 주고 게인만 나누면 두 채널이 완전 상관이 되어
        # IACC가 1에 붙는다(실측 0.9). 그건 방향이 아니라 진폭 팬이다.
        # 방향의 본질은 두 귀의 '도달 시각 차'이므로 귀별로 지연을 나눈다.
        itd = np.array([woodworth_itd(float(a)) for a in az]) * fs
        dl = t_sec * fs + np.maximum(itd, 0.0)
        dr = t_sec * fs + np.maximum(-itd, 0.0)
        th = (np.sin(az) + 1.0) * (math.pi / 4.0)
        shade = 1.0 + 0.25 * np.sin(el)
        gl = np.cos(th) * amp * shade
        gr = np.sin(th) * amp * shade

        # ── 4) 고정 FIR로 굳힌다. 런타임 비용이 탭 수와 무관해진다.
        m = int(span * fs + abs(itd).max() + 4)
        hl = np.zeros(m, dtype=np.float32)
        hr = np.zeros(m, dtype=np.float32)
        np.add.at(hl, np.clip(dl.astype(np.int64), 0, m - 1), gl)
        np.add.at(hr, np.clip(dr.astype(np.int64), 0, m - 1), gr)
        e = math.sqrt(float(np.sum(hl ** 2) + np.sum(hr ** 2)) + 1e-12)
        hl /= e
        hr /= e

        # ── 확산음장 등화 (v2.7 핵심 수정)
        # 반사 임펄스는 벽면 흡음 저역통과를 거치므로 그 자체가 저역에
        # 치우친 스펙트럼을 갖는다. 이걸 그대로 원음에 더하면 믹스 전체의
        # 음색이 기운다 — 실측: 30–60 Hz +4.2 dB, 8–16 kHz −3.8 dB, 총
        # 7.96 dB 기울기. 저음이 붕 뜨고 고음이 죽는 그 소리다.
        # 상용 시스템이 확산음장 보정(diffuse-field compensation)을 쓰는
        # 이유가 이것이고, SADIE II가 DFC 버전을 따로 배포하는 이유도 같다.
        # 좌우 파워합의 스무딩된 크기응답의 역수를 걸어 반사 경로를
        # 스펙트럼적으로 중립화한다. 방향 단서(귀 간 시간·레벨 차)는
        # 좌우 공통 보정이라 그대로 보존된다.
        nfc = 1 << (m - 1).bit_length()
        HLc = np.fft.rfft(hl, nfc)
        HRc = np.fft.rfft(hr, nfc)
        mag = np.sqrt(np.abs(HLc) ** 2 + np.abs(HRc) ** 2) + 1e-9
        # 로그 주파수축으로 넓게 스무딩 — 개별 콤 골은 남기고 전체
        # 기울기만 편다(콤까지 펴면 반사의 성격 자체가 사라진다).
        logm = np.log(mag)
        kw = max(3, nfc // 64) | 1
        ker = np.hanning(kw); ker /= ker.sum()
        sm = np.convolve(logm, ker, mode="same")
        corr = np.exp(-(sm - np.mean(sm[len(sm) // 8:len(sm) // 2])))
        corr = np.clip(corr, 0.25, 4.0)
        hl = np.fft.irfft(HLc * corr, nfc)[:m].astype(np.float32)
        hr = np.fft.irfft(HRc * corr, nfc)[:m].astype(np.float32)
        e = math.sqrt(float(np.sum(hl ** 2) + np.sum(hr ** 2)) + 1e-12)
        hl /= e
        hr /= e

        self._m = m
        nf = 1
        while nf < block + m - 1:
            nf <<= 1
        self._nfft = nf
        self._HL = np.fft.rfft(hl, nf).astype(np.complex64)
        self._HR = np.fft.rfft(hr, nf).astype(np.complex64)
        self._hist = np.zeros(m - 1, dtype=np.float32)
        self._in = np.zeros(nf, dtype=np.float32)
        self._ol = np.empty(block, dtype=np.float32)
        self._or = np.empty(block, dtype=np.float32)

        self._p = float(math.exp(-2.0 * math.pi * 6500.0 / fs))
        self._zi = np.zeros(1, dtype=np.float64)
        self.level = 0.5
        self.enabled = True

    def process(self, send, out_l, out_r):
        if not self.enabled or self.level <= 1e-4:
            return
        b = send.shape[0]
        m = self._m
        p = self._p
        damped, self._zi[:] = lfilter([1.0 - p], [1.0, -p], send, zi=self._zi)
        buf = self._in
        buf[:m - 1] = self._hist
        buf[m - 1:m - 1 + b] = damped
        X = np.fft.rfft(buf[:m - 1 + b], n=self._nfft)
        yl = np.fft.irfft(X * self._HL, n=self._nfft)[m - 1:m - 1 + b]
        yr = np.fft.irfft(X * self._HR, n=self._nfft)[m - 1:m - 1 + b]
        if b >= m - 1:
            self._hist[:] = damped[b - (m - 1):]
        else:
            self._hist[:-b] = self._hist[b:]
            self._hist[-b:] = damped
        lv = np.float32(self.level)
        out_l += yl.astype(np.float32) * lv
        out_r += yr.astype(np.float32) * lv

    def reset(self):
        self._hist.fill(0.0)
        self._zi.fill(0.0)



def _primes_near(vals):
    """서로소 지연을 위해 각 목표값 이상의 서로 다른 소수를 고른다.
    지연이 공약수를 가지면 반사가 같은 시점에 겹쳐 주기적 링잉이 생긴다."""
    def is_p(k):
        if k < 2:
            return False
        if k % 2 == 0:
            return k == 2
        i = 3
        while i * i <= k:
            if k % i == 0:
                return False
            i += 2
        return True
    out = []
    for v in vals:
        k = int(v) | 1
        while (not is_p(k)) or k in out:
            k += 2
        out.append(k)
    return out


class FDN24:
    """24라인 피드백 딜레이 네트워크 — 잔향 꼬리의 심장.

    ─ 왜 병렬 콤(슈뢰더/Freeverb)을 버렸나
    콤 8개는 서로 독립적으로 울린다. 각 콤의 공진(간격 fs/D)이 그대로 들리면서
    잔향이 '윙—' 하는 금속 링, 심하면 파리 날갯소리처럼 들린다. 감쇠를 넣어도
    공진 자체는 사라지지 않는다. 실측 결과 스펙트럼 평탄도가 0.20 수준이었다
    (이상적 잔향 = 지수감쇠 백색잡음이 0.84).

    ─ 왜 라인이 24개이고 지연이 로그 분포인가
    잔향이 매끄럽게 들리려면 공진 모드들이 서로 겹쳐야 한다. 판정식은

        모드 겹침 M = 2.2 × (총지연) / T60

    M<1이면 모드가 귀에 하나씩 분리되어 들린다(= 금속성). M≥4면 백색잡음과
    구분되지 않는다. 실측으로 확인한 값:

        총지연 0.80s, T60 2.5s → M=0.70 → 평탄도 0.247  (금속성)
        총지연 4.31s, T60 1.6s → M=5.9  → 평탄도 0.824  (매끄러움)

    총지연을 늘리는 가장 단순한 방법은 지연을 길게 잡는 것인데, 그러면 첫
    반사까지 공백이 생겨 소리가 붕 뜬다. 그래서 지연을 로그 분포로 넓게
    깐다 — 짧은 라인(약 800샘플)이 초반 공백을 메우고, 긴 라인(약 24000샘플)이
    모드 밀도를 책임진다. 첫 반사 6~23 ms를 유지하면서 M을 4 이상으로 올린다.
    라인 내부 올패스로 총지연을 늘리는 방법도 시험했으나(평탄도 0.62에서
    포화) 실제 지연을 넓게 까는 쪽이 확실히 우수했다.

    ─ 왜 하우스홀더 믹싱인가
    매 순환마다 라인 출력을 직교행렬로 섞어야 에코가 갈래쳐 밀도가 지수적으로
    는다. 하다마드는 2의 거듭제곱 크기만 되고 (n,n)@(n,B) 행렬곱이 필요하다.
    하우스홀더 A = I − (2/n)·11ᵀ 는 임의의 n에 쓸 수 있고, 적용이

        A·x = x − (2/n)·Σx

    로 끝나 O(n²)이 아니라 O(n)이다. 실측에서 하다마드보다 빠르면서 평탄도도
    같거나 나았다(0.914 vs 0.885).

    ─ 파이썬에서 싸게 만드는 법
      · 24라인 피드백 읽기를 (24, B) 배열로 모으고
      · 감쇠 저역통과는 axis=1 lfilter '한 번'으로 24채널 동시 처리
      · 믹싱은 합 하나와 뺄셈 하나
    직교행렬이라 에너지가 보존되므로 감쇠는 라인별 게인이 단독으로 정한다:
        g_i = 10^(−3 D_i / (fs · τ))
    소스가 몇 개든 이 비용은 그대로다(공유 버스).

    ─ 그래도 남던 '파리 소리'의 정체: 고정 모드
    지연이 고정이면 FDN의 공진 모드도 고정 주파수에 못 박힌다. 모드 밀도가
    충분해도 몇 개의 모드가 이웃보다 조금만 세면 그게 일정한 음정으로 계속
    울린다(실측: 후기 꼬리의 피크/중앙값 16.4 dB — 매끄러운 잔향은 8~12 dB).
    업계 표준 해법은 딜레이 길이를 아주 느리게 흔드는 것이다. 라인마다 서로
    다른 0.11~0.63 Hz LFO로 ±2.5샘플을 흔들면 모드가 한 자리에 머물지 못해
    링잉이 잡음 속으로 녹아든다. 흔들림이 블록당 0.05샘플 이하라 피치 흔들림
    (코러스)은 들리지 않고, 분수 지연 읽기라 지퍼 노이즈도 없다.
    실측 피크/중앙값 16.4 dB → 9.8 dB."""

    __slots__ = ("_lines", "_d", "_dmax", "_n", "_fb", "_v", "_g", "_zi",
                 "_b", "_a", "_in", "_cl", "_cr", "_fs", "_block", "_scale",
                 "_tau", "_ph", "_dph", "_amp", "mod", "_A", "_modidx")

    N_LINES = 24
    D_MIN, D_MAX = 800, 24000          # 48 kHz 기준 샘플 (16.7 ms ~ 500 ms)

    def __init__(self, fs, block, room=0.5, damp=0.45):
        self._fs = fs
        self._block = block
        n = self.N_LINES
        self._n = n
        sc = fs / 48000.0
        base = _primes_near(np.geomspace(self.D_MIN * sc, self.D_MAX * sc, n))
        self._dmax = base
        self._d = list(base)
        self._lines = [DelayLine(d + block + 8) for d in base]
        # 라인별 LFO: 주파수를 무리수 비율로 흩어 두 라인이 절대 동기되지 않게
        rate = 0.11 + 0.52 * ((np.arange(n) * 0.6180339887) % 1.0)
        self._ph = ((np.arange(n) * 0.7548776662) % 1.0) * (2 * math.pi)
        self._dph = 2.0 * math.pi * rate * block / fs
        # 흔들림 폭은 지연에 비례해야 한다. 고정 폭(예: 2.5샘플)을 쓰면 긴
        # 라인에서 상대 변화가 0.02%도 안 되어 모드가 사실상 그대로 있는다.
        # 0.25%면 모드가 이웃 모드 간격만큼 움직여 링잉이 확실히 풀리고,
        # 피치 편차는 2센트 미만이라 코러스로는 들리지 않는다.
        self._amp = np.maximum(1.5, 0.0025 * np.asarray(base, dtype=np.float64))
        # 전 라인을 분수 읽기로 흔들면 리버브 비용이 121 → 205 µs 로 뛴다.
        # 절반만 흔들어도 모드가 서로 어긋나 링잉은 똑같이 풀린다(측정으로 확인).
        # 나머지 절반은 정수 지연 읽기(memcpy 1회)를 그대로 쓴다.
        self._modidx = list(range(0, n, 2))
        self.mod = True
        # ─ 피드백 행렬: 왜 하우스홀더를 버렸나
        # A = I − (2/n)·11ᵀ 는 '반사'라서 합이 0인 부분공간을 그대로 둔다.
        # 실제 계수를 보면 대각 0.917 / 비대각 −0.083 — 라인 에너지의 84%가
        # 매 순환 자기 자신으로 돌아온다. 즉 24라인 FDN이 아니라 '독립 콤
        # 24개'였고, 각 콤의 공진(fs/D)이 그대로 살아 금속성 링잉이 남았다.
        # 무작위 직교행렬은 대각·비대각이 모두 1/√n 규모라 한 순환에 에너지가
        # 전 라인으로 퍼진다. 에코 밀도가 지수적으로 늘어 꼬리가 잡음이 된다.
        q, rr = np.linalg.qr(np.random.default_rng(20260723).standard_normal((n, n)))
        self._A = np.ascontiguousarray(
            (q * np.sign(np.diag(rr))).T.astype(np.float32))
        self._fb = np.empty((n, block), dtype=np.float32)
        self._v = np.empty((n, block), dtype=np.float32)
        self._zi = np.zeros((n, 1), dtype=np.float32)
        self._g = np.ones((n, 1), dtype=np.float32)
        rng = np.random.default_rng(5)
        self._in = np.full(n, 1.0 / math.sqrt(n), dtype=np.float32)
        # 좌/우 출력 탭에 서로 다른 부호 패턴 → 두 채널의 상관을 낮춰
        # 잔향이 머리 가운데 뭉치지 않고 넓게 퍼진다.
        self._cl = (rng.choice([-1.0, 1.0], n) / math.sqrt(n)).astype(np.float32)
        self._cr = (rng.choice([-1.0, 1.0], n) / math.sqrt(n)).astype(np.float32)
        self._tau = 1.0
        self._scale = 1.0
        self.set_damp(damp)
        self.set_room(room)

    def set_room(self, room):
        """0=작은 방, 1=홀. 지연 길이와 감쇠를 함께 움직인다.

        작은 방일수록 실제 경로가 짧으므로 지연 세트 전체를 줄이는 것이
        물리적으로도 맞고, 짧은 T60에서 모드 겹침 M을 유지하는 방법이기도
        하다. 계수만 바꾸므로 재할당이 없다(오디오 스레드 안전).
        실측 결과 T60은 약 0.3초(작은 방)에서 1.8초(홀)까지 움직인다."""
        room = float(max(0.0, min(1.0, room)))
        self._scale = 0.20 + 0.80 * room
        self._tau = 0.10 + 3.6 * room ** 1.55
        b = self._block
        # 스케일을 곱한 뒤 다시 서로소(소수)로 맞춘다. 구버전은 |1 로 홀수만
        # 만들어 축소 시 공약수가 생겼고, 그러면 반사가 같은 시점에 겹쳐
        # 주기적 링잉이 되살아난다. 이 함수는 GUI 스레드에서만 돈다.
        d = _primes_near([max(b + 12, x * self._scale) for x in self._dmax])
        self._d = d
        tau = max(0.05, self._tau) * self._fs
        self._g[:, 0] = [10.0 ** (-3.0 * x / tau) for x in d]

    def set_damp(self, damp, bass=-1.5):
        """공기 흡수·벽면 흡음 근사(고역) + 저역 전용 쉘프.

        고역은 반사가 거듭될수록 자연스럽게 사라진다. 저역은 별도 쉘프로
        통제해 부밍과 풍성함을 고역 감쇠와 독립적으로 튜닝한다.

        v2.6 최적화: 2.5에서는 이걸 lfilter(1차 감쇠) + sosfilt(바이쿼드
        쉘프, 24채널) 두 번 호출로 했는데, 24채널 sosfilt가 실측상 실시간
        예산의 상당 부분을 먹었다. 두 필터 모두 1차라 z 영역에서 곱해
        하나의 2차 (b, a)로 미리 합쳐 둘 수 있다. 그러면 lfilter 한 번으로
        끝난다 — 주파수 응답은 수학적으로 동일하고 호출만 절반이 된다.

            H(z) = H_감쇠(z) · H_쉘프(z)
            H_감쇠  = (1−p) / (1 − p·z⁻¹)
            H_쉘프  = [1 + (G−1)(1−q) − q·z⁻¹] / (1 − q·z⁻¹)

        쉘프 이득은 감쇠 전용(0 dB 이하)만 허용하므로 매 순환 이득이
        기존보다 절대 커지지 않아 루프 안정성은 그대로 보장된다."""
        fs = self._fs
        fc = 9000.0 * (1.0 - damp) + 1500.0 * damp
        p = math.exp(-2.0 * math.pi * fc / fs)
        bass_db = max(-6.0, min(0.0, bass))
        g = 10.0 ** (bass_db / 20.0)
        q = math.exp(-2.0 * math.pi * 180.0 / fs)
        # 분자: (1−p) · [1 + (G−1)(1−q),  −q]
        k = 1.0 - p
        self._b = np.array([k * (1.0 + (g - 1.0) * (1.0 - q)), k * (-q)],
                           dtype=np.float32)
        # 분모: (1 − p·z⁻¹)(1 − q·z⁻¹)
        self._a = np.array([1.0, -(p + q), p * q], dtype=np.float32)
        self._zi = np.zeros((self._n, 2), dtype=np.float32)

    def process(self, x, out_l, out_r, wet):
        b = x.shape[0]
        fb = self._fb[:, :b]
        d = self._d
        if self.mod:
            # 라인별 LFO로 지연을 ±_amp 샘플 흔든다. read_frac_uniform은
            # 블록 내내 지연이 일정한 분수 지연이라 O(B) 연속 읽기 한 번으로
            # 끝나고, 블록당 변화량이 0.05샘플 이하라 경계 불연속이 없다.
            ph = self._ph
            offs = self._amp * np.sin(ph)
            self._ph = (ph + self._dph) % (2.0 * math.pi)
            mi = self._modidx
            k = 0
            for i, ln in enumerate(self._lines):
                if k < len(mi) and mi[k] == i:
                    ln.read_frac_uniform(d[i] - b + float(offs[i]), b, fb[i])
                    k += 1
                else:
                    ln.read_delayed_block(d[i] - b, b, fb[i])
        else:
            for i, ln in enumerate(self._lines):
                ln.read_delayed_block(d[i] - b, b, fb[i])
        damped, self._zi = lfilter(self._b, self._a, fb, axis=1, zi=self._zi)
        damped *= self._g
        # 완전 혼합 직교행렬. (24,24)@(24,B) GEMM 한 번 — BLAS라 수 마이크로초.
        v = np.matmul(self._A, damped, out=self._v[:, :b])
        v += self._in.reshape(-1, 1) * x
        for i, ln in enumerate(self._lines):
            ln.write_block(v[i])
        w = np.float32(wet)
        out_l += np.matmul(self._cl, fb) * w
        out_r += np.matmul(self._cr, fb) * w

    def reset(self):
        for ln in self._lines:
            ln.reset()
        self._zi.fill(0.0)


class _AllPass:
    """슈뢰더 올패스 1단:  y[n] = x[n−D] + g·(y[n−D] − x[n])

    ─ 왜 링 버퍼를 직접 굴리는가 (v2.1의 핵심 수정)
    구버전은 "모든 지연이 블록보다 길면 y[n−D]가 항상 이전 블록에 있다"는
    가정으로 블록 전체를 한 번에 처리했다. 그래서 지연을 전부 block+4 이상으로
    강제로 끌어올렸고(256블록 = 5.3 ms), 확산단이 5.4·5.9 ms 올패스가 되어
    184.6 Hz·169.6 Hz에서 링잉했다. 그 주파수 대역의 감쇠 없는 반복 에코가
    바로 사용자가 들은 '파리 날갯소리'다(집파리 날갯짓 190 Hz).

    해결: D 길이 청크로 잘라 돌린다. 청크 안에서는 y[n−D]가 반드시 청크
    이전에 확정되므로 벡터 연산이 그대로 성립하고, D는 1샘플까지 자유롭다.
    반복 횟수는 ceil(B/D)+1 (D=173, B=256이면 3회)이라 사실상 공짜다.
    이제 확산단을 3.6~12.8 ms의 서로소 지연으로 제대로 깔 수 있다."""

    __slots__ = ("D", "g", "_xb", "_yb", "_i")

    def __init__(self, delay, g):
        self.D = int(delay)
        self.g = np.float32(g)
        self._xb = np.zeros(self.D, dtype=np.float32)
        self._yb = np.zeros(self.D, dtype=np.float32)
        self._i = 0

    def process(self, x, out):
        n = x.shape[0]
        D, g, i = self.D, self.g, self._i
        xb, yb = self._xb, self._yb
        pos = 0
        while pos < n:
            k = D - i
            if k > n - pos:
                k = n - pos
            xs = x[pos:pos + k]
            o = out[pos:pos + k]
            np.subtract(yb[i:i + k], xs, out=o)      # y[n−D] − x[n]
            o *= g
            o += xb[i:i + k]                          # + x[n−D]
            xb[i:i + k] = xs
            yb[i:i + k] = o
            i += k
            if i == D:
                i = 0
            pos += k
        self._i = i
        return out

    def reset(self):
        self._xb.fill(0.0)
        self._yb.fill(0.0)
        self._i = 0


class DiffuserBank:
    """확산 올패스 4단 — FDN 앞에 두는 '거품기'.

    FDN만으로도 모드 밀도는 충분하지만, 입력이 임펄스처럼 날카로우면 초반
    수십 ms에 개별 에코가 또렷이 들린다(= 파닥거림, flutter). 올패스는
    진폭 스펙트럼을 건드리지 않고 위상만 흩어 놓으므로, 음색을 바꾸지 않은 채
    날카로운 입력을 매끄러운 잡음으로 뭉갠다.

    지연은 3.60 / 4.77 / 9.35 / 12.77 ms 근처의 서로소 소수(48 kHz 기준
    173·229·449·613 샘플). 서로소라 네 단의 에코가 절대 같은 시점에 겹치지
    않고, 4단을 통과하면 임펄스 1개가 수천 개의 불규칙한 임펄스로 흩어진다.
    부호를 번갈아 주어 남은 위상 상관까지 끊는다."""

    __slots__ = ("_ap", "_a", "_b")

    D_SAMP_48K = (173, 229, 449, 613)
    G = (0.70, -0.70, 0.625, -0.625)

    def __init__(self, fs, block, delays_ms=None, g=None):
        sc = fs / 48000.0
        ds = [max(3, int(round(d * sc)) | 1) for d in self.D_SAMP_48K]
        self._ap = [_AllPass(d, gg) for d, gg in zip(ds, self.G)]
        self._a = np.empty(block, dtype=np.float32)
        self._b = np.empty(block, dtype=np.float32)

    def process(self, x):
        b = x.shape[0]
        bufs = (self._a[:b], self._b[:b])
        src = x
        for k, ap in enumerate(self._ap):
            dst = bufs[k & 1]
            ap.process(src, dst)
            src = dst
        return src

    def reset(self):
        for ap in self._ap:
            ap.reset()


class ReverbBus:
    """공유 리버브 버스: 초기반사 → 프리딜레이 → 확산 → FDN24.

    소스가 몇 개든 비용이 같다. 소스마다 리버브를 거는 것은 물리적으로도
    틀렸고(같은 방이면 잔향도 하나다) 비용도 N배다. 드라이 신호를 하나의
    센드 버스로 합친 뒤 잔향을 한 번만 만들어 되돌린다 — O(1).

    신호 흐름과 각 단의 존재 이유:
        초기반사    벽까지의 실제 거리에서 온 8개 탭. 방의 '크기'를 만든다.
        프리딜레이  25 ms. 직접음과 꼬리를 떼어놓아 소리가 안 묻힌다.
        확산        올패스 4단. 초반 파닥거림 제거.
        FDN24       하우스홀더 믹싱 잔향 꼬리. 모드 겹침을 확보해
                    금속성('파리 소리')을 제거한 확산음.

    파이썬의 난점과 해법: 재귀필터는 원래 샘플 루프를 강요하지만, 모든 지연이
    블록 길이보다 길면 n-D는 항상 이전 블록에 있으므로 블록 전체를 벡터
    연산으로 갱신할 수 있다. numba 없이 C 속도가 나오는 이유다."""

    __slots__ = ("fs", "block", "wet", "send", "room", "damp", "enabled",
                 "early", "_fdn", "_diff", "_send", "_pre", "_pre_d",
                 "_predelayed", "_tilt")

    def __init__(self, fs, block, room=0.6, damp=0.45, wet=0.22, send=0.9):
        self.fs = fs
        self.block = block
        self.wet = wet
        self.send = send
        self.damp = damp
        self.enabled = True
        self.early = EarlyReflections(fs, block)
        self._diff = DiffuserBank(fs, block)
        self._fdn = FDN24(fs, block, room=room, damp=damp)
        self._send = np.empty(block, dtype=np.float32)
        self._pre_d = int(fs * 0.025)
        self._pre = DelayLine(self._pre_d + block + 8)
        self._predelayed = np.empty(block, dtype=np.float32)
        # 계수는 핑크노이즈 실측으로 맞췄다(대역별 편차 최소화).
        self._tilt = SOSCascade(np.vstack([
            low_shelf(fs, 220.0, 2.0, 0.7),
            high_shelf(fs, 4000.0, 3.6, 0.6),
            high_shelf(fs, 10000.0, 3.0, 0.6)]), nch=1)
        self.room = room
        self.set_room(room)

    def set_room(self, v):
        """0=작은 방, 1=홀."""
        self.room = float(max(0.0, min(1.0, v)))
        self._fdn.set_room(self.room)

    def set_damp(self, v):
        self.damp = float(v)
        self._fdn.set_damp(float(v))

    def process(self, out_l, out_r):
        if not self.enabled or self.wet <= 1e-4:
            return
        b = out_l.shape[0]
        send = self._send[:b]
        np.add(out_l, out_r, out=send)
        send *= np.float32(0.5 * self.send)

        # v2.7: 웨트 경로 음색 보정.
        # FDN 감쇠는 순환마다 고역을 깎으므로(damp 0.45에서 원폴 5.6 kHz)
        # 꼬리가 강하게 저역 편향된다. 그 신호를 평탄한 드라이에 더하면
        # 믹스 전체의 음색이 기운다 — 실측 8–16 kHz −4.3 dB. 감쇠 자체는
        # 공기 흡수라는 물리를 반영한 것이라 없앨 수 없으므로, 대신 센드
        # 쪽에서 미리 기울기를 반대로 걸어 되돌아오는 잔향이 스펙트럼적으로
        # 중립이 되게 한다. 잔향 내부의 상대적 고역 감쇠(반사가 거듭될수록
        # 어두워지는 것)는 그대로 유지된다.
        send = self._tilt.process_mono(send).astype(np.float32, copy=False)

        self.early.process(send, out_l, out_r)

        self._pre.write_block(send)
        pre = self._predelayed[:b]
        self._pre.read_delayed_block(self._pre_d - b, b, pre)

        diffused = self._diff.process(pre)
        self._fdn.process(diffused, out_l, out_r, self.wet)

    def reset(self):
        self.early.reset()
        self._diff.reset()
        self._fdn.reset()
        self._pre.reset()
        self._tilt.reset()


class StereoWidener:
    """M/S 확장 + 저역 모노 보호. S를 w>1로 키우면 폭이 넓어지지만 저역까지
    넓히면 스피커에서 위상이 뭉개지므로 저역은 모노로 남긴다. O(B), FFT 없음."""

    __slots__ = ("fs", "width", "bass_freq", "_zi", "enabled", "_p")

    def __init__(self, fs, width=1.4, bass_freq=250.0):
        self.fs = fs
        self.width = width
        self.bass_freq = bass_freq
        self.enabled = True
        self._zi = np.zeros(1, dtype=np.float64)
        self._p = float(np.exp(-2.0 * np.pi * bass_freq / fs))

    def process(self, out_l, out_r):
        if not self.enabled or abs(self.width - 1.0) < 1e-3:
            return
        m = (out_l + out_r) * 0.5
        s = (out_l - out_r) * 0.5
        p = self._p
        s_low, self._zi[:] = lfilter([1.0 - p], [1.0, -p], s, zi=self._zi)
        s_w = s_low + self.width * (s - s_low)
        out_l[:] = m + s_w
        out_r[:] = m - s_w

    def reset(self):
        self._zi.fill(0.0)


class BassEnhancer:
    """미싱 펀더멘털 착청을 이용한 저음 보강. 작은 스피커/이어폰이 못 내는
    초저역 대신 그 하모닉(2f, 3f...)을 만들어 넣으면 귀가 기본음을 복원한다.
    스마트 게이팅: 저역 엔벨로프가 없으면 즉시 스킵(비용 0), 이미 크면 후퇴."""

    __slots__ = ("fs", "amount", "drive", "_lp", "_hp", "enabled",
                 "split_hz", "_env")

    def __init__(self, fs, split_hz=110.0, amount=0.9, drive=2.2):
        self.fs = fs
        self.amount = amount
        self.drive = drive
        self.split_hz = split_hz
        self.enabled = True
        self._lp = SOSCascade(low_pass(fs, split_hz, 0.707), nch=1)
        self._hp = SOSCascade(high_pass(fs, split_hz, 0.707), nch=1)
        self._env = 0.0

    def process(self, out_l, out_r):
        if not self.enabled or self.amount <= 1e-3:
            return
        mono = (out_l + out_r) * 0.5
        low = self._lp.process_mono(mono)
        rms = float(np.sqrt(np.mean(low * low) + 1e-12))
        self._env += 0.3 * (rms - self._env)
        if self._env < 1e-4:
            return
        gate = self._env / (self._env + 0.15)
        d = self.drive
        # 미싱 펀더멘털은 2f·3f·4f... 가 다 있어야 귀가 기본음을 복원한다.
        # 구버전은 tanh(x) 와 x·|x| 를 썼는데 둘 다 홀함수라 2f·4f 가
        # 아예 안 생겼다(측정: 2배음 −111 dB). 홀수 배음만 얹으면 저음이
        # 굵어지는 게 아니라 그냥 찌그러진다. |x| 를 섞어 짝수 배음을 만든다
        # (DC 는 뒤의 하이패스가 제거).
        odd = np.tanh(d * low)
        even = np.abs(low) * d * 0.8
        harm = odd * 0.55 + even * 0.75
        harm = self._hp.process_mono(harm.astype(np.float32))
        # 입력 저역 크기에 맞춰 정규화. 안 하면 드라이브가 셀수록 보강음이
        # 원음보다 커져 저음이 터진다.
        hr = float(np.sqrt(np.mean(harm * harm) + 1e-12))
        if hr > 1e-9:
            harm *= np.float32(min(2.0, rms / hr))
        amt = np.float32(self.amount * gate)
        out_l += harm * amt
        out_r += harm * amt

    def reset(self):
        self._lp.reset()
        self._hp.reset()
        self._env = 0.0


class _Band:
    __slots__ = ("kind", "f", "q", "static_db", "thr_db", "ratio",
                 "cur_db", "_flt", "_det", "_fs", "_env")

    def __init__(self, fs, kind, f, q, static_db, thr_db, ratio):
        self._fs = fs
        self.kind = kind
        self.f = f
        self.q = q
        self.static_db = static_db
        self.thr_db = thr_db
        self.ratio = ratio
        self.cur_db = static_db
        self._env = 0.0
        self._flt = SOSCascade(self._design(static_db), nch=2)
        self._det = SOSCascade(np.vstack([high_pass(fs, f / 1.4, 1.0),
                                          low_pass(fs, f * 1.4, 1.0)]), nch=1)

    def _design(self, g):
        if self.kind == "peak":
            return peaking(self._fs, self.f, g, self.q)
        if self.kind == "low_shelf":
            return low_shelf(self._fs, self.f, g, self.q)
        return high_shelf(self._fs, self.f, g, self.q)

    def process(self, buf2, mono_ref):
        band = self._det.process_mono(mono_ref)
        rms = float(np.sqrt(np.mean(band * band) + 1e-12))
        self._env += 0.4 * (rms - self._env)
        over = 20.0 * math.log10(self._env + 1e-9) - self.thr_db
        target = self.static_db - max(0.0, over) * (1.0 - 1.0 / self.ratio)
        if abs(target - self.cur_db) > 0.5:          # 0.5 dB 넘게 움직일 때만
            self.cur_db = target                     # 계수 재설계 (캐시)
            self._flt.set_sos(self._design(target))
        return self._flt.process2(buf2)

    def reset(self):
        self._flt.reset()
        self._det.reset()
        self._env = 0.0


class DynamicEQ:
    """정적 밴드는 전부 하나의 SOS 캐스케이드로 합쳐 sosfilt 호출 1번으로
    끝낸다(작은 블록에선 scipy 호출당 파이썬 오버헤드가 지배적). 계수가
    실시간으로 변하는 동적 밴드만 별도 호출. 검출은 블록 레이트."""

    __slots__ = ("fs", "enabled", "_mono", "_static", "_dyn")

    def __init__(self, fs):
        self.fs = fs
        self.enabled = True
        self._mono = None
        self._static = SOSCascade(np.vstack([low_shelf(fs, 90, 1.5, 0.7),
                                             high_shelf(fs, 11000, 1.0, 0.7)]),
                                  nch=2)
        # v2.5: 저역 다이나믹 밴드 추가. 3.2 kHz 한 밴드만으로는 저역이
        # 소스 수가 늘 때 뭉치는 걸(베이스 버짐) 못 잡는다. 90 Hz 로우쉘프를
        # 온화한 2:1 비율로 걸어 저역이 튈 때만 살짝 눌러준다 — 상시 컷이
        # 아니라 필요할 때만 개입하는 마스터링 체인의 '글루' 컴프레션.
        self._dyn = [_Band(fs, "low_shelf", 90, 0.7, 0.0, -14.0, 2.0),
                     _Band(fs, "peak", 3200, 1.2, 0.0, -18.0, 3.0)]

    def process(self, mix):
        if not self.enabled:
            return
        buf = self._static.process2(mix)
        b = mix.shape[1]
        if self._mono is None or self._mono.shape[0] < b:
            self._mono = np.empty(b, dtype=np.float32)
        mono = self._mono[:b]
        np.add(buf[0], buf[1], out=mono)
        mono *= np.float32(0.5)
        for band in self._dyn:
            buf = band.process(buf, mono)
        mix[:] = buf

    def reset(self):
        self._static.reset()
        for b in self._dyn:
            b.reset()


class Loudness:
    """스트리밍 라우드니스 정규화 + 룩어헤드 트루피크 리미터.

    ─ 라우드니스
    EBU R128 통합 라우드니스는 전체 구간 게이팅이 필요해 실시간엔 못 쓴다.
    대신 BS.1770 K-웨이팅(38 Hz 하이패스 + 1.5 kHz +4 dB 하이셸프) 후 약
    400 ms 지수창 평균제곱으로 라우드니스를 추정하고, 슬루 제한된 AGC로
    타깃까지 천천히 붙인다. K-웨이팅은 검출 경로에만 걸리므로 출력 음색은
    변하지 않는다.

    ─ 왜 룩어헤드인가
    이득 곡선을 원폴로 부드럽게 만들면 어택이 늦어 트랜지언트가 그대로 새어
    나간다(실측: 실링 0.98 설정에서 출력 피크 1.36까지 넘어갔다). 해결책은
    순서를 뒤집는 것:

        1) 샘플별 필요 이득을 구한 뒤
        2) 슬라이딩 최솟값 필터를 먼저 걸어 '앞으로 올 피크'를 미리 반영하고
        3) 그 다음에 한 폴로 부드럽게 만든다(이제 릴리스만 매끄러워진다)
        4) 오디오를 룩어헤드 창의 절반만큼 지연시켜 이득과 시간을 맞춘다

    최솟값 필터가 어택을 담당하므로 오버슈트가 원리적으로 불가능하고, 스무딩이
    뒤에 오므로 이득 곡선이 계단지지 않아 왜곡도 없다. 비용은 O(B)."""

    LOOKAHEAD = 64                     # 샘플 (48 kHz에서 1.33 ms)

    __slots__ = ("fs", "target_lufs", "ceiling", "_kw", "_ms", "_gain_db",
                 "_a_ms", "max_slew_db", "gain_range", "enabled", "block",
                 "_la", "_hist_g", "_dl_l", "_dl_r", "_g_prev", "_buf",
                 "_lb", "_la_c", "_tp_taps", "_tp_up", "_tp_delay")

    def __init__(self, fs, block, target_lufs=-16.0, ceiling=0.97):
        self.fs = fs
        self.block = block
        self.target_lufs = target_lufs
        self.ceiling = ceiling
        self.enabled = True
        self._kw = SOSCascade(np.vstack([high_pass(fs, 38.0, 0.5),
                                         high_shelf(fs, 1500.0, 4.0, 0.707)]),
                              nch=2)
        self._ms = 1e-6
        self._gain_db = 0.0
        bps = fs / block
        self._a_ms = 1.0 - math.exp(-1.0 / (0.4 * bps))
        self.max_slew_db = 3.0 / bps
        self.gain_range = (-12.0, 12.0)
        la = self.LOOKAHEAD
        self._la = la
        self._hist_g = np.ones(la, dtype=np.float32)    # 이전 블록 이득 꼬리
        self._dl_l = DelayLine(la + block + 8)
        self._dl_r = DelayLine(la + block + 8)
        a = 0.25
        self._lb = np.array([a], dtype=np.float64)
        self._la_c = np.array([1.0, -(1.0 - a)], dtype=np.float64)
        self._g_prev = np.array([1.0 - a], dtype=np.float64)   # 초기 이득 1.0
        self._buf = np.empty((2, block), dtype=np.float32)
        # v2.5: 트루피크(인터샘플 피크) 근사 검출. 샘플 피크만 보면 재생 시
        # DAC 재구성 필터가 샘플 사이에서 실링을 넘는 오버슈트를 만들 수
        # 있다(로시 코덱을 거치면 더 심해짐). 4배 오버샘플로 샘플 사이
        # 피크까지 잡아 실링 판정에 반영한다. 필터는 1회만 설계해 재사용
        # 하므로(블록마다 재설계 X) 오디오 스레드 비용은 upfirdn 호출뿐이다.
        up = 4
        self._tp_up = up
        taps = firwin(63, cutoff=1.0 / up, window=("kaiser", 8.6)) * up
        self._tp_taps = taps.astype(np.float32)
        self._tp_delay = (len(taps) - 1) // 2

    def process(self, mix):
        if not self.enabled:
            return
        b = mix.shape[1]
        # ---- 라우드니스 AGC ----
        det = self._kw.process2(mix)
        msq = float(np.mean(det * det) + 1e-12)
        self._ms += self._a_ms * (msq - self._ms)
        lufs = -0.691 + 10.0 * math.log10(self._ms + 1e-12)
        err = self.target_lufs - lufs
        step = max(-self.max_slew_db, min(self.max_slew_db, err))
        g = max(self.gain_range[0], min(self.gain_range[1],
                                        self._gain_db + step))
        self._gain_db = float(g)
        mix *= np.float32(10.0 ** (g / 20.0))

        # ---- 룩어헤드 리미터 (트루피크 검출) ----
        la = self._la
        sample_peak = np.maximum(np.abs(mix[0]), np.abs(mix[1]))
        # v2.6: 4배 오버샘플은 비싸다(실측상 실시간 예산의 무시 못 할 몫).
        # 그런데 인터샘플 피크가 샘플 피크를 넘는 폭은 물리적으로 유한하고,
        # 실제 음악에서 +3 dB(×1.41)를 넘는 경우는 사실상 없다. 블록의 샘플
        # 피크가 실링보다 그 배수만큼 아래면 오버샘플을 해봐야 결과가 바뀌지
        # 않으므로 통째로 건너뛴다. 일반적인 음량의 음악은 대부분 이쪽으로
        # 빠져 오버샘플 비용이 0이 되고, 실제로 실링에 붙는 순간에만 정밀
        # 검출이 돌아간다. 판정을 보수적으로(1.41) 잡아 안전 마진을 남긴다.
        sp_max = float(sample_peak.max())
        if sp_max * 1.41 < self.ceiling:
            peak = sample_peak
        else:
            up = self._tp_up
            ov0 = upfirdn(self._tp_taps, mix[0], up=up, down=1)
            ov1 = upfirdn(self._tp_taps, mix[1], up=up, down=1)
            d0 = self._tp_delay
            ov0 = ov0[d0:d0 + b * up]
            ov1 = ov1[d0:d0 + b * up]
            true_peak = np.maximum(np.abs(ov0),
                                   np.abs(ov1)).reshape(b, up).max(axis=1)
            peak = np.maximum(sample_peak, true_peak[:b])
        desired = np.minimum(1.0, self.ceiling / (peak + 1e-9))
        # 이전 블록 꼬리를 앞에 붙여 블록 경계에서도 창이 끊기지 않게
        ext = np.concatenate((self._hist_g, desired))
        self._hist_g = desired[-la:].astype(np.float32) if b >= la else \
            np.concatenate((self._hist_g[b:], desired)).astype(np.float32)
        gmin = minimum_filter1d(ext, size=la * 2 + 1, mode="nearest")[la:la + b]
        # 이제 스무딩은 한 폴 하나면 된다. 최솟값 필터가 이미 '앞으로 올 피크'를
        # 반영해 이득을 미리 내려놨으므로, 한 폴이 느려도 오버슈트가 없다
        # (한 폴 정착 ~9샘플 ≪ 룩어헤드 64샘플). 스칼라 루프 없이 lfilter 한 번.
        out, self._g_prev = lfilter(self._lb, self._la_c, gmin, zi=self._g_prev)
        out = out.astype(np.float32)

        # 오디오를 룩어헤드만큼 지연시켜 이득과 정렬
        self._dl_l.write_block(mix[0])
        self._dl_r.write_block(mix[1])
        buf = self._buf[:, :b]
        self._dl_l.read_delayed_block(la, b, buf[0])
        self._dl_r.read_delayed_block(la, b, buf[1])
        buf *= out
        mix[:] = buf

    @property
    def gain_db(self):
        return self._gain_db

    def reset(self):
        self._kw.reset()
        self._ms = 1e-6
        self._gain_db = 0.0
        self._hist_g.fill(1.0)
        self._dl_l.reset()
        self._dl_r.reset()
        self._g_prev[:] = 1.0 - 0.25


# ═══════════════════════════════════════════════════════════════════════════
#  6. 소스 / 청취자 / 엔진
# ═══════════════════════════════════════════════════════════════════════════
class Source:
    """모노 소스 = 오디오 공급원 + 공간 상태.
    오디오는 루프 버퍼 또는 pull 콜백(라이브 신스/디코더)에서 온다.
    pull(n)은 미리 잡아둔 스크래치에 채우므로 호출당 할당이 없다."""

    __slots__ = ("pos", "vel", "gain", "cone_inner", "cone_outer",
                 "cone_outer_gain", "forward", "active",
                 "_buf", "_read", "_cb", "_scratch")

    def __init__(self, block, buffer=None, callback=None, gain=1.0):
        self.pos = np.zeros(3, dtype=np.float64)
        self.vel = np.zeros(3, dtype=np.float64)
        self.gain = gain
        self.forward = np.array([0.0, 0.0, 1.0])
        self.cone_inner = math.pi
        self.cone_outer = math.pi
        self.cone_outer_gain = 0.25
        self.active = True
        self._cb = callback
        self._read = 0
        self._scratch = np.zeros(block, dtype=np.float32)
        self._buf = (np.ascontiguousarray(buffer, dtype=np.float32)
                     if buffer is not None else None)

    def pull(self, n):
        out = self._scratch[:n]
        if self._cb is not None:
            self._cb(out)
            return out
        if self._buf is None:
            out.fill(0.0)
            return out
        buf = self._buf
        L = buf.shape[0]
        r = self._read
        if r + n <= L:
            out[:] = buf[r:r + n]
            self._read = (r + n) % L
        else:
            first = L - r
            out[:first] = buf[r:]
            out[first:] = buf[:n - first]
            self._read = n - first
        return out

    def set_buffer(self, buffer):
        self._buf = np.ascontiguousarray(buffer, dtype=np.float32)
        self._read = 0

    def directivity_gain(self, to_listener_dir):
        """소스 콘 감쇠. O(1)."""
        if self.cone_inner >= math.pi:
            return 1.0
        f = self.forward
        c = float(np.dot(f, to_listener_dir) / (np.linalg.norm(f) + 1e-12))
        ang = math.acos(max(-1.0, min(1.0, c)))
        if ang <= self.cone_inner:
            return 1.0
        if ang >= self.cone_outer:
            return self.cone_outer_gain
        t = (ang - self.cone_inner) / (self.cone_outer - self.cone_inner + 1e-9)
        return (1.0 - t) + t * self.cone_outer_gain


class Listener:
    """청취자 자세 + 캐시된 머리 좌표계. 방향을 바꿀 때만 정규직교 기저를
    다시 만들고, 블록마다 하는 일은 내적 몇 번뿐이다."""

    __slots__ = ("pos", "vel", "_fwd", "_up", "_right")

    def __init__(self):
        self.pos = np.zeros(3, dtype=np.float64)
        self.vel = np.zeros(3, dtype=np.float64)
        self._fwd = np.array([0.0, 0.0, 1.0])
        self._up = np.array([0.0, 1.0, 0.0])
        self._right = np.array([1.0, 0.0, 0.0])

    def set_pose(self, pos=None, forward=None, up=None, vel=None):
        if pos is not None:
            self.pos[:] = pos
        if vel is not None:
            self.vel[:] = vel
        if forward is not None or up is not None:
            f = np.asarray(forward if forward is not None else self._fwd, float)
            u = np.asarray(up if up is not None else self._up, float)
            f = f / (np.linalg.norm(f) + 1e-12)
            # 좌표계는 X=오른쪽, Y=위, Z=정면(왼손계). 이 규약에서 오른쪽
            # 벡터는 cross(up, forward)다. cross(forward, up)로 쓰면 부호가
            # 뒤집혀 화면 오른쪽의 소리가 왼쪽 귀에서 들린다.
            r = np.cross(u, f)
            r = r / (np.linalg.norm(r) + 1e-12)
            self._fwd, self._right, self._up = f, r, np.cross(f, r)

    @property
    def basis(self):
        return self._right, self._up, self._fwd


class _Voice:
    __slots__ = ("src", "spat", "dop", "conv", "az", "el", "dist", "vrad",
                 "dgain", "scratch")

    def __init__(self, src, fs, block):
        self.src = src
        self.spat = SourceVoice(fs, block)
        self.dop = DopplerVoice(fs, block)
        self.conv = None
        self.az = 0.0
        self.el = 0.0
        self.dist = 1.0
        self.vrad = 0.0
        self.dgain = 1.0
        self.scratch = np.zeros(block, dtype=np.float32)


class SpatialAudioEngine:
    """블록당 파이프라인 (B 샘플, 스테레오 출력):

        소스마다: pull → [도플러] → HRTF(컨볼루션 또는 파라메트릭) → 믹스
        마스터  : 초기반사 + 리버브(공유) → 스테레오 확장 → 스마트 베이스
                  → 다이나믹 EQ → 라우드니스/리미터 → (B,2) 패킹

    CPU 오토스케일러: 블록 처리시간/블록 길이의 EMA를 부하로 보고, 히스테리시스
    임계를 넘으면 비싼 단계를 우선순위대로 끈다. 하드웨어가 무엇이든 스스로
    맞춰 들어가므로 드롭아웃이 나지 않는다.
        낮추는 순서 : 도플러 → 초기반사 → 리버브 → 다이나믹 EQ → 컨볼루션 HRTF
        되돌리는 순서: 역순"""

    def __init__(self, fs=FS_DEFAULT, block=BLOCK_DEFAULT, max_sources=64,
                 quality="high"):
        self.fs = fs
        self.block = block
        self.max_sources = max_sources
        self.listener = Listener()
        self._voices = {}
        self._next_id = 0
        self._lock = threading.Lock()

        self.reverb = ReverbBus(fs, block)
        self.widener = StereoWidener(fs)
        self.bass = BassEnhancer(fs)
        self.eq = DynamicEQ(fs)
        self.loudness = Loudness(fs, block)

        self._mix = np.zeros((2, block), dtype=np.float32)
        self._out = np.zeros((block, 2), dtype=np.float32)

        # v2.6: 목표 CPU 점유율. 오토스케일러가 이 값을 넘지 않도록 기능을
        # 스스로 끄고 켠다. 기계마다 성능이 다르므로 고정된 절대 시간이
        # 아니라 비율로 잡아야 어느 PC에서도 같은 약속이 지켜진다.
        self.cpu_budget = CPU_BUDGET

        if _HAS_NUMBA:
            threading.Thread(target=_warm_numba, daemon=True).start()

        # HRIR 뱅크는 무거우니 백그라운드에서 만들고, 준비되면 자동으로 승격
        self.bank = None
        self.f_conv = False
        self.want_conv = (quality == "high")
        if self.want_conv:
            threading.Thread(target=self._build_bank, daemon=True).start()

        self.f_doppler = True
        self.f_reverb = True
        self.f_eq = True

        self._load = 0.0
        self._block_sec = block / fs
        self._hi = 0
        self._lo = 0
        self.autoscale = True
        self._stream = None

    def _build_bank(self):
        t0 = time.perf_counter()
        bank = HRIRBank(self.fs, self.block)
        with self._lock:
            self.bank = bank
            self.f_conv = self.want_conv
        self.bank_build_ms = (time.perf_counter() - t0) * 1e3

    # ---------------- 소스 관리 ----------------
    def add_source(self, source):
        with self._lock:
            sid = self._next_id
            self._next_id += 1
            self._voices[sid] = _Voice(source, self.fs, self.block)
            return sid

    def remove_source(self, sid):
        with self._lock:
            self._voices.pop(sid, None)

    def set_listener(self, **kw):
        with self._lock:
            self.listener.set_pose(**kw)

    # ---------------- 기하 스냅샷 (락 안, 마이크로초 단위) ----------------
    def _snapshot(self):
        right, up, fwd = self.listener.basis
        lpos = self.listener.pos
        lvel = self.listener.vel
        for v in self._voices.values():
            s = v.src
            if not s.active:
                v.dgain = 0.0
                continue
            rel = s.pos - lpos
            dist = float(np.linalg.norm(rel)) + 1e-9
            inv = 1.0 / dist
            xr = float(np.dot(rel, right))
            zf = float(np.dot(rel, fwd))
            yu = float(np.dot(rel, up))
            v.az = math.atan2(xr, zf)
            v.el = math.asin(max(-1.0, min(1.0, yu * inv)))
            v.dist = dist
            v.vrad = float(np.dot(s.vel - lvel, rel) * inv)
            v.dgain = s.directivity_gain(-rel * inv) * s.gain

    # ---------------- 블록 처리 ----------------
    def process_block(self):
        t0 = time.perf_counter()
        b = self.block
        mix = self._mix
        mix.fill(0.0)
        l = mix[0]
        r = mix[1]

        with self._lock:
            self._snapshot()
            voices = list(self._voices.values())
            bank = self.bank
            use_conv = self.f_conv and bank is not None

        dop_on = self.f_doppler
        for v in voices:
            if v.dgain <= 1e-6:
                continue
            mono = v.src.pull(b)
            if dop_on:
                v.dop.process(mono, v.dist, v.scratch)   # 전파지연 = 도플러
                mono = v.scratch
            if use_conv:
                if v.conv is None:
                    v.conv = ConvVoice(bank, b)
                d = v.dist if v.dist > 0.3 else 0.3
                v.conv.process(mono, v.az, v.el, v.dgain / d, l, r)
            else:
                v.spat.render(mono, v.az, v.el, v.dist, v.dgain, l, r)

        if self.f_reverb:
            self.reverb.process(l, r)
        self.widener.process(l, r)
        self.bass.process(l, r)
        if self.f_eq:
            self.eq.process(mix)
        self.loudness.process(mix)

        out = self._out
        out[:, 0] = l
        out[:, 1] = r

        dt = time.perf_counter() - t0
        self._load += 0.1 * (dt / self._block_sec - self._load)
        if self.autoscale:
            self._scale()
        return out

    def _scale(self):
        """v2.6: 임계값을 하드코딩(0.55/0.30)하지 않고 cpu_budget 기준으로
        잡는다. 사람마다 CPU가 다르므로 '몇 마이크로초'가 아니라 '그 기계에서
        실제로 몇 %를 먹는가'가 유일하게 의미 있는 기준이다."""
        budget = self.cpu_budget
        load = self._load
        if load > budget:
            self._hi += 1
            self._lo = 0
        elif load < budget * 0.55:
            self._lo += 1
            self._hi = 0
        else:
            self._hi = self._lo = 0
        if self._hi >= 8:
            self._hi = 0
            if self.f_doppler:
                self.f_doppler = False
            elif self.reverb.early.enabled:
                self.reverb.early.enabled = False
            elif self.f_reverb:
                self.f_reverb = False
            elif self.f_eq:
                self.f_eq = False
            elif self.f_conv:
                self.f_conv = False          # 최후의 수단 (실측상 이득이 작다)
        elif self._lo >= 20:
            self._lo = 0
            if not self.f_conv and self.want_conv and self.bank is not None:
                self.f_conv = True
            elif not self.f_eq:
                self.f_eq = True
            elif not self.f_reverb:
                self.f_reverb = True
            elif not self.reverb.early.enabled:
                self.reverb.early.enabled = True
            elif not self.f_doppler:
                self.f_doppler = True

    @property
    def cpu_load(self):
        """블록 처리시간 / 실시간 EMA. ×100이 1코어 점유율(%)."""
        return self._load

    @property
    def quality_label(self):
        if self.f_conv:
            return "컨볼루션 HRTF"
        return "고속 HRTF" if self.bank is None else "고속 HRTF (절전)"

    def reset(self):
        for v in self._voices.values():
            v.spat.reset()
            v.dop.reset()
            if v.conv is not None:
                v.conv.reset()
        self.reverb.reset()
        self.widener.reset()
        self.bass.reset()
        self.eq.reset()
        self.loudness.reset()


# ═══════════════════════════════════════════════════════════════════════════
#  7. 테스트 신호 / WAV 입출력
# ═══════════════════════════════════════════════════════════════════════════
_PINK_B = np.array([0.049922035, -0.095993537, 0.050612699, -0.004408786])
_PINK_A = np.array([1.0, -2.494956002, 2.017265875, -0.522189400])


def _loop_fade(x, fs, ms=60.0):
    """루프 경계 클릭 제거. 끝 ms 구간을 앞머리와 크로스페이드하고 잘라낸다."""
    n = int(fs * ms / 1000.0)
    if n * 2 >= x.shape[0]:
        return x
    f = np.linspace(0.0, 1.0, n, dtype=np.float32)
    y = x[:-n].copy()
    y[:n] = x[:n] * f + x[-n:] * (1.0 - f)
    return y


def _peak_norm(x, peak=0.8):
    """데모 파형 정규화. 위쪽 바이쿼드용 _norm 과 이름이 겹치면 안 된다."""
    m = float(np.max(np.abs(x))) + 1e-12
    return (x / m * peak).astype(np.float32)


def _vehicle(fs, f0=46.0, dur=3.4):
    """차량 통과음. 톤 성분은 기본파와 2배음(46·92 Hz)까지만 남기고 나머지는
    전부 노이즈로 만든다.

    ─ 왜 배음을 잘라내는가
    처음엔 16배음까지 가산합성했는데, 58 Hz의 3·4배음이 174·232 Hz —
    정확히 곤충 날갯짓 대역이었다. 좁은 대역에 강한 톤이 계속 울리면
    사람 귀는 그걸 벌레로 듣는다(측정 순음성 34.6 dB). 46 Hz로 낮추고
    2배음까지만 남기면 톤 성분이 92 Hz 이하로 내려가 그 대역이 비고,
    나머지 질감은 노면 소음(광대역 노이즈)이 담당한다."""
    n = int(fs * dur)
    t = np.arange(n, dtype=np.float64) / fs
    rng = np.random.default_rng(11)
    tone = (np.sin(2 * np.pi * f0 * t) +
            0.45 * np.sin(2 * np.pi * 2 * f0 * t + 1.1))
    w = rng.standard_normal(n)
    rumble = sosfilt(np.vstack([low_pass(fs, 130.0), low_pass(fs, 130.0)]), w)
    body = sosfilt(np.vstack([high_pass(fs, 220.0), low_pass(fs, 900.0)]),
                   rng.standard_normal(n))
    tire = sosfilt(np.vstack([high_pass(fs, 900.0), low_pass(fs, 6000.0)]),
                   rng.standard_normal(n))
    y = tone * 0.30 + rumble * 2.2 + body * 0.9 + tire * 0.55
    y *= 1.0 + 0.10 * np.sin(2 * np.pi * 2.3 * t) + 0.07 * np.sin(2 * np.pi * 3.7 * t)
    return _loop_fade(_peak_norm(y, 0.85), fs)


def _probe(fs, dur=4.0):
    """경로 확인용 중립 신호.

    음정이 있는 데모음(종·마림바 따위)은 전부 걷어냈다. 이유는 두 가지다.
    ① 이 프로그램의 본체는 시스템 소리를 3D로 바꾸는 것이지 소리를 만드는
       게 아니다. 켜자마자 자기가 만든 소리가 울리면 앞뒤가 안 맞는다.
    ② 정위 확인에는 광대역 잡음이 객관적으로 가장 정확하다. 귓바퀴 노치와
       머리 그림자 단서가 전 대역에 걸쳐 있어서, 배음이 듬성듬성한 악기음은
       그 단서 대부분을 비워 둔 채로 방향을 판단하게 만든다.
    즉 이건 '데모'가 아니라 오디오 경로가 살아 있는지 보는 계측 신호다."""
    n = int(fs * dur)
    rng = np.random.default_rng(5)
    w = rng.standard_normal(n)
    y = lfilter(_PINK_B, _PINK_A, w)
    t = np.arange(n, dtype=np.float64) / fs
    y *= 1.0 + 0.35 * np.sin(2 * np.pi * 0.31 * t)
    return _loop_fade(_peak_norm(y, 0.7), fs)


_DEMO_CACHE = {}


def demo_wave(kind, fs, freq):
    """데모 파형을 한 번만 합성해 캐시한다. 오디오 스레드는 읽기만 한다."""
    key = (kind, fs, round(float(freq), 2))
    buf = _DEMO_CACHE.get(key)
    if buf is None:
        buf = _vehicle(fs) if kind == "vehicle" else _probe(fs)
        _DEMO_CACHE[key] = buf
    return buf


class Gen:
    """데모용 신호 생성기. 전부 미리 잡아둔 블록에 in-place로 채운다.

    probe / vehicle 은 초기화 때 한 번 합성한 파형을 순환
    재생한다(블록당 비용 = memcpy 1회). 실시간 오실레이터를 돌리지 않으므로
    대역제한이 보장되고 앨리어싱이 원천적으로 없다."""

    __slots__ = ("fs", "kind", "freq", "level", "_ph", "_t", "_rng", "_zi",
                 "_buf", "_read", "_lfo", "_s", "loop", "done")

    SYNTH = ("probe", "vehicle")

    def __init__(self, fs, block, kind="probe", freq=200.0, level=0.35, seed=0):
        self.fs = fs
        self.kind = kind
        self.freq = freq
        self.level = level
        self._ph = 0.0
        self._lfo = 0.0
        self._t = np.arange(block, dtype=np.float64)
        self._rng = np.random.default_rng(seed)
        self._zi = np.zeros(3)
        self._s = np.zeros(block, dtype=np.float64)
        self._read = 0
        self.loop = True
        self.done = False
        self._buf = demo_wave(kind, fs, freq) if kind in self.SYNTH else None

    def set_file(self, samples, loop=False):
        """음악 재생. loop=False면 끝까지 가고 멈춘다(구버전은 무한 반복)."""
        self._buf = np.ascontiguousarray(samples, dtype=np.float32)
        self._read = 0
        self.kind = "file"
        self.loop = loop
        self.done = False

    @property
    def length_sec(self):
        return 0.0 if self._buf is None else self._buf.shape[0] / self.fs

    @property
    def pos_sec(self):
        return 0.0 if self._buf is None else self._read / self.fs

    def _play_buf(self, out, gain):
        n = out.shape[0]
        b = self._buf
        L = b.shape[0]
        r = self._read
        if r >= L:
            out.fill(0.0)
            self.done = True
            return
        if r + n <= L:
            out[:] = b[r:r + n]
            r += n
            if r == L and self.loop:
                r = 0
        elif self.loop:
            first = L - r
            out[:first] = b[r:]
            out[first:] = b[:n - first]
            r = n - first
        else:
            first = L - r
            out[:first] = b[r:]
            out[first:] = 0.0
            r = L
            self.done = True
        self._read = r
        out *= gain

    def __call__(self, out):
        n = out.shape[0]
        k = self.kind
        fs = self.fs
        if k == "file" and self._buf is not None:
            self._play_buf(out, self.level * 2.4)
        elif self._buf is not None:                   # 합성 데모 파형
            self._play_buf(out, self.level * 2.0)
        elif k == "sine":
            inc = 2 * math.pi * self.freq / fs
            np.sin(self._ph + inc * self._t[:n], out=self._s[:n])
            out[:] = self._s[:n] * self.level
            self._ph = (self._ph + inc * n) % (2 * math.pi)
        elif k == "pink":
            w = self._rng.standard_normal(n)
            y, self._zi = lfilter(_PINK_B, _PINK_A, w, zi=self._zi)
            out[:] = (y * self.level * 3.0).astype(np.float32)
        else:
            out.fill(0.0)


def load_audio(path, fs):
    """WAV를 모노 다운믹스 + 선형 리샘플로 읽는다(표준 라이브러리만 사용)."""
    with wave.open(path, "rb") as w:
        nch, sw, sr, nfr = (w.getnchannels(), w.getsampwidth(),
                            w.getframerate(), w.getnframes())
        raw = w.readframes(nfr)
    if sw == 2:
        a = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif sw == 4:
        a = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    elif sw == 1:
        a = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128) / 128.0
    else:
        raise ValueError("지원하지 않는 비트 심도")
    if nch > 1:
        a = a.reshape(-1, nch).mean(axis=1)
    if sr != fs:
        a = np.interp(np.linspace(0, len(a) - 1, int(len(a) * fs / sr)),
                      np.arange(len(a)), a).astype(np.float32)
    return (a / (float(np.max(np.abs(a))) + 1e-9) * 0.7).astype(np.float32)


def save_wav(path, pcm, fs):
    x = np.clip(pcm, -1.0, 1.0)
    with wave.open(path, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(fs)
        w.writeframes((x * 32767.0).astype("<i2").tobytes())


# ═══════════════════════════════════════════════════════════════════════════
#  8. 플레이어 — 엔진 소유, 출력 스트림 또는 시뮬레이션 클럭, 레벨 측정
# ═══════════════════════════════════════════════════════════════════════════
class Voice3D:
    """GUI가 다루는 소스 하나: 좌표 + 움직임 + 신호원."""

    __slots__ = ("sid", "src", "gen", "color", "orbit", "radius", "speed",
                 "height", "angle", "flyby")

    def __init__(self, sid, src, gen, color):
        self.sid = sid
        self.src = src
        self.gen = gen
        self.color = color
        self.orbit = True
        self.flyby = False
        self.radius = 3.0
        self.speed = 0.5
        self.height = 0.0
        self.angle = 0.0

    def step(self, dt):
        if self.flyby:
            x = self.src.pos[0] + self.speed * dt
            if x > 22.0:
                x = -22.0
            self.src.pos[0] = x
            self.src.pos[1] = self.height
            self.src.pos[2] = 3.0
            self.src.vel[0] = self.speed
            self.src.vel[1] = self.src.vel[2] = 0.0
        elif self.orbit:
            self.angle += self.speed * dt
            r, a = self.radius, self.angle
            ca, sa = math.cos(a), math.sin(a)
            self.src.pos[0] = r * sa
            self.src.pos[1] = self.height
            self.src.pos[2] = r * ca
            v = r * self.speed
            self.src.vel[0] = v * ca
            self.src.vel[1] = 0.0
            self.src.vel[2] = -v * sa
        else:
            self.src.vel[:] = 0.0


_PALETTE = ("#0e8f86", "#6247ea", "#d99a00", "#e0387a",
            "#2e9e5b", "#2f7fd1", "#e07a1f", "#a855f7")


def default_output_fs(fallback=FS_DEFAULT, device=None):
    """장치가 지원하지 않는 샘플레이트로 열면 스트림이 아예 안 열리거나
    OS가 강제 리샘플링한다. 엔진을 장치 기본값에 맞춰 만드는 게 가장 안전하다."""
    try:
        import sounddevice as sd
        d = sd.query_devices(device if device is not None else None,
                             kind="output")
        return int(d["default_samplerate"])
    except Exception:
        return fallback


class Player:
    """오디오 스레드와 GUI 사이의 유일한 접점.
    GUI는 Monitor의 스칼라 몇 개만 읽고, 오디오 콜백은 락을 잡지 않는다."""

    TAIL_SEC = 2.6          # 잔향 꼬리를 끝까지 흘려보낼 시간
    FADE_SEC = 1.1          # 마지막 페이드아웃

    def __init__(self, fs=FS_DEFAULT, block=BLOCK_DEFAULT, device=None):
        self.fs = fs
        self.block = block
        self.device = device
        self.finished = False
        self._tail = -1
        self._tail_n = 1
        self._fade_prev = 1.0
        self._ramp = np.linspace(0.0, 1.0, block,
                                 dtype=np.float32).reshape(-1, 1)
        self.eng = SpatialAudioEngine(fs=fs, block=block, quality="high")
        self.eng.set_listener(pos=[0, 0, 0], forward=[0, 0, 1], up=[0, 1, 0])
        self.voices = []
        self.mode = None
        self.latency = 0.0
        self.error = ""
        self.master_gain = 1.0
        self.peak = 0.0
        self.blocks = 0
        self._n = 0
        self._stream = None
        self._sim = None
        self._run = False
        self._left = None
        self.device_name = ""
        self.opened_with = ""


    # ---- 소스 ----
    def add(self, kind="probe", freq=200.0, radius=3.0, angle=0.0, gain=0.8):
        gen = Gen(self.fs, self.block, kind=kind, freq=freq, seed=self._n + 1)
        src = Source(block=self.block, callback=gen, gain=gain)
        sid = self.eng.add_source(src)
        v = Voice3D(sid, src, gen, _PALETTE[self._n % len(_PALETTE)])
        v.radius = radius
        v.angle = angle
        self._n += 1
        v.step(0.0)
        self.voices.append(v)
        return v

    def clear(self):
        for v in self.voices:
            self.eng.remove_source(v.sid)
        self.voices.clear()
        self._n = 0

    def step(self, dt):
        for v in self.voices:
            v.step(dt)

    # ---- 렌더 + 레벨 측정 ----
    def _music_done(self):
        """파일 소스가 하나라도 있고 전부 끝났는가."""
        any_file = False
        for v in self.voices:
            g = v.gen
            if g.kind == "file":
                any_file = True
                if not g.done:
                    return False
        return any_file

    def _render(self):
        out = self.eng.process_block()
        if self.master_gain != 1.0:
            out *= self.master_gain

        # ── 종결 처리: 음악이 끝나면 잔향 꼬리를 끝까지 흘린 뒤 페이드아웃
        if self._tail < 0:
            if self._music_done():
                self._tail_n = max(1, int(self.TAIL_SEC * self.fs / self.block))
                self._tail = self._tail_n
                self._fade_prev = 1.0
        elif self._tail > 0:
            self._tail -= 1
            nf = max(1, int(self.FADE_SEC * self.fs / self.block))
            g1 = 1.0 if self._tail > nf else self._tail / nf
            g0 = self._fade_prev
            self._fade_prev = g1
            if g0 != 1.0 or g1 != 1.0:
                # 블록당 계단 게인은 블록레이트(172~188 Hz) 잡음이 된다.
                # 샘플 단위 램프로 이어 붙인다.
                out *= np.float32(g0) + np.float32(g1 - g0) * self._ramp
            if self._tail == 0:
                self.finished = True
        else:
            out[:] = 0.0

        p = float(np.max(np.abs(out)))
        self.peak = p if p > self.peak else self.peak * 0.85 + p * 0.15
        self.blocks += 1
        return out

    def rewind(self):
        """처음부터 다시. 재생이 끝난 뒤 다시 누르면 이게 먼저 돈다."""
        for v in self.voices:
            g = v.gen
            g._read = 0
            g.done = False
        self.eng.reset()
        self.finished = False
        self._tail = -1
        self._fade_prev = 1.0
        self.peak = 0.0

    @property
    def music_length(self):
        return max((v.gen.length_sec for v in self.voices
                    if v.gen.kind == "file"), default=0.0)

    @property
    def music_pos(self):
        return max((v.gen.pos_sec for v in self.voices
                    if v.gen.kind == "file"), default=0.0)

    # ---- 트랜스포트 ----
    def start(self):
        """스트림을 연다. 구버전은 실패하면 조용히 sim 모드로 빠져서 화면은
        정상인데 소리만 안 나고, 윈도우 볼륨 믹서에도 앱이 안 뜬다(스트림을
        연 앱만 표시되기 때문). 이제 설정을 단계적으로 낮춰가며 재시도하고,
        전부 실패하면 이유를 그대로 남긴다."""
        self.error = ""
        self.device_name = ""
        self.opened_with = ""
        self._left = None
        try:
            import sounddevice as sd
        except Exception as ex:
            self.error = f"sounddevice 미설치 ({ex.__class__.__name__}) — pip install sounddevice"
            self._start_sim()
            return

        try:
            info = sd.query_devices(self.device, kind="output")
            self.device_name = str(info["name"])
            dev_fs = int(info["default_samplerate"])
        except Exception as ex:
            self.error = f"출력 장치를 찾을 수 없음: {ex}"
            self._start_sim()
            return

        blk = self.block
        # 위에서부터 시도. 아래로 갈수록 요구가 느슨하다.
        cfgs = [
            ("저지연 " + str(blk), dict(blocksize=blk, latency="low")),
            ("기본지연 " + str(blk), dict(blocksize=blk)),
            ("기본지연 " + str(blk * 4), dict(blocksize=blk * 4)),
            ("장치 자동", dict(blocksize=0)),
        ]
        errs = []
        for label, kw in cfgs:
            try:
                st = sd.OutputStream(samplerate=self.fs, channels=2,
                                     dtype="float32", callback=self._callback,
                                     device=self.device, **kw)
                st.start()
            except Exception as ex:
                errs.append(f"{label}: {str(ex).splitlines()[0][:60]}")
                continue
            self._stream = st
            lat = st.latency
            self.latency = float(lat[1] if isinstance(lat, (tuple, list)) else lat)
            self.mode = "audio"
            self.opened_with = label
            if self.fs != dev_fs:
                self.error = (f"장치 기본 {dev_fs} Hz인데 엔진은 {self.fs} Hz로 "
                              f"열렸습니다(리샘플링됨)")
            return
        self.error = "장치 열기 실패 · " + " / ".join(errs[:2])
        self._start_sim()

    def _callback(self, outdata, frames, t, status):
        """프레임 수가 블록과 달라도 절대 무음이 되지 않게 한다.
        구버전은 frames != block 이면 0을 출력하고 리턴해서, 호스트가 다른
        크기를 요구하는 순간 영구 무음이 됐다."""
        try:
            pos = 0
            left = self._left
            if left is not None:
                k = left.shape[0]
                if k > frames:
                    outdata[:frames] = left[:frames]
                    self._left = left[frames:].copy()
                    return
                outdata[:k] = left
                self._left = None
                pos = k
            while pos < frames:
                blk = self._render()
                k = blk.shape[0]
                room = frames - pos
                if k > room:
                    outdata[pos:frames] = blk[:room]
                    self._left = blk[room:].copy()
                    return
                outdata[pos:pos + k] = blk
                pos += k
        except Exception as ex:                      # 콜백 예외 = 스트림 중단
            outdata[:] = 0.0
            self.error = f"콜백 오류: {ex}"

    def _start_sim(self):
        self._run = True
        self._sim = threading.Thread(target=self._loop, daemon=True)
        self._sim.start()
        self.latency = self.block / self.fs
        self.mode = "sim"

    def _loop(self):
        period = self.block / self.fs
        nxt = time.perf_counter()
        while self._run:
            self._render()
            nxt += period
            d = nxt - time.perf_counter()
            if d > 0:
                time.sleep(d)
            else:
                nxt = time.perf_counter()

    def stop(self):
        self._run = False
        self._left = None
        if self._sim is not None:
            self._sim.join(timeout=0.5)
            self._sim = None
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None
        self.mode = None

    @property
    def running(self):
        return self.mode is not None

    # ---- 오프라인 렌더 ----
    def render_to(self, path, seconds=None):
        was = self.mode
        self.stop()
        if seconds is None:
            ml = self.music_length
            seconds = (ml + self.TAIL_SEC) if ml > 0 else 12.0
        self.rewind()
        nb = int(seconds * self.fs / self.block)
        buf = np.empty((nb * self.block, 2), dtype=np.float32)
        dt = self.block / self.fs
        t0 = time.perf_counter()
        for i in range(nb):
            self.step(dt)
            buf[i * self.block:(i + 1) * self.block] = self._render()
        el = time.perf_counter() - t0
        save_wav(path, buf, self.fs)
        self.rewind()
        if was:
            self.start()
        return el, seconds / el


# ═══════════════════════════════════════════════════════════════════════════
#  9. GUI — 조작은 최소, 나머지는 전부 자동
#     화면에 있는 것: 재생 버튼 1개 · 프리셋 4개 · 슬라이더 3개 · 무대 1개.
#     HRTF 품질, 도플러, EQ, 라우드니스, 오토스케일은 사용자가 만질 필요가
#     없도록 엔진이 알아서 켜고 끈다.
# ═══════════════════════════════════════════════════════════════════════════
BG = "#f6f7fb"          # v2.5: 다크 → 화이트 톤. 완전 순백 대신 살짝 눌러 눈부심 방지
CARD = "#ffffff"
CARD_HI = "#eef1f7"
LINE = "#dde3ec"
TXT = "#151a24"
DIM = "#6b7688"
CYAN = "#0e8f86"        # 기존 민트(#31e0c8)는 흰 배경에서 대비가 부족해 진한 틸로 조정
VIOLET = "#6247ea"       # 기존 라벤더보다 채도·명도를 낮춰 흰 배경에서도 또렷하게
RED = "#e0384a"


def _rrect(cv, x0, y0, x1, y1, r=12, **kw):
    return cv.create_polygon(
        [x0 + r, y0, x1 - r, y0, x1, y0, x1, y0 + r, x1, y1 - r, x1, y1,
         x1 - r, y1, x0 + r, y1, x0, y1, x0, y1 - r, x0, y0 + r, x0, y0],
        smooth=True, splinesteps=10, **kw)


class BigButton:
    """큰 눌림 영역 + 호버. 캔버스 아이템 2개로 끝나므로 다시 그릴 게 없다."""

    def __init__(self, parent, x, y, w, h, text, command, fill=CYAN,
                 fg="#08131a", font=None, sub=None):
        self.cv = parent
        self.command = command
        self.fill = fill
        self.rect = _rrect(parent, x, y, x + w, y + h, r=14, fill=fill,
                           outline="")
        ty = y + h / 2 if sub is None else y + h / 2 - 9
        self.txt = parent.create_text(x + w / 2, ty, text=text, fill=fg,
                                      font=font)
        self.sub = (parent.create_text(x + w / 2, y + h / 2 + 12, text=sub,
                                       fill=fg, font=("", 8))
                    if sub else None)
        for it in filter(None, (self.rect, self.txt, self.sub)):
            parent.tag_bind(it, "<Button-1>", lambda e: self.command())
            parent.tag_bind(it, "<Enter>", lambda e: self._hi(True))
            parent.tag_bind(it, "<Leave>", lambda e: self._hi(False))

    def _hi(self, on):
        self.cv.itemconfig(self.rect, fill=_mix(self.fill, "#ffffff",
                                                0.18 if on else 0.0))
        self.cv.config(cursor="hand2" if on else "")

    def set(self, text=None, fill=None, fg=None):
        if text is not None:
            self.cv.itemconfig(self.txt, text=text)
        if fill is not None:
            self.fill = fill
            self.cv.itemconfig(self.rect, fill=fill)
        if fg is not None:
            self.cv.itemconfig(self.txt, fill=fg)


def _mix(c1, c2, t):
    a = [int(c1[i:i + 2], 16) for i in (1, 3, 5)]
    b = [int(c2[i:i + 2], 16) for i in (1, 3, 5)]
    return "#%02x%02x%02x" % tuple(int(x + (y - x) * t) for x, y in zip(a, b))


class BigSlider:
    """라벨 + 값 + 두꺼운 트랙. 잡기 쉬우라고 세로 40px 전체가 히트 영역."""

    def __init__(self, parent, x, y, w, label, lo, hi, val, command,
                 font=None, color=CYAN):
        self.cv = parent
        self.lo, self.hi = lo, hi
        self.command = command
        self.value = val
        self.x0, self.x1 = x, x + w
        self.y = y + 26
        parent.create_text(x, y, text=label, anchor="w", fill=DIM, font=font)
        self.val_t = parent.create_text(x + w, y, text="", anchor="e",
                                        fill=TXT, font=font)
        self.track = parent.create_line(x, self.y, x + w, self.y, fill=LINE,
                                        width=7, capstyle="round")
        self.fill = parent.create_line(x, self.y, x, self.y, fill=color,
                                       width=7, capstyle="round")
        self.knob = parent.create_oval(0, 0, 0, 0, fill=color, outline=BG,
                                       width=3)
        self.hit = parent.create_rectangle(x - 10, y - 12, x + w + 10, y + 42,
                                           fill="", outline="")
        for it in (self.track, self.fill, self.knob, self.hit):
            parent.tag_bind(it, "<Button-1>", self._drag)
            parent.tag_bind(it, "<B1-Motion>", self._drag)
        self._paint()

    def _drag(self, e):
        t = (e.x - self.x0) / max(1, self.x1 - self.x0)
        t = 0.0 if t < 0 else (1.0 if t > 1 else t)
        self.value = self.lo + t * (self.hi - self.lo)
        self._paint()
        self.command(self.value)

    def set(self, v):
        self.value = v
        self._paint()

    def _paint(self):
        t = (self.value - self.lo) / (self.hi - self.lo + 1e-9)
        x = self.x0 + t * (self.x1 - self.x0)
        self.cv.coords(self.fill, self.x0, self.y, x, self.y)
        self.cv.coords(self.knob, x - 11, self.y - 11, x + 11, self.y + 11)
        self.cv.itemconfig(self.val_t, text=f"{int(round(t * 100))}")


class OmegaApp:
    """단일 창. 캔버스 하나 위에 전부 올리고, 프레임마다 좌표만 갱신한다."""

    W, H = 960, 660
    STAGE = 420

    def __init__(self, fs=None, block=BLOCK_DEFAULT, device=None):
        import tkinter as tk
        from tkinter import filedialog
        if fs is None:
            fs = default_output_fs(device=device)
        self.tk = tk
        self.filedialog = filedialog
        if sys.platform == "win32":
            try:
                import ctypes
                ctypes.windll.shcore.SetProcessDpiAwareness(1)
            except Exception:
                pass

        self.player = Player(fs=fs, block=block, device=device)
        self.device = device
        self.live = None
        self._spatial_preset = "standard"
        self.root = tk.Tk()
        self.root.title("OMEGA SPATIAL")
        self.root.configure(bg=BG)
        self.root.geometry(f"{self.W}x{self.H}")
        self.root.resizable(False, False)

        fam = self._family()
        self.f_title = (fam, 20, "bold")
        self.f_big = (fam, 15, "bold")
        self.f_mid = (fam, 11, "bold")
        self.f_sm = (fam, 10)
        self.f_xs = (fam, 9)

        self.cv = tk.Canvas(self.root, width=self.W, height=self.H, bg=BG,
                            highlightthickness=0, bd=0)
        self.cv.pack(fill="both", expand=True)
        self._dots = {}
        self._drag = None
        self._frame = 0
        self._last = time.perf_counter()
        self._silence_since = None

        self._build()
        self._room(0.42)
        self.player.clear()          # 기본 소스 없음. 이 앱은 소리를 만들지 않는다.
        self.player.master_gain = 1.0
        self._sync_dots()
        self.cv.itemconfig(
            self.status,
            text="▶ 를 누르고 음악이나 영상을 재생하십시오")
        self.root.protocol("WM_DELETE_WINDOW", self._close)
        self.root.after(50, self._tick)

    def _family(self):
        from tkinter import font as tkfont
        fams = set(tkfont.families(self.root))
        for f in ("Pretendard", "Malgun Gothic", "맑은 고딕", "Noto Sans KR",
                  "Segoe UI", "DejaVu Sans"):
            if f in fams:
                return f
        return "TkDefaultFont"

    # ------------------------------------------------------------ 화면
    def _build(self):
        cv = self.cv
        cv.create_text(34, 34, text="OMEGA SPATIAL", anchor="w", fill=TXT,
                       font=self.f_title)
        cv.create_text(232, 38, text="입체음향 엔진", anchor="w", fill=CYAN,
                       font=self.f_xs)
        cv.create_text(34, 62, text="헤드폰을 쓰고 재생을 누르세요. "
                                    "소리가 머리 밖에서 돌아다닙니다.",
                       anchor="w", fill=DIM, font=self.f_sm)

        # ---- 무대 ----
        sx, sy, s = 34, 92, self.STAGE
        self.stage = (sx, sy, s)
        _rrect(cv, sx, sy, sx + s, sy + s, r=18, fill=CARD, outline=LINE)
        cx, cy = sx + s / 2, sy + s / 2
        self.center = (cx, cy)
        self.px_per_m = (s / 2 - 40) / 6.0                # 6 m = 무대 반지름
        for rm in (2, 4, 6):
            r = rm * self.px_per_m
            cv.create_oval(cx - r, cy - r, cx + r, cy + r, outline=LINE)
        self.pulse = cv.create_oval(0, 0, 0, 0, outline=CYAN, width=2)
        cv.create_oval(cx - 13, cy - 13, cx + 13, cy + 13, fill=CYAN,
                       outline=BG, width=3)
        cv.create_line(cx, cy - 13, cx, cy - 30, fill=CYAN, width=3)
        cv.create_text(cx, cy + 30, text="나", fill=DIM, font=self.f_xs)
        cv.create_text(sx + 16, sy + 18, text="소리를 끌어서 옮겨보세요",
                       anchor="w", fill=DIM, font=self.f_xs)
        cv.bind("<Button-1>", self._press)
        cv.bind("<B1-Motion>", self._motion)
        cv.bind("<ButtonRelease-1>", lambda e: setattr(self, "_drag", None))

        # ---- 오른쪽: 재생 ----
        rx = sx + s + 26
        rw = self.W - rx - 34
        self.btn_play = BigButton(cv, rx, sy, rw, 74, "▶   실시간 3D",
                                  self.toggle_live, fill=CYAN, font=self.f_big)

        # ---- 프리셋 4개 ----
        cv.create_text(rx, sy + 96, text="무엇을 들어볼까요", anchor="w",
                       fill=DIM, font=self.f_xs)
        pw, ph, gap = (rw - 12) / 2, 62, 12
        py = sy + 110
        specs = (("공간 넓이", "정확 / 표준 / 넓게", self.cycle_preset),
                 ("내 음악", "WAV 파일 3D 변환", self.preset_music),
                 ("파일로 저장", "변환 결과 WAV", self.save),
                 ("경로 점검", "잡음 4초 · 소리 확인용", self.path_check))
        for i, (t, sub, cmd) in enumerate(specs):
            x = rx + (i % 2) * (pw + gap)
            y = py + (i // 2) * (ph + gap)
            BigButton(cv, x, y, pw, ph, t, cmd, fill=CARD_HI, fg=TXT,
                      font=self.f_mid, sub=sub)

        # ---- 슬라이더 3개 ----
        sl_y = py + 2 * (ph + gap) + 24
        e = self.player.eng
        self.sl_vol = BigSlider(cv, rx, sl_y, rw, "볼륨", 0.0, 1.4, 1.0,
                                self._vol, font=self.f_sm)
        self.sl_room = BigSlider(cv, rx, sl_y + 66, rw, "공간감", 0.0, 1.0,
                                 0.42, self._room, font=self.f_sm,
                                 color=VIOLET)
        self.sl_bass = BigSlider(cv, rx, sl_y + 132, rw, "저음", 0.0, 2.0,
                                 e.bass.amount, self._bass, font=self.f_sm,
                                 color=VIOLET)

        # ---- 아래: 레벨바 + 상태 + 작은 버튼 ----
        by = sy + s + 22
        cv.create_line(34, by, self.W - 34, by, fill=LINE)
        self.level_bg = cv.create_rectangle(34, by + 14, 34 + 300, by + 22,
                                            fill=CARD_HI, outline="")
        self.level = cv.create_rectangle(34, by + 14, 34, by + 22,
                                         fill=CYAN, outline="")
        self.status = cv.create_text(34, by + 50, text="", anchor="w",
                                     fill=DIM, font=self.f_sm)
        BigButton(cv, self.W - 300, by + 4, 120, 34, "음악 열기",
                  self.preset_music, fill=CARD_HI, fg=TXT, font=self.f_xs)
        BigButton(cv, self.W - 168, by + 4, 134, 34, "WAV로 저장",
                  self.save, fill=CARD_HI, fg=TXT, font=self.f_xs)

    # ------------------------------------------------------------ 동작
    def path_check(self):
        """오디오 경로 확인용. 4초만, 낮은 레벨로, 끝나면 자동 정지.

        여기서 나는 잡음은 '데모'가 아니라 계측 신호다. 스피커까지 신호가
        도달하는지만 보는 용도이므로 길게 틀 이유가 없다."""
        p = self.player
        if self.live is not None:
            self.toggle_live()
        p.clear()
        v = p.add("probe", 200.0, radius=2.6, gain=0.35)
        v.speed = 0.9
        p.rewind()
        if not p.running:
            p.start()
            self.btn_play.set("■   정지", fill=RED, fg="#1a0508")
        self.cv.itemconfig(self.status, text="경로 점검 · 4초 후 자동 정지")
        self.root.after(4000, self._path_check_end)

    def _path_check_end(self):
        if self.live is None and self.player.running:
            self.player.stop()
            self.player.clear()
            self._sync_dots()
            self.btn_play.set("▶   실시간 3D", fill=CYAN, fg="#08131a")
            self.cv.itemconfig(self.status, text="경로 점검 완료")

    def cycle_preset(self):
        """공간 넓이 프리셋 순환. 셋 다 음색 편차 1.5 dB 이내라 마음껏 바꿔도
        음악이 망가지지 않는다."""
        order = ("accurate", "standard", "wide")
        label = {"accurate": "정확 — 정위 우선", "standard": "표준",
                 "wide": "넓게 — 포위감 우선"}
        self._spatial_preset = order[(order.index(self._spatial_preset) + 1)
                                     % len(order)]
        if self.live is not None:
            apply_preset(self.live.field, self.live.reverb,
                         self._spatial_preset)
        self.cv.itemconfig(self.status,
                           text=f"공간 넓이 · {label[self._spatial_preset]}")

    def toggle_live(self):
        """시스템 소리 실시간 3D 변환 시작/정지.

        v2.6: 첫 번째 후보 장치가 실패하면(예: WDM-KS 배타 접근 거부) 그
        자리에서 포기하고 콘솔로 떠넘기던 걸 없앴다. 이제 후보 전체를
        순서대로 자동 시도하고, 실패해도 GUI 안에서 평범한 말로 알려준다.
        '초보자가 콘솔을 열 필요가 아예 없어야 한다'가 원칙이다."""
        if self.live is not None:
            self.live.stop()
            self.live = None
            self.btn_play.set("▶   실시간 3D", fill=CYAN, fg="#08131a")
            self.cv.itemconfig(self.status, text="실시간 3D 정지")
            self._silence_since = None
            return
        try:
            import sounddevice as sd
        except Exception:
            self.cv.itemconfig(
                self.status,
                text="오디오 기능이 설치되지 않았습니다 · pip install sounddevice")
            return
        try:
            oh = sd.query_devices(self.device, kind="output")["hostapi"]
        except Exception:
            oh = None
        cand = list_loopback_candidates(prefer_hostapi=oh)
        if not cand:
            self.cv.itemconfig(
                self.status,
                text="시스템 소리를 캡처할 장치를 찾지 못했습니다 · "
                     "Windows 소리 설정에서 '스테레오 믹스'나 가상 케이블을 켜보세요")
            return
        if self.player.running:
            self.toggle()
        try:
            fs = int(sd.query_devices(self.device,
                                      kind="output")["default_samplerate"])
        except Exception:
            fs = self.player.eng.fs

        # 후보를 순서대로 전부 시도 — 하나가 배타 접근 등으로 실패해도
        # 다음 후보로 자동으로 넘어간다(사용자가 번호를 골라줄 필요 없음).
        last_err = None
        for in_dev, dinfo in cand:
            try:
                live = LiveSpatializer(fs, LIVE_BLOCK, in_dev, self.device,
                                       preset=self._spatial_preset)
                lat = live.start()
            except Exception as ex:
                last_err = (dinfo["name"], ex)
                continue
            self.live = live
            self._silence_since = None
            tot = (lat + live.field.latency_sec) * 1000
            self.btn_play.set("■   정지", fill=RED, fg="#1a0508")
            self.cv.itemconfig(
                self.status,
                text=f"실시간 3D · {live.mode} · {dinfo['name'][:22]}"
                     f" · 지연 {tot:.0f} ms")
            return

        # 후보를 다 시도해도 전부 실패한 경우만 여기로 온다.
        detail = f"'{last_err[0][:20]}' 열기 실패" if last_err else "알 수 없는 오류"
        self.cv.itemconfig(
            self.status,
            text=f"실시간 3D를 시작하지 못했습니다({detail}) · "
                 f"다른 프로그램이 마이크를 쓰고 있는지 확인 후 다시 눌러보세요")

    def close(self):
        if self.live is not None:
            self.live.stop()
            self.live = None
        self.player.stop()

    def toggle(self):
        p = self.player
        if p.running:
            p.stop()
            self.btn_play.set("▶   재생", fill=CYAN, fg="#08131a")
        else:
            if p.finished:
                p.rewind()
            p.start()
            self.btn_play.set("■   정지", fill=RED, fg="#1a0508")

    def _vol(self, v):
        self.player.master_gain = float(v)

    def _room(self, v):
        """슬라이더 하나로 '공간감' 전체를 움직인다. 잔향량·방 크기·초기반사를
        따로 두면 일반 사용자는 조합을 못 맞춘다. 0=무향실, 1=대형 홀."""
        t = float(v)
        e = self.player.eng
        e.reverb.wet = 0.55 * t
        e.reverb.set_room(t)
        e.reverb.early.level = 0.06 + 0.20 * t

    def _bass(self, v):
        self.player.eng.bass.amount = float(v)

    def save(self):
        path = self.filedialog.asksaveasfilename(
            defaultextension=".wav", filetypes=[("WAV", "*.wav")],
            initialfile="omega_spatial.wav")
        if not path:
            return
        el, rt = self.player.render_to(path)
        self.cv.itemconfig(self.status,
                           text=f"저장 완료 · {el:.1f}초에 렌더 "
                                f"(실시간의 {rt:.0f}배 속도)")

    # ---- 프리셋 ----
    def preset_orbit(self):
        p = self.player
        p.clear()
        v = p.add("probe", 200.0, radius=2.6)
        v.speed = 0.9
        self._sync_dots()

    def preset_surround(self):
        p = self.player
        p.clear()
        for i in range(8):
            v = p.add("probe", 200.0, radius=2.2 + 0.4 * (i % 3),
                      angle=i * math.pi / 4, gain=0.30)
            v.speed = 0.10 + 0.03 * i
            v.height = ((i % 3) - 1) * 1.1
        self._sync_dots()

    def preset_flyby(self):
        p = self.player
        p.clear()
        v = p.add("vehicle", 58.0)
        v.orbit = False
        v.flyby = True
        v.speed = 22.0                       # m/s → 도플러가 확실히 들린다
        v.src.pos[:] = [-22.0, 0.0, 3.0]
        self._sync_dots()

    def preset_music(self):
        path = self.filedialog.askopenfilename(filetypes=[("WAV 파일", "*.wav")])
        if not path:
            return
        try:
            data = load_audio(path, self.player.fs)
        except Exception as ex:
            self.cv.itemconfig(self.status, text=f"열 수 없는 파일: {ex}")
            return
        p = self.player
        p.clear()
        v = p.add("file", radius=2.4, gain=0.9)
        v.gen.set_file(data, loop=False)      # 끝까지 재생하고 종료
        v.speed = 0.35
        p.rewind()
        self._sync_dots()
        m, sec = divmod(int(v.gen.length_sec), 60)
        self.cv.itemconfig(self.status,
                           text=f"불러옴 · {m}:{sec:02d} · 끝까지 재생 후 자동 정지")
        if not p.running:
            self.toggle()

    # ---- 무대 상호작용 ----
    def _w2s(self, x, z):
        cx, cy = self.center
        return cx + x * self.px_per_m, cy - z * self.px_per_m

    def _s2w(self, sx, sy):
        cx, cy = self.center
        return (sx - cx) / self.px_per_m, (cy - sy) / self.px_per_m

    def _press(self, e):
        best, bd = None, 26 ** 2
        for v in self.player.voices:
            x, y = self._w2s(v.src.pos[0], v.src.pos[2])
            d = (x - e.x) ** 2 + (y - e.y) ** 2
            if d < bd:
                best, bd = v, d
        if best is not None:
            self._drag = best
            best.orbit = False
            best.flyby = False

    def _motion(self, e):
        v = self._drag
        if v is None:
            return
        x, z = self._s2w(e.x, e.y)
        d = math.hypot(x, z)
        if d > 6.5:                                   # 무대 밖으로는 못 나감
            x, z = x / d * 6.5, z / d * 6.5
        v.src.pos[0] = x
        v.src.pos[2] = z
        v.radius = math.hypot(x, z)
        v.angle = math.atan2(x, z)

    def _sync_dots(self):
        cv = self.cv
        for items in self._dots.values():
            for it in items:
                cv.delete(it)
        self._dots.clear()
        for v in self.player.voices:
            halo = cv.create_oval(0, 0, 0, 0, outline=v.color, width=1)
            dot = cv.create_oval(0, 0, 0, 0, fill=v.color, outline=BG, width=3)
            self._dots[v] = (halo, dot)

    # ------------------------------------------------------------ 루프
    def _tick(self):
        now = time.perf_counter()
        dt = min(0.2, now - self._last)
        self._last = now
        p = self.player
        if self.live is not None:
            ls = self.live
            if ls.peak < 0.003:
                if self._silence_since is None:
                    self._silence_since = now
            else:
                self._silence_since = None
            if self._silence_since is not None and now - self._silence_since > 2.5:
                hint = (" · 소리가 안 들려요? Windows 소리 설정에서 출력 장치를 "
                        "확인해 보세요(스피커/헤드폰이 실제로 선택돼 있어야 합니다)")
            else:
                hint = ""
            self.cv.itemconfig(
                self.status,
                text=f"실시간 3D · {ls.mode} · {self._spatial_preset}"
                     f" · 피크 {ls.peak:.2f}"
                     + (f" · 끊김 {ls.over + ls.under}회" if (ls.over or ls.under) else "")
                     + hint)
        p.step(dt)
        if p.finished and p.running:
            p.stop()
            self.btn_play.set("▶   재생", fill=CYAN, fg="#08131a")
            self.cv.itemconfig(self.status, text="재생 완료")

        lvl = min(1.0, p.peak)
        for v, (halo, dot) in self._dots.items():
            x, y = self._w2s(v.src.pos[0], v.src.pos[2])
            h = 1.0 + 0.25 * v.src.pos[1]              # 높이 → 크기
            r = 9 * max(0.6, h)
            self.cv.coords(dot, x - r, y - r, x + r, y + r)
            hr = r + 4 + 12 * lvl
            self.cv.coords(halo, x - hr, y - hr, x + hr, y + hr)
        cx, cy = self.center
        pr = 20 + 26 * lvl
        self.cv.coords(self.pulse, cx - pr, cy - pr, cx + pr, cy + pr)
        self.cv.itemconfig(self.pulse, outline=_mix(BG, CYAN, 0.25 + 0.5 * lvl))
        self.cv.coords(self.level, 34, self._lv_y(), 34 + 300 * lvl,
                       self._lv_y() + 8)

        self._frame += 1
        if self._frame % 8 == 0:
            self._status()
        self.root.after(33, self._tick)

    def _lv_y(self):
        return self.stage[1] + self.STAGE + 36

    def _status(self):
        p = self.player
        e = p.eng
        if not p.running:
            txt = "정지됨 — 재생을 누르세요"
        elif p.mode == "sim":
            txt = "무음(계산만 함) · " + (p.error or "오디오 장치를 열지 못했습니다")
        else:
            txt = (f"재생 중 · {p.device_name[:22]} · {p.opened_with}"
                   f" · {e.quality_label} · CPU {e.cpu_load * 100:.0f}%"
                   f" · 지연 {p.latency * 1000:.0f} ms")
            if p.music_length > 0:
                cm, cs = divmod(int(p.music_pos), 60)
                tm, ts = divmod(int(p.music_length), 60)
                txt += f" · {cm}:{cs:02d} / {tm}:{ts:02d}"
            if p.error:
                txt += " · " + p.error
        self.cv.itemconfig(self.status, text=txt)

    def _close(self):
        self.player.stop()
        self.root.destroy()

    def run(self):
        self.root.mainloop()


# ═══════════════════════════════════════════════════════════════════════════
#  10. 성능 측정 / 진입점
# ═══════════════════════════════════════════════════════════════════════════
def _bench_fn(fn, warmup=60, iters=400):
    import gc
    for _ in range(warmup):
        fn()
    gc.disable()
    ts = np.empty(iters)
    for i in range(iters):
        t0 = time.perf_counter()
        fn()
        ts[i] = time.perf_counter() - t0
    gc.enable()
    us = ts * 1e6
    return float(np.median(us)), float(np.percentile(us, 95))


def bench(fs=FS_DEFAULT, block=BLOCK_DEFAULT):
    rng = np.random.default_rng(0)
    blk_us = block / fs * 1e6
    print(f"OMEGA SPATIAL {__version__} — {fs} Hz, 블록 {block} "
          f"({blk_us:.0f} µs/블록)\n")
    x = (rng.standard_normal(block) * 0.2).astype(np.float32)
    xl = (rng.standard_normal(block) * 0.2).astype(np.float32)
    xr = (rng.standard_normal(block) * 0.2).astype(np.float32)
    rows = []

    sv = SourceVoice(fs, block)
    l = np.zeros(block, np.float32)
    r = np.zeros(block, np.float32)
    st = [0.0]

    def k_par():
        l.fill(0); r.fill(0); st[0] += 0.02
        sv.render(x, math.sin(st[0]), 0.2, 3.0, 0.5, l, r)
    rows.append(("파라메트릭 HRTF (소스 1개)", *_bench_fn(k_par)))

    t0 = time.perf_counter()
    bank = HRIRBank(fs, block)
    build = (time.perf_counter() - t0) * 1e3
    cvv = ConvVoice(bank, block)

    def k_conv():
        l.fill(0); r.fill(0); st[0] += 0.02
        cvv.process(x, math.sin(st[0]), 0.2, 0.5, l, r)
    rows.append(("컨볼루션 HRTF (소스 1개)", *_bench_fn(k_conv)))

    dv = DopplerVoice(fs, block)
    ob = np.empty(block, np.float32)
    rows.append(("도플러", *_bench_fn(lambda: dv.process(x, 8.0, ob))))

    er = EarlyReflections(fs, block)
    rows.append(("초기 반사 8탭", *_bench_fn(
        lambda: er.process(x, xl.copy(), xr.copy()))))
    rb = ReverbBus(fs, block)
    rb.early.enabled = False
    rows.append(("리버브 (공유 버스)", *_bench_fn(
        lambda: rb.process(xl.copy(), xr.copy()))))
    wd = StereoWidener(fs)
    rows.append(("스테레오 확장", *_bench_fn(
        lambda: wd.process(xl.copy(), xr.copy()))))
    be = BassEnhancer(fs)
    rows.append(("스마트 베이스", *_bench_fn(
        lambda: be.process(xl.copy(), xr.copy()))))
    s2 = np.stack([xl, xr]).copy()
    eq = DynamicEQ(fs)
    m = s2.copy()
    rows.append(("다이나믹 EQ", *_bench_fn(
        lambda: (np.copyto(m, s2), eq.process(m)))))
    ld = Loudness(fs, block)
    m2 = s2.copy()
    rows.append(("라우드니스 + 리미터", *_bench_fn(
        lambda: (np.copyto(m2, s2), ld.process(m2)))))

    print(f"{'커널':30s}{'중앙값 µs':>11s}{'p95 µs':>9s}{'CPU%':>8s}")
    print("-" * 58)
    for n, md, p95 in rows:
        print(f"{n:30s}{md:11.1f}{p95:9.1f}{md / blk_us * 100:7.2f}%")
    print(f"\nHRIR 뱅크 생성(1회): {build:.0f} ms, "
          f"{bank.N_AZ}×{bank.N_EL}셀 × {bank.TAPS}탭")

    print(f"\n{'전체 엔진':30s}{'중앙값 µs':>11s}{'CPU%':>8s}{'여유':>8s}")
    print("-" * 58)
    sig = (rng.standard_normal(fs) * 0.1).astype(np.float32)
    for q in ("high", "fast"):
        for nsrc in (1, 4, 8, 16):
            eng = SpatialAudioEngine(fs=fs, block=block, quality=q)
            if q == "high":
                eng.bank = bank
                eng.f_conv = True
            eng.autoscale = False
            for i in range(nsrc):
                s = Source(block=block, buffer=sig, gain=0.4)
                s.pos[:] = [math.cos(i) * 3, 0.2 * i, math.sin(i) * 3 + 1]
                s.vel[:] = [6.0, 0, 0]
                eng.add_source(s)
            eng.set_listener(pos=[0, 0, 0], forward=[0, 0, 1], up=[0, 1, 0])
            md, _ = _bench_fn(eng.process_block, 40, 200)
            tag = ("컨볼루션" if q == "high" else "고속") + f" HRTF · {nsrc}소스"
            print(f"{tag:30s}{md:11.1f}{md / blk_us * 100:7.2f}%"
                  f"{blk_us / md:7.1f}×")


def render_demo(path, seconds=12.0, fs=FS_DEFAULT, block=BLOCK_DEFAULT):
    p = Player(fs=fs, block=block)
    for _ in range(80):                    # HRIR 뱅크가 준비될 때까지 대기
        if p.eng.bank is not None:
            break
        time.sleep(0.05)
    v = p.add("engine", 95.0, radius=2.6)
    v.speed = 1.1
    el, rt = p.render_to(path, seconds)
    print(f"{path} 저장 · {seconds:.0f}초를 {el:.2f}초에 렌더 "
          f"(실시간의 {rt:.0f}배)")


def audio_report():
    """소리가 안 날 때 원인을 한 번에 보여준다."""
    print(f"OMEGA SPATIAL {__version__} · python {sys.version.split()[0]}")
    try:
        import sounddevice as sd
    except Exception as ex:
        print(f"  sounddevice 없음 → {ex}")
        print("  해결:  pip install sounddevice")
        return
    print(f"  sounddevice {sd.__version__}")
    try:
        out = sd.query_devices(kind="output")
    except Exception as ex:
        print(f"  기본 출력 장치 없음 → {ex}")
        return
    print(f"  기본 출력  : {out['name']}")
    print(f"  기본 SR    : {int(out['default_samplerate'])} Hz, "
          f"최대 {out['max_output_channels']}ch")
    for rate in sorted({48000, 44100, int(out["default_samplerate"])}):
        try:
            sd.check_output_settings(samplerate=rate, channels=2,
                                     dtype="float32")
            print(f"  {rate:6d} Hz 2ch float32 : 사용 가능")
        except Exception as ex:
            print(f"  {rate:6d} Hz 2ch float32 : 실패 — "
                  f"{str(ex).splitlines()[0][:60]}")
    print("\n  시스템 소리 캡처 후보 (이걸로 --live --in <번호>)")
    cands = list_loopback_candidates()
    if cands:
        for i, d in cands:
            print(f"    [{i}] {d['name'][:42]}  "
                  f"{int(d['default_samplerate'])} Hz  {d['max_input_channels']}ch  "
                  f"[{hostapi_name(d['hostapi'])}]")
    else:
        print("    없음 — VB-CABLE 설치 또는 스테레오 믹스 활성화 필요")
    print("\n  전체 입력 장치")
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] > 0:
            print(f"    [{i}] {d['name'][:38]}  "
                  f"{int(d['default_samplerate'])} Hz  {d['max_input_channels']}ch  "
                  f"[{hostapi_name(d['hostapi'])}]")
    print("\n  전체 출력 장치")
    for i, d in enumerate(sd.query_devices()):
        if d["max_output_channels"] > 0:
            print(f"    [{i}] {d['name'][:38]}  "
                  f"{int(d['default_samplerate'])} Hz  "
                  f"{d['max_output_channels']}ch  [{hostapi_name(d['hostapi'])}]")


def selftest(seconds=6.0, device=None):
    """엔진을 실제로 스피커로 내보내 본다. 여기서 소리가 나면 엔진은 정상이고,
    안 나면 장치/드라이버 문제다."""
    fs = default_output_fs(device=device)
    p = Player(fs=fs, block=BLOCK_DEFAULT, device=device)
    v = p.add("probe", 200.0, radius=2.6)
    v.speed = 1.2
    p.start()
    print(f"모드={p.mode}  장치={p.device_name or '-'}  "
          f"설정={p.opened_with or '-'}  SR={fs}  "
          f"지연={p.latency * 1000:.0f} ms")
    if p.error:
        print(f"오류: {p.error}")
    if p.mode != "audio":
        print("→ 스트림이 안 열렸습니다. --devices 로 원인을 확인하십시오.")
        p.stop()
        return
    print(f"{seconds:.0f}초 동안 소리가 머리 주위를 돌아야 정상입니다...")
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        time.sleep(0.03)
        p.step(0.03)
    print(f"블록 {p.blocks}개 출력, 피크 {p.peak:.3f}, "
          f"CPU {p.eng.cpu_load * 100:.0f}%")
    if p.blocks < 10:
        print("→ 콜백이 거의 안 불렸습니다. 장치/드라이버 문제입니다.")
    elif p.peak < 1e-4:
        print("→ 콜백은 도는데 출력이 0입니다. 엔진 쪽 문제입니다.")
    p.stop()


def isolate(device=None):
    """'파리 소리'가 어디서 나오는지 귀로 가르는 도구.

    단계를 하나씩 들려주고, 어느 단계부터 소리가 나는지만 확인하면 원인이
    한 번에 좁혀진다. 1~3번은 DSP를 하나도 안 거친 생 신호다."""
    try:
        import sounddevice as sd
    except Exception as ex:
        print("sounddevice 없음:", ex)
        return
    fs = default_output_fs(device=device)
    B = BLOCK_DEFAULT

    def raw(kind, freq, sec):
        g = Gen(fs, B, kind=kind, freq=freq)
        n = int(sec * fs / B) * B
        out = np.empty(n, dtype=np.float32)
        for i in range(n // B):
            g(out[i * B:(i + 1) * B])
        return np.column_stack([out, out]) * 0.7

    def spatial(kind, freq, sec, reverb, moving, quality="high"):
        eng = SpatialAudioEngine(fs=fs, block=B, quality=quality)
        eng.autoscale = False
        eng.f_reverb = reverb
        eng.set_listener(pos=[0, 0, 0], forward=[0, 0, 1], up=[0, 1, 0])
        time.sleep(0.4)                       # HRIR 뱅크 완성 대기
        g = Gen(fs, B, kind=kind, freq=freq)
        src = Source(block=B, callback=g, gain=0.8)
        eng.add_source(src)
        nb = int(sec * fs / B)
        out = np.empty((nb * B, 2), dtype=np.float32)
        ang = 0.0
        for i in range(nb):
            if moving:
                ang += 1.2 * B / fs
                src.pos[0] = 2.6 * math.sin(ang)
                src.pos[2] = 2.6 * math.cos(ang)
            else:
                src.pos[:] = [0.0, 0.0, 2.6]
            out[i * B:(i + 1) * B] = eng.process_block()
        return out

    steps = [
        ("1. 계측 신호 원본 — DSP 전혀 안 거침",
         lambda: raw("probe", 200.0, 4.0)),
        ("2. 사인파 원본 440Hz — DSP 없음",
         lambda: raw("sine", 440.0, 3.0)),
        ("3. 핑크노이즈 원본 — DSP 없음",
         lambda: raw("pink", 200.0, 3.0)),
        ("4. 계측신호 + 공간화(정지) · 리버브 OFF",
         lambda: spatial("probe", 200.0, 4.0, False, False)),
        ("5. 계측신호 + 공간화(회전) · 리버브 OFF",
         lambda: spatial("probe", 200.0, 4.0, False, True)),
        ("6. 계측신호 + 공간화(회전) + 리버브  = selftest 와 동일",
         lambda: spatial("probe", 200.0, 4.0, True, True)),
        ("7. 핑크노이즈 + 공간화(회전) + 리버브",
         lambda: spatial("pink", 200.0, 4.0, True, True)),
    ]
    print(f"장치 {sd.query_devices(device, kind='output')['name']} · {fs} Hz")
    print("각 단계마다 파리 소리가 '들린다/안 들린다'만 적어 주십시오.\n")
    for label, fn in steps:
        print("  " + label, flush=True)
        seg = np.ascontiguousarray(np.clip(fn(), -1.0, 1.0), dtype=np.float32)
        sd.play(seg, fs, device=device)
        sd.wait()
        time.sleep(0.5)
    print("\n판정")
    print("  1번에서 이미 들림      → 데모 신호 자체가 파리 소리. 엔진 정상.")
    print("  4번부터 들림          → HRTF 공간화 단계 문제.")
    print("  5번부터 들림          → 소스가 움직일 때만 생기는 블록 경계 잡음.")
    print("  6번부터 들림          → 리버브 문제.")
    print("  7번에서도 들림        → 신호와 무관한 상시 잡음.")


SPATIAL_PRESETS = {
    # (spread, ambience, amb_hrtf, early, wet)
    #  IACC(낮을수록 넓음) / 음색 편차  — 원본 IACC 0.353 기준 실측
    # early 값 주의: 2.7에서 초기반사가 6탭 → 220탭 정규화 FIR로 바뀌면서
    # 같은 숫자가 전혀 다른 양을 뜻하게 됐다. 구값(1.30)을 그대로 두었더니
    # 모노(정중앙) 신호의 좌우 상관이 0.99 → 0.25로 무너졌다 — 보컬과 킥이
    # 중앙에서 흩어져 흐물거리는 소리가 된다. 아래 값은 중앙 상관 0.93 이상을
    # 유지하는 선에서 실측으로 정한 것이다. Dolby가 말한 "높은 DR 비"와도
    # 같은 방향이다: 외재화는 잔향을 늘려서가 아니라 반사의 질로 얻는다.
    "accurate": (2.0, 1.6, 0.30, 0.10, 0.22),   # 정위 우선
    "standard": (3.0, 1.6, 0.18, 0.15, 0.25),   # 기본
    "wide":     (3.5, 1.6, 0.05, 0.22, 0.30),   # 포위감 우선
}


def apply_preset(field, reverb, name="standard"):
    sp, amb, ahr, early, wet = SPATIAL_PRESETS.get(
        name, SPATIAL_PRESETS["standard"])
    field.spread = sp
    field.ambience = amb
    field.amb_hrtf = ahr
    reverb.early.level = early
    reverb.wet = wet
    return name


class LiveSpatializer:
    """시스템 오디오를 받아 3D로 만들어 헤드폰으로 내보내는 실시간 경로.

    ─ 신호 경로
        모든 앱 → (Windows 기본 출력 = 가상 장치) → 그 장치의 캡처 엔드포인트
                → 여기(듀플렉스 스트림) → 실제 헤드폰

    ─ 왜 듀플렉스 한 스트림인가
        입력·출력을 별도 스트림으로 열면 두 클럭이 미세하게 달라 몇 분에 한 번씩
        버퍼가 넘치거나 마르면서 툭툭 끊긴다. 하나의 듀플렉스 스트림은 콜백
        하나가 입력과 출력을 동시에 받아 샘플 동기가 보장된다.

    ─ 지연
        STFT (nfft−hop) + 장치 지연. 영상과 함께 볼 때 립싱크가 깨지지 않도록
        STFT 추가 지연을 16 ms 선에서 묶어 둔다."""

    def __init__(self, fs, block, in_dev, out_dev, room=0.28,
                 preset="standard"):
        self.fs = fs
        self.block = block
        self.in_dev = in_dev
        self.out_dev = out_dev
        self.bank = HRIRBank(fs, block)
        self.field = SpatialField(fs, block, self.bank._synth)
        self.reverb = ReverbBus(fs, block)
        self.reverb.set_room(room)
        apply_preset(self.field, self.reverb, preset)
        self.preset = preset
        self.loud = Loudness(fs, block, target_lufs=-15.0)
        self.stream = None
        self.in_stream = None
        self.out_stream = None
        self.mode = ""
        self.in_name = ""
        self.out_name = ""
        self.same_api = True
        self._ratio = 1.0
        self._primed = False
        self.under = 0
        self._last = np.zeros((block, 2), dtype=np.float32)
        self.error = ""
        self.blocks = 0
        self.peak = 0.0
        self.over = 0
        self.bypass = False
        self.f_field = True
        self.f_reverb = True
        self.cpu_budget = CPU_BUDGET
        self._load = 0.0
        self._hi = 0
        self._lo = 0
        self._cool = 0
        self._l = np.zeros(block, dtype=np.float32)
        self._r = np.zeros(block, dtype=np.float32)
        self._mix = np.zeros((2, block), dtype=np.float32)

    def _callback(self, indata, outdata, frames, t, status):
        try:
            if status:
                self.over += 1
            if frames != self.block:            # 예상 밖 크기 → 그대로 통과
                outdata[:] = indata
                return
            self._render_into(indata, outdata)
        except Exception as ex:
            outdata[:] = 0.0
            self.error = f"콜백 오류: {ex}"

    # ---- 스트림 열기 ----------------------------------------------------
    def start(self):
        """호스트 API가 다르면 듀플렉스를 아예 시도하지 않는다.

        PortAudio 듀플렉스(sd.Stream)는 입력과 출력이 같은 호스트 API일 것을
        요구한다. 다르면 'Illegal combination of I/O devices' 로 실패한다.
        윈도우에는 MME / WASAPI / WDM-KS 가 공존하고 가상 캡처 장치와 실제
        헤드폰이 서로 다른 API에 잡히는 게 흔하다. 그러니 미리 확인해서
        다르면 곧장 분리 스트림으로 간다 — 실패할 걸 알면서 던지고 에러를
        보여줄 이유가 없다.

        분리 스트림은 입력·출력이 서로 다른 클럭이라 링 버퍼로 잇는다.
        샘플레이트까지 다르면 그건 드리프트가 아니라 음정이 틀어지는
        문제라 읽기 쪽에서 비율 리샘플링을 한다."""
        import sounddevice as sd
        self.mode = ""
        errs = []
        ind = sd.query_devices(self.in_dev, kind="input")
        outd = sd.query_devices(self.out_dev, kind="output")
        same_api = ind["hostapi"] == outd["hostapi"]
        self.in_name = str(ind["name"])
        self.out_name = str(outd["name"])
        self.same_api = same_api

        if same_api:
            try:
                st = sd.Stream(samplerate=self.fs, blocksize=self.block,
                               channels=2, dtype="float32",
                               device=(self.in_dev, self.out_dev),
                               callback=self._callback)
                st.start()
                self.stream = st
                self.mode = "듀플렉스"
                lat = st.latency
                return float(lat[1] if isinstance(lat, (tuple, list)) else lat)
            except Exception as ex:
                errs.append("듀플렉스: " + str(ex).splitlines()[0][:90])

        # ── 분리 스트림 + 링 버퍼
        cap = self.block * 32
        self._ring = np.zeros((cap, 2), dtype=np.float32)
        self._cap = cap
        self._wi = 0
        self._rpos = 0.0
        self._fill = 0.0
        self._ratio = 1.0
        self._primed = False
        self.under = 0

        for in_fs in (self.fs, int(ind["default_samplerate"])):
            try:
                self.in_stream = sd.InputStream(
                    samplerate=in_fs, blocksize=self.block, channels=2,
                    dtype="float32", device=self.in_dev, callback=self._in_cb)
                self.out_stream = sd.OutputStream(
                    samplerate=self.fs, blocksize=self.block, channels=2,
                    dtype="float32", device=self.out_dev,
                    callback=self._out_cb)
                self._ratio = in_fs / float(self.fs)
                self.in_stream.start()
                self.out_stream.start()
                self.stream = self.out_stream
                self.mode = "분리" if self._ratio == 1.0 else \
                            f"분리+리샘플 {in_fs}→{self.fs}"
                lat = self.out_stream.latency
                return float(lat[1] if isinstance(lat, (tuple, list)) else lat)
            except Exception as ex:
                errs.append(f"분리({in_fs}Hz): " + str(ex).splitlines()[0][:80])
                self._close_all()
            if in_fs == int(ind["default_samplerate"]):
                break
        raise RuntimeError(" / ".join(errs))

    def _in_cb(self, indata, frames, t, status):
        if status:
            self.over += 1
        cap = self._cap
        w = self._wi
        n = min(frames, cap)
        end = w + n
        if end <= cap:
            self._ring[w:end] = indata[:n]
        else:
            k = cap - w
            self._ring[w:] = indata[:k]
            self._ring[:end - cap] = indata[k:n]
        self._wi = end % cap
        self._fill = min(float(cap - 2), self._fill + n)

    def _out_cb(self, outdata, frames, t, status):
        try:
            if status:
                self.over += 1
            b = self.block
            need = b * self._ratio
            # 선충전: 처음부터 곧바로 꺼내 쓰면 버퍼가 계속 바닥을 쳐서
            # 툭툭 끊긴다. 목표 수위까지 채운 뒤에 재생을 시작한다.
            if not self._primed:
                if self._fill < need * self.PRIME_BLOCKS:
                    outdata[:] = 0.0
                    return
                self._primed = True
            if frames != b:
                outdata[:] = 0.0
                return
            if self._fill < need + 2.0:
                # 순간적으로 말랐을 때는 무음보다 직전 블록 유지가 덜 들린다
                outdata[:] = self._last
                self.under += 1
                return
            cap = self._cap
            # 클럭 드리프트 보정: 너무 차면 앞쪽을 조금 버린다
            if self._fill > cap * 0.75:
                drop = self._fill - cap * 0.5
                self._rpos += drop
                self._fill -= drop
            # 읽기 위치는 쓰기 위치 기준 (fill 만큼 뒤)
            base = (self._wi - self._fill) % cap
            pos = base + np.arange(b, dtype=np.float64) * self._ratio
            i0 = np.floor(pos).astype(np.intp)
            fr = (pos - i0).astype(np.float32).reshape(-1, 1)
            a = self._ring[i0 % cap]
            c = self._ring[(i0 + 1) % cap]
            blk = a + (c - a) * fr
            self._fill -= need
            self._render_into(blk, outdata)
            self._last[:] = outdata
        except Exception as ex:
            outdata[:] = 0.0
            self.error = f"출력 콜백 오류: {ex}"

    def _render_into(self, indata, outdata):
        t0 = time.perf_counter()
        b = self.block
        l, r = self._l, self._r
        l[:] = indata[:, 0]
        r[:] = indata[:, 1] if indata.shape[1] > 1 else indata[:, 0]
        if not self.bypass:
            if self.f_field:
                self.field.process(l, r)
            if self.f_reverb:
                self.reverb.process(l, r)
            mix = self._mix
            mix[0] = l
            mix[1] = r
            self.loud.process(mix)
            l[:] = mix[0]
            r[:] = mix[1]
        outdata[:, 0] = l
        outdata[:, 1] = r
        self.blocks += 1
        p = float(max(np.max(np.abs(l)), np.max(np.abs(r))))
        self.peak = p if p > self.peak else self.peak * 0.9 + p * 0.1
        # ── CPU 예산 거버너 ────────────────────────────────────────────
        # v2.6: 실시간 경로에는 오토스케일러가 아예 없었다. 엔진 쪽에만
        # 있었는데 정작 제품이 실제로 도는 경로는 여기다. 사용자 PC 성능을
        # 알 수 없으므로 절대 시간이 아니라 '이 기계에서 실제로 몇 %를
        # 먹는가'를 측정해서 예산 안으로 들어오게 스스로 낮춘다.
        dt = (time.perf_counter() - t0) * self.fs / b
        self._load += 0.02 * (dt - self._load)
        self._govern()

    def _govern(self):
        """예산 초과가 계속되면 비싼 단계부터 끄고, 여유가 지속되면 되돌린다.

        데드밴드가 핵심이다. 임계를 예산에 딱 붙여 놓으면, 예산 바로 아래
        (예: 9.3%)에서 안정적으로 도는 상태인데도 순간 스파이크가 EMA를
        잠깐 밀어 올릴 때마다 카운터가 쌓여 결국 멀쩡한 기능까지 꺼진다
        (실측: 절전 단계에서 안정 동작 중이었는데도 공간화까지 내려갔다).
        내릴 때는 예산의 110%, 올릴 때는 60%를 기준으로 삼아 두 임계 사이에
        넓은 무반응 구간을 둔다.

        두 번째는 쿨다운이다. 부하 측정이 EMA(시정수 약 50블록)라, 리버브를
        끈 직후에도 EMA는 한동안 끄기 전의 높은 값을 그대로 들고 있다. 그
        사이에 카운터가 다시 차면 '아직 효과도 안 나타난 조치'를 근거로
        공간화까지 내려간다(실측: 실제로 그렇게 무너졌다). 제어 루프는
        자기 조치의 결과가 관측될 때까지 기다려야 한다. 단계를 바꾼 뒤
        EMA 시정수보다 넉넉히 긴 동안 추가 변경을 막는다.

        끄는 순서는 실측 비용과 가치를 함께 보고 정했다.
            리버브(4.8%)  먼저 — 공간감 보조 요소라 없어도 3D는 유지된다
            공간화(6.2%)  나중 — 이게 제품의 본체다. 최후에만 내린다."""
        if self._cool > 0:
            self._cool -= 1
            return
        load = self._load
        budget = self.cpu_budget
        if load > budget * 1.10:
            self._hi += 1
            self._lo = 0
        elif load < budget * 0.60:
            self._lo += 1
            self._hi = 0
        else:
            self._hi = self._lo = 0
        if self._hi >= 12:
            self._hi = self._lo = 0
            if self.f_reverb:
                self.f_reverb = False
                self._cool = self.COOLDOWN
            elif self.f_field:
                self.f_field = False
                self._cool = self.COOLDOWN
        elif self._lo >= 40:
            self._hi = self._lo = 0
            if not self.f_field:
                self.f_field = True
                self._cool = self.COOLDOWN
            elif not self.f_reverb:
                self.f_reverb = True
                self._cool = self.COOLDOWN

    COOLDOWN = 150      # 블록. EMA 시정수(≈50)보다 넉넉히 길게.

    @property
    def cpu(self):
        """블록 처리시간 / 실시간. ×100이 1코어 점유율(%)."""
        return self._load

    @property
    def quality_label(self):
        if self.f_field and self.f_reverb:
            return "최고"
        if self.f_field:
            return "절전"
        return "최소"

    PRIME_BLOCKS = 4.0

    def _close_all(self):
        for a in ("in_stream", "out_stream"):
            st = getattr(self, a, None)
            if st is not None:
                try:
                    st.stop()
                    st.close()
                except Exception:
                    pass
                setattr(self, a, None)

    def stop(self):
        if self.stream is not None and self.stream is not self.out_stream:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass
        self.stream = None
        self._close_all()


def list_loopback_candidates(prefer_hostapi=None):
    """시스템 소리를 잡아올 수 있는 입력 장치 후보.

    prefer_hostapi 를 주면 그 호스트 API 장치를 앞에 놓는다. PortAudio 듀플렉스는
    입력·출력이 같은 API여야 열리므로, 출력과 같은 API의 캡처 장치를 먼저
    고르면 성공률이 크게 올라간다."""
    import sounddevice as sd
    pats = ("stereo mix", "스테레오 믹스", "what u hear", "wave out",
            "cable output", "vb-audio", "voicemeeter", "sonar", "vad",
            "loopback", "믹스")
    out = []
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] < 2:
            continue
        nm = d["name"].lower()
        if any(p in nm for p in pats):
            out.append((i, d))
    if prefer_hostapi is not None:
        out.sort(key=lambda t: 0 if t[1]["hostapi"] == prefer_hostapi else 1)
    return out


def hostapi_name(idx):
    import sounddevice as sd
    try:
        return sd.query_hostapis(idx)["name"]
    except Exception:
        return "?"


LIVE_BLOCK = 512      # 실시간 경로 기본 블록


def live(in_dev=None, out_dev=None, fs=None, block=None,
         seconds=None, preset="standard"):
    """블록을 256이 아니라 512로 잡는 이유
    콜백 부담률은 두 경우 모두 13% 정도로 같지만, 마감 시간이 5.3 ms 에서
    10.7 ms 로 두 배가 된다. 파이썬 GC 정지나 윈도우 스케줄링 지터가 한 번
    끼어들었을 때 마감을 넘길 확률이 크게 줄고, 그 마감 초과가 곧 '지지직'
    이다. 블루투스처럼 지연이 이미 100 ms 인 장치에서는 32 ms 를 더 써서
    끊김을 없애는 쪽이 명백히 낫다."""
    block = block or LIVE_BLOCK
    try:
        import sounddevice as sd
    except Exception as ex:
        print("sounddevice 없음:", ex)
        return
    if in_dev is None:
        try:
            oh = sd.query_devices(out_dev if out_dev is not None else None,
                                  kind="output")["hostapi"]
        except Exception:
            oh = None
        cand = list_loopback_candidates(prefer_hostapi=oh)
        if not cand:
            print("시스템 소리를 잡을 입력 장치를 못 찾았습니다.\n")
            print("아래 중 하나를 하십시오.")
            print("  A. 이미 깔린 SteelSeries Sonar 사용")
            print("     Windows 소리 설정 → 출력 = Sonar 가상 장치")
            print("     그 다음  --live --in <해당 캡처 장치 번호>")
            print("  B. VB-CABLE 설치 (무료)")
            print("     출력 = CABLE Input, --live --in <CABLE Output 번호>")
            print("  C. Realtek 스테레오 믹스 활성화")
            print("     소리 설정 → 녹음 → 사용 안 함 장치 표시 → 스테레오 믹스 사용")
            print("\n입력 장치 목록은 --devices 로 확인하십시오.")
            return
        in_dev = cand[0][0]
        print(f"입력 자동 선택: [{in_dev}] {cand[0][1]['name']}")
    if fs is None:
        # 엔진은 출력 장치 기준으로 돈다. 헤드폰이 못 받는 레이트로 열면
        # 스트림이 안 열리거나 OS가 강제 리샘플링한다. 입력 쪽 레이트가
        # 다르면 링 버퍼 읽기에서 비율 리샘플링으로 맞춘다.
        fs = int(sd.query_devices(out_dev if out_dev is not None else None,
                                  kind="output")["default_samplerate"])
    ind = sd.query_devices(in_dev)
    outd = sd.query_devices(out_dev if out_dev is not None else None,
                            kind="output")
    if ind["name"] == outd["name"]:
        print("입력과 출력이 같은 장치입니다. 처리한 소리가 다시 입력으로 들어와")
        print("폭주합니다. 출력을 실제 헤드폰으로 바꾸십시오 (--out <번호>).")
        return
    print(f"입력  [{in_dev}] {ind['name']}  ({hostapi_name(ind['hostapi'])})")
    print(f"출력  {outd['name']}  ({hostapi_name(outd['hostapi'])})")
    if ind["hostapi"] != outd["hostapi"]:
        print("  호스트 API가 다릅니다 → 듀플렉스는 건너뛰고 분리 스트림으로 엽니다")
    ls = LiveSpatializer(fs, block, in_dev, out_dev, preset=preset)
    print(f"SR {fs} · STFT 추가지연 {ls.field.latency_sec * 1000:.0f} ms")
    try:
        lat = ls.start()
    except Exception as ex:
        print("\n스트림 열기 실패")
        for line in str(ex).split(" / "):
            print("  " + line)
        print("\n  --in <번호> --out <번호> 로 직접 지정해 보십시오.")
        print("  장치 번호는 --devices 로 확인합니다.")
        return
    print(f"{ls.mode} 스트림 · 장치 지연 {lat * 1000:.0f} ms · "
          f"총 {lat * 1000 + ls.field.latency_sec * 1000:.0f} ms")
    print("음악이나 영상을 재생하십시오. Ctrl+C 로 종료.")
    t0 = time.perf_counter()
    try:
        while seconds is None or time.perf_counter() - t0 < seconds:
            time.sleep(0.5)
            el = time.perf_counter() - t0
            print(f"\r  {el:5.1f}s · 블록 {ls.blocks} · 피크 {ls.peak:.3f}"
                  f" · 드롭 {ls.over}   ", end="", flush=True)
            if ls.error:
                print("\n" + ls.error)
                break
    except KeyboardInterrupt:
        pass
    finally:
        ls.stop()
        print("\n종료")


def spatialize_file(src, dst, fs=None, block=BLOCK_DEFAULT, width=1.35,
                    ambience=None, room=0.28, wet=None, normalize=False,
                    preset="standard"):
    """WAV 파일을 3D로 변환해 저장. 장치 라우팅 없이 결과를 먼저 확인하는 용도."""
    import wave
    with wave.open(src, "rb") as w:
        sr = w.getframerate()
        nch = w.getnchannels()
        sw = w.getsampwidth()
        raw = w.readframes(w.getnframes())
    if sw == 2:
        a = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif sw == 4:
        a = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"{sw * 8}bit WAV 는 지원하지 않습니다")
    a = a.reshape(-1, nch)
    L = a[:, 0].copy()
    R = a[:, 1].copy() if nch > 1 else L.copy()
    fs = fs or sr
    if fs != sr:
        raise ValueError(f"파일 {sr} Hz, 요청 {fs} Hz — 샘플레이트가 다릅니다")

    bank = HRIRBank(fs, block)
    field = SpatialField(fs, block, bank._synth)
    rev = ReverbBus(fs, block)
    rev.set_room(room)
    apply_preset(field, rev, preset)
    field.width = width
    if ambience is not None:
        field.ambience = ambience
    if wet is not None:
        rev.wet = wet
    # 기본은 레벨 매칭. 라우드니스 정규화를 걸어 두면 원본보다 커져서
    # A/B 할 때 '크니까 좋다'로 착각하게 된다. 공간감이 실제로 좋아졌는지
    # 판단하려면 같은 크기여야 한다.
    loud = Loudness(fs, block, target_lufs=-15.0) if normalize else None

    tail = int((field.latency_sec + 2.0) * fs)
    n = ((L.shape[0] + tail) // block) * block
    L = np.pad(L, (0, max(0, n - L.shape[0])))[:n]
    R = np.pad(R, (0, max(0, n - R.shape[0])))[:n]
    out = np.empty((n, 2), dtype=np.float32)
    mix = np.zeros((2, block), dtype=np.float32)
    t0 = time.perf_counter()
    for i in range(n // block):
        l = L[i * block:(i + 1) * block].copy()
        r = R[i * block:(i + 1) * block].copy()
        field.process(l, r)
        rev.process(l, r)
        if loud is not None:
            mix[0] = l
            mix[1] = r
            loud.process(mix)
            l, r = mix[0], mix[1]
        out[i * block:(i + 1) * block, 0] = l
        out[i * block:(i + 1) * block, 1] = r
    el = time.perf_counter() - t0
    if loud is None:
        # 입력과 같은 RMS 로 맞춘다(레벨 매칭 A/B용)
        src_rms = math.sqrt(float(np.mean(L ** 2) + np.mean(R ** 2)) * 0.5) + 1e-12
        dst_rms = math.sqrt(float(np.mean(out ** 2))) + 1e-12
        out *= np.float32(min(4.0, src_rms / dst_rms))
        pk = float(np.max(np.abs(out)))
        if pk > 0.98:
            out *= np.float32(0.98 / pk)
    save_wav(dst, out, fs)
    print(f"{dst} 저장 · {n / fs:.1f}초를 {el:.1f}초에 처리 "
          f"(실시간의 {n / fs / el:.0f}배) · 지연 {field.latency_sec * 1000:.0f} ms")




# ═══════════════════════════════════════════════════════════════════════════
#  10. 청감 튜닝 하네스
#
#  TUNE 의 값들은 물리 상수가 아니라 경험값이다. 물리적으로 정확한 HRTF는
#  실제로 답답하게 들리고 앞뒤 구분도 애매하다 — 상용 제품이 실제보다 넓고
#  시원하게 의도적으로 과장하는 이유가 이것이다. 정답은 계산이 아니라 청취로만
#  나온다. 이 하네스는 그 청취가 오염되지 않게 두 가지를 강제한다.
#
#    1) 라우드니스 정합. 0.3 dB만 커도 사람은 거의 항상 그쪽을 "좋다"고 고른다.
#       음질이 아니라 크기를 듣는 것이다. 비교 전에 BS.1770 K-웨이팅으로 맞춘다.
#    2) 블라인드. 어느 쪽이 '새로 고친 것'인지 알면 그쪽 편을 든다. A/B를 X/Y로
#       무작위 배정하고 정체는 판정 후에 공개한다.
#
#  이 둘이 없으면 수천 번을 비교해도 좋아지는 게 아니라 표류한다.
#
#      python OMEGA_SPATIAL.py --tune-list
#      python OMEGA_SPATIAL.py --tune 음악.wav --param shadow_depth --value 9000
#      python OMEGA_SPATIAL.py --tally 5 7
# ═══════════════════════════════════════════════════════════════════════════
def tune_k_weight(x, fs):
    """ITU-R BS.1770 K-웨이팅. 사람이 느끼는 크기에 맞춰 잰다.
    단순 RMS로 맞추면 저역이 많은 쪽이 손해를 본다."""
    from scipy.signal import bilinear, lfilter
    # 1단: 고역 쉘프 (머리 회절 근사)
    f0, G, Q = 1681.97, 3.999, 0.7071
    K = math.tan(math.pi * f0 / fs)
    Vh = 10 ** (G / 20.0)
    Vb = Vh ** 0.4996
    a0 = 1.0 + K / Q + K * K
    b = np.array([(Vh + Vb * K / Q + K * K), 2.0 * (K * K - Vh),
                  (Vh - Vb * K / Q + K * K)]) / a0
    a = np.array([1.0, 2.0 * (K * K - 1.0) / a0,
                  (1.0 - K / Q + K * K) / a0])
    y = lfilter(b, a, x)
    # 2단: 저역 차단 (RLB)
    f0 = 38.13547087602444
    Q = 0.5003270373238773
    K = math.tan(math.pi * f0 / fs)
    a0 = 1.0 + K / Q + K * K
    b2 = np.array([1.0, -2.0, 1.0])
    a2 = np.array([1.0, 2.0 * (K * K - 1.0) / a0, (1.0 - K / Q + K * K) / a0])
    return lfilter(b2, a2, y)


def tune_lufs(stereo, fs):
    """대략적인 통합 라우드니스(LUFS). 상대 비교용이라 절대 정확도는 불필요."""
    acc = 0.0
    for ch in range(stereo.shape[1]):
        y = tune_k_weight(stereo[:, ch].astype(np.float64), fs)
        acc += np.mean(y * y)
    return -0.691 + 10.0 * math.log10(acc + 1e-20)


def tune_match_loudness(a, b, fs):
    """b를 a의 라우드니스에 맞춘다. 반환: 정합된 b, 적용된 이득(dB)."""
    la, lb = tune_lufs(a, fs), tune_lufs(b, fs)
    g_db = la - lb
    g = 10.0 ** (g_db / 20.0)
    out = b * g
    peak = np.max(np.abs(out))
    if peak > 0.999:                      # 클리핑 방지: 양쪽을 같이 낮춘다
        s = 0.999 / peak
        return out * s, g_db, s
    return out, g_db, 1.0


# ═══════════════════════════════════════════════════════════════════════════
#  엔진 렌더링
# ═══════════════════════════════════════════════════════════════════════════

TUNE_PARAMS = {
    # 이름            (경로, 기본값, 최소, 최대, 설명)
    "shadow_depth": ("TUNE", 16500.0, 4000.0, 17500.0,
                     "먼 귀 고역 감쇠 깊이. 크면 좌우 분리가 강해지지만 답답해진다"),
    "elev_tilt":    ("TUNE", 0.45, 0.0, 1.2,
                     "위아래 명암차. 크면 고도 구분이 뚜렷하나 음색이 흔들린다"),
    "itd_scale":    ("TUNE", 1.0, 0.5, 2.0,
                     "ITD 배율. 1보다 크면 좌우가 과장되어 넓게 들린다"),
    "width":        ("field", 1.35, 0.5, 2.5, "무대 폭"),
    "ambience":     ("field", 1.6, 0.0, 3.0, "포위감"),
    "spread":       ("field", 3.0, 1.0, 6.0, "정위 ↔ 포위 균형"),
    "crossfeed":    ("field", 0.22, 0.0, 0.6, "좌우 교차량. 헤드폰 피로도에 영향"),
    "mix":          ("field", 1.0, 0.0, 1.0, "원본 ↔ 완전 공간화"),
    "wet":          ("reverb", None, 0.0, 0.6, "리버브 양. 많으면 넓지만 탁해진다"),
    "early":        ("early", 0.5, 0.0, 1.5, "초기 반사 양. 거리감을 만든다"),
}



def tune_read_wav(path):
    with wave.open(path, "rb") as w:
        fs = w.getframerate()
        ch = w.getnchannels()
        sw = w.getsampwidth()
        raw = w.readframes(w.getnframes())
    if sw != 2:
        raise SystemExit("16비트 WAV만 지원합니다. 변환 후 다시 시도하십시오.")
    d = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    d = d.reshape(-1, ch)
    if ch == 1:
        d = np.repeat(d, 2, axis=1)
    return d[:, :2].copy(), fs


def tune_write_wav(path, x, fs):
    pcm = np.clip(x, -1.0, 1.0)
    with wave.open(path, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(fs)
        w.writeframes((pcm * 32767).astype(np.int16).tobytes())



def tune_binomial_p(wins, trials):
    """귀무가설(차이 없음, 50:50)에서 이 정도 편중이 나올 확률(양측).
    이게 0.05보다 크면 '더 좋다'고 말할 근거가 없다 — 그냥 우연이다."""
    if trials == 0:
        return 1.0
    k = max(wins, trials - wins)
    tail = 0.0
    for i in range(k, trials + 1):
        tail += math.comb(trials, i) * 0.5 ** trials
    return min(1.0, 2.0 * tail)


def tune_verdict(wins, trials):
    p = tune_binomial_p(wins, trials)
    if trials < 7:
        return f"판정 보류 — {trials}회는 부족합니다 (최소 7회)"
    if p > 0.05:
        return (f"차이 없음 ({wins}/{trials}, p={p:.3f}) — "
                f"우연과 구분되지 않습니다. 이 파라미터는 여기서 안 들립니다")
    side = "B" if wins * 2 < trials else "A"
    return f"{side} 우세 ({wins}/{trials}, p={p:.3f}) — 통계적으로 유의합니다"




def tune_render(pcm, fs, block, overrides, preset="standard"):
    saved = dict(TUNE)
    for k, v in overrides.items():
        if k in TUNE_PARAMS and TUNE_PARAMS[k][0] == "TUNE":
            TUNE[k] = v
    try:
        bank = HRIRBank(fs, block)
        field = SpatialField(fs, block, bank._synth)
        reverb = ReverbBus(fs, block)
        reverb.set_room(0.28)
        apply_preset(field, reverb, preset)
        loud = Loudness(fs, block, target_lufs=-15.0)
        for k, v in overrides.items():
            if k not in TUNE_PARAMS:
                continue
            where = TUNE_PARAMS[k][0]
            if where == "field":
                setattr(field, k, v)
            elif where == "reverb":
                setattr(reverb, k, v)
            elif where == "early":
                reverb.early.level = v
        n = len(pcm)
        out = np.zeros((n, 2), dtype=np.float32)
        mix = np.zeros((2, block), dtype=np.float32)
        for i in range(0, n - block, block):
            l = pcm[i:i + block, 0].copy()
            r = pcm[i:i + block, 1].copy()
            field.process(l, r)
            reverb.process(l, r)
            mix[0] = l
            mix[1] = r
            loud.process(mix)
            out[i:i + block, 0] = mix[0]
            out[i:i + block, 1] = mix[1]
        return out
    finally:
        TUNE.clear()
        TUNE.update(saved)


def tune_list():
    print("\n조정 가능한 파라미터\n" + "=" * 74)
    for k, (where, dflt, lo, hi, desc) in TUNE_PARAMS.items():
        d = "엔진 기본" if dflt is None else f"{dflt:g}"
        print(f"  {k:<14} 기본 {d:<10} 범위 {lo:g}~{hi:g}")
        print(f"  {'':<14} {desc}")
    print()


def tune_run(wav, param, value, start=0.0, dur=25.0, out="tune_out",
             preset="standard"):
    import os, json, random
    if param not in TUNE_PARAMS:
        raise SystemExit(f"모르는 파라미터: {param}  (--tune-list 로 확인)")
    pcm, fs = tune_read_wav(wav)
    i0 = int(start * fs)
    i1 = min(len(pcm), i0 + int(dur * fs))
    seg = pcm[i0:i1]
    if len(seg) < fs:
        raise SystemExit("구간이 너무 짧습니다.")
    os.makedirs(out, exist_ok=True)
    print(f"\n{fs} Hz · 구간 {start:.0f}~{start + dur:.0f}초")
    print(f"{param}: 기본 vs {value:g}\n  A안 렌더링...")
    a = tune_render(seg, fs, LIVE_BLOCK, {}, preset)
    print("  B안 렌더링...")
    b = tune_render(seg, fs, LIVE_BLOCK, {param: value}, preset)
    la, lb = tune_lufs(a, fs), tune_lufs(b, fs)
    b_m, g_db, scale = tune_match_loudness(a, b, fs)
    a_m = a * scale
    print(f"\n  라우드니스  A {la:.2f} LUFS · B {lb:.2f} LUFS")
    print(f"  정합 보정   B에 {g_db:+.2f} dB → 이제 같은 크기입니다")
    if abs(g_db) > 1.0:
        print(f"  주의: 원래 {abs(g_db):.1f} dB 차이가 났습니다. 정합 없이")
        print("        들었으면 큰 쪽을 '좋다'고 골랐을 겁니다.")
    labels = ["X", "Y"]
    random.shuffle(labels)
    mapping = {labels[0]: "A(기본)", labels[1]: f"B({param}={value:g})"}
    tune_write_wav(os.path.join(out, labels[0] + ".wav"), a_m, fs)
    tune_write_wav(os.path.join(out, labels[1] + ".wav"), b_m, fs)
    key = os.path.join(out, "정답.json")
    with open(key, "w", encoding="utf-8") as f:
        json.dump({"mapping": mapping, "param": param,
                   "base": TUNE_PARAMS[param][1], "test": value},
                  f, ensure_ascii=False, indent=2)
    print(f"\n  {out}/X.wav 와 {out}/Y.wav 를 만들었습니다.")
    print(f"  라우드니스는 정합돼 있고 정체는 {key} 에 봉인돼 있습니다.\n")
    print("  1. X와 Y를 번갈아, 같은 구간을 여러 번 왕복해 들으십시오.")
    print("  2. 판정을 내린 뒤에만 정답.json 을 여십시오.")
    print("  3. 최소 7회 반복해야 우연과 구분됩니다:")
    print("       python OMEGA_SPATIAL.py --tally 이긴횟수 전체횟수\n")


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    fs, block, device = None, BLOCK_DEFAULT, None
    for i, a in enumerate(argv):
        if a.startswith("--fs="):
            fs = int(a.split("=")[1])
        elif a.startswith("--block="):
            block = int(a.split("=")[1])
        elif a.startswith("--device="):
            device = int(a.split("=")[1])
        elif a == "--device" and i + 1 < len(argv):
            device = int(argv[i + 1])
    if "--tune-list" in argv:
        tune_list()
        return
    if "--tally" in argv:
        i = argv.index("--tally")
        print(tune_verdict(int(argv[i + 1]), int(argv[i + 2])))
        return
    if "--tune" in argv:
        i = argv.index("--tune")
        wav = argv[i + 1]
        def _opt(name, cast, dflt=None):
            if name in argv:
                return cast(argv[argv.index(name) + 1])
            return dflt
        param = _opt("--param", str)
        value = _opt("--value", float)
        if not param or value is None:
            raise SystemExit("--param 과 --value 를 지정하십시오 "
                             "(--tune-list 로 목록 확인)")
        tune_run(wav, param, value,
                 start=_opt("--start", float, 0.0),
                 dur=_opt("--dur", float, 25.0),
                 out=_opt("--out", str, "tune_out"),
                 preset=_opt("--preset", str, "standard"))
        return
    if "--devices" in argv:
        audio_report()
        return
    if "--selftest" in argv:
        selftest(device=device)
        return
    if "--isolate" in argv:
        isolate(device=device)
        return
    if "--live" in argv:
        ind = outd = None
        for i, a in enumerate(argv):
            if a == "--in" and i + 1 < len(argv):
                ind = int(argv[i + 1])
            elif a == "--out" and i + 1 < len(argv):
                outd = int(argv[i + 1])
        pre = "standard"
        blk = None
        for a in argv:
            if a.startswith("--preset="):
                pre = a.split("=")[1]
            elif a.startswith("--block="):
                blk = int(a.split("=")[1])
        live(in_dev=ind, out_dev=outd if outd is not None else device,
             fs=fs, block=blk, preset=pre)
        return
    if "--spatialize" in argv:
        i = argv.index("--spatialize")
        src = argv[i + 1]
        dst = argv[i + 2] if len(argv) > i + 2 else "spatial_out.wav"
        kw = {}
        for j, a in enumerate(argv):
            if a.startswith("--width="):
                kw["width"] = float(a.split("=")[1])
            elif a.startswith("--ambience="):
                kw["ambience"] = float(a.split("=")[1])
            elif a.startswith("--wet="):
                kw["wet"] = float(a.split("=")[1])
            elif a == "--normalize":
                kw["normalize"] = True
            elif a.startswith("--preset="):
                kw["preset"] = a.split("=")[1]
        spatialize_file(src, dst, block=block, **kw)
        return
    if "--bench" in argv:
        bench(fs or FS_DEFAULT, block)
        return
    if "--render" in argv:
        i = argv.index("--render")
        path = argv[i + 1] if len(argv) > i + 1 else "omega_spatial.wav"
        sec = 12.0
        if "--sec" in argv:
            sec = float(argv[argv.index("--sec") + 1])
        render_demo(path, sec, fs or FS_DEFAULT, block)
        return
    OmegaApp(fs=fs, block=block, device=device).run()


if __name__ == "__main__":
    main()

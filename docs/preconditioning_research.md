## 1. hemistone/diffvg 코드베이스 구조 · 어떻게 돌아가는지 요약

### 1.1 전체 레이어

당신 레포는 원본 diffvg(BachiLi/diffvg)를 그대로 가져온 뒤, **Python 프론트엔드(pydiffvg)를 꽤 크게 리팩토링 + Bézier Splatting backend 추가**한 형태야.([GitHub][1])

대략 레이어는 이렇게 보면 돼:

1. **C++/CUDA core (`diffvg` 모듈)**

   * `Scene`, `Circle`, `Path`, `Rect`, `ShapeGroup`, `Filter` 등 low-level 타입과 래스터라이저 & backward kernel이 들어있음.([GitHub][2])
   * PyTorch에서 넘기는 flat argument list를 받아서 실제 GPU/CPU에서 렌더링.

2. **Python 프론트엔드 (`pydiffvg` 패키지)**

   주요 서브모듈:

   * `device.py`

     * `set_use_gpu`, `get_device` 등 **디바이스/CPU·CUDA 선택을 중앙집중 관리**.
     * `DIFFVG_DEVICE`, `DIFFVG_FORCE_CPU`, `DIFFVG_FORCE_GPU` 같은 env var도 처리.([GitHub][3])

   * `shape.py`

     * `Circle, Ellipse, Path, Polygon, Rect, ShapeGroup` 등 **파라미터 컨테이너**.
     * SVG path string을 바로 `Path` 객체 리스트로 바꿔주는 `from_svg_path` 포함.([GitHub][4])

   * `serialization.py`

     * `(canvas_w, canvas_h, shapes, shape_groups, …)` → **C++ core가 이해하는 flat args 리스트**로 변환.
     * `keep_on_device=True`인 경우에는 GPU 텐서 상태 유지해서 **splat backend가 바로 Torch tensor를 쓸 수 있게 설계**.([GitHub][5])

   * `render_pytorch.py`

     * 원래의 `RenderFunction` 역할을 하는 `BaselineRenderFunction` (torch.autograd.Function) 구현.
     * forward 에서 flat args를 풀어 C++ `Scene` 구성 → `Scene.render` 실행.([GitHub][2])
     * `OutputType.color / sdf` 지원.

   * `backends/registry.py`, `backend.py`, `renderer.py`

     * **backend 선택 레이어**:

       * backend 이름: `"baseline"`(원래 diffvg), `"splat"`(Bézier splatting).([GitHub][6])
       * `SplatConfig(K, R, rho, tile, depth_policy)` 구조체로 splat 파라미터 관리.([GitHub][6])
     * `Renderer` 클래스는

       * `Renderer(backend="baseline" | "splat")`
       * `serialize_scene(...)` → 선택된 backend의 `serialize_scene` 호출
       * `apply(...)` → 선택된 backend의 forward (`RenderAPI.apply`) 호출.([GitHub][7])

   * `splat/*`

     * **Bézier Splatting 구현체**:

       * `ScenePayload`, `PathSpec`, `GaussianBatch` 등 타입 정의.([GitHub][8])
       * `geometry.py`에서 diffvg scene → path segment 리스트로 변환 (`_gather_specs`, `_build_segments`).([GitHub][9])
       * `gauss.py`에서 Bézier segment들을 따라가며 **2D Gaussian 샘플 세트(mu, sigma_x, sigma_y, theta, color, opacity)**로 변환.([GitHub][10])
       * `triton_splat.py` + `splat/triton/*` 에서 Triton 커널 연동 (타일링된 2D Gaussian compositing, backward).([GitHub][11])
       * `render_splat.py`가 이 모든걸 감싸서 **splat 전용 `SplatRenderFunction` forward/backward** + fallback 시 baseline으로 위임.([GitHub][12])

   * `optimize/*`

     * SVG 기반 최적화 경로를 high-level API로 제공:

       * `OptimizableSvg` (SVG 파서 + scene builder + per-primitive optimizer)([GitHub][13])
       * `SvgOptimizationDriver(optimize())` : 간단한 루프 드라이버.([GitHub][14])
       * `SvgOptimizationSettings` : lr / 어떤 파라미터를 최적화할지 등에 대한 설정.([GitHub][15])
     * `optimize_svg.py`는 legacy shim.

   * 기타: `parse_svg.py`, `image.py`(의존성 적은 PNG writer), `color.py`, `pixel_filter.py` 등.

3. **예제 / 앱 (`apps/`)**

   * 원본 diffvg의 예제들을 대부분 보존하고, 당신의 워크플로우에 맞춰 일부 수정된 것으로 보임 (예: backend 선택 사용 등).

### 1.2 Bézier Splatting backend의 제약

`splat` 백엔드는 속도 대신 **지원하는 scene 형태에 제한**이 있음:([GitHub][9])

* **지원하는 것**

  * `Path` 기반 open curve stroke (is_closed=False, stroke만 있는 경우)
  * 일정 두께의 stroke (`stroke_width`는 scalar 텐서 1개)
  * closed path + fill (단, stroke는 아직 제한 있음)
  * constant RGBA 색상만 (gradient, texture X)
  * `shape_to_canvas == I` (transform 미지원)

* **지원하지 않는 것 (→ baseline으로 fallback)**

  * circle/ellipse/rect 등 non-path primitive
  * per-point thickness (Path.thickness 텐서)
  * 거리 근사용 path (`use_distance_approx=True`)
  * transform이 들어간 ShapeGroup
  * background compositing, SDF query (현재는 color only)([GitHub][12])

**Pre-conditioning에서 생성할 Path는 이 제약을 만족시키는 쪽으로 설계하는 게 좋음**. 그래야 splat backend를 그대로 활용해서 속도 이득을 받음.

---

## 2. 문제 재정의: 왜 pre-conditioning이 필요한지

당신의 목표:

> raster 사진 (가족사진/프로필/배경 등) → “스케치 느낌”의 벡터 그래픽 → g-code → pen plotter 출력

여기서 diffvg(+Bézier splatting)는 “**벡터 파라미터를 gradient descent로 튜닝하는 마지막 stage**”야.
그런데 지금은 보통:

* 무작위/균일 그리드의 Path 초기화 →
* 수백 iteration 동안 **“대충 edge 근처로 모이는 작업”**에 GPU 시간을 다 씀.

문제 포인트:

* splat backend로 per-iteration cost는 많이 줄였지만,
  **iteration 수(500 iter 등)는 여전히 크고, 초기 수십~수백 iter이 “구조 잡기”에만 쓰이는 느낌**.
* 실제로 필요한 건:

  * 초기엔 **대충 괜찮은 stroke 배치** (정확도 < 속도)
  * 이후 diffvg/splat으로 **미세 조정(fine-tuning)**

즉, 우리가 원하는 pre-conditioning 단계는:

> “저렴한 CPU 이미지 처리로 topology/geometry의 뼈대를 미리 잡아서,
> diffvg가 geometry를 끌어오는 구간(예: 0~300 iter)을 50~100 iter로 줄이는 것”

---

## 3. 저연산 raster → stroke line 알고리즘 후보들 정리

여기까지를 기반으로, 실제로 쓸만한 **CPU pre-conditioning 후보**를 정리해보면:

### 3.1 XDoG + 벡터화 (Potrace/직접 구현)

* **XDoG (Extended Difference of Gaussians)**

  * 고전적인 DoG edge filter를 예술적 라인드로잉 스타일로 확장한 필터.([GitHub][16])
  * 장점

    * pure convolution + pointwise 연산 → OpenCV/NumPy로 **매우 빠름 (HD도 수십 ms 수준)**.
    * 파라미터 (`sigma, k, gamma, epsilon, phi`) 로 라인 두께/노이즈/contrast 조절 용이.
  * 출력: **고대비(edge 중심) 흑백 이미지**.

* **XDoG → 이진 edge map**

  * thresholding + morphology (open/close)로 “깨끗한” binary edge 만들어 Potrace나 skeletonization에 사용.

* **벡터화**

  * 옵션 A: Potrace / pypotrace 등 사용하여 closed path polygon + cubic Bézier로 벡터화.([GitHub][17])
  * 옵션 B: skeletonization 후 polyline 추출 (3.3 참고).

* 장점:

  * 구현 난이도 낮고 속도 빠르며, topology를 잘 보존.
  * diffvg의 Path 초기화에 바로 먹이기 좋음.

* 단점:

  * 에지 흐름(flow)을 고려하지 않기 때문에 **stroke coherence**는 DrawingBot 계열보다는 떨어질 수 있음.

### 3.2 Coherent Line Drawing (ETF + FDoG)

* Kang et al. 의 **Coherent Line Drawing** / Flow-based DoG (FDoG)는
  gradient 기반 **Edge Tangent Flow(ETF)** vector field를 만든 뒤, 그 방향을 따라 anisotropic DoG를 돌려 **길고 부드러운 라인을 얻는 알고리즘**.([University of Missouri–St. Louis][18])

* 장점:

  * 끊어진 edge들을 ETF smoothing으로 잘 이어서, **길고 coherent한 stroke** 생성.
  * noise에 강하고 “일러스트 느낌”으로 안티-클러터.

* 단점:

  * 구현이 XDoG보다는 복잡하고, ETF 반복 smoothing 과정 때문에 CPU 비용이 더 큼.
  * 하지만 여전히 “GPU 수백 iteration”보다는 훨씬 싸다.

* practical note:

  * XDoG vs FDoG 중 하나만 택한다기보다,

    * **빠른 1차: XDoG**,
    * 필요하면 **고품질 모드: ETF+FDoG** 정도로 모드 나눌 수 있음.

### 3.3 skeletonization (thinning) 기반 stroke centerline 추출

XDoG / FDoG가 만들어 준 binary edge를 **1-pixel wide centerline** 으로 줄이는 단계.

* 후보:

  * `skimage.morphology.skeletonize` / `thin` API  (Lee 94, 등)([Scikit-Image][19])
  * Zhang-Suen thinning 알고리즘의 Python 구현들([GitHub][17])

* 역할:

  * edge “밴드”를 그래프 구조로 줄여서, **polyline 추출의 준비 단계**.
  * 출력은 boolean 2D 배열; 각 픽셀은 노드, 8-neighborhood로 edge 연결.

### 3.4 DrawingBotV3 스타일 Path Finding (Sketch Lines PFM 참고)

DrawingBotV3의 Free Path Finding Modules 중 **“Sketch Lines”**가 바로 당신이 말한 CPU만으로도 꽤 빠른 stroke generator.([Drawing Bot V3 Documentation][20])

문서만 보면 핵심 아이디어는:

* “**Lightened/Working image**” 개념:

  * path가 지나간 곳은 밝게 만들거나 마스킹해서, **이미 그려진 곳을 피하면서** 다음 stroke를 찾음.
* Seed:

  * 가장 어두운 픽셀, edge map, sobel map 등에서 seed를 선택.
* Local step:

  * 주변 몇 픽셀 중에서 **gradient or variance가 낮은 방향**/edge 방향을 따라 이동.
  * Canny/Sobel edge power, Luminance power, Directionality 등 가중치 조합으로 방향 결정.([Drawing Bot V3 Documentation][21])

이걸 그대로 베끼지는 않아도, **비슷한 greedy path growing** 을 Python으로 적당히 구현하면:

* XDoG/FDoG로 나온 edge map + gradient 방향 field(또는 ETF)를 이용해
* “가장 어두운 곳에서 시작 → edge 방향으로 일정 길이까지 걷기 → 지나간 곳은 마스킹”

를 반복해서, 꽤 괜찮은 polyline stroke 세트를 얻을 수 있음.

장점:

* **극도로 빠름** (Canny+greedy stepping 수준)
* 긴 coherent stroke들이 잘 나옴 (pen-plotter friendly)

단점:

* 순차 알고리즘이라 완전 SIMD는 아니지만, Python에서도 큰 병목은 아님 (중요한 길이만 추출하면 됨).

### 3.5 polyline → Bézier Path fitting

위까지로 polyline 세트를 얻으면, 이제 diffvg가 먹을 수 있는 `Path(num_control_points, points, ...)`로 바꿔야 함.

* 1단계: polyline simplification

  * Ramer–Douglas–Peucker (RDP) 알고리즘 등으로 points 수를 줄이기.
  * tolerance를 키우면 곡선이 더 부드러워지고 segment 수가 줄어 듦.

* 2단계: Bézier fitting

  * 완전한 least-squares Bézier fitting (Schneider 1990 등)을 구현하면 좋지만,
    pre-conditioning 용도로는:

    * segment마다 `num_control_points`를 0(직선) 또는 2(cubic)으로만 두고,
    * “curvature 큰 구간”만 cubic, 나머진 straight line으로 처리해도 충분할 것.
  * cubic 세그먼트의 초기 control point는

    * 예: P0, P1, P2, P3 네 점일 때

      * C1 = P0 + α(P1−P0)
      * C2 = P3 + β(P2−P3)
      * α, β를 1/3 근처 값으로 놓고 대략적인 곡률만 맞춰줌.

* 3단계: diffvg `Path` 구성

  * `num_control_points`: [0, 2, 0, 2, …]
  * `points`: [first_point, cp1, cp2, end, cp1, cp2, end, ...] 형태
    (원래 diffvg 포맷과 동일; segment 간 end/start 공유 주의).([GitHub][4])

이 정도만 해도 초기 path geometry는 “사진의 에지를 대충 따라가는 stroke set”으로 상당히 근접할 거고, 이후 diffvg/splat이 control point, stroke_width, 색깔만 fine-tuning 하면 됨.

### 3.6 NN 기반 outline/edge: TEED / Anyline(MistoLine)

XDoG/Canny 같은 전통적 edge detector 외에, **NN 기반 edge/outline detector**를 preconditioning의 `edge_mask` 생성기로 사용하는 선택지도 있음.

* **TEED**: 파라미터 수가 매우 작은(58K 수준) CNN edge detector로, 사진에서 “사람 관점의 윤곽선”이 더 안정적으로 잡히는 경우가 많음.([TEED paper][22])
* **Anyline(MistoLine)**: SDXL 생태계에서 유통되는 프리프로세서로, TEED 계열 edge 결과 + lineart intensity 기반 후처리를 결합해 outline을 보강하는 형태가 흔함.([ComfyUI-Anyline][23])

도입 관점에서는 기존 파이프라인(XDoG → skeleton → polyline → Path)에서 **XDoG를 TEED/Anyline으로 교체**하는 수준으로 시작할 수 있고,
구현/라이선스/가중치/의존성 이슈는 별도 문서에 정리해두었음:

* `docs/teed_anyline.md`

---

## 4. hemistone/diffvg에 끼워 넣을 pre-conditioning 설계 (Python 중심)

이제 실제로 **당신 레포 구조를 기준으로** pre-conditioning을 어떻게 붙일지 설계해볼게.

### 4.1 모듈 구조 제안

`pydiffvg` 안에 아래처럼 새 서브패키지를 두는 걸 추천:

```text
pydiffvg/
  precondition/
    __init__.py
    config.py
    xdog.py
    edges.py
    skeleton.py
    vectorize.py
    init_paths.py
```

역할:

* `config.py`: Preconditioning 설정값 (dataclass)
* `xdog.py`: XDoG / FDoG 등 edge 필터
* `edges.py`: thresholding, morphology, edge mask 클린업
* `skeleton.py`: skeletonization + graph traversal → polylines
* `vectorize.py`: polyline simplification + Bézier fitting
* `init_paths.py`: 최종적으로 `List[Path]`, `List[ShapeGroup]` 만들어주는 high-level API

### 4.2 설정 객체

```python
# pydiffvg/precondition/config.py
from dataclasses import dataclass

@dataclass
class PreconditionConfig:
    # XDoG/FDoG 관련
    use_fdog: bool = False
    sigma: float = 1.0
    k: float = 1.6
    gamma: float = 0.98
    epsilon: float = 0.1
    phi: float = 10.0

    # edge post-processing
    min_component_area: int = 32
    morph_open_radius: int = 1
    morph_close_radius: int = 1

    # skeleton & polyline
    skeleton_method: str = "skimage"  # or "zhang_suen"
    max_path_length: int = 1024   # pixel steps
    min_path_length: int = 16
    simplify_epsilon: float = 1.5

    # path sampling
    max_paths: int = 2000
    path_sort: str = "darkest_first"  # or "longest_first"

    # stroke & color
    base_stroke_width: float = 1.5
    max_stroke_width: float = 3.0
    stroke_width_gamma: float = 1.5  # dark 영역에서 더 굵게
    sample_color: bool = True
```

### 4.3 XDoG edge map 생성

```python
# pydiffvg/precondition/xdog.py
import cv2
import numpy as np
from .config import PreconditionConfig

def xdog_edges(gray: np.ndarray, cfg: PreconditionConfig) -> np.ndarray:
    """
    gray: [H, W], float32, [0, 1]
    return: uint8 binary image (0 or 255), edges=255
    """
    sigma, k, gamma = cfg.sigma, cfg.k, cfg.gamma
    eps, phi = cfg.epsilon, cfg.phi

    g1 = cv2.GaussianBlur(gray, (0, 0), sigma)
    g2 = cv2.GaussianBlur(gray, (0, 0), sigma * k)

    dog = g1 - gamma * g2

    # Winnemöller XDoG식에 맞추되, preconditioning 용으로는
    # 간단한 thresholding + sign flip 조합으로 충분
    edge = np.zeros_like(dog, dtype=np.uint8)
    edge[dog < eps] = 255

    # 작고 잡음성 component 제거 및 morphological 정리
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                       (2 * cfg.morph_open_radius + 1,
                                        2 * cfg.morph_open_radius + 1))
    edge = cv2.morphologyEx(edge, cv2.MORPH_OPEN, kernel)
    if cfg.morph_close_radius > 0:
        kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                            (2 * cfg.morph_close_radius + 1,
                                             2 * cfg.morph_close_radius + 1))
        edge = cv2.morphologyEx(edge, cv2.MORPH_CLOSE, kernel2)

    return edge
```

(ETF/FDoG를 쓰고 싶다면, `gauss-sketch-book` 같은 구현을 참고해 flow 기반으로 확장 가능.([GitHub][16]))

### 4.4 skeletonization + polyline 추출

```python
# pydiffvg/precondition/skeleton.py
import numpy as np
from skimage.morphology import skeletonize
from collections import deque
from .config import PreconditionConfig

NEIGHBORS8 = [(-1,-1), (-1,0), (-1,1),
              ( 0,-1),         ( 0,1),
              ( 1,-1), ( 1,0), ( 1,1)]

def binary_skeleton(edge_u8: np.ndarray, cfg: PreconditionConfig) -> np.ndarray:
    # edge_u8: 0/255
    binary = (edge_u8 > 0).astype(bool)
    skel = skeletonize(binary)  # 1px-wide centerline:contentReference[oaicite:28]{index=28}
    return skel

def _neighbors(skel: np.ndarray, y: int, x: int):
    H, W = skel.shape
    for dy, dx in NEIGHBORS8:
        ny, nx = y + dy, x + dx
        if 0 <= ny < H and 0 <= nx < W and skel[ny, nx]:
            yield ny, nx

def skeleton_to_polylines(skel: np.ndarray,
                          cfg: PreconditionConfig) -> list[list[tuple[int,int]]]:
    """
    skel: bool [H, W]
    return: list of polyline as [(y,x), ...]
    """
    H, W = skel.shape
    visited = np.zeros_like(skel, dtype=bool)
    polylines: list[list[tuple[int,int]]] = []

    # degree 계산
    degree = np.zeros_like(skel, dtype=np.uint8)
    ys, xs = np.nonzero(skel)
    for y, x in zip(ys, xs):
        d = 0
        for _ in _neighbors(skel, y, x):
            d += 1
        degree[y, x] = d

    # endpoint(=degree 1)에서 시작
    endpoints = [(y, x) for y, x in zip(ys, xs) if degree[y, x] == 1]

    def trace_from(start_y, start_x):
        path = [(start_y, start_x)]
        visited[start_y, start_x] = True
        y, x = start_y, start_x
        while len(path) < cfg.max_path_length:
            candidates = [(ny, nx) for ny, nx in _neighbors(skel, y, x)
                          if not visited[ny, nx]]
            if not candidates:
                break
            if len(candidates) == 1:
                ny, nx = candidates[0]
            else:
                # 여러 후보면, degree==2 우선으로 (junction은 나중에)
                c2 = [c for c in candidates if degree[c] == 2]
                ny, nx = (c2[0] if c2 else candidates[0])
            path.append((ny, nx))
            visited[ny, nx] = True
            y, x = ny, nx
        return path

    # 1) endpoint 기반 path
    for y, x in endpoints:
        if visited[y, x]:
            continue
        p = trace_from(y, x)
        if len(p) >= cfg.min_path_length:
            polylines.append(p)

    # 2) 순환 루프 (모든 점 degree==2)도 처리
    for y, x in zip(ys, xs):
        if visited[y, x] or degree[y, x] != 2:
            continue
        loop = trace_from(y, x)
        if len(loop) >= cfg.min_path_length:
            polylines.append(loop)

    return polylines
```

여기까지가 “XDoG → binary edge → skeleton → polyline” 부분.

### 4.5 polyline simplification + Path 생성

```python
# pydiffvg/precondition/vectorize.py
import numpy as np
import torch
import pydiffvg
from .config import PreconditionConfig

def _rdp(points: np.ndarray, epsilon: float) -> np.ndarray:
    """
    단순 RDP 구현 (points: [N, 2]).
    """
    if points.shape[0] < 3:
        return points

    # line P0-Pn에서 각 점까지의 거리
    P0, Pn = points[0], points[-1]
    v = Pn - P0
    v_norm = np.linalg.norm(v) + 1e-8
    distances = np.abs(np.cross(v, points - P0) / v_norm)

    idx = np.argmax(distances)
    dmax = distances[idx]

    if dmax < epsilon:
        return np.vstack([P0, Pn])
    left = _rdp(points[:idx+1], epsilon)
    right = _rdp(points[idx:], epsilon)
    return np.vstack([left[:-1], right])

def polylines_to_paths(polylines: list[list[tuple[int,int]]],
                       image_rgb: np.ndarray,
                       cfg: PreconditionConfig,
                       canvas_w: int,
                       canvas_h: int,
                       device: torch.device) -> tuple[list[pydiffvg.Path], list[pydiffvg.ShapeGroup]]:
    """
    skeleton-based polylines → diffvg Path / ShapeGroup 리스트로 변환
    """
    H, W, _ = image_rgb.shape
    gray = image_rgb.mean(axis=2)

    # path 우선순위: 어두운 영역+길이
    scored = []
    for poly in polylines:
        pts = np.array([[x, y] for (y, x) in poly], dtype=np.float32)
        ys = np.clip(pts[:, 1].astype(int), 0, H-1)
        xs = np.clip(pts[:, 0].astype(int), 0, W-1)
        luminance = gray[ys, xs]
        # “어두울수록” score 높게
        score = (1.0 - luminance.mean()) * len(poly)
        scored.append((score, poly))

    scored.sort(key=lambda t: t[0], reverse=True)
    if cfg.max_paths is not None:
        scored = scored[:cfg.max_paths]

    shapes: list[pydiffvg.Path] = []
    groups: list[pydiffvg.ShapeGroup] = []

    for idx, (_, poly) in enumerate(scored):
        pts = np.array([[x, y] for (y, x) in poly], dtype=np.float32)
        # 이미지 좌표(픽셀)를 diffvg canvas 좌표로 정규화 (원하면 scale)
        # 여기서는 1:1 매핑으로 둠
        pts = _rdp(pts, cfg.simplify_epsilon)

        if pts.shape[0] < 2:
            continue

        num_control_points = []
        out_points = []

        # 간단하게: 전부 straight line segment로만 구성
        out_points.append(pts[0])
        for i in range(1, pts.shape[0]):
            out_points.append(pts[i])
            num_control_points.append(0)  # 각 segment는 직선

        points_tensor = torch.tensor(out_points, device=device, dtype=torch.float32)
        ncp_tensor = torch.tensor(num_control_points, device=device, dtype=torch.int32)

        # stroke width: 해당 path가 지나가는 luminance 평균으로 결정
        ys = np.clip(pts[:, 1].astype(int), 0, H-1)
        xs = np.clip(pts[:, 0].astype(int), 0, W-1)
        lum_mean = gray[ys, xs].mean()
        darkness = 1.0 - lum_mean
        width = cfg.base_stroke_width + (cfg.max_stroke_width - cfg.base_stroke_width) * (darkness ** cfg.stroke_width_gamma)
        width_tensor = torch.tensor(width, device=device)

        path = pydiffvg.Path(
            num_control_points=ncp_tensor,
            points=points_tensor,
            is_closed=False,
            stroke_width=width_tensor,
            id=f"pre_skel_{idx}",
            use_distance_approx=False,
        )
        shapes.append(path)

        # color 샘플링
        if cfg.sample_color:
            rgb = image_rgb[ys, xs].mean(axis=0) / 255.0
            stroke_color = torch.tensor([rgb[0], rgb[1], rgb[2], 1.0],
                                        dtype=torch.float32, device=device)
        else:
            stroke_color = torch.tensor([0.0, 0.0, 0.0, 1.0],
                                        dtype=torch.float32, device=device)

        group = pydiffvg.ShapeGroup(
            shape_ids=torch.tensor([len(shapes)-1], device=device, dtype=torch.int32),
            fill_color=None,
            stroke_color=stroke_color,
            shape_to_canvas=torch.eye(3, device=device),
            id=f"pre_skel_group_{idx}",
        )
        groups.append(group)

    return shapes, groups
```

여기서는 Bézier 세그먼트가 아니라 전부 straight line만 썼는데, 이 상태만으로도 **diffvg/splat으로 이어서 최적화하기에는 충분한 topology 초기화**가 된다.
추후엔 curvature 기준으로 cubic segment를 추가해서 splatting quality를 더 끌어올릴 수 있음.

### 4.6 high-level: 이미지 → 초기 diffvg Scene

```python
# pydiffvg/precondition/init_paths.py
import cv2
import numpy as np
import torch
import pydiffvg

from .config import PreconditionConfig
from .xdog import xdog_edges
from .skeleton import binary_skeleton, skeleton_to_polylines
from .vectorize import polylines_to_paths

def build_preconditioned_scene(image_path: str,
                               cfg: PreconditionConfig,
                               device: torch.device | None = None):
    """
    1) 이미지 로드
    2) XDoG edge
    3) skeleton → polyline
    4) polyline → pydiffvg.Path / ShapeGroup
    5) diffvg scene args serialize
    """
    if device is None:
        device = pydiffvg.get_device()

    bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(image_path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb_f = rgb.astype(np.float32) / 255.0
    gray = rgb_f.mean(axis=2)

    edge = xdog_edges(gray, cfg)
    skel = binary_skeleton(edge, cfg)
    polylines = skeleton_to_polylines(skel, cfg)

    H, W, _ = rgb.shape
    shapes, groups = polylines_to_paths(polylines, rgb, cfg, W, H, device)

    # splat backend 사용 준비 (device-resident serialize)
    renderer = pydiffvg.Renderer(backend="splat")
    scene_args = renderer.serialize_scene(
        W, H, shapes, groups,
        keep_on_device=True,
        device=device,
    )

    return {
        "width": W,
        "height": H,
        "shapes": shapes,
        "shape_groups": groups,
        "scene_args": scene_args,
        "renderer": renderer,
    }
```

---

## 5. diffvg/splat 최적화 루프에의 통합

### 5.1 기본 루프 구조

```python
import torch
import pydiffvg
from pydiffvg.precondition import PreconditionConfig, build_preconditioned_scene

def optimize_from_precondition(image_path: str,
                               iterations: int = 200,
                               backend: str = "splat"):
    cfg = PreconditionConfig()
    pydiffvg.set_backend(backend)

    device = pydiffvg.get_device()
    target = ...  # target image를 torch.Tensor [H,W,4]로 로드 (알파 1.0)

    scene = build_preconditioned_scene(image_path, cfg, device=device)
    W, H = scene["width"], scene["height"]
    shapes, groups = scene["shapes"], scene["shape_groups"]
    renderer = scene["renderer"]

    # 최적화 대상 파라미터 설정
    params = []
    for p in shapes:
        p.points.requires_grad_(True)
        params.append(p.points)
        p.stroke_width.requires_grad_(True)
        params.append(p.stroke_width)
    for g in groups:
        if isinstance(g.stroke_color, torch.Tensor):
            g.stroke_color.requires_grad_(True)
            params.append(g.stroke_color)

    opt = torch.optim.Adam(params, lr=1e-1)

    for t in range(iterations):
        opt.zero_grad()

        scene_args = renderer.serialize_scene(
            W, H, shapes, groups,
            keep_on_device=True,
            device=device,
            cache_key="main",  # renderer 캐시 활용
            invalidate_cache=(t == 0),
        )

        img = renderer.apply(
            W, H,
            2, 2,        # num_samples_x, num_samples_y
            t,           # seed
            None,        # background_image
            *scene_args
        )

        # target과의 L2 / perceptual loss 등
        loss = ((img[..., :3] - target[..., :3]) ** 2).mean()

        loss.backward()
        opt.step()

        if t % 10 == 0:
            print(f"[{t}/{iterations}] loss={loss.item():.4f}")

    return shapes, groups
```

이 루프는 당신이 지금 쓰고 있는 diffvg 최적화 루프와 거의 동일하지만,

* **초기 shapes/groups가 pre-conditioning 결과**라는 점이 다름.
* backend를 `"splat"`로 두면, `render_splat`의 Triton 기반 Gaussian splatting이 사용됨.([GitHub][12])

### 5.2 성능 기대치 (개략적인 추론)

정확한 수치는 직접 벤치마크해야 하지만, 보통:

* **XDoG + skeleton + polyline 추출**

  * CPU에서 1024×1024 기준 수십 ~ 100ms 안에 들어올 가능성이 큼 (cv2 + skimage 기준).

* 기존:

  * random/grid init + diffvg만으로 500 iter 정도 필요했다면,

* 제안 파이프라인:

  * pre-conditioning 100ms + diffvg/splat 150~200 iter 정도로 수렴 가능 →
    전체 wall-clock이 **x2~x5 이상 줄어드는 것**을 목표로 삼을 수 있음.

---

## 6. 코드 에이전트용 TODO 체크리스트

코드 에이전트에 바로 넘기기 좋은 형태로, 해야 할 작업을 요약해볼게.

### 6.1 pydiffvg 확장

1. `pydiffvg/precondition` 패키지 생성
2. `config.py`에 `PreconditionConfig` dataclass 정의
3. `xdog.py` 구현

   * 입력: float32 gray image [H,W]
   * 출력: uint8 binary edge [H,W] (0/255)
4. `skeleton.py`

   * `binary_skeleton(edge_u8, cfg)` → bool skeleton
   * `skeleton_to_polylines(skel, cfg)` → list[list[(y,x)]]
5. `vectorize.py`

   * `_rdp(points, epsilon)` (Douglas–Peucker)
   * `polylines_to_paths(polylines, image_rgb, cfg, canvas_w, canvas_h, device)`
     → `List[pydiffvg.Path]`, `List[pydiffvg.ShapeGroup]`
   * splat backend 제약 준수:

     * Path only
     * stroke_width scalar
     * shape_to_canvas == I
     * fill_color=None, stroke_color constant RGBA
6. `init_paths.py`

   * `build_preconditioned_scene(image_path, cfg, device=None)`
     → dict(width, height, shapes, shape_groups, scene_args, renderer)

### 6.2 예제 / 벤치마크

1. `apps/precondition_vectorize.py` (새 스크립트)

   * 입력: 이미지 경로
   * 옵션: `--backend baseline|splat`, `--iters`, `--cfg_json` 등
   * 두 가지 모드:

     * a) 기존 random init 대비 pre-conditioning init 비교
     * b) baseline vs splat backend 비교

2. PSNR / SSIM 측정 및 로그 출력으로 iteration별 수렴 곡선 비교.

### 6.3 향후 확장 아이디어 (선택)

* ETF + FDoG 기반 **coherent line drawing 모드** 추가 (고품질 옵션).([University of Missouri–St. Louis][18])
* TEED / Anyline(MistoLine) 기반 NN edge/outline을 `edge_mask` 생성기로 도입 (윤곽선 품질 개선, iteration 감소 기대): `docs/teed_anyline.md`
* DrawingBotV3-style **PFM-like path grower** 구현:

  * working/luminance map 사용
  * seed selection (darkest / edge / sobel)
  * step 방향을 gradient/ETF/edge power 기반으로 선택
* splat backend의 `DepthPolicy.small_first` 옵션을 pre-conditioning path 길이나 stroke width 기반으로 자동 설정해서,
  겹치는 stroke의 painterly ordering 제어.([GitHub][6])

---

요약하면,

* 지금 hemistone/diffvg는 **backend abstraction + Bézier splatting + Triton 커널**까지 이미 잘 깔려 있고,
* 여기에 **XDoG → skeleton → polyline → Path** 파이프라인을 얹으면

  * “대충 topology를 잘 잡아주는 저렴한 CPU 전처리”를 만들 수 있음.
* 이 구조를 쓰면,

  * originally 500 iter needed → precondition 후 150~200 iter 정도로 줄이는 걸 현실적인 목표로 삼을 수 있고,
  * splat backend와도 제약을 맞추기 쉬움.

에이전트에게는 위의 모듈 구조 + 함수 시그니처를 그대로 넘겨서, 단계별로 구현하게 하면 될 것 같아.

[1]: https://github.com/BachiLi/diffvg/raw/master/pydiffvg/__init__.py "raw.githubusercontent.com"
[2]: https://github.com/Hemistone/diffvg/raw/master/pydiffvg/render_pytorch.py "raw.githubusercontent.com"
[3]: https://github.com/Hemistone/diffvg/raw/master/pydiffvg/device.py "raw.githubusercontent.com"
[4]: https://github.com/Hemistone/diffvg/raw/master/pydiffvg/shape.py "raw.githubusercontent.com"
[5]: https://github.com/Hemistone/diffvg/raw/master/pydiffvg/serialization.py "raw.githubusercontent.com"
[6]: https://github.com/Hemistone/diffvg/raw/master/pydiffvg/backend.py "raw.githubusercontent.com"
[7]: https://github.com/Hemistone/diffvg/raw/master/pydiffvg/renderer.py "raw.githubusercontent.com"
[8]: https://github.com/Hemistone/diffvg/raw/master/pydiffvg/splat/types.py "raw.githubusercontent.com"
[9]: https://github.com/Hemistone/diffvg/raw/master/pydiffvg/splat/geometry.py "raw.githubusercontent.com"
[10]: https://github.com/Hemistone/diffvg/raw/master/pydiffvg/splat/gauss.py "raw.githubusercontent.com"
[11]: https://github.com/Hemistone/diffvg/raw/master/pydiffvg/triton_splat.py "raw.githubusercontent.com"
[12]: https://github.com/Hemistone/diffvg/raw/master/pydiffvg/render_splat.py "raw.githubusercontent.com"
[13]: https://github.com/Hemistone/diffvg/raw/master/pydiffvg/optimize/core.py "raw.githubusercontent.com"
[14]: https://github.com/Hemistone/diffvg/raw/master/pydiffvg/optimize/driver.py "raw.githubusercontent.com"
[15]: https://github.com/Hemistone/diffvg/raw/master/pydiffvg/optimize/settings.py "raw.githubusercontent.com"
[16]: https://github.com/Shorakie/gauss-sketch-book?utm_source=chatgpt.com "Shorakie/gauss-sketch-book - GitHub"
[17]: https://github.com/linbojin/Skeletonization-by-Zhang-Suen-Thinning-Algorithm?utm_source=chatgpt.com "GitHub - linbojin/Skeletonization-by-Zhang-Suen-Thinning-Algorithm ..."
[18]: https://www.umsl.edu/~kangh/Papers/kang_npar07_hi.pdf?utm_source=chatgpt.com "Coherent Line Drawing"
[19]: https://scikit-image.org/docs/stable/auto_examples/edges/plot_skeleton.html?utm_source=chatgpt.com "Skeletonize — skimage 0.25.2 documentation - scikit-image"
[20]: https://docs.drawingbotv3.com/en/latest/about.html?utm_source=chatgpt.com "Drawing Bot V3 — Drawing Bot V3 1.6.10 documentation"
[21]: https://docs.drawingbotv3.com/en/latest/pfms.html?utm_source=chatgpt.com "5. Path Finding Modules — Drawing Bot V3 1.6.10 documentation"
[22]: https://arxiv.org/abs/2308.06468 "Tiny and Efficient Model for the Edge Detection Generalization (TEED)"
[23]: https://github.com/TheMistoAI/ComfyUI-Anyline "ComfyUI-Anyline"

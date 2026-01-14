## TEED / Anyline(MistoLine) 기반 preconditioning 조사 노트

이 문서는 DiffVG preconditioning(초기 stroke 배치) 품질을 높이기 위해, 전통적 edge detector(XDoG/Canny 등) 대신 **NN 기반 edge/outline 추출기**를 도입하는 방안으로서 **TEED**와 SDXL 생태계의 **Anyline(MistoLine)**을 조사한 기록이다.

핵심 요약:

* **TEED**는 58K 파라미터 수준의 매우 작은 CNN edge detector로, 일반적인 사진에서 “사람이 기대하는 윤곽선”을 비교적 깔끔하게 뽑아내는 편이다.
* SDXL/ComfyUI에서 말하는 **Anyline**은 보통 “TEED 계열(edge) + lineart intensity 기반 보강”을 합친 **프리프로세서 파이프라인**이며, 실제 배포 가중치로는 `MTEED.pth`가 많이 쓰인다.
* 우리 프로젝트 관점에서는 “preconditioning의 `edge_mask` 생성기만 교체”하면 되므로, 구조적으로 도입이 쉽다.
* 다만, “어떤 가중치를 기본으로 지원할지(라이선스)”와 “splat backend 제약(폐곡선 stroke)”을 같이 고려해야 한다.

---

## 1. TEED란?

TEED(Tiny and Efficient Edge Detector)는 edge detection의 단순성/효율/일반화(generalization)를 목표로 한 소형 CNN이다.[1]

* 파라미터 수가 매우 작아(논문/레포 기준 58K 수준) CPU/GPU 모두에서 inference 비용이 낮은 편
* BIPED 등에서 학습된 가중치로 crisp edge-map을 생성
* ControlNet annotator 생태계에서도 “softedge/lineart” 계열의 대체로 널리 사용됨

### 1.1 구현체 관찰 (ControlNet-aux 계열)

ControlNet annotator를 PyPI로 포장한 `controlnet-aux` 구현을 기준으로 보면, TEED detector의 forward 흐름은 대략 아래 형태다.

1. 입력 이미지를 `detect_resolution` 기준으로 리사이즈(최소 변 기준, 64 배수로 반올림)
2. 모델이 multi-scale logits(여러 장의 1ch 맵)를 출력
3. 각 맵을 입력 해상도로 리사이즈 후 stack → 평균
4. `sigmoid` 후 필요하면 계단화(`safe_step`) 적용
5. 0~255 uint8 edge-map으로 변환

이 결과는 “연속적인 edge strength(probability-like)”에 가까워서, 우리 쪽에서는 후속 단계(skeletonization)를 위해 **bool edge_mask로 thresholding**을 추가로 수행해야 한다.

---

## 2. Anyline(MistoLine)이란?

SDXL/ComfyUI에서 말하는 Anyline은 “TEED 계열로 뽑은 윤곽선 + 추가적인 라인 강화”를 결합한 프리프로세서로 유통되는 경우가 많다.[2]

대표 구현(ComfyUI-Anyline / controlnet-aux 계열)에서 확인되는 구성은 아래와 같다.

1. **MTEED 결과(TEED 계열 edge)**:
   * TEED와 동일한 네트워크 구조(TED)를 사용하되, 별도 가중치(`MTEED.pth`)를 로드해서 edge-map을 생성
2. **lineart intensity 결과(강도 기반 라인 보강)**:
   * 입력 RGB에서 가우시안 블러된 이미지와 원본 차이를 사용해 “어두운 선”을 강조하는 intensity map 생성
   * small object 제거 등으로 잡음 정리
3. 두 레이어를 마스킹 기반으로 합성(screen 유사)하여 최종 결과를 생성

즉, Anyline은 “단일 네트워크”라기보다 **edge 네트워크 출력 + 전통적 후처리 조합**으로 보는 편이 안전하다.

---

## 3. 라이선스/가중치 이슈 (도입 시 가장 중요)

### 3.1 TEED (권장 기본)

* TEED 원본 레포: MIT 라이선스[3]
* `fal-ai/teed` 등 일부 배포 가중치도 MIT로 표기되는 경우가 있음(배포처의 모델 카드/라이선스 표기 확인 필요)

따라서 “TEED 코드 + TEED(MIT) 가중치” 조합은 **프로젝트 기본 옵션**으로 두기 상대적으로 부담이 적다.

### 3.2 MistoLine / Anyline(MTEED.pth) (옵션 권장)

* `TheMistoAI/MistoLine`의 `MTEED.pth`는 **CreativeML Open RAIL++-M(OpenRAIL++)** 계열 라이선스로 배포된다.[4]
* 이 라이선스는 “오픈”이긴 하지만 용도 제한(Use-based restrictions)이 포함될 수 있으므로,
  * 레포에 가중치를 포함해 배포하거나,
  * 라이브러리 기본 기능으로 자동 다운로드/자동 사용
  같은 형태는 신중해야 한다.

실무적으로는:

* **사용자가 로컬에 내려받아 둔 가중치 경로를 직접 지정하는 옵션**으로만 지원
* 문서에 라이선스/제한사항을 명시

를 권장한다.

---

## 4. 우리 preconditioning 파이프라인에 어떻게 끼우나?

현재 preconditioning의 핵심 흐름은 아래다.

* `edge_mask` 생성 (현재: XDoG 기반)
* `skeletonize_edges(edge_mask)` → `skeleton_to_polylines(...)`
* `polylines_to_paths(...)`로 diffvg `Path`/`ShapeGroup` 생성

TEED/Anyline 도입은 대부분 “edge_mask 생성기 교체”로 끝난다. 구체적으로:

1. `pydiffvg/precondition/xdog.py:xdog_edges()`와 동급의 함수(예: `teed_edges()` 또는 `anyline_edges()`) 추가
2. `pydiffvg/precondition/init_paths.py:build_preconditioned_scene()`에서 `cfg`에 따라 edge backend 선택
3. TEED/Anyline이 주는 grayscale edge map(0~1 또는 0~255)을 **bool edge_mask로 thresholding** + morphology 정리

### 4.1 splat backend 제약과의 상호작용(중요)

skeleton 기반 vectorize 과정에서 loop가 많아지면 폐곡선 path가 만들어질 수 있는데, splat backend는 “closed path에 stroke”를 지원하지 않는다.

따라서 TEED/Anyline처럼 윤곽이 잘 잡히는 edge detector를 붙이면, 오히려 loop가 늘어나서 splat fallback이 발생할 수 있다.

대응 전략(선택지):

* “splat 사용 시” preconditioning 단계에서 path를 **강제로 open curve로 만들기**(loop cut 또는 `is_closed=False` 강제)
* loop는 fill로만 쓰고 stroke는 open curve만 쓰도록 정책 분리(추가 구현 필요)

현재는 `force_open_paths` 옵션으로 closed stroke를 강제로 열도록 지원하며, splat backend에서는 자동으로 활성화된다.

---

## 5. 구현 전략

### 5.0 추천 흐름(도입 순서)

Anyline을 “바로 통째로” 붙이기보다, 아래 순서로 가는 편이 실험 비용 대비 효율이 좋다.

1. **TEED부터 도입**: preconditioning의 `edge_mask` 생성기만 `xdog` → `teed`로 교체하고, threshold/morphology 및 `detect_resolution`을 튜닝해서 “iteration 감소”가 실제로 발생하는지 먼저 확인한다.
2. **Anyline식 후처리만 추가**: TEED 출력에 intensity 기반 lineart 보강(가우시안 블러 기반) + small-object 제거 + layer combine(screen 유사)을 옵션으로 얹어, “보강이 skeleton/polyline 품질에 도움이 되는지”를 비교한다. (이 단계는 별도 가중치가 필요 없다.)
3. **MTEED(MistoLine) 가중치는 옵션**: 추가 개선이 필요할 때만 `MTEED.pth`를 실험하되, 라이선스(OpenRAIL++) 특성상 “사용자가 로컬 경로로 제공”하는 형태로만 지원하는 것을 권장한다.

### 5.1 최소 의존성(권장)

* TEED 모델(`TED`)을 **순수 PyTorch**로 로드/추론
* 리사이즈는 PIL/torch로 처리 (OpenCV, einops 의존을 피함)
* 가중치는 기본적으로 **로컬 경로 지정**을 요구하고, HuggingFace 다운로드는 optional로만 지원

장점: `pydiffvg/precondition`의 “가벼운 유틸리티” 성격을 유지하기 쉽다.

### 5.2 외부 패키지 기반(빠른 실험용)

* `controlnet-aux` 같은 패키지가 설치되어 있으면 그 detector를 그대로 호출
* Anyline(MTEED)도 동일한 인터페이스로 실험 가능

장점: SD 생태계와 동일 결과를 재현하기 쉽고, 실험이 빠르다.  
단점: 의존성(opencv/einops/hf hub 등)이 커지고, 자동 다운로드까지 엮이면 배포/재현성이 복잡해진다.

---

## 6. 실험 체크리스트

* XDoG vs TEED vs Anyline(MTEED) 간:
  * preconditioning init render의 구조적 유사도(초기 loss)
  * 목표 “iteration 감소”가 실제로 발생하는지(예: 500 → 150~200)
  * path 수/길이 분포가 skeleton graph에 어떻게 영향을 주는지
  * splat fallback 비율(unsupported path 비율)이 늘지는 않는지
* TEED threshold mode:
  * `fixed`: 기존처럼 단일 threshold로 edge mask 생성
  * `hysteresis`: high/low threshold를 사용해 연결된 약한 edge를 보존
  * `quantile`: edge strength 상위 q 비율만 유지
  * `otsu`: Otsu 자동 임계값
  * CLI 예시: `--teed-threshold-mode hysteresis --teed-hysteresis-low-ratio 0.5`
* thresholding 전략:
  * 고정 threshold vs quantile/otsu 기반 threshold
  * morphology 파라미터(min_component_area/open/close)가 skeleton 품질에 미치는 영향

---

## 7. 다음 작업 후보 (현재 상태 기록)

* Anyline 스타일 보강(TEED + lineart intensity 합성): **적용**
  * `teed_lineart` + `teed_lineart_blur_sigma/strength/combine`로 blur+screen/max/add 보강 지원
* Threshold 고급화 확장: **적용**
  * `teed_threshold_mode=quantile|otsu` + `teed_threshold_quantile` 추가
  * quantile preset: `configs/precondition/teed_detail_quantile.toml`
  * otsu preset: `configs/precondition/teed_detail_otsu.toml`
* Skeleton/Polyline 품질 개선: **부분 적용**
  * `merge_polylines` + `merge_distance/merge_angle_deg` 옵션으로 단편 merge
  * `force_open_paths`로 splat 폐곡선 회피
  * dynamic merge/split은 후순위
* path 수 자동 보정: **적용**
  * `precondition_target_paths_min/max` 범위를 기준으로 내부 해상도를 자동 조정(업/다운스케일)하여 path 수를 맞춤
  * 기본 범위는 `PRECONDITION_TARGET_PATHS_MIN_DEFAULT`/`PRECONDITION_TARGET_PATHS_MAX_DEFAULT`를 따르며 필요 시 CLI에서 min/max만 조정
  * preconditioning은 항상 자동 보정 모드로 동작하며 min-side 수동 옵션은 제거됨
* TEED detect_res/threshold/safe_steps 스윕: **미적용**
  * 이미지 유형(건축/인물/자연)별 추천 범위 확보
* 전역 스타일 옵션 확장: **부분 적용**
  * fixed-stroke-width는 추가됨
  * 색 샘플링/모노톤 플래그 강화는 미적용

---

## 8. 고급 라인 모드 방향 (2026-01-13 메모)

* 향후 고도화는 **Flowline(path growing)**과 **ETF/FDoG**에 집중.
* skeleton graph 개선은 시각적 품질 한계 때문에 제외.
* ETF/FDoG는 CPU-only로는 무거울 수 있어, GPU 가속을 전제로 검토하는 쪽이 합리적.

---

[1]: https://arxiv.org/abs/2308.06468 "Tiny and Efficient Model for the Edge Detection Generalization (TEED)"
[2]: https://github.com/TheMistoAI/ComfyUI-Anyline "ComfyUI-Anyline"
[3]: https://github.com/xavysp/TEED/blob/master/LICENSE "TEED LICENSE (MIT)"
[4]: https://huggingface.co/TheMistoAI/MistoLine/blob/main/LICENSE.md "MistoLine LICENSE (CreativeML Open RAIL++-M)"

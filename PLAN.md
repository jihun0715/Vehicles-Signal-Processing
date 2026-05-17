# PLAN: OpenCV Optical Flow 테스트 파이프라인

## 목표

이미지 시퀀스에 대해 1차 optical flow 파이프라인을 구현한다.

파이프라인의 역할은 다음과 같다.

1. `test_images/`에서 이미지 프레임들을 읽는다.
2. `config.py`에서 지정한 OpenCV optical flow method를 선택한다.
3. 연속된 이미지 쌍에 대해 optical flow를 계산한다.
   - `t0 -> t1`
   - `t1 -> t2`
   - `t2 -> t3`
   - ...
4. `test_vision.py`에서 결과를 시각화한다.

이번 단계에서는 optical flow 생성과 시각화까지만 구현한다.
RANSAC homography 추정과 camera motion decomposition은 이후 단계에서 추가한다.

## 대상 파일

```text
Vehicles-Signal-Processing/
├── config.py
├── test_vision.py
├── test_images/
└── vision/
    └── optical_flow.py
```

## Optical Flow Method

`config.py`에서 아래 값으로 사용할 optical flow method를 선택할 수 있게 한다.

```python
OPTICAL_FLOW_METHOD = "dis_medium"
```

지원할 method:

```text
"dis_ultrafast"
"dis_fast"
"dis_medium"
"farneback"
"lucas_kanade"
"tvl1"
```

기본 추천값:

```python
OPTICAL_FLOW_METHOD = "dis_medium"
```

## config.py 계획

`config.py`에는 간단한 전역 설정값을 둔다.

### Pseudocode

```python
TEST_IMAGE_DIR = "test_images"

OPTICAL_FLOW_METHOD = "dis_medium"

RESIZE_WIDTH = None

SHOW_FLOW_STEP = True
FLOW_VIS_SCALE = 1.0
```

설정값 의미:

- `TEST_IMAGE_DIR`: 테스트 이미지 시퀀스 폴더 경로
- `OPTICAL_FLOW_METHOD`: 사용할 optical flow backend
- `RESIZE_WIDTH`: 빠른 테스트를 위한 선택적 resize width
- `SHOW_FLOW_STEP`: 각 이미지 쌍의 결과를 순서대로 보여줄지 여부
- `FLOW_VIS_SCALE`: flow vector 시각화 배율

## optical_flow.py 계획

`vision/optical_flow.py`는 하나의 공통 API를 제공한다.

주요 함수:

```python
compute_optical_flow(prev_img, curr_img, method="dis_medium")
```

반환 형식:

```python
flow_result = {
    "method": method,
    "flow": flow,
    "points_prev": points_prev,
    "points_curr": points_curr,
    "status": status,
}
```

Dense method의 경우:

- `flow`: `H x W x 2` numpy array
- `points_prev`: `None`
- `points_curr`: `None`
- `status`: `None`

Lucas-Kanade의 경우:

- `flow`: `None`
- `points_prev`: 이전 프레임의 tracking point
- `points_curr`: 현재 프레임의 tracking point
- `status`: tracking 성공 여부

이런 반환 구조를 사용하면 이후 RANSAC homography 단계에서 dense flow와 sparse flow를 같은 흐름으로 처리하기 쉽다.

## 공통 전처리

### Grayscale 변환

```python
def to_gray(img):
    if img has 3 channels:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img
```

### 선택적 Resize

```python
def resize_if_needed(img, resize_width):
    if resize_width is None:
        return img

    scale = resize_width / original_width
    resized_height = int(original_height * scale)
    return cv2.resize(img, (resize_width, resized_height))
```

## DIS Optical Flow

DIS는 첫 번째 dense optical flow method로 사용하기에 가장 적합하다.
CPU에서도 빠르고, 품질과 속도의 균형이 좋다.

### Pseudocode

```python
def compute_dis(prev_gray, curr_gray, method):
    if method == "dis_ultrafast":
        cv_preset = cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST
    elif method == "dis_fast":
        cv_preset = cv2.DISOPTICAL_FLOW_PRESET_FAST
    else:
        cv_preset = cv2.DISOPTICAL_FLOW_PRESET_MEDIUM

    optical_flow = cv2.DISOpticalFlow_create(cv_preset)
    flow = optical_flow.calc(prev_gray, curr_gray, None)

    return flow
```

## Farneback Optical Flow

Farneback은 구현이 단순한 dense optical flow baseline으로 사용한다.

### Pseudocode

```python
def compute_farneback(prev_gray, curr_gray):
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray,
        curr_gray,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0,
    )

    return flow
```

## Lucas-Kanade Optical Flow

Lucas-Kanade는 sparse optical flow이다.
전체 픽셀의 flow 대신 feature point의 이동을 추적한다.

이 출력은 이후 RANSAC homography 추정에 바로 사용할 수 있다.

### Pseudocode

```python
def compute_lucas_kanade(prev_gray, curr_gray):
    points_prev = cv2.goodFeaturesToTrack(
        prev_gray,
        maxCorners=1000,
        qualityLevel=0.01,
        minDistance=7,
        blockSize=7,
    )

    if points_prev is None:
        return None, None, None

    points_curr, status, error = cv2.calcOpticalFlowPyrLK(
        prev_gray,
        curr_gray,
        points_prev,
        None,
    )

    valid = status.reshape(-1) == 1

    good_prev = points_prev.reshape(-1, 2)[valid]
    good_curr = points_curr.reshape(-1, 2)[valid]

    return good_prev, good_curr, status
```

## Dual TV-L1 Optical Flow

Dual TV-L1은 optional method로 둔다.
OpenCV contrib 모듈이 필요할 수 있으므로, 사용할 수 없을 때는 명확한 에러 메시지를 낸다.

### Pseudocode

```python
def compute_tvl1(prev_gray, curr_gray):
    if not hasattr(cv2, "optflow"):
        raise RuntimeError("Dual TV-L1 requires opencv-contrib-python")

    optical_flow = cv2.optflow.DualTVL1OpticalFlow_create()
    flow = optical_flow.calc(prev_gray, curr_gray, None)

    return flow
```

## Method Switch

`compute_optical_flow()`는 `method` 값에 따라 내부 구현을 선택한다.

### Pseudocode

```python
def compute_optical_flow(prev_img, curr_img, method, config):
    prev_gray = to_gray(prev_img)
    curr_gray = to_gray(curr_img)

    if method in ["dis_ultrafast", "dis_fast", "dis_medium"]:
        flow = compute_dis(prev_gray, curr_gray, method)
        return dense_result(method, flow)

    if method == "farneback":
        flow = compute_farneback(prev_gray, curr_gray)
        return dense_result(method, flow)

    if method == "lucas_kanade":
        points_prev, points_curr, status = compute_lucas_kanade(prev_gray, curr_gray)
        return sparse_result(method, points_prev, points_curr, status)

    if method == "tvl1":
        flow = compute_tvl1(prev_gray, curr_gray)
        return dense_result(method, flow)

    raise ValueError(f"Unknown optical flow method: {method}")
```

## Flow 시각화

Dense method와 sparse method는 결과 형식이 다르기 때문에 시각화 함수도 나누어 둔다.

## Dense Flow 시각화

Dense optical flow는 HSV color map으로 시각화한다.

### Pseudocode

```python
def visualize_dense_flow(flow):
    u = flow[..., 0]
    v = flow[..., 1]

    magnitude, angle = cv2.cartToPolar(u, v)

    hsv = zeros_like_bgr_image
    hsv[..., 0] = angle * 180 / pi / 2
    hsv[..., 1] = 255
    hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)

    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr
```

디버깅용 vector 시각화:

```python
def draw_flow_vectors(image, flow, step=16):
    for y, x in grid_points:
        dx, dy = flow[y, x]
        draw line from (x, y) to (x + dx, y + dy)
```

## Sparse Flow 시각화

Lucas-Kanade 결과는 추적된 point의 이동을 선과 점으로 그린다.

### Pseudocode

```python
def visualize_sparse_flow(image, points_prev, points_curr):
    output = image.copy()

    for prev_pt, curr_pt in zip(points_prev, points_curr):
        x0, y0 = prev_pt
        x1, y1 = curr_pt

        cv2.line(output, (x0, y0), (x1, y1), color)
        cv2.circle(output, (x1, y1), radius, color)

    return output
```

## test_vision.py 계획

`test_vision.py`는 테스트 이미지를 읽고, 연속 프레임 사이의 optical flow를 계산한 뒤 결과를 보여준다.

### Pseudocode

```python
from pathlib import Path
import cv2

import config
from vision.optical_flow import compute_optical_flow
from vision.optical_flow import visualize_flow


def load_image_sequence(image_dir):
    image_paths = sorted all jpg/png/jpeg/bmp files

    if len(image_paths) < 2:
        raise RuntimeError("Need at least 2 images in test_images/")

    images = []
    for path in image_paths:
        image = cv2.imread(str(path))
        if image is None:
            continue
        images.append((path, image))

    return images
```

```python
def main():
    images = load_image_sequence(config.TEST_IMAGE_DIR)

    for idx in range(len(images) - 1):
        prev_path, prev_img = images[idx]
        curr_path, curr_img = images[idx + 1]

        result = compute_optical_flow(
            prev_img,
            curr_img,
            method=config.OPTICAL_FLOW_METHOD,
            config=config,
        )

        vis = visualize_flow(curr_img, result)

        cv2.imshow("optical flow", vis)
        key = cv2.waitKey(0)

        if key == ord("q"):
            break

    cv2.destroyAllWindows()
```

## 이후 단계: RANSAC Homography

이번 구현 범위는 아니지만, optical flow 출력은 이후 RANSAC homography 추정을 고려해서 설계한다.

Dense method의 경우:

```python
sample flow vectors
convert valid flow pixels to point pairs:
    points_prev = [[x, y], ...]
    points_curr = [[x + u, y + v], ...]
run cv2.findHomography(points_prev, points_curr, cv2.RANSAC)
```

Lucas-Kanade의 경우:

```python
use points_prev and points_curr directly
run cv2.findHomography(points_prev, points_curr, cv2.RANSAC)
```

향후 추가할 함수:

```python
def flow_result_to_point_matches(result):
    if dense:
        return sampled point pairs from flow

    if sparse:
        return points_prev, points_curr
```

## 구현 순서

1. `config.py`에 optical flow 관련 설정값을 추가한다.
2. `vision/optical_flow.py`에 이미지 전처리 helper를 구현한다.
3. DIS optical flow를 구현한다.
4. Farneback optical flow를 구현한다.
5. Lucas-Kanade optical flow를 구현한다.
6. Dual TV-L1 optical flow를 optional로 구현하고, 사용 불가능할 경우 명확한 에러 메시지를 낸다.
7. Dense와 sparse 결과 시각화 helper를 구현한다.
8. `test_vision.py`를 구현한다.
9. `test_images/`에 최소 2장의 이미지가 있는 상태에서 테스트한다.

## 실행 명령

Repository root에서 실행:

```bash
python test_vision.py
```

Docker 내부에서 실행:

```bash
cd /workspace/Vehicles-Signal-Processing
python test_vision.py
```

## 참고 사항

- `test_images/`에는 최소 2장의 이미지 파일이 필요하다.
- DIS는 CPU에서 빠르게 dense optical flow를 얻기 위한 기본 method로 둔다.
- Lucas-Kanade는 dense flow field가 아니라 sparse point match를 반환하므로 별도로 처리한다.
- Dual TV-L1은 `opencv-contrib-python`이 없으면 사용할 수 없으므로 graceful failure가 필요하다.

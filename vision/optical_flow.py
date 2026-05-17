import cv2
import numpy as np


def to_gray(img):
    if img is None:
        raise ValueError("Input image is None")

    if img.ndim == 2:
        return img

    if img.ndim == 3:
        if img.shape[2] == 3:
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if img.shape[2] == 4:
            return cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)

    raise ValueError(f"Unsupported image shape: {img.shape}")


def resize_if_needed(img, resize_width):
    if resize_width is None:
        return img

    height, width = img.shape[:2]
    if width <= 0:
        raise ValueError("Image width must be positive")

    resize_width = int(resize_width)
    if resize_width <= 0:
        raise ValueError("RESIZE_WIDTH must be positive or None")

    scale = resize_width / width
    resized_height = max(1, int(round(height * scale)))
    return cv2.resize(img, (resize_width, resized_height), interpolation=cv2.INTER_AREA)


def _resize_to_size(img, size):
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)


def _get_resize_width(config):
    if config is None:
        return None
    return getattr(config, "RESIZE_WIDTH", None)


def _prepare_gray_pair(prev_img, curr_img, config):
    resize_width = _get_resize_width(config)

    prev_gray = to_gray(prev_img)
    curr_gray = to_gray(curr_img)

    prev_height, prev_width = prev_gray.shape[:2]
    curr_height, curr_width = curr_gray.shape[:2]

    if resize_width is None:
        if prev_gray.shape != curr_gray.shape:
            raise ValueError(
                "prev_img and curr_img must have the same size when RESIZE_WIDTH is None"
            )
        return prev_gray, curr_gray, (1.0, 1.0), (1.0, 1.0)

    resized_prev = resize_if_needed(prev_gray, resize_width)
    target_height, target_width = resized_prev.shape[:2]
    target_size = (target_width, target_height)
    resized_curr = _resize_to_size(curr_gray, target_size)

    prev_scale = (prev_width / target_width, prev_height / target_height)
    curr_scale = (curr_width / target_width, curr_height / target_height)

    return resized_prev, resized_curr, prev_scale, curr_scale


def _scale_dense_flow_to_original(flow, curr_img, curr_scale):
    scale_x, scale_y = curr_scale
    if scale_x == 1.0 and scale_y == 1.0:
        return flow

    curr_height, curr_width = curr_img.shape[:2]
    scaled_flow = cv2.resize(flow, (curr_width, curr_height), interpolation=cv2.INTER_LINEAR)
    scaled_flow[..., 0] *= scale_x
    scaled_flow[..., 1] *= scale_y
    return scaled_flow


def _scale_points(points, scale):
    if points is None:
        return points

    points = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if points.size == 0:
        return points

    scale_x, scale_y = scale
    scaled = points.copy()
    scaled[:, 0] *= scale_x
    scaled[:, 1] *= scale_y
    return scaled


def dense_result(method, flow):
    return {
        "method": method,
        "flow": flow,
        "points_prev": None,
        "points_curr": None,
        "status": None,
    }


def sparse_result(method, points_prev, points_curr, status):
    return {
        "method": method,
        "flow": None,
        "points_prev": points_prev,
        "points_curr": points_curr,
        "status": status,
    }


def compute_dis(prev_gray, curr_gray, method):
    if not hasattr(cv2, "DISOpticalFlow_create"):
        raise RuntimeError("DIS optical flow is not available in this OpenCV build")

    if method == "dis_ultrafast":
        cv_preset = cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST
    elif method == "dis_fast":
        cv_preset = cv2.DISOPTICAL_FLOW_PRESET_FAST
    elif method == "dis_medium":
        cv_preset = cv2.DISOPTICAL_FLOW_PRESET_MEDIUM
    else:
        raise ValueError(f"Unknown DIS method: {method}")

    optical_flow = cv2.DISOpticalFlow_create(cv_preset)
    return optical_flow.calc(prev_gray, curr_gray, None)


def compute_farneback(prev_gray, curr_gray):
    return cv2.calcOpticalFlowFarneback(
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


def compute_lucas_kanade(prev_gray, curr_gray):
    empty_points = np.empty((0, 2), dtype=np.float32)
    empty_status = np.empty((0,), dtype=np.uint8)

    points_prev = cv2.goodFeaturesToTrack(
        prev_gray,
        maxCorners=1000,
        qualityLevel=0.01,
        minDistance=7,
        blockSize=7,
    )

    if points_prev is None:
        return empty_points, empty_points, empty_status

    points_curr, status, _ = cv2.calcOpticalFlowPyrLK(
        prev_gray,
        curr_gray,
        points_prev,
        None,
    )

    if points_curr is None or status is None:
        return empty_points, empty_points, empty_status

    valid = status.reshape(-1) == 1
    prev_flat = points_prev.reshape(-1, 2)
    curr_flat = points_curr.reshape(-1, 2)
    finite = np.isfinite(prev_flat).all(axis=1) & np.isfinite(curr_flat).all(axis=1)
    valid = valid & finite

    good_prev = prev_flat[valid].astype(np.float32)
    good_curr = curr_flat[valid].astype(np.float32)
    filtered_status = status.reshape(-1)[valid].astype(np.uint8)

    return good_prev, good_curr, filtered_status


def compute_tvl1(prev_gray, curr_gray):
    if hasattr(cv2, "optflow") and hasattr(cv2.optflow, "DualTVL1OpticalFlow_create"):
        optical_flow = cv2.optflow.DualTVL1OpticalFlow_create()
        return optical_flow.calc(prev_gray, curr_gray, None)

    if hasattr(cv2, "DualTVL1OpticalFlow_create"):
        optical_flow = cv2.DualTVL1OpticalFlow_create()
        return optical_flow.calc(prev_gray, curr_gray, None)

    raise RuntimeError("Dual TV-L1 requires opencv-contrib-python")


def compute_optical_flow(prev_img, curr_img, method="dis_medium", config=None):
    prev_gray, curr_gray, prev_scale, curr_scale = _prepare_gray_pair(
        prev_img,
        curr_img,
        config,
    )

    if prev_gray.shape != curr_gray.shape:
        raise ValueError("prev_gray and curr_gray must have the same size")

    if method in ["dis_ultrafast", "dis_fast", "dis_medium"]:
        flow = compute_dis(prev_gray, curr_gray, method)
        flow = _scale_dense_flow_to_original(flow, curr_img, curr_scale)
        return dense_result(method, flow)

    if method == "farneback":
        flow = compute_farneback(prev_gray, curr_gray)
        flow = _scale_dense_flow_to_original(flow, curr_img, curr_scale)
        return dense_result(method, flow)

    if method == "lucas_kanade":
        points_prev, points_curr, status = compute_lucas_kanade(prev_gray, curr_gray)
        points_prev = _scale_points(points_prev, prev_scale)
        points_curr = _scale_points(points_curr, curr_scale)
        return sparse_result(method, points_prev, points_curr, status)

    if method == "tvl1":
        flow = compute_tvl1(prev_gray, curr_gray)
        flow = _scale_dense_flow_to_original(flow, curr_img, curr_scale)
        return dense_result(method, flow)

    raise ValueError(f"Unknown optical flow method: {method}")


def _as_bgr_image(image):
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

    return image.copy()


def visualize_dense_flow(flow):
    if flow is None:
        raise ValueError("flow must not be None")

    if flow.ndim != 3 or flow.shape[2] != 2:
        raise ValueError(f"Expected flow shape HxWx2, got {flow.shape}")

    u = flow[..., 0].astype(np.float32)
    v = flow[..., 1].astype(np.float32)

    magnitude, angle = cv2.cartToPolar(u, v)

    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
    hsv[..., 0] = np.clip(angle * 180.0 / np.pi / 2.0, 0, 179).astype(np.uint8)
    hsv[..., 1] = 255
    hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(
        np.uint8
    )

    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def draw_flow_vectors(image, flow, step=16, scale=1.0):
    output = _as_bgr_image(image)

    if flow is None:
        return output

    if flow.ndim != 3 or flow.shape[2] != 2:
        return output

    flow_height, flow_width = flow.shape[:2]
    if output.shape[:2] != (flow_height, flow_width):
        output = cv2.resize(output, (flow_width, flow_height), interpolation=cv2.INTER_AREA)

    step = max(1, int(step))
    scale = float(scale)

    for y in range(step // 2, flow_height, step):
        for x in range(step // 2, flow_width, step):
            dx, dy = flow[y, x]
            x2 = int(round(x + dx * scale))
            y2 = int(round(y + dy * scale))
            cv2.line(output, (x, y), (x2, y2), (0, 255, 0), 1, cv2.LINE_AA)
            cv2.circle(output, (x, y), 1, (0, 0, 255), -1, cv2.LINE_AA)

    return output


def visualize_sparse_flow(image, points_prev, points_curr):
    output = _as_bgr_image(image)

    if points_prev is None or points_curr is None:
        return output

    points_prev = np.asarray(points_prev, dtype=np.float32).reshape(-1, 2)
    points_curr = np.asarray(points_curr, dtype=np.float32).reshape(-1, 2)

    if len(points_prev) == 0 or len(points_curr) == 0:
        return output

    for prev_pt, curr_pt in zip(points_prev, points_curr):
        if not np.isfinite(prev_pt).all() or not np.isfinite(curr_pt).all():
            continue

        x0, y0 = np.round(prev_pt).astype(int)
        x1, y1 = np.round(curr_pt).astype(int)

        cv2.line(output, (x0, y0), (x1, y1), (0, 255, 0), 1, cv2.LINE_AA)
        cv2.circle(output, (x1, y1), 3, (0, 0, 255), -1, cv2.LINE_AA)

    return output


def visualize_flow(image, result, config=None):
    if result["flow"] is not None:
        flow = result["flow"]
        dense_vis = visualize_dense_flow(flow)
        base = _as_bgr_image(image)

        if base.shape[:2] != dense_vis.shape[:2]:
            height, width = dense_vis.shape[:2]
            base = cv2.resize(base, (width, height), interpolation=cv2.INTER_AREA)

        vis = cv2.addWeighted(base, 0.45, dense_vis, 0.55, 0)
        scale = getattr(config, "FLOW_VIS_SCALE", 1.0) if config is not None else 1.0
        return draw_flow_vectors(vis, flow, scale=scale)

    return visualize_sparse_flow(
        image,
        result["points_prev"],
        result["points_curr"],
    )

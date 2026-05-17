from pathlib import Path

import cv2

import config
from vision.optical_flow import compute_optical_flow
from vision.optical_flow import visualize_flow


IMAGE_EXTENSIONS = ("*.jpg", "*.jpeg", "*.png", "*.bmp")


def _resolve_image_dir(image_dir):
    image_dir = Path(image_dir)
    if image_dir.is_absolute():
        return image_dir
    return Path(__file__).resolve().parent / image_dir


def load_image_sequence(image_dir):
    image_dir = _resolve_image_dir(image_dir)

    image_paths = []
    for pattern in IMAGE_EXTENSIONS:
        image_paths.extend(image_dir.glob(pattern))
        image_paths.extend(image_dir.glob(pattern.upper()))

    image_paths = sorted(set(image_paths))

    images = []
    for path in image_paths:
        image = cv2.imread(str(path))
        if image is None:
            print(f"Skipping unreadable image: {path}")
            continue
        images.append((path, image))

    if len(images) < 2:
        raise RuntimeError(
            f"Need at least 2 valid images in {image_dir}, found {len(images)}"
        )

    return images


def main():
    images = load_image_sequence(config.TEST_IMAGE_DIR)

    for idx in range(len(images) - 1):
        prev_path, prev_img = images[idx]
        curr_path, curr_img = images[idx + 1]

        print(f"[{idx}] {prev_path.name} -> {curr_path.name}")

        result = compute_optical_flow(
            prev_img,
            curr_img,
            method=config.OPTICAL_FLOW_METHOD,
            config=config,
        )

        vis = visualize_flow(curr_img, result)

        if config.SHOW_FLOW_STEP:
            try:
                cv2.imshow("optical flow", vis)
                key = cv2.waitKey(0)
            except cv2.error as exc:
                raise RuntimeError("cv2.imshow failed. Check GUI/display support.") from exc

            if key == ord("q"):
                break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

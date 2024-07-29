import os

# NPU cache
os.environ["XLNX_ENABLE_CACHE"] = "1"

import cv2

from library.config import Config
from library.inference import Inference


def main():
    config = Config()

    # Model
    keypoint_score_th = config.getfloat("inference", "keypoint_score_th")
    input_size = config.getint("model", "input_size")
    model_path = "models/converted/" + config.get("inference", "model_path")

    # NPU
    enable_npu = config.getboolean("inference", "enable_npu")

    inference = Inference(model_path, enable_npu)

    # Image folders for inference
    image_folder = config.get("inference", "image_folder")
    image_folder_predicted = config.get("inference", "image_folder_predicted")
    if not os.path.exists(image_folder_predicted):
        os.makedirs(image_folder_predicted)

    images = os.listdir(image_folder)
    print(f"Found {len(images)} images in {image_folder}")

    mean_scores = []
    for image_file in images:
        image = cv2.imread(os.path.join(image_folder, image_file))
        image_width, image_height = image.shape[1], image.shape[0]

        keypoints = inference.run(
            input_size,
            image,
        )

        # Extracts the scores (3rd column) from the keypoints array
        scores = keypoints[:, 2]

        # Mean score per image
        mean_score_per_image = sum(scores) / len(scores)
        print(f"Mean score: {mean_score_per_image} in {image_file}")
        mean_scores.append(mean_score_per_image)

        for keypoint in keypoints:
            x, y = int(keypoint[0] * image_width), int(keypoint[1] * image_height)
            score = keypoint[2]
            if score > keypoint_score_th:
                cv2.circle(image, (x, y), 20, (255, 255, 255), -1)
                cv2.circle(image, (x, y), 10, (0, 0, 0), -1)

        # Save the image
        cv2.imwrite(os.path.join(image_folder_predicted, image_file), image)

    # Mean score of all images
    mean_score_all = sum(mean_scores) / len(mean_scores)
    print(f"Mean score all: {mean_score_all}")


if __name__ == "__main__":
    _dir = os.path.dirname(__file__)
    os.chdir(_dir)

    main()

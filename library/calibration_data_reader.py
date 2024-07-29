import os

import cv2
from onnxruntime.quantization import CalibrationDataReader
from library.inference import Inference


class ImageDataReader(CalibrationDataReader):
    """
    A class that reads image data for calibration.
    """

    def __init__(
        self,
        image_folder,
        input_size: int,
        process_num=100,
        model_input_name="input",
        preprocess_image_astype="int32",
    ):
        self.image_folder = image_folder
        self.input_size = input_size
        self.model_input_name = model_input_name
        self.process_count = 0
        self.process_num = process_num
        self.preprocess_image_astype = preprocess_image_astype

        # Files in the image_folder to be enumerated
        images = os.listdir(image_folder)
        self.enumerate_images = iter(images)

        # Count the number of images
        self.image_count = len(images)
        print(f"Found {self.image_count} images in {image_folder}")

    def get_next(self):
        """
        generate the input data dict for ONNXinferenceSession run
        """
        # Limit the number of images to be processed
        if self.process_count >= self.process_num:
            return None

        image_file = next(self.enumerate_images, None)
        if image_file is None:
            return None

        # Read image and preprocess
        image = cv2.imread(os.path.join(self.image_folder, image_file))
        image_data = Inference.preprocess(
            image, self.input_size, self.preprocess_image_astype
        )

        # Print progress
        self.process_count += 1
        if self.process_count % 100 == 0:
            print(f"Processed {self.process_count} images")

        return {self.model_input_name: image_data}

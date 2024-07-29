import os

import cv2
import numpy as np
import onnxruntime


class Inference:

    def __init__(self, model_path: str, enable_npu: bool = True):
        self.model_path = model_path

        if enable_npu:
            providers = ["VitisAIExecutionProvider"]
            provider_options = [
                {"config_file": "vaip_config.json", "cacheDir": "models/.cache"}
            ]
        else:
            providers = ["CPUExecutionProvider"]
            provider_options = None

        self.onnx_session = onnxruntime.InferenceSession(
            self.model_path,
            providers=providers,
            provider_options=provider_options,
        )

    @staticmethod
    def preprocess(input_image, input_size: int, image_astype="int32"):
        input_image = cv2.resize(input_image, dsize=(input_size, input_size))
        # OpenCV uses BGR, but, the model requires RGB
        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
        input_image = input_image.reshape(-1, input_size, input_size, 3)
        input_image = input_image.astype(image_astype)
        return input_image

    @staticmethod
    def postprocess(outputs, image_width, image_height):
        """
        Postprocess the outputs from the model
        :return: keypoints (list of [x, y, score])
        """
        # Remove unnecessary dimensions
        keypoints = np.squeeze(outputs)

        # Replace keypoints[n][1] and keypoints[n][0]
        keypoints[:, [0, 1]] = keypoints[:, [1, 0]]
        return keypoints

    def run(self, input_size, image):
        image_width, image_height = image.shape[1], image.shape[0]

        # Preprocess
        input_image = self.preprocess(image, input_size)

        # Inference
        input_name = self.onnx_session.get_inputs()[0].name
        output_name = self.onnx_session.get_outputs()[0].name
        outputs = self.onnx_session.run([output_name], {input_name: input_image})

        # Postprocess
        keypoints = self.postprocess(outputs, image_width, image_height)

        return keypoints

    def print_model_info(self):
        inputs = self.onnx_session.get_inputs()
        outputs = self.onnx_session.get_outputs()
        print("----------------------------------------")
        print("Model Path:", self.model_path)
        print("shape:", "len(inputs)=", len(inputs), "len(outputs)=", len(outputs))
        print("inputs[0]:", inputs[0])
        print("outputs[0]:", outputs[0])
        print("----------------------------------------")

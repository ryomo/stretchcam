import json
import os
import time

# NPU cache
os.environ["XLNX_ENABLE_CACHE"] = "1"

import cv2
import simpleaudio as sa
from vidgear.gears import CamGear

from library.config import Config
from library.draw_keypoints import DrawKeypoints
from library.inference import Inference

STREAM_TYPE = "camera"  # "camera" or "youtube"
KEYPOINTS_FILEPATH = "keypoints/keypoints_"
KEYPOINTS_JSON_EXTENSION = ".json"

class Predict:
    enable_keypoints = True
    enable_stretch = True

    def __init__(self):
        config = Config()

        # Model
        self.keypoint_score_th = config.getfloat("inference", "keypoint_score_th")
        self.input_size = config.getint("model", "input_size")
        model_path = "models/converted/" + config.get("inference", "model_path")

        # Prepare the inference
        enable_npu = config.getboolean("inference", "enable_npu")
        self.inference = Inference(model_path, enable_npu)

        self.keypoints = None

        self.draw_keypoints = DrawKeypoints(self.keypoint_score_th)

        self._prepare_stream(config)

    def _prepare_stream(self, config):
        # Camera
        self.cap = cv2.VideoCapture(config.getint("camera", "device"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.getint("camera", "width"))
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.getint("camera", "height"))

        # Inference per second to calculate the sleep time
        self.inference_per_second_max = config.getfloat("inference", "inference_per_second_max")

        # Prepare pose
        self.load_keypoints()

        # Prepare sound
        self.wave_obj = sa.WaveObject.from_wave_file("assets/sounds/pop.wav")
        self.play_obj = None

    def run(self):
        start_time = time.time()

        # Capture
        ret, frame = self.cap.read()
        if not ret:
            return False
        frame = cv2.flip(frame, 1)  # Mirror

        # Inference
        # Note: keypoints[num][0] is the x-coordinate, keypoints[num][1] is the y-coordinate, and keypoints[num][2] is the score
        inference_start_time = time.time()
        self.keypoints = self.inference.run(
            self.input_size,
            frame,
        )
        inference_elapsed_time = time.time() - inference_start_time

        # Draw keypoints
        if self.enable_keypoints:
            frame = self.draw_keypoints.draw(
                frame,
                inference_elapsed_time,
                self.keypoints,
            )

        # Stretch
        if self.enable_stretch:
            if len(self.loaded_keypoints_list) > 0:
                # Draw loaded keypoints
                loaded_keypoints = self.loaded_keypoints_list[0]
                self.draw_keypoints.draw_loaded_keypoints(frame, loaded_keypoints)

                # Check distance between keypoints and loaded keypoints
                retval = self.draw_keypoints.are_keypoints_close(self.keypoints, loaded_keypoints)
                if retval:
                    self.loaded_keypoints_list.pop(0)

                    # Play sound
                    if self.play_obj is None or self.play_obj.is_playing() is False:
                        self.play_obj = self.wave_obj.play()

            else:
                # Show "completed" message on the screen
                frame = self.draw_keypoints.draw_completed_message(frame, "You have completed!")

        # Sleep
        elapsed_time = time.time() - start_time
        sleep_time = max(1.0 / self.inference_per_second_max - elapsed_time, 0)
        time.sleep(sleep_time)

        return frame

    def save_keypoints(self):
        """
        Save the keypoints to a file
        """
        # Find the next available filename
        counter = 0
        while os.path.exists(KEYPOINTS_FILEPATH + str(counter) + KEYPOINTS_JSON_EXTENSION):
            counter += 1
        # Save the keypoints
        with open(KEYPOINTS_FILEPATH + str(counter) + KEYPOINTS_JSON_EXTENSION, "w") as f:
            json.dump(self.keypoints.tolist(), f)

    def load_keypoints(self):
        counter = 0
        self.loaded_keypoints_list = []
        while os.path.exists(KEYPOINTS_FILEPATH + str(counter) + KEYPOINTS_JSON_EXTENSION):
            with open(KEYPOINTS_FILEPATH + str(counter) + KEYPOINTS_JSON_EXTENSION, "r") as f:
                loaded_keypoints = json.load(f)
                self.loaded_keypoints_list.append(loaded_keypoints)
            counter += 1

    def release(self):
        self.cap.release()
        cv2.destroyAllWindows()


class PredictYoutube(Predict):

    def _prepare_stream(self, config):
        # YouTube video
        options = {
            "STREAM_RESOLUTION": "480p",
        }
        self.stream = CamGear(
            source="https://www.youtube.com/watch?v=vuGnzLxRvZM",
            stream_mode=True,
            logging=True,
            **options,
        ).start()

    def run(self):
        # Capture
        frame = self.stream.read()
        if frame is None:
            return False

        # Inference
        inference_start_time = time.time()
        self.keypoints = self.inference.run(
            self.input_size,
            frame,
        )
        inference_elapsed_time = time.time() - inference_start_time

        # Draw keypoints
        if self.enable_keypoints:
            frame = self.draw_keypoints.draw(
                frame,
                inference_elapsed_time,
                self.keypoints,
            )

        return frame


def main():
    if STREAM_TYPE == "camera":
        predict = Predict()
    elif STREAM_TYPE == "youtube":
        predict = PredictYoutube()

    while True:
        frame = predict.run()
        if frame is False:
            break

        # Exit by pressing the ESC key
        key = cv2.waitKey(1)
        if key == 27:  # ESC
            return False

        # Show image
        cv2.imshow("MoveNet(singlepose) Demo", frame)

    predict.release()


if __name__ == "__main__":
    _dir = os.path.dirname(__file__)
    os.chdir(_dir)

    main()

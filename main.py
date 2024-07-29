import os

from library.config import Config

# Set the Kivy home directory (Need to be done before importing Kivy)
os.environ["KIVY_HOME"] = os.path.dirname(__file__) + "/.kivy"

# NPU cache
os.environ["XLNX_ENABLE_CACHE"] = "1"

import cv2
from kivy.app import App
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.properties import ObjectProperty
from kivy.uix.image import Image
from kivy.uix.widget import Widget

from predict import STREAM_TYPE, Predict, PredictYoutube


class VideoContainer(Image):

    def __init__(self, **kwargs):
        super(VideoContainer, self).__init__(**kwargs)

        if STREAM_TYPE == "camera":
            self.predict = Predict()
        elif STREAM_TYPE == "youtube":
            self.predict = PredictYoutube()

        Clock.schedule_interval(self.update, 1.0 / 30)

    def update(self, dt):
        frame = self.predict.run()

        # Convert the frame to the kivy texture
        buf = cv2.flip(frame, 0).tobytes()
        texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt="bgr")
        texture.blit_buffer(buf, colorfmt="bgr", bufferfmt="ubyte")

        # Display the texture
        self.texture = texture


class MainView(Widget):

    video_container = ObjectProperty(None)
    keypoints_button = ObjectProperty(None)
    toggle_stretch_button = ObjectProperty(None)
    save_keypoints_button = ObjectProperty(None)

    def __init__(self, **kwargs):
        super(MainView, self).__init__(**kwargs)

        self.predict = self.video_container.predict

        # Hide the save button
        config = Config()
        if not config.getboolean("inference", "enable_save_keypoints_button"):
            self.save_keypoints_button.opacity = 0
            self.save_keypoints_button.disabled = True

    def toggle_keypoints(self):
        self.predict.enable_keypoints = not self.predict.enable_keypoints
        if self.predict.enable_keypoints:
            self.keypoints_button.text = "Hide Keypoints"
        else:
            self.keypoints_button.text = "Show Keypoints"

    def toggle_stretch(self):
        self.predict.enable_stretch = not self.predict.enable_stretch
        if self.predict.enable_stretch:
            self.predict.load_keypoints()
            self.toggle_stretch_button.text = "Stop Stretch"
        else:
            self.toggle_stretch_button.text = "Start Stretch"

    def save_keypoints(self):
        self.predict.save_keypoints()


class MainApp(App):
    def build(self):
        return MainView()


if __name__ == "__main__":
    _dir = os.path.dirname(__file__)
    os.chdir(_dir)

    MainApp().run()

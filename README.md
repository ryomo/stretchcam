# StretchCam

## Requirements

* Windows 11
* [Ryzen AI Software 1.1](https://ryzenai.docs.amd.com/en/1.1/inst.html)


## How to install

Open a Anaconda Powershell Prompt and run the following commands:

```powershell
git clone https://github.com/ryomo/stretchcam.git
cd stretchcam

# Create a new conda environment from existing Ryzen AI environment
conda create --name stretchcam --clone <your-ryzen-ai-env>
conda activate stretchcam

# Install Kivy and other dependencies
conda install kivy=2.1.0 -c conda-forge
# NOTE: If `opencv-python` is already installed, uninstall it first to avoid conflicts with `opencv-contrib-python`.
pip install -r requirements.txt
```


## How to use

```powrshell
conda activate stretchcam
python main.py
```


## Configuration (Optional)

The default configuration file is `config/default.ini`. If you wish to modify the configuration, you can edit this file.
Alternatively, you can create a `config/user.ini` file to override the settings in the default configuration.


## Self quantization (Optional)

1. Put calibration images in `datasets/subdirectory` (e.g. `datasets/mypose`).
    * The calibration images which should contain poses can be taken by the camera.
    * 10 images seems to be enough. But if you want to put more images, you need to change `calibration_image_count = 100` in the configuration file.

2. Add the following lines to the configuration file:
    ```ini
    [quantization]
    calibration_image_dir = datasets/mypose/
    ```

3. Run the following command:

    ```powershell
    python quantize.py
    ```


## Bonus features

### Pose estimation from images

The source code of this version is very simple, making it easy to understand.

1. Put images in `datasets/subdirectory` (e.g. `datasets/mypose`).
2. Open the configuration file and edit `[inference]` section's `image_folder` to the directory where the images are stored (e.g. `datasets/mypose`). You can also edit `image_folder_predicted` to the directory where the predicted images will be stored (e.g. `datasets/mypose-predicted`).
3. Run `python predict_imagefiles.py` to predict poses from the images.

### Pose estimation from YouTube video

Change `STREAM_TYPE = "camera"` in `predict.py` to `STREAM_TYPE = "youtube"`, and run `python main.py`.

### Make stretch poses by yourself

1. Edit the configuration file and set the following values:
    * `enable_save_keypoints_button = True`
    * `model_path = singlepose-thunder.onnx`
    * `keypoint_score_th = 0.3`

2. Run the application and click the "Save keypoints" button to save the keypoints of the poses you want to stretch. The keypoints will be saved in the `keypoints` directory.

3. Edit the file number in the `keypoints` directory. (e.g. `keypoints/keypoints_0.json` is the first pose to be loaded.)


## Uninstall

* Remove the directory where you installed StretchCam.
* (Optional) Remove the model cache directory which is `C:\Users\<USERNAME>\.cache\kagglehub\models\google\movenet`.


## Licenses

* This project is released under the MIT License.
* This project highly depends on the following projects.


### MoveNet

* Open source pose estimation model
* Licensed under the Apache License 2.0
* [link](https://www.kaggle.com/models/google/movenet)

### tf2onnx

* Open source tool to convert TensorFlow models to ONNX
* Licensed under the Apache License 2.0
* [link](https://github.com/onnx/tensorflow-onnx)

### OpenCV

* Open source computer vision library
* Licensed under the Apache License 2.0
* [link](https://github.com/opencv/opencv)

### Kivy

* Open source UI framework
* Licensed under the MIT License
* [link](https://github.com/kivy/kivy)

### Other projects

* See `requirements.txt` for other dependencies.

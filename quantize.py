import os

# Disable NPU cache
# https://ryzenai.docs.amd.com/en/1.1/modelrun.html#environment-variables
# NOTE: Enabling the cache skips the compilation process, so you may not notice errors from the compilation process.
os.environ["XLNX_ENABLE_CACHE"] = "0"

import vai_q_onnx
from onnxruntime.quantization import shape_inference

from library.config import Config
from library.calibration_data_reader import ImageDataReader
from library.inference import Inference


def main():
    config = Config()
    input_size = config.getint("model", "input_size")
    model_name = config.get("model", "model_name")

    converted_model_dir = "models/converted/"
    preprocessed_model_suffix = config.get("quantize", "preprocessed_model_suffix")
    quantized_model_suffix = config.get("quantize", "quantized_model_suffix")
    onnx_model_ext = ".onnx"

    image_dir = config.get("quantize", "calibration_image_dir")
    image_count = config.getint("quantize", "calibration_image_count")

    enable_step0 = config.getboolean("quantize", "enable_step0")
    enable_step1 = config.getboolean("quantize", "enable_step1")
    enable_step2 = config.getboolean("quantize", "enable_step2")
    enable_step3 = config.getboolean("quantize", "enable_step3")

    # NPU
    enable_npu = config.getboolean("inference", "enable_npu")

    ###############################
    # Step-0: Download the model
    ###############################

    if enable_step0:
        import shutil
        import kagglehub

        path = kagglehub.model_download("google/movenet/tensorFlow2/" + model_name)
        # print("Path to model files:", path)

        # Copy downloaded model files to the model directory
        shutil.copytree(path, "models/downloaded/" + model_name, dirs_exist_ok=True)

    ###############################
    # Step-1: Convert the model
    ###############################

    if enable_step1:
        import subprocess

        input_model_dir = "models/downloaded/" + model_name
        output_model = converted_model_dir + model_name + onnx_model_ext

        # Convert the model to ONNX format
        # https://github.com/onnx/tensorflow-onnx
        completed_process = subprocess.run(
            ["python", "-m", "tf2onnx.convert", "--opset", "13", "--saved-model", input_model_dir, "--output", output_model],  # fmt: skip
        )
        completed_process.check_returncode()
        print(completed_process.stdout)

        # Print model info
        inference = Inference(output_model, enable_npu)
        inference.print_model_info()

    ###############################
    # Step-2: Pre-processing on the float model
    ###############################

    if enable_step2:
        input_model = converted_model_dir + model_name + onnx_model_ext
        output_model = (
            converted_model_dir
            + model_name
            + preprocessed_model_suffix
            + onnx_model_ext
        )

        # https://ryzenai.docs.amd.com/en/1.1/vai_quant/vai_q_onnx.html#recommended-pre-processing-on-the-float-model
        shape_inference.quant_pre_process(
            input_model,
            output_model,
            auto_merge=True,  # If False (by default), 'Incomplete symbolic shape inference' exception will be raised.
        )

        # Print model info
        inference = Inference(output_model, enable_npu)
        inference.print_model_info()

    ###############################
    # Step-3: Quantize the model
    ###############################

    if enable_step3:
        calibration_data_reader = ImageDataReader(image_dir, input_size, image_count)

        input_model = (
            converted_model_dir
            + model_name
            + preprocessed_model_suffix
            + onnx_model_ext
        )
        output_model = (
            converted_model_dir + model_name + quantized_model_suffix + onnx_model_ext
        )

        # Quantizing Using the vai_q_onnx API
        # https://ryzenai.docs.amd.com/en/1.1/vai_quant/vai_q_onnx.html#quantizing-using-the-vai-q-onnx-api
        vai_q_onnx.quantize_static(
            input_model,
            output_model,
            calibration_data_reader,
            # fmt: off
            # https://black.readthedocs.io/en/stable/usage_and_configuration/the_basics.html#ignoring-sections
            op_types_to_quantize=[
                # To avoid the error:
                # > F20240527 19:49:13.959744 42452 ReplaceConstPass.cpp:88]
                # > Check failed: xir::create_data_type<float>() == op_const->get_output_tensor()->get_data_type()
                # > || xir::create_data_type<double>() == op_const->get_output_tensor()->get_data_type()
                # > The data type of xir::Op{name = Resize__349:0_vaip_161_transfered_DwDeConv_weights, type = const}'s output tensor,
                # > xir::Tensor{name = Resize__349:0_vaip_161_transfered_DwDeConv_weights, type = INT32, shape = {1, 4, 4, 64}}
                # > only supports float now.
                #
                # Op Type,      Node Count, Note
                "Conv",         # 74
                "Clip",         # 35
                # "Add",        # 19        error
                # "Unsqueeze",  # 12
                # "Cast",       # 10
                # "Reshape",    #  9
                # "Relu",       #  7
                # "Sub",        #  6
                # "Mul",        #  5
                # "Concat",     #  5
                # "Transpose",  #  4
                # "Squeeze",    #  4
                # "GatherND",   #  4
                # "Split",      #  3
                # "Resize",     #  3        error
                # "Div",        #  3
                # "Sigmoid",    #  2
                # "Pow",        #  2
                # "ArgMax",     #  2
                # "Sqrt",       #  1        error
            ],
            # fmt: on
            # Recommended settings for CNNs on NPU
            # https://ryzenai.docs.amd.com/en/1.1/vai_quant/vai_q_onnx.html#cnns-on-npu
            quant_format=vai_q_onnx.QuantFormat.QDQ,
            calibrate_method=vai_q_onnx.PowerOfTwoMethod.MinMSE,
            activation_type=vai_q_onnx.QuantType.QUInt8,
            weight_type=vai_q_onnx.QuantType.QInt8,
            enable_ipu_cnn=True,
            # Enable CLE for better accuracy
            # https://ryzenai.docs.amd.com/en/1.1/vai_quant/vai_q_onnx.html#quantizing-using-cross-layer-equalization
            include_cle=True,
            extra_options={
                "ActivationSymmetric": True,
                "ReplaceClip6Relu": True,
                "CLESteps": 1,
                "CLEScaleAppendBias": True,
            },
        )

        # Print model info
        inference = Inference(output_model, enable_npu)
        inference.print_model_info()


if __name__ == "__main__":
    _dir = os.path.dirname(__file__)
    os.chdir(_dir)

    main()
    print("Done.")

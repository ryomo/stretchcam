from library.calibration_data_reader import ImageDataReader


def main():
    image_folder = "datasets/mypose/"
    input_size = 256
    reader = ImageDataReader(image_folder, input_size)
    data = reader.get_next()
    print(data)


if __name__ == "__main__":
    main()

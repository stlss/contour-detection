from contour_detection import ContourDetector
from datetime import datetime
from PIL import Image
import numpy as np
import pandas as pd


# Возвращает случайное изображение размерностью (n, m).
def generate_random_image(n: int, m: int) -> Image:
    arr_image = np.random.randint(255, size=(n, m, 3), dtype=np.uint8)
    image = Image.fromarray(arr_image)
    return image


# Возвращает время работы создания экземпляра класса ContourDetector и создания контурного изображения.
def measure_time(image: Image, params: tuple[int, int]) -> float:
    time_start = datetime.now()

    contour_detector = ContourDetector(image)
    _ = contour_detector.get_contour_image(params[0], params[1])

    time_end = datetime.now()
    return (time_end - time_start).total_seconds()


# Возвращает дата-фрейм, содержащий время работы создания экземпляра класса ContourDetector и создания контурного
# изображения при различных размерах исходных изображений и параметров.
def measure_times(params: list[tuple[int, int]], sizes_images: list[tuple[int, int]]) -> pd.DataFrame:
    time_params = [np.empty(shape=len(sizes_images)) for _ in range(len(params))]

    for j in range(len(sizes_images)):
        n, m = sizes_images[j]
        image = generate_random_image(n, m)

        for i in range(len(params)):
            time_params[i][j] = measure_time(image, params[i])

    time_params = [np.round(time_params_, 2) for time_params_ in time_params]
    dict_ = {"size_image": sizes_images}

    for params_, time_params_ in zip(params, time_params):
        dict_[f"({params_[0]}, {params_[1]})"] = time_params_

    return pd.DataFrame(dict_)

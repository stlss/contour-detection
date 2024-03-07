from PIL import Image
from collections import deque
import numpy as np


# Возвращает инверсионное изображение.
def inverse_image(contour_image: Image):
    array_image = np.asarray(contour_image)
    array_image = np.vectorize(lambda a, b: np.uint8(abs(int(a) - int(b))))(array_image, 255)
    return Image.fromarray(array_image)


# Обнаружитель контура, хранит матрицу яркости и размерность массива исходного изображения.
class ContourDetector:
    def __init__(self, image: Image):
        # Трёхмерный массива исходного изображения.
        array_image = np.asarray(image)

        # Матрица яркости и размерность массива исходного изображения.
        self._matrix_brightness = _get_matrix_brightness(array_image)
        self._shape = array_image.shape

    # Возвращает контурное изображение на основе исходного и двух параметров.
    def get_contour_image(self, contrast: int, size_component: int) -> Image:
        # Матрица контрастности исходного изображения на основе матрицы контрастности.
        matrix_contrast = _get_matrix_contrast(self._matrix_brightness, contrast)

        # Компоненты связностей контрастных пикселей.
        components = _get_connectivity_components(matrix_contrast)
        components = _clear_connectivity_components(components, size_component)

        # Трёхмерный массива контурного изображения и само изображение.
        array_contour_image = _get_array_contour_image(components, self._shape)
        contour_image = Image.fromarray(array_contour_image)

        return contour_image


# Возвращает матрицу яркости, полученную из массива изображения.
def _get_matrix_brightness(array_image: np.array) -> np.array:
    r, g, b = np.float16(0.2126), np.float16(0.7152), np.float16(0.0722)
    return r * array_image[:, :, 0] + g * array_image[:, :, 1] + b * array_image[:, :, 2]


# Сдвиги, при помощи их будем получать соседние пиксели.
# Сдвиги идут по часовой стрелки, начиная со сдвига для верхнего левого соседа.
_shifts = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, 1),
    (1, 1),
    (1, 0),
    (1, -1),
    (0, -1)
)


# Возвращает матрицу контрастности, полученную из матрицы яркости и параметра контрастности.
def _get_matrix_contrast(matrix_brightness: np.array, contrast: int) -> np.array:
    shape = matrix_brightness.shape
    matrix_contrast = np.zeros(shape=shape, dtype=np.bool_)
    contrast = np.float16(contrast)

    # Пиксель является контрастным, если хотя бы с одним соседним пикселем он контрастирует.
    def _is_contrast_pixel():
        for shift_i, shift_j in _shifts:
            if matrix_brightness[i, j] - matrix_brightness[i + shift_i, j + shift_j] >= contrast:
                return True
        return False

    for i in range(1, shape[0] - 1):
        for j in range(1, shape[1] - 1):
            matrix_contrast[i, j] = _is_contrast_pixel()

    return matrix_contrast


# Возвращает список компонент связностей контрастных пикселей, полученный из матрицы контрастности.
def _get_connectivity_components(matrix_contrast: np.array) -> list[list[tuple[int, int]]]:
    shape = matrix_contrast.shape
    used = np.zeros(shape=shape, dtype=np.bool_)  # used[i, j] - был ли (i, j)-ый пиксель посещён или будет ли посещён.
    components = list()

    # Пиксель является корректным, если его координаты не выходят за диапазоны.
    def is_correct_pixel():
        return 0 <= i < shape[0] and 0 <= j < shape[1]

    for i in range(shape[0]):
        for j in range(shape[1]):
            # Если пиксель контрастный, не был и не будет посещён, то запускаем из него поиск в ширину.
            if matrix_contrast[i, j] and not used[i, j]:
                component = [(i, j)]  # Новая компонента связности.
                queue = deque()  # Очередь пикселей.

                queue.append((i, j))  # Добавляем стартовый пиксель.
                used[i, j] = True  # Отмечаем, что он был посещён.

                # Пока очередь не пуста.
                while queue:
                    # Извлекаем координаты очередного пикселя из очереди.
                    x, y = queue[0]
                    queue.popleft()

                    # Перебираем его соседей.
                    for shift_x, shift_y in _shifts:
                        # Координаты соседа.
                        x_, y_ = x + shift_x, y + shift_y

                        # Если сосед корректный, контрастный и не был посещён, то ...
                        if is_correct_pixel() and matrix_contrast[x_, y_] and not used[x_, y_]:
                            component.append((x_, y_))  # Добавляем соседа в компоненту.
                            queue.append((x_, y_))  # Добавляем соседа в очередь.
                            used[x_, y_] = True  # Отмечаем, что сосед будет посещён.

                components.append(component)

    return components


# Возвращает список компонент связностей контрастных пикселей, которые имеют число пикселей >= size_component.
def _clear_connectivity_components(components: list[list[tuple[int, int]]], size_component: int) \
        -> list[list[tuple[int, int]]]:
    return [component for component in components if len(component) >= size_component]


# Возвращает массив контурного изображения, полученного из списка компонент связностей.
def _get_array_contour_image(components: list[list[tuple[int, int]]], shape: tuple[int, int, int]) -> np.array:
    array_image = np.empty(shape=shape, dtype=np.uint8)

    # Делаем все пиксели белыми.
    white, black = np.uint8(255), np.uint8(0)
    array_image = np.vectorize(lambda _: white)(array_image)

    # Делаем пиксели из компонент связностей чёрными.
    for component in components:
        for i, j in component:
            array_image[i, j, 0] = array_image[i, j, 1] = array_image[i, j, 2] = black

    return array_image

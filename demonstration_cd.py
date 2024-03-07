from contour_detection import ContourDetector
from PIL import Image
import matplotlib.pyplot as plt


# Возвращает изображение по указанному пути.
def get_image(path: str) -> Image:
    with Image.open(path) as image:
        image.load()
    return image


# Возвращает список контурных изображений на основе исходного изображения и передаваемых параметров.
def get_contour_images(image: Image, params: list[tuple[int, int]]) -> list[Image]:
    contour_detector = ContourDetector(image)
    return [contour_detector.get_contour_image(contrast, size_component) for contrast, size_component in params]


# Сохраняет контурные изображения в папку images/contour с именами {name_original_image}_{contrast}_{size_component}.
def save_contour_images(name_original_image: str, contour_images: list[Image], params: list[tuple[int, int]]) -> None:
    for image, params_ in zip(contour_images, params):
        image.save(f"images/contour/{name_original_image}_{params_[0]}_{params_[1]}.png")


# Показывает изображение как отдельный график.
def show_image(number: int, title_: str, image: Image) -> None:
    plt.subplot(number)
    plt.title(title_)
    plt.imshow(image)


# Возвращает заголовок графика для контурного изображения.
def get_title_contour_image(params: tuple[int, int]) -> str:
    return f"contrast = {params[0]} size_component = {params[1]}"

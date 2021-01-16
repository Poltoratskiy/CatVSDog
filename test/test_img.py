import os

import pytest
from PIL import Image

from model_func import standardize_img


# model_func.standardize_img tests
@pytest.fixture
def test_img_list():
    test_dir = os.path.join(os.path.dirname(__file__), "test_images")
    test_files = [os.path.join(test_dir, path) for path in os.listdir(test_dir)]
    test_imgs = []
    for name in test_files:
        try:
            img = Image.open(name)
        except IOError:
            print(f"Error: {name} is not image!")
            continue
        test_imgs.append(img)

    yield test_imgs

    while test_imgs:
        img = test_imgs.pop()
        img.close()


def test_standardize_img(test_img_list):
    for img in test_img_list:
        img = standardize_img(img)
        assert img.size == (224, 224)
        assert img.mode == "RGB"

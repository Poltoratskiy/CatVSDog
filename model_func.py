from typing import Tuple

import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow import keras
from tensorflow.keras.preprocessing.image import img_to_array


def standardize_img(img: Image) -> Image:
    """
    standardize_img function takes PIL.Image object as input
    and create new PIL.Image object from this with a size=(224,224),
    format="PNG" and mode "RGB"

    :param img: original PIL.Image object
    :return: reformatted PIL.Image object
    """
    reformatted_img = img.resize((224, 224)).convert(mode="RGB")
    return reformatted_img


def load_tf_model(path: str) -> keras.Model:
    """
    Load serialized keras model from the directory \`path\`

    :param path: directory with serialized keras model
    :return: serialized keras model
    """
    print(f"Loading keras model from {path}")
    return keras.models.load_model(path)


def is_cat(model: keras.Model, img: Image) -> bool:
    """
    Returns True if Cat on the picture, Dog otherwise

    :param model: Loaded model for CatVSDog classification
    :param img: PIL PNG image with size 224x224
        example:
        >>> print(img)
        <PIL.PngImagePlugin.PngImageFile image mode=RGB size=224x224 at ...>
    :return:
    """
    img_array = tf.cast(img_to_array(img), tf.float32) / 255.0
    img_expended = np.expand_dims(img_array, axis=0)
    return model.predict(img_expended)[0][0] < 0.5


def classify(model: keras.Model, img: Image):
    """
    A function that prepares an image for classification and
    runs classifier.
    :param model: Model for image classification.
    :param img: Original image
    :return: altered image and classification result.
    """
    img = standardize_img(img)
    if is_cat(model, img):
        animal = "cat"
    else:
        animal = "dog"
    return img, animal

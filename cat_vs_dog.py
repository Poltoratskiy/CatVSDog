import io
import os
import sys
import time
import urllib.request
from collections import defaultdict
from concurrent.futures.thread import ThreadPoolExecutor
from functools import partial
from typing import Iterator
from urllib.error import URLError

import PIL.Image as Image
from tensorflow import keras

import arg_parser
import model_func
from cfg.config import user_agent


def create_dir(*args: str) -> str:
    dir_path = os.path.join(*args)
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    return dir_path


def get_links_count(filename: str) -> int:
    """
    A function that counts amount of links in the file,
    so we can properly name our images.
    :param filename: str
    :return: int
    """
    count = 0
    if not os.path.exists(filename):
        raise FileNotFoundError(f"{filename} does not exist.")
    else:
        with open(filename, "r") as f:
            for _ in f:
                count += 1
    return count


def get_link(filename: str):
    """
    A generator that provides the index of the link and the link.
    :param filename: str
    :return: dict
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"{filename} does not exist.")
    else:
        links_count = get_links_count(filename)
        digits = len(str(links_count))
        with open(filename, "r") as f:
            for idx, line in enumerate(f):
                yield {"idx": idx, "link": line, "digits": digits}


def download_image(url: str, count: int, digits: int) -> Image:
    """
    A function that downloads an image from the Internet
    :param url: str
    :param count: int
    :param digits: int
    :return: image: Image
    """
    count_filled = str(count).zfill(digits)

    req = urllib.request.Request(url, headers=user_agent)

    with urllib.request.urlopen(req) as response:
        img_bytes = response.read()
        size = len(img_bytes)
        image = Image.open(io.BytesIO(img_bytes))
        image.filename = count_filled
        # image.save(count_filled, "PNG")
    return image, size


def process_image(args: dict, model: keras.Model) -> dict:
    """
    A function that runs the pipeline of downloading an image, passing it to the neural network,
    and then saving it to an appropriate directory.
    :param args: a dictionary that contains the link and its index in a text file
    :return: result: a dictionary that contains the status of downloading, a message suitable for
    the status, the size of the downloaded image, whether it's cat or not
    """

    result = {"status": False, "message": "", "size": 0, "cat": False}
    if "idx" in args and "link" in args and "digits" in args:
        idx = args["idx"]

        if not isinstance(idx, int):
            result["message"] = f"{idx} must be of type int."
            return result

        link = args["link"]
        if not isinstance(link, str):
            result["message"] = (f"{link} must be of type str.",)
            return result

        digits = args["digits"]
        if not isinstance(digits, int):
            result["message"] = (f"{digits} must be of type int.",)
            return result
    else:
        result["message"] = f"{args} not valid arguments."
        return result

    cat_dir = os.path.join(os.getcwd(), "cats")
    dog_dir = os.path.join(os.getcwd(), "dogs")
    try:
        img, size = download_image(link, idx, digits)
        result["status"] = True
        result["message"] = f"{idx} {link} downloaded successfully."
        result["size"] = size
        img_name = img.filename + ".png"
        img, result["cat"] = model_func.classify(cat_model, img)

        if result["cat"]:
            result["message"] += f" {img_name} is cat."
            img_name = os.path.join(cat_dir, img_name)
        else:
            result["message"] += f" {img_name} is dog."
            img_name = os.path.join(dog_dir, img_name)
        img.save(img_name, "PNG")

    except ValueError:
        result["message"] = f"{idx} {link} A Value Error occurred."
    except URLError:
        result["message"] = f"{idx} {link} A URLError occurred."
    except Exception as e:
        result["message"] = f"{idx} {link} Unexpected error occurred."

    print(result["message"])

    return result


def create_report(results: Iterator[dict], start: float, end: float):
    stat = {"files_count": 0, "errors": 0, "downloaded": 0}
    counter = defaultdict(int)
    for i in results:
        if i["status"]:
            stat["files_count"] += 1
            stat["downloaded"] += i["size"]
            counter[i["cat"]] += 1
        else:
            stat["errors"] += 1
    print("Total time: ", end - start)
    print("Files downloaded: ", stat["files_count"])
    print("Data downloaded in bytes: ", stat["downloaded"])
    print("Cats: ", counter[True])
    print("Dogs: ", counter[False])
    print("Errors count: ", stat["errors"])


if __name__ == "__main__":
    # параметры запуска тут, пока прописаны вручную
    file_name, threads = arg_parser.get_arguments()
    print("File name is", file_name)
    print("Thread numb is", threads)

    if threads <= 0:
        print(
            f"Error: Number of threads expected to be positive but {threads} found!",
            file=sys.stderr,
        )
        sys.exit()
    curr_dir = os.getcwd()
    cat_dir = create_dir(curr_dir, "cats")
    dog_dir = create_dir(curr_dir, "dogs")
    cat_model = model_func.load_tf_model(os.path.join(curr_dir, "model"))
    process_image_with_model = partial(process_image, model=cat_model)

    start = time.time()
    with ThreadPoolExecutor(max_workers=threads) as pool:
        results = pool.map(process_image_with_model, get_link(filename=file_name))
    end = time.time()
    create_report(results, start, end)

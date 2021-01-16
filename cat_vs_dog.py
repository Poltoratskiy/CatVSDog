import io
import os
import sys
import time
import urllib.request
from collections import Counter
from concurrent.futures.thread import ThreadPoolExecutor
from functools import partial
from typing import Iterator, Tuple
from urllib.error import URLError

import PIL.Image as Image
from tensorflow import keras

import arg_parser
import model_func
from cfg.config import user_agent


def check_file(filename: str) -> bool:
    """
    A function that checks if file exists and can be opened.
    :param filename: the name of file to be checked
    :return: True if file can be opened else False.
    """
    try:
        f = open(filename, "r")
        f.close()
    except FileNotFoundError:
        print(f"Error: {filename} not found!", file=sys.stderr)
        return False
    except IsADirectoryError:
        print(
            f"Error: {filename} expected to be file, but directory is found.",
            file=sys.stderr,
        )
        return False
    except Exception as e:
        print(
            f"Error: Unexpected error! Program terminated with message: {e}",
            file=sys.stderr,
        )
        return False
    return True


def check_input(filename: str, threads: int) -> bool:
    """
    A function that checks if number of threads is positive and file with filename
    can be opened.
    :param filename: name of the file to be checked.
    :param threads: number of threads to be checked.
    :return: True if input parameters are correct else False.
    """
    if threads <= 0:
        print(
            f"Error: Number of threads expected to be positive but {threads} found!",
            file=sys.stderr,
        )
        return False
    return check_file(filename)


def create_dir(*args: str) -> str:
    """
    A function that creates directories.
    :param args: relative path of the directory to be created
    :return: dir path
    """
    dir_path = os.path.join(*args)
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    return dir_path


def get_links_count(filename: str) -> int:
    """
    A function that counts amount of links in the file,
    so we can properly name our images.
    :param filename: name of the file with URL links
    :return: amount of links
    """
    count = 0
    if not check_file(filename):
        print("Program terminated.", file=sys.stderr)
        sys.exit()
    with open(filename, "r") as f:
        for _ in f:
            count += 1
    return count


def get_link(filename: str):
    """
    A generator that provides the index of the link and the link.
    :param filename: name of the file with URL links.
    :return: dict with the index, link, and the amount of digits for file naming
    """
    if not check_file(filename):
        print("Program terminated.", file=sys.stderr)
        sys.exit()

    links_count = get_links_count(filename)
    digits = len(str(links_count))
    with open(filename, "r") as f:
        for idx, line in enumerate(f):
            yield {"idx": idx, "link": line, "digits": digits}


def download_image(url: str, count: int, digits: int) -> Image:
    """
    A function that downloads an image from the Internet
    :param url: URL of the image
    :param count: index of the URL in the text file.
    :param digits: amount of digits in the number of links
    :return: image: downloaded Image object
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


def check_arguments(args) -> Tuple[int, str, int, str]:
    """
    A function that checks arguments of process_image()
    :param args: a dictionary that contains the link and its index in a text file
    :return: tuple with link data and suitable message (empty if everything is fine)
    """
    if "idx" in args and "link" in args and "digits" in args:
        idx = args["idx"]

        if not isinstance(idx, int):
            return 0, "", 0, f"{idx} must be of type int."

        link = args["link"]
        if not isinstance(link, str):
            return 0, "", 0, f"{link} must be of type str."

        digits = args["digits"]
        if not isinstance(digits, int):
            return 0, "", 0, f"{digits} must be of type int."

    else:
        return 0, "", 0, f"{args} not valid arguments."

    return idx, link, digits, ""


def process_image(args: dict, cat_model: keras.Model) -> dict:
    """
    A function that runs the pipeline of downloading an image, passing it to the neural network,
    and then saving it to an appropriate directory.
    :param args: a dictionary that contains the link and its index in a text file
    :return: result: a dictionary that contains the status of downloading, a message suitable for
    the status, the size of the downloaded image, whether it's cat or not
    """
    buff = sys.stdout
    result = {"status": False, "message": "", "size": 0, "animal": ""}
    idx, link, digits, message = check_arguments(args)
    if message:
        buff = sys.stderr
        print(message, file=buff)
        result["message"] = message
        return result

    try:
        img, size = download_image(link, idx, digits)
        result["status"] = True
        result["message"] = f"{idx} {link} downloaded successfully. "
        result["size"] = size
        img_name = img.filename + ".png"
        img, result["animal"] = model_func.classify(cat_model, img)

        result["message"] += f"{img_name} is {result['animal']}."
        img_name = os.path.join(f"{result['animal']}s", img_name)
        img.save(img_name, "PNG")

    except ValueError as e:
        buff = sys.stderr
        result[
            "message"
        ] = f"{idx} {link} A Value Error occurred with message: \n{e}.\n"
    except URLError as e:
        buff = sys.stderr
        result["message"] = f"{idx} {link} A URLError occurred with message: \n{e}.\n"
    except Exception as e:
        buff = sys.stderr
        result[
            "message"
        ] = f"{idx} {link} Unexpected error occurred with message: \n{e}.\n"

    print(result["message"], file=buff)

    return result


def create_report(results: Iterator[dict], elapsed_time: float):
    """
    A function that creates a report of the pipeline containing amount of
    files downloaded, errors occurred, images that were recognized as cats and dogs.
    :param results: small report of one link processing
    :param elapsed_time: time of computing
    """
    stat = {"files_count": 0, "errors": 0, "downloaded": 0}
    animals = []
    for i in results:
        if i["status"]:
            stat["files_count"] += 1
            stat["downloaded"] += i["size"]
            animals.append(i["animal"])
        else:
            stat["errors"] += 1
    counter = Counter(animals)
    print("Total time:", elapsed_time, "s")
    print("Files downloaded:", stat["files_count"])
    print("Data downloaded in bytes:", stat["downloaded"])
    print("Cats:", counter["cat"])
    print("Dogs:", counter["dog"])
    print("Errors count:", stat["errors"])


if __name__ == "__main__":
    # параметры запуска тут, пока прописаны вручную
    file_name, threads = arg_parser.get_arguments()
    print("File name is", file_name)
    print("Thread numb is", threads)

    if not check_input(filename=file_name, threads=threads):
        sys.exit()

    curr_dir = os.getcwd()
    cat_dir = create_dir(curr_dir, "cats")
    dog_dir = create_dir(curr_dir, "dogs")
    cat_model = model_func.load_tf_model(os.path.join(curr_dir, "model"))
    process_image_with_model = partial(process_image, cat_model=cat_model)

    start = time.time()
    with ThreadPoolExecutor(max_workers=threads) as pool:
        results = pool.map(process_image_with_model, get_link(filename=file_name))
    end = time.time()
    elapsed_time = end - start
    create_report(results, elapsed_time)

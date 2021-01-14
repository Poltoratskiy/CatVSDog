import io
import os
import time
import urllib.request
from concurrent.futures.thread import ThreadPoolExecutor
from typing import Iterator
from urllib.error import URLError

import PIL.Image as Image
from cfg.config import user_agent


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
    name = "images/" + count_filled

    req = urllib.request.Request(url, headers=user_agent)
    with urllib.request.urlopen(req) as response:
        img_bytes = response.read()
        size = len(img_bytes)
        image = Image.open(io.BytesIO(img_bytes))
        image.filename = name
        image.save(name, "PNG")
    return image, size


def process_image(args: dict) -> dict:
    """
    A function that runs the pipeline of downloading an image, passing it to the neural network,
    and then saving it to an appropriate directory.
    :param args: a dictionary that contains the link and its index in a text file
    :return: result: a dictionary that contains the status of downloading, a message suitable for
    the status, the size of the downloaded image, whether it's cat or not
    """
    if "idx" in args and "link" in args and "digits" in args:
        idx = args["idx"]
        if not isinstance(idx, int):
            result = {
                "status": False,
                "message": f"{idx} must be of type int.",
                "size": 0,
                "cat": False,
                "dog": False,
            }
            return result

        link = args["link"]
        if not isinstance(link, str):
            result = {
                "status": False,
                "message": f"{link} must be of type str.",
                "size": 0,
                "cat": False,
                "dog": False,
            }
            return result

        digits = args["digits"]
        if not isinstance(digits, int):
            result = {
                "status": False,
                "message": f"{digits} must be of type int.",
                "size": 0,
                "cat": False,
                "dog": False,
            }
            return result

    else:
        result = {"status": False, "message": f"{args} not valid arguments.", "size": 0, "cat": False, "dog": False}
        return result

    try:
        img, size = download_image(link, idx, digits)
        result = {
            "status": True,
            "message": f"{idx} {link} downloaded successfully.",
            "size": size,
            "cat": False,
            "dog": False,
        }
    except ValueError:
        result = {
            "status": False,
            "message": f"{idx} {link} A Value Error occurred.",
            "size": 0,
            "cat": False,
            "dog": False,
        }
    except URLError:
        result = {
            "status": False,
            "message": f"{idx} {link} A URLError occurred.",
            "size": 0,
            "cat": False,
            "dog": False,
        }
    except Exception as e:
        result = {
            "status": False,
            "message": f"{idx} {link} Unexpected error occurred.",
            "size": 0,
            "cat": False,
            "dog": False,
        }

    print(result["message"])

    # здесь происходит обработка изображения и вызов нейронки примерно так:
    # меняем размер
    # загружаем модель
    # сохраняем по директориям
    return result


def create_report(results: Iterator[dict], start: float, end: float):
    stat = {"files_count": 0, "errors": 0, "downloaded": 0, "cats": 0, "dogs": 0}
    for i in results:
        if i["status"]:
            stat["files_count"] += 1
            stat["downloaded"] += i["size"]
        else:
            stat["errors"] += 1
    print("Total time: ", end - start)
    print("Files downloaded: ", stat["files_count"])
    print("Data downloaded in bytes: ", stat["downloaded"])
    print("Cats: ", stat["cats"])
    print("Dogs: ", stat["dogs"])
    print("Errors count: ", stat["errors"])


if __name__ == "__main__":
    # параметры запуска тут, пока прописаны вручную
    file_name = "urllist.txt"
    threads = 4
    start = time.time()
    with ThreadPoolExecutor(max_workers=threads) as pool:
        results = pool.map(process_image, get_link(filename=file_name))
    end = time.time()
    create_report(results, start, end)

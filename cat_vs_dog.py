import os
import time
import urllib.request
from multiprocessing.pool import ThreadPool
from urllib.error import URLError


def get_links(filename):
    if not os.path.exists(filename):
        raise FileNotFoundError("Such file does not exist.")
    else:
        with open(filename, "r") as f:
            for line in f:
                yield line


def download_image(url: str, count: int):
    """
    A function that downloads an image from the Internet
    :param url: str
    :param count: int
    :return: image: response body of the URL
    """
    opener = urllib.request.build_opener()
    opener.addheaders = [
        (
            "User-Agent",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.88 Safari/537.36",
        )
    ]
    urllib.request.install_opener(opener)

    if len(str(count)) <= 5:
        name = "images/" + f"{count:05d}"
    else:
        name = "images/" + str(count)

    with urllib.request.urlopen(url) as response:
        image = response.read()
        # size = len(image)
        with open(name, "wb") as f:
            f.write(image)
    return image


def downloader(idx, link):
    result = dict()
    try:
        img = download_image(link, idx)
        result = {"status": True, "message": f"{idx} {link} downloaded successfully."}
    except Exception as e:
        result = {"status": False, "message": f"{idx} {link} An error occurred"}
    # здесь происходит обработка изображения и вызов нейронки примерно так:
    # process_image(img)
    # model = load_tf_model(path)
    # dir = "cat" if is_cat(model, img) else "dog"
    return result


if __name__ == "__main__":
    start = time.time()
    # параметры запуска тут, пока прописаны вручную
    filename = "urllist.txt"
    threads = 4
    stat = {"files_count": 0, "errors": 0, "downloaded": 0, "cats": 0, "dogs": 0}
    with ThreadPool(processes=threads) as pool:
        for i in pool.starmap(downloader, enumerate(get_links(filename))):
            if i["status"]:
                stat["files_count"] += 1
            else:
                stat["errors"] += 1
            print(i["message"])
    end = time.time()
    print("Total time: ", end - start)
    print("Files downloaded: ", stat["files_count"])
    print("Data downloaded in bytes: ", stat["downloaded"])
    print("Cats: ", stat["cats"])
    print("Dogs: ", stat["dogs"])
    print("Errors count: ", stat["errors"])

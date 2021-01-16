# Image Downloader and CatVSDog Classifier
A simple program for downloading images from the Internet and following image classification.

## Before start

All packages needed are presented in file: [requirements.txt](requirements.txt)
```Bash
pip3 install -r requirements.txt
```
or
```Bash
sudo pip3 install -r requirements.txt
```
Also you need CUDA 10 for correct program execution.

## How to execute

```Bash
python3 cat_vs_dog.py urllist.txt -t 4
```
where urllist.txt is the file with URL links (one link per line). --threads or -t is an optional parameter to set
number of threads to run program in parallel (by default 1).

## Result

Two directories with classified images: 
- cats - directory with cat images
- dogs - directory with dogs

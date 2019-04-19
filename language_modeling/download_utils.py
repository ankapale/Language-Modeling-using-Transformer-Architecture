import csv
import itertools
import logging
import os
import pickle
import random
import string
import sys
import tempfile
import zipfile
from os import makedirs
from os.path import dirname
from os.path import exists
from sys import stdout

import imageio
import numpy as np
import requests
import torch
from scipy.ndimage import zoom
from collections import namedtuple

class Constant:
    # SYS Constant
    SYS_LINUX = 'linux'
    SYS_WINDOWS = 'windows'
    SYS_GOOGLE_COLAB = 'goog_colab'

    # Google drive downloader
    CHUNK_SIZE = 32768
    DOWNLOAD_URL = "https://docs.google.com/uc?export=download"


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None

def save_response_content(response, destination):
    with open(destination, "wb") as f:
        for chunk in response.iter_content(Constant.CHUNK_SIZE):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)

def download_file_from_google_drive(file_id, dest_path, verbose=False):
    """
    Downloads a shared file from google drive into a given folder.
    Optionally unzips it.
    Refact from:
    https://github.com/ndrplz/google-drive-downloader/blob/master/google_drive_downloader/google_drive_downloader.py
    Args:
        verbose:
        file_id: str
            the file identifier.
            You can obtain it from the sharable link.
        dest_path: str
            the destination where to save the downloaded file.
            Must be a path (for example: './downloaded_file.txt')
    """

    destination_directory = dirname(dest_path)
    if len(destination_directory) > 0 and not exists(destination_directory):
        makedirs(destination_directory)

    session = requests.Session()

    if verbose:
        print('Downloading file with Google ID {} into {}... '.format(file_id, dest_path), end='')
    stdout.flush()

    response = session.get(Constant.DOWNLOAD_URL, params={'id': file_id}, stream=True)

    token = get_confirm_token(response)
    if token:
        params = {'id': file_id, 'confirm': token}
        response = session.get(Constant.DOWNLOAD_URL, params=params, stream=True)

    save_response_content(response, dest_path)
    if verbose:
        print('Download completed.')
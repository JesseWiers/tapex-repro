# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import tarfile
import requests
import shutil
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)

# Resources are obtained and modified from https://github.com/pytorch/fairseq/tree/master/examples/bart
RESOURCE_DICT = {
    "bart.large": "https://dl.fbaipublicfiles.com/fairseq/models/bart.large.tar.gz",
    "tapex.large": "https://github.com/microsoft/Table-Pretraining/releases/download/pretrained-model/tapex.large.tar.gz",
    "bart.base": "https://dl.fbaipublicfiles.com/fairseq/models/bart.base.tar.gz",
    "tapex.base": "https://github.com/microsoft/Table-Pretraining/releases/download/pretrained-model/tapex.base.tar.gz"
}

DEFAULT_ENCODER_JSON = "https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/encoder.json"
DEFAULT_VOCAB_BPE = "https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/vocab.bpe"


def download_file(url, output_dir=None, chunk_size=8192, retries=3):
    """
    Download file with retry logic
    """
    for attempt in range(retries):
        try:
            r = requests.get(url, stream=True, timeout=300)  # 5 minute timeout
            r.raise_for_status()
            
            total_size = int(r.headers.get('content-length', 0))
            filename = os.path.basename(url)
            
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                filename = os.path.join(output_dir, filename)
                
            with open(filename, 'wb') as f:
                with tqdm(total=total_size, unit='iB', unit_scale=True) as pbar:
                    for chunk in r.iter_content(chunk_size=chunk_size):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
            return filename
            
        except (requests.exceptions.ChunkedEncodingError, 
                requests.exceptions.ConnectionError) as e:
            if attempt == retries - 1:  # Last attempt
                raise e
            print(f"Download failed, retrying... ({attempt + 1}/{retries})")
            time.sleep(1)  # Wait a bit before retrying


def download_model_weights(resource_dir, resource_name):
    abs_resource_dir = os.path.abspath(resource_dir)
    logger.info("Downloading `model.pt` and `dict.txt` to `{}` ...".format(abs_resource_dir))
    download_url = RESOURCE_DICT[resource_name]
    # download file into resource folder, the file ends with .tar.gz
    file_path = download_file(download_url, resource_dir)
    # unzip files into resource_folder
    tar = tarfile.open(file_path, "r:gz")
    names = tar.getnames()
    for name in names:
        read_f = tar.extractfile(name)
        # if is a file
        if read_f:
            # open a file with the same name
            file_name = os.path.split(name)[-1]
            write_f = open(os.path.join(resource_dir, file_name), "wb")
            write_f.write(read_f.read())
    tar.close()
    logger.info("Copying `dict.txt` to `dict.src.txt` and `dict.tgt.txt` ...")
    # copy dict.txt into dict.src.txt and dict.
    shutil.copy(os.path.join(resource_dir, "dict.txt"),
                os.path.join(resource_dir, "dict.src.txt"))
    shutil.copy(os.path.join(resource_dir, "dict.txt"),
                os.path.join(resource_dir, "dict.tgt.txt"))


def download_bpe_files(resource_dir):
    abs_resource_dir = os.path.abspath(resource_dir)
    logger.info("Downloading `vocab.bpe` and `encoder.json` to `{}` ...".format(abs_resource_dir))
    download_file(DEFAULT_VOCAB_BPE, resource_dir)
    download_file(DEFAULT_ENCODER_JSON, resource_dir)

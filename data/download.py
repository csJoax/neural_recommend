import os
from urllib.request import urlretrieve
from os.path import isfile, isdir
from tqdm import tqdm
import zipfile
import hashlib


def _unzip(save_path, _, database_name, data_path):
    """
    解压
    :param save_path: The path of the gzip files
    :param database_name: Name of database
    :param data_path: Path to extract to
    :param _: HACK - Used to have to same interface as _ungzip
    """
    print('Extracting {}...'.format(database_name))
    with zipfile.ZipFile(save_path) as zf:
        zf.extractall(data_path)


_extract_name_fn = {
    'zip': _unzip
}


def download_extract(database_name: str, data_path: str,
                     url: str, hash_code: str = None,
                     extract_method=None, remove_zipfile=False):
    """
    下载提取数据
    :param database_name: Database name
    """

    extract_path = os.path.join(data_path, database_name)
    file_name = url.split('/')[-1]
    save_path = os.path.join(data_path, file_name)

    if os.path.exists(extract_path):
        print('Found {} Data'.format(database_name))
        return

    if not os.path.exists(data_path):
        os.makedirs(data_path)

    if not os.path.exists(save_path):
        with DLProgress(unit='B', unit_scale=True, miniters=1, desc='Downloading {}'.format(database_name)) as pbar:
            urlretrieve(url, save_path, pbar.hook)

    if hash_code:
        assert hashlib.md5(open(save_path, 'rb').read()).hexdigest() == hash_code, \
            '{} file is corrupted.  Remove the file and try again.'.format(save_path)

    os.makedirs(extract_path)

    if not extract_method:
        extract_method = file_name.split('.', 1)[-1]
        if extract_method in _extract_name_fn:
            extract_fn = _extract_name_fn[extract_method]
        else:
            raise ValueError("无法解压，未知的压缩格式。")
    try:
        extract_fn(save_path, extract_path, database_name, data_path)
    except Exception as err:
        # Remove extraction folder if there is an error
        shutil.rmtree(extract_path)
        raise err

    # Remove compressed data
    if remove_zipfile:
        os.remove(save_path)

    print('Done.')


# Tqdm 是一个快速，可扩展的Python进度条
class DLProgress(tqdm):
    """
    下载时处理进度条
    """
    last_block = 0

    def hook(self, block_num=1, block_size=1, total_size=None):
        """
        A hook function that will be called once on establishment of the network connection and
        once after each block read thereafter.
        :param block_num: A count of blocks transferred so far
        :param block_size: Block size in bytes
        :param total_size: The total size of the file. This may be -1 on older FTP servers which do not return
                            a file size in response to a retrieval request.
        """
        self.total = total_size
        self.update((block_num - self.last_block) * block_size)
        self.last_block = block_num

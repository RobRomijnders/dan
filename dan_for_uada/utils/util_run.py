import zipfile
import os
import logging
import datetime
from glob import glob
import tensorflow as tf
import shutil


def maybe_remove_log_dir(dirname):
    """
    empties the directory if its name is only a digit

    The logging of checkpoints halts when the directory already contains log files.
    So for quick prototyping, log to a directory whose name is only a digit
    :param dirname:
    :return:
    """

    # First find the name of the pending directory
    if dirname[-1] == '/':
        dirname = dirname[:-1]
    name = os.path.basename(dirname)

    # If the name is only a digit, then empty it
    if name.isdigit():
        for file in glob(os.path.join(dirname, '*')):
            try:
                os.remove(file)
            except IsADirectoryError:
                shutil.rmtree(file)


def setup_logger(dirname, name=''):
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    # create logger
    logger = logging.getLogger('semantic_segmentation')
    logger.setLevel(logging.DEBUG)

    # create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    # create formatter
    formatter = logging.Formatter('%(asctime)s [%(levelname)s]: %(message)s')

    # add formatter to ch
    ch.setFormatter(formatter)

    # create file handler
    file_handler = logging.FileHandler(os.path.join(dirname, 'semantic_segmentation_%s_%s.log' %
                                                    (name, datetime.datetime.now().isoformat())))
    print(os.path.join(dirname, 'semantic_segmentation%s.log' % name))
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(ch)

    # Tensorflow does its own logging
    # So we add our file handler also to tensorflow's logging
    tf_logger = logging.getLogger('tensorflow')
    tf_logger.addHandler(file_handler)
    return logger


def zipit(path, archname):
    """
    Inspired from https://gist.github.com/felixSchl/d38b455df8bf83a78d3d
    :param path: directory that you want to zip
    :param archname: name of the archive
    :return:
    """
    archive = zipfile.ZipFile(archname, "w", zipfile.ZIP_DEFLATED)
    if os.path.isdir(path):
        _zippy(path, path, archive)
    else:
        _, name = os.path.split(path)
        archive.write(path, name)
    archive.close()


def _zippy(base_path, path, archive):
    paths = os.listdir(path)
    for p in paths:
        p = os.path.join(path, p)
        if os.path.isdir(p):
            _zippy(base_path, p, archive)
        else:
            if os.path.splitext(p)[1] == '.py':
                archive.write(p, os.path.relpath(p, base_path))


if __name__ == '__main__':
    # Example use
    zipit('/home/mps/Documents/semantic-segmentation-fork/semantic-segmentation', 'test.zip')

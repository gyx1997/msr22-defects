import os


def rmdir(dir):
    """
    Remove a directory recursively.

    :param dir: The directory to be removed.
    """
    dir = dir  # type: str
    dir = dir.replace("\\", "/")
    if os.path.isdir(dir):
        for subdir in os.listdir(dir):
            rmdir(os.path.join(dir, subdir))
        if os.path.exists(dir):
            os.rmdir(dir)
    else:
        if os.path.exists(dir):
            os.remove(dir)
    return

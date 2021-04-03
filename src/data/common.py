
import zipfile


class ZipReader(object):
    file_dict = dict()

    def __init__(self):
        super(ZipReader, self).__init__()

    @staticmethod
    def build_file_dict(path):
        file_dict = ZipReader.file_dict
        if path in file_dict:
            return file_dict[path]
        else:
            file_handle = zipfile.ZipFile(path, mode='r', allowZip64=True)
            file_dict[path] = file_handle
            return file_dict[path]

    @staticmethod
    def imread(path, image_name):
        zfile = ZipReader.build_file_dict(path)
        data = zfile.read(image_name)
        im = Image.open(io.BytesIO(data))
        return im



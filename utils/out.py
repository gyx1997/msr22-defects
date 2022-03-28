import sys
import time


class Out:
    logging_file_descriptor = None
    data_file_descriptor = None

    @classmethod
    def init(cls, **kwargs):
        """
        Initialize the output file(s) as well as stdout.

        :key logging: Required. The name of file where logging info will be outputted to.
        :key data: Required. The name of file where data will be outputted to.
        """
        if cls.logging_file_descriptor is None and cls.data_file_descriptor is None:

            logging_filename = kwargs.get("logging", None)
            if logging_filename is None:
                raise ValueError("Parameter 'logging' required.")

            data_filename = kwargs.get("data", None)
            if data_filename is None:
                raise ValueError("Parameter 'data' required.")

            cls.logging_file_descriptor = open(logging_filename, "w+")
            cls.data_file_descriptor = open(data_filename, "w+")

        else:
            raise RuntimeError("Out cannot be initialized more than once.")

    @classmethod
    def _time_str(cls):
        return time.strftime('%y-%m-%d %H:%M:%S', time.localtime(time.time()))

    @classmethod
    def write_time(cls):
        current_time = cls._time_str()
        for file in (sys.stdout, cls.logging_file_descriptor):
            file.write("  " + current_time + "\n")

    @classmethod
    def flush(cls):
        for file in (sys.stdout, cls.logging_file_descriptor):
            file.flush()

    @classmethod
    def write(cls, log=""):
        for file in (sys.stdout, cls.logging_file_descriptor):
            file.write(log + "\n")

    @classmethod
    def write_data(cls, data):
        cls.data_file_descriptor.write(data + "\n")
        cls.data_file_descriptor.flush()

    @classmethod
    def close(cls):
        cls.logging_file_descriptor.close()
        cls.logging_file_descriptor = None
        cls.data_file_descriptor.close()
        cls.data_file_descriptor = None

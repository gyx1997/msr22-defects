import os.path

import javalang
import pandas
import numpy
import copy
import pickle

from tqdm import tqdm

from preprocess.java_parser.ast2seq import AST2Seq, BaseAST2Seq
import preprocess.java_parser.javafile
from utils.out import Out


class Project:
    """
    Class of a single PROMISE project.
    """

    def __init__(self, **kwargs):
        """
        Constructor of a single project of PROMISE dataset.
        :param dataset: String of dataset name. In this case must be "promise".
        :param name: The project name string formatted as `name`-`version`
        """
        super().__init__()

        # Declare the data structure(s) for package information of the dataset.
        self.__modules = []
        self.__c2jf = {}  # Class name to JavaFile => c2jf
        self.__c2bl = {}  # Class name to Buggy Label => c2bl
        self.__c2s = {}  # Class name to Sequence => c2s
        self.__c2tokens = {}  # Class name to Tokens => c2tok
        self.__c2sm = {}  # Class name to Static Metrics => c2sm
        self.__c2vsp = {}  # Class name To Vectorized Sequence after Padding.
        self.__c2ast = {}  # Class name To Abstract Syntax Tree => c2ast.
        # Declare the meta data.
        self.meta_info = pandas.DataFrame()
        # Declare the numerical vectors
        self.dict_c2sv = {}  # Class name to Sequence Vectors => c2sv
        self.dict_c2botv = {}  # Class name to Bag of Token Vectors => c2botv

        # Get the dataset object.
        self.dataset = kwargs.get("dataset", None)
        if self.dataset is None:
            raise ValueError("Dataset object must be specified.")

        # Get the dataset path.
        self.dataset_path = self.dataset.data_dir

        # Get the project name from parameters.
        project_name = kwargs.get("name", None)
        if project_name is None:
            raise ValueError("Project name must be specified.")

        # Check the validness of project name.
        project_name_splited = project_name.split("-")
        if len(project_name_splited) != 2:
            raise ValueError("Invalid project name string `{}`. "
                             "It should be specified as the format of `name`-`version`."
                             .format(project_name))

        # Storage the project name and version in class variables.
        self.project_name, self.project_version = project_name_splited

        # Get the full path of the project.
        self.project_dir = "".join(
            [self.dataset_path,
             os.path.sep,
             self.project_name,
             os.path.sep,
             self.project_version,
             os.path.sep]
        )

        # Check the existence of the directory.
        if not os.path.exists(self.project_dir):
            raise FileExistsError("Project `{}-{}` does not exist in directory `{}`."
                                  .format(self.project_name,
                                          self.project_version,
                                          self.project_dir))

        # Check the existence of Metadata. Note that metadata can be downloaded
        # in mirror of 'repo.openscience.us' on Github.
        metadata_filename = self.project_dir + Dataset.metadata_filename
        if not os.path.isfile(metadata_filename):
            raise RuntimeError("Metadata of project '{}-{}' missing."
                               .format(self.project_name,
                                       self.project_version))

        # Get node types to be captured.
        self.java2seq = kwargs.get("ast2seq", None)
        if self.java2seq is None:
            self.java2seq = AST2Seq(accepted_tokens=[])
            raise UserWarning("None of AST Nodes will be captured since the `node_type` parameter is set to empty "
                              "list `[]`.")

        # Load the metadata of the project
        self.__load_meta_data(metadata_filename)

        # Get the sequence representation for each package in metadata.
        self.__load_java_source_files()

    @property
    def file(self):
        """
        Returns the dictionary which maps class name to
        'preprocess.java_parser.JavaFile'.
        """
        return self.__c2jf

    @property
    def classes(self):
        return self.__modules

    @property
    def buggy_labels(self):
        """
        Get the dictionary which maps class name to buggy label.
        """
        return self.__c2bl

    @property
    def token_sequences(self):
        """
        Get the dictionary which maps class name to code element sequence.
        """
        return self.__c2s

    @property
    def tokens(self):
        return self.__c2tokens

    @property
    def static_metrics(self):
        """
        Get the dictionary which maps class name to static metrics.
        """
        return self.__c2sm

    @property
    def abstract_syntax_tree(self):
        return self.__c2ast

    def __load_meta_data(self, metadata_filename):
        """
        Load the meta data of the dataset.
        """
        Out.write("Now loading metadata from project `{}-{}`".format(self.project_name, self.project_version))

        # Load meta data of a dataset.
        self.meta_info = pandas.read_csv(metadata_filename)

        Out.write("{} instances found.".format(self.meta_info.shape[0]))

        # Split traditional metrics, package name and buggy label.
        for index, tup in self.meta_info.iterrows():
            traditional_static_metrics = tup.drop(["name", "name.1", "version", "bug"])
            # Transform the count of bugs into the buggy flag.
            buggy_flag = 0
            if tup["bug"] > 0:
                buggy_flag = 1
            # Get the class name
            class_name = tup["name.1"]
            # Update class member variables.
            self.__c2bl[class_name] = buggy_flag
            self.__c2sm[class_name] = traditional_static_metrics

    def __load_java_source_files(self):
        """
        Helper method to load java source file and convert it into token sequence.
        """

        # Get Java2Seq Transformer.
        # Note that accepted tokens should be specified with the Java2Seq object.
        if not isinstance(self.java2seq, BaseAST2Seq):
            raise TypeError("`preprocess.java_parser.ast2seq.AST2Seq` expected, got {}.".format(type(self.java2seq)))

        Out.write("Now loading java source files.")

        # Declare a dict to map java_parser class name to sequence
        dict_c2s = {}
        dict_c2tok = {}
        dict_c2ast = {}
        dict_c2file = {}

        # Walk through the source code directory and parse each java source file.
        for root, dirs, files in os.walk(self.project_dir):
            for f in files:
                # Get the current filename, and process it if it is a java_parser file.
                current_filename = os.path.join(root, f).replace("/", os.path.sep)
                if current_filename.endswith(".java"):
                    try:
                        # Parse java source file and get tokens and the token sequence.
                        java_src = preprocess.java_parser.javafile.JavaFile.parse(current_filename)
                        class_name = java_src.class_name
                        # Parse AST and capture the specified nodes.
                        tokens, token_sequence = self.java2seq.parse(java_src)
                        # Build the accepted token set.
                        captured_tokens = {}
                        for token in token_sequence:
                            captured_tokens[token.str(show_prefix=False)] = token
                        dict_c2ast[class_name] = java_src.ast
                        dict_c2s[class_name] = token_sequence
                        dict_c2tok[class_name] = tokens
                        dict_c2file[class_name] = java_src
                    except RuntimeError:
                        Out.write("Java source file without a class '{}'. Ignored.".format(current_filename))
                    except javalang.parser.JavaSyntaxError:
                        Out.write("Invalid java source file. '{}'. Ignored.".format(current_filename))
                    except UnicodeDecodeError:
                        Out.write("File unicode error '{}'. Ignored.".format(current_filename))

        # First declare the counter for missing/corrupted file.
        missing_count = 0
        # For each record in meta data, try to find the corresponding sequence for each class,
        # or set the missing flag.
        for class_name in self.__c2bl.keys():
            if dict_c2ast.__contains__(class_name):
                # Update the project's data.
                self.__c2s[class_name] = dict_c2s[class_name]
                self.__c2tokens[class_name] = dict_c2tok[class_name]
                self.__c2ast[class_name] = dict_c2ast[class_name]
                self.__c2jf[class_name] = dict_c2file[class_name]
                self.__modules.append(class_name)
            else:
                missing_count += 1
                Out.write("Missing or corrupted file {}.".format(class_name))

        Out.write("Total Missing or corrupted file {}.".format(missing_count))

    def data(self, **kwargs):
        """
        Get filenames, matrix of sequence representation,
        matrix of bag of tokens, matrix of traditional static metrics
        and vector of labels.

        :key show_token_prefix: If it is set True, the string of a token will include its prefix.
        :rtype: tuple
        """

        show_prefix = kwargs.get("show_token_prefix", False)

        classes = []
        matrix_sequence = []
        matrix_static_metrics = []
        vector_labels = []
        java_files = []

        for class_name in self.classes:
            classes.append(class_name)
            java_files.append(self.file[class_name])
            class_token_sequence = []

            for token in self.token_sequences[class_name]:
                class_token_sequence.append(token.str(show_prefix=show_prefix))
            matrix_sequence.append(class_token_sequence)

            matrix_static_metrics.append(self.static_metrics[class_name])
            vector_labels.append(self.buggy_labels[class_name])

        # Convert python list to numpy ndarray.
        npmat_traditional = numpy.array(matrix_static_metrics)
        npvec_labels = numpy.array(vector_labels)

        return classes, matrix_sequence, npmat_traditional, npvec_labels, java_files

    def raw_sequence(self, **kwargs):
        """
        Get the raw token sequence of a class.
        """
        class_name = kwargs.get("class_name", "")
        if self.__c2s.__contains__(class_name):
            return self.__c2s[class_name]
        raise ValueError("Invalid Class Name.")


class Dataset:
    """
    Dataset of 2 paired-projects which are training project and test project.
    """

    metadata_filename = "metadata.csv"

    def __init__(self, **kwargs):
        """
        Constructor of paired-project dataset with source code.
        """

        # Declare the global token dictionary.
        self.dict_gt2i = {}
        self.dict_i2gt = []

        # Parameters initialization.

        # Get dataset path.
        self.data_dir = kwargs.get("data_dir", None)
        # Get the minimum appearance of a token to be captured.
        self.minimum_appearance = kwargs.get("min_appear", 3)

        # Get the java_parser object.
        self.java2seq = kwargs.get("ast2seq", None)
        if self.java2seq is None:
            raise ValueError("Parameter java_parser must not be None.")

        # Check the correctness of data path.
        if self.data_dir is None:
            raise RuntimeError("Data path must be specified.")
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError("Cannot found the specified data path '{}'.".format(self.data_dir))

        # Load source project and target project.
        self.source_project = Project(name=kwargs.get("source_project", ""), dataset=self, ast2seq=self.java2seq)
        self.target_project = Project(name=kwargs.get("target_project", ""), dataset=self, ast2seq=self.java2seq)

    @property
    def global_token_count(self):
        return len(self.dict_gt2i)

    @property
    def global_tokens(self):
        return self.dict_i2gt

    @staticmethod
    def save(dataset, filename):
        """
        Save a given dataset.
        """
        file = open(filename, "wb")
        pickle.dump(dataset, file)
        file.close()

    @staticmethod
    def load(filename):
        """
        Load a dataset.
        """
        file = open(filename, "rb")
        result = pickle.load(file)
        file.close()
        return result

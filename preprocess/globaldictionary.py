import numpy

import preprocess.dataset
from utils.out import Out
from tqdm import tqdm


class GlobalDictionary:
    """
    Global dictionary of captured tokens from all source code files.
    """

    def __init__(self, **kwargs):
        """
        Constructor of GlobalDictionary.

        :key projects: The projects used for dictionary construction.

        :key minimum_appearance: [Optional] The minimum appearance of a token which should be captured.
            Default is 3 if it is not specified by following the common practice.
        :key verbose: [Optional] The flag for debugging output. Default is True.
        """

        self.projects = kwargs.get("projects", {})
        if len(self.projects) == 0:
            raise ValueError("At least 1 project should be specified.")
        self.minimum_appearance = kwargs.get("minimum_appearance", 3)
        self.include_type = kwargs.get("include_type", False)
        self.verbose = True

        # self.global_tokens storages the count of a token
        self.__token_count = {}

        # Add a dummy <PADDING> token to the dictionary.
        # i2t -> index 2 token, t2i -> token 2 index, idf -> inverse document frequency.
        self.__i2t = ["<PADDING>"]
        self.__t2i = {"<PADDING>": 0}
        self.__idf = {}

        # Build global tokens
        self.__build_global_tokens()

        # Build inverse document frequency
        self.__build_inv_doc_freq()

        # Get max length of those sequences.
        self.__seq_max_length = self.__get_sequence_max_length()

    def contains_token(self, token_name):
        """
        Test whether a token appears.

        :param token_name: The name of the given token.
        :rtype: bool
        """
        return self.__token_count.__contains__(token_name)

    def token_appearance_count(self, token_name):
        """
        Get the number of appearance for a given token.

        :param token_name: The name of the given token.
        :rtype: int
        """
        return self.__token_count[token_name]

    @property
    def id2token(self):
        """
        A list of index2token.

        :rtype: list
        """
        return self.__i2t

    @property
    def token2id(self):
        """
        A dictionary of token2index.
        """
        return self.__t2i

    def get_token_id(self, token_name):
        """
        Get the integer id of a specific token.

        :param token_name: The queried token.
        :return: An integer which represents the id.
        """
        return self.__t2i.get(token_name, -1)

    @property
    def token_count(self):
        """
        Get the count of the tokens which satisfy the minimum appearance criterion in all projects.
        """
        return len(self.__i2t)

    @property
    def inverse_document_frequency(self):
        return self.__idf

    @property
    def max_sequence_length(self):
        """
        Get the maximum length of all code token sequences.
        """
        return self.__seq_max_length

    def __build_global_tokens(self):
        """
        Build global token list for Seq2Vec.
        """
        Out.write_time()
        Out.write("Building global tokens...")

        # Initialize the global token dictionary.
        self.__token_count = {}

        # Count the tokens
        for project in self.projects:
            self.__count_project_tokens(project)

        # Build the tokens
        self.__build_project_tokens()

    def __count_project_tokens(self, project):
        """
        Count the tokens appeared in a given project.
        """

        if not isinstance(project, preprocess.dataset.Project):
            raise TypeError("Only type of 'preprocess.dataset.Project' can be considered as a project.")

        # Process each token of the project
        for _, module_tokens in project.tokens.items():
            for token_name, token_count in module_tokens.items():
                if not self.__token_count.__contains__(token_name):
                    self.__token_count[token_name] = token_count
                else:
                    self.__token_count[token_name] += token_count

    def __build_project_tokens(self):
        """
        Add tokens appeared in a given project to the global token list with minimum appearance criterion.
        :rtype: None
        """
        # Iterate all the possible tokens, and then select the tokens satisfying the
        # minimum appearance criterion.

        for token, count in (
           tqdm(self.__token_count.items())
           if self.verbose else self.__token_count.items()
        ):
            if count >= self.minimum_appearance:
                self.__t2i[token] = len(self.__i2t)
                self.__i2t.append(token)

    def __get_sequence_max_length(self):
        """
        Get the maximum length of all token sequences in all project(s).
        """
        max_len = 0
        for project in self.projects:
            for seq in project.token_sequences.values():
                if len(seq) > max_len:
                    max_len = len(seq)
        return max_len

    def __calc_token_inv_doc_freq(self, token):
        """
        Calculate the inverse document frequency (known as IDF) of a given token.
        """

        num_modules_with_token = 1  # Laplace correction.
        num_modules = 0

        for project in self.projects:
            for module, module_tokens in project.tokens.items():
                num_modules += 1
                for token_name, token_count in module_tokens.items():
                    if token_name == token:
                        num_modules_with_token += 1
                        break

        return num_modules / num_modules_with_token

    def __build_inv_doc_freq(self):
        """
        Build inverse document frequency.
        """
        Out.write_time()
        Out.write("Building inverse document frequency...")

        for token in (
                tqdm(self.id2token) if self.verbose else self.id2token
        ):
            inv_doc_freq = self.__calc_token_inv_doc_freq(token)
            self.__idf[token] = inv_doc_freq

        self.__idf["<PADDING>"] = 0

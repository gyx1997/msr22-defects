"""
Parsers for transforming Java abstract syntax tree into token
sequence by pre-order traversal.
"""

import javalang
import javalang.tree
import javalang.ast

import preprocess.java_parser.javafile


class ASTParser:
    """
    Base file for parsing java source file to sequence.
    """
    tokenizers = {}

    def __init__(self, **kwargs):

        # Initialize tokenizers.
        self.tokenizers = {}
        tokenizers = kwargs.get("accepted_tokens", [])
        self.init_tokenizers(tokenizers)

    def init_tokenizers(self, accepted_tokens=None):
        if accepted_tokens is None:
            accepted_tokens = []
        for tokenizer in accepted_tokens:
            self.tokenizers[tokenizer.type()] = tokenizer

    def tokenize(self, node, trace=None):
        """
        Tokenize a given javalang AST node.

        :param node: The node to be tokenized.
        :param trace: The node trace.
        :returns: string token if the node needs to be captured, otherwise None.
        :rtype: preprocess.java_parser.tokenizers.Token
        """
        node_type = type(node)
        return self.tokenizers[node_type].tokenize(node=node, trace=trace) \
            if self.tokenizers.__contains__(node_type) \
            else None


class BaseAST2Seq(ASTParser):
    """
    Base class for transforming a java source file into a token sequence.
    Instances of this class will result in empty token sequence, since no
    walk method of AST is specified.
    """
    tokenizers = {}

    def __init__(self, **kwargs):
        # Initialize tokenizers.
        super().__init__(**kwargs)

    def parse(self, java_file):
        if not isinstance(java_file, preprocess.java_parser.javafile.JavaFile):
            raise ValueError("Only `preprocesss.java_parser.javafile.JavaFile` could be parsed.")

        # Get the token sequence of this file.
        tokens, token_seq = self.walk(java_file.ast)
        return tokens, token_seq

    def walk(self, ast_root):
        return {}, []

    pass


class AST2Seq(BaseAST2Seq):
    """
    Convertor by pre-order traversing the AST nodes.
    """

    def __init__(self, **kwargs):

        # Initialize
        super().__init__(**kwargs)

    def walk(self, ast_root):

        # The final token sequence.
        token_sequence = []
        # The final token dictionary (for Bag of Tokens)
        tokens = {}

        # Walk in the AST and parse each node.
        for node in ast_root:
            # Get the current node.
            currentNode = node[1]
            # Parse the node.
            token = self.tokenize(currentNode, node[0])
            # Accepted tokens will be added into the 'token_sequence'.
            if token is not None:
                token_sequence.append(token)
                # Counter increment of current token
                if tokens.__contains__(token):
                    tokens[token.str(show_prefix=False)] += 1
                else:
                    tokens[token.str(show_prefix=False)] = 1

        return tokens, token_sequence


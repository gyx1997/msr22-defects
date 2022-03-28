import javalang
import javalang.tree


class JavaFile:
    """
    Class of a java source file.
    """

    def __init__(self, **kwargs):
        self.__class_name = kwargs.get("class_name", None)
        if self.__class_name is None:
            raise ValueError("Java class name must be specified.")

        self.__ast = kwargs.get("ast", None)
        if not isinstance(self.__ast, javalang.tree.Node):
            raise ValueError("AST must be type or subclass of `javalang.tree.Node`.")

        self.__tokens = kwargs.get("tokens", [])

    @property
    def class_name(self):
        """
        Returns the class name defined in the java file.
        """
        return self.__class_name

    @property
    def ast(self):
        """
        Returns the abstract syntax tree of the java file.
        """
        return self.__ast

    @property
    def tokens(self):
        """
        Returns the tokens (after lexical analysis) of the java file.
        :return:
        """
        return self.__tokens

    @staticmethod
    def parse(filename=None):
        if filename is None:
            raise ValueError("Java source filename must be specified when parsing a java file.")

        # Read java source file.
        java_file = open(filename, "r", encoding='utf-8')
        java_text = java_file.read()
        java_file.close()

        # Use package 'javalang' to parse a java file into the Abstract Syntax Tree (AST).
        ast = javalang.parse.parse(java_text)
        tokens_generator = javalang.tokenizer.tokenize(java_text)
        tokens = []
        for token in tokens_generator:
            tokens.append(token)
        # Check whether it is a valid java file.
        if len(ast.types) != 1:
            raise RuntimeError("Invalid Java source file {}.".format(filename))

        class_name = ast.types[0].name
        package_name = ast.package.name if hasattr(ast.package, "name") else ""
        # Get the full name of this class of the file.
        class_name = package_name + "." + class_name
        return JavaFile(class_name=class_name, ast=ast, tokens=tokens)

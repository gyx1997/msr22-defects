import javalang
import javalang.tree
from abc import ABC, ABCMeta, abstractmethod


class Token:
    """
    Class of tokens.
    """

    def __init__(self, **kwargs):
        """
        Constructor of Token.

        :key token_str: Required. The string representation of a token.
        :key token_type: Required. The type of the token.
        :key line: Required. The line of current token.
        """
        self.token_str = kwargs.get("token_str", None)
        if self.token_str is None:
            raise ValueError("Parameter 'token_str' must not be None.")

        self.token_type = kwargs.get("token_type", None)
        if self.token_type is None:
            raise ValueError("Parameter 'token_type' must not be None.")

        self.line = kwargs.get("token_line", -1)
        self.statement_node = kwargs.get("statement_node", None)

    def __str__(self):
        """
        Overrides the internal str function.
        """
        return self.str(show_prefix=True)

    def str(self, **kwargs):
        """
        Convert a token into a string representation.

        :key show_prefix: Optional. '[Prefix]TokenString' will be returned if
            'show_prefix' is set True. Default False.
        :rtype: str
        """
        show_prefix = kwargs.get("show_prefix", False)
        if show_prefix:
            return "[" + str(self.token_type) + "]" + self.token_str
        return self.token_str

    def type(self):
        """
        Get the type of the token.

        :rtype: str
        """
        return self.token_type


class BaseTokenizer(metaclass=ABCMeta):
    """
    Base tokenizer for javalang nodes.
    """

    # Defines the nodes which are considered as statements.
    statement_nodes = [
        # javalang.tree.Statement,
        # javalang.tree.Statement is not included since its output for control flow nodes
        # such as if-statements are not stable, i.e., sometimes the condition clause is
        # assumed as part of the if-statement, which will be thought as one statement,
        # while sometimes not.
        javalang.tree.StatementExpression,
        javalang.tree.LocalVariableDeclaration,
        javalang.tree.FieldDeclaration,
        javalang.tree.ReturnStatement,
        javalang.tree.ThrowStatement,
        javalang.tree.ForControl,
        javalang.tree.EnhancedForControl,
        javalang.tree.FormalParameter,
    ]

    @classmethod
    def _is_statement_node(cls, node):
        for statement_node_type in cls.statement_nodes:
            if isinstance(node, statement_node_type):
                return True
        return False

    @staticmethod
    def type():
        """
        Get the class type of current tokenizer.
        """
        raise RuntimeError("Cannot call method of abstract class.")

    @classmethod
    def tokenize(cls, **kwargs):
        """
        Get the token object of current node.
        """

        node = kwargs.get("node", None)
        if node is None:
            raise ValueError()

        trace = kwargs.get("trace", [])
        #if trace is None:
        #    raise ValueError()

        # Get the node type and node string.
        node_type, node_str = cls()._parse_node(node)

        # Get the node's line. Since some types of nodes do not have attribute _position,
        # we need to iterate its trace and get its parent node's _position. Hence, we can
        # only get line(s) of nodes. The column(s) are not accurate, and we ignore them.

        node_line = -1
        if hasattr(node, "_position"):
            node_line = node._position.line
        else:
            i = len(trace) - 1
            while i >= 0:
                if hasattr(trace[i], "_position"):
                    node_line = trace[i]._position.line
                    break
                i -= 1

        # Get the node's statement.

        statement_node = None
        i = len(trace) - 1
        while i >= 0:
            for statement_node_type in cls.statement_nodes:
                if isinstance(trace[i], statement_node_type):
                    statement_node = trace[i]
                    break
            if statement_node is not None:
                # In this situation, the statement node has been found.
                break
            i -= 1

        # If no statement node of current node is found, we thought itself as a statement node.
        if statement_node is None:
            statement_node = node

        return Token(token_str=node_str,
                     token_type=node_type,
                     token_line=node_line,
                     statement_node=statement_node)

    @classmethod
    @abstractmethod
    def _parse_node(cls, node):
        """
        Protected method for parsing node(s) of java abstract syntax tree.

        :param node: The current node to be parsed.
        :return: A tuple with node type and node string.
        """
        raise RuntimeError("Cannot call method of abstract class.")


class BreakStatement(BaseTokenizer):
    """
    Tokenizer for BreakStatement.
    """

    @staticmethod
    def type():
        return type(javalang.tree.BreakStatement())

    @classmethod
    def _parse_node(cls, node):
        return "breakStatement", "break"


class Import(BaseTokenizer):
    """
    Tokenizer for Import in javalang.
    """

    @staticmethod
    def type():
        return type(javalang.tree.Import())

    @classmethod
    def _parse_node(cls, node):
        if node is None:
            raise ValueError("Node must be specified when parsing an import node.")
        return "Import", node.path


class DoStatement(BaseTokenizer):
    """
    Tokenizer for DoStatement in javalang.
    """

    @staticmethod
    def type():
        return type(javalang.tree.DoStatement())

    @classmethod
    def _parse_node(cls, node):
        return "DoStatement", "do"


class ForStatement(BaseTokenizer):
    """
    Tokenizer for ForStatement in javalang.
    """

    @staticmethod
    def type():
        return type(javalang.tree.ForStatement())

    @classmethod
    def _parse_node(cls, node):
        return "ForStatement", "for"


class ForControl(BaseTokenizer):
    """
    Tokenizer for ForControl in javalang.
    This type of node may be unnecessary since every for loop
    would have a ForControl or EnhancedForControl after the ForStatement.
    Anyway, tokenizer for this type of node is provided here.
    """

    @staticmethod
    def type():
        return type(javalang.tree.ForControl())

    @classmethod
    def _parse_node(cls, node):
        return "ForControl", "forControl"


class EnhancedForControl(BaseTokenizer):
    """
    Tokenizer for EnhancedForControl in javalang.
    This type of node may be unnecessary since every for loop
    would have a ForControl or EnhancedForControl after the ForStatement.
    Anyway, tokenizer for this type of node is provided here.
    """

    @staticmethod
    def type():
        return type(javalang.tree.EnhancedForControl())

    @classmethod
    def _parse_node(cls, node):
        return "EnhancedForControl", "enhancedForControl"


class WhileStatement(BaseTokenizer):
    """
    Tokenizer for WhileStatement.
    """

    @staticmethod
    def type():
        return type(javalang.tree.WhileStatement())

    @classmethod
    def _parse_node(cls, node):
        return "WhileStatement", "while"


class ContinueStatement(BaseTokenizer):
    """
    Tokenizer for ContinueStatement.
    """

    @staticmethod
    def type():
        return type(javalang.tree.ContinueStatement())

    @classmethod
    def _parse_node(cls, node):
        return "ContinueStatement", "continue"


class IfStatement(BaseTokenizer):
    """
    Tokenizer for IfStatement.
    """

    @staticmethod
    def type():
        return type(javalang.tree.IfStatement())

    @classmethod
    def _parse_node(cls, node):
        return "IfStatement", "if"


class SwitchStatement(BaseTokenizer):
    """
    Tokenizer for SwitchStatement.
    """

    @staticmethod
    def type():
        return type(javalang.tree.SwitchStatement())

    @classmethod
    def _parse_node(cls, node):
        return "SwitchStatement", "switch"


class SwitchStatementCase(BaseTokenizer):
    """
    Tokenizer for SwitchStatementCase.
    """

    @staticmethod
    def type():
        return type(javalang.tree.SwitchStatementCase())

    @classmethod
    def _parse_node(cls, node):
        return "SwitchStatementCase", "case"


class ThrowStatement(BaseTokenizer):
    """
    Tokenizer for ThrowStatement.
    """

    @staticmethod
    def type():
        return type(javalang.tree.ThrowStatement())

    @classmethod
    def _parse_node(cls, node):
        return "ThrowStatement", "throw"


class AssertStatement(BaseTokenizer):
    """
    Tokenizer for AssertStatement.
    """

    @staticmethod
    def type():
        return type(javalang.tree.AssertStatement())

    @classmethod
    def _parse_node(cls, node):
        return "AssertStatement", "assert"


class ReturnStatement(BaseTokenizer):
    """
    Tokenizer for ReturnStatement.
    """

    @staticmethod
    def type():
        return type(javalang.tree.ReturnStatement())

    @classmethod
    def _parse_node(cls, node):
        return "ReturnStatement", "return"


class SynchronizedStatement(BaseTokenizer):
    """
    Tokenizer for SynchronizedStatement.
    """

    @staticmethod
    def type():
        return type(javalang.tree.SynchronizedStatement())

    @classmethod
    def _parse_node(cls, node):
        return "SynchronizedStatement", "synchronized"


class TryStatement(BaseTokenizer):
    """
    Tokenizer for TryStatement.
    """

    @staticmethod
    def type():
        return type(javalang.tree.TryStatement())

    @classmethod
    def _parse_node(cls, node):
        return "TryStatement", "try"


class TryResource(BaseTokenizer):
    """
    Tokenizer for TryResource.
    """

    @staticmethod
    def type():
        return type(javalang.tree.TryResource())

    @classmethod
    def _parse_node(cls, node):
        return "TryResource", "tryResource"


class CatchClause(BaseTokenizer):
    """
    Tokenizer for TryStatement.
    """

    @staticmethod
    def type():
        return type(javalang.tree.CatchClause())

    @classmethod
    def _parse_node(cls, node):
        return "CatchClause", "catch"


class CatchClauseParameter(BaseTokenizer):
    """
    Tokenizer for TryStatement.
    """

    @staticmethod
    def type():
        return type(javalang.tree.CatchClauseParameter())

    @classmethod
    def _parse_node(cls, node):
        return "CatchClauseParameter", "catchParameter"


class BlockStatement(BaseTokenizer):
    """
    Tokenizer for BlockStatement.
    """

    @staticmethod
    def type():
        return type(javalang.tree.BlockStatement())

    @classmethod
    def _parse_node(cls, node):
        return "BlockStatement", "blockStatement"


""" Invocation-related Nodes. """


class MethodInvocation(BaseTokenizer):
    """
    Tokenizer for MethodInvocation.
    """

    @staticmethod
    def type():
        return type(javalang.tree.MethodInvocation())

    @classmethod
    def _parse_node(cls, node):
        if node is None:
            raise ValueError("Node must be specified when parsing a MethodInvocation node.")
        # Make full method name.
        if node.qualifier is not None and len(node.qualifier) > 0:
            methodName = node.qualifier + "." + node.member
        else:
            methodName = node.member
        methodName = node.member
        return "MethodInvocation", methodName


class SuperMethodInvocation(BaseTokenizer):
    """
    Tokenizer for MethodInvocation.
    """

    @staticmethod
    def type():
        return type(javalang.tree.SuperMethodInvocation())

    @classmethod
    def _parse_node(cls, node):
        if node is None:
            raise ValueError("Node must be specified when parsing a SuperMethodInvocation node.")
        # Make full method name.
        if node.qualifier is not None and len(node.qualifier) > 0:
            methodName = node.qualifier + "." + node.member
        else:
            methodName = node.member
        methodName = node.member
        return "SuperMethodInvocation", methodName


class ClassCreator(BaseTokenizer):
    """
    Tokenizer for ClassCreator.
    """

    @staticmethod
    def type():
        return type(javalang.tree.ClassCreator())

    @classmethod
    def _parse_node(cls, node):
        if node is None:
            raise ValueError("Node must be specified when parsing a ClassCreator node.")
        # Get full name of class created.
        classType = node.type
        className = ""
        while classType is not None:
            # className += (classType.name + ".")
            className = classType.name
            classType = classType.sub_type
        # className = className[:-1]  # Remove the last dot.
        return "ClassCreator", className


""" Declaration-related Nodes. """


class MethodDeclaration(BaseTokenizer):
    """
    Tokenizer for MethodDeclaration.
    """

    @staticmethod
    def type():
        return type(javalang.tree.MethodDeclaration())

    @classmethod
    def _parse_node(cls, node):
        if node is None:
            raise ValueError("Node must be specified when parsing a MethodDeclaration node.")
        return "MethodDeclaration", node.name


class EnumDeclaration(BaseTokenizer):
    """
    Tokenizer for EnumDeclaration.
    """

    @staticmethod
    def type():
        return type(javalang.tree.EnumDeclaration())

    @classmethod
    def _parse_node(cls, node):
        if node is None:
            raise ValueError("Node must be specified when parsing a EnumDeclaration node.")
        return "EnumDeclaration", node.name


class InterfaceDeclaration(BaseTokenizer):
    """
    Tokenizer for InterfaceDeclaration.
    """

    @staticmethod
    def type():
        return type(javalang.tree.InterfaceDeclaration())

    @classmethod
    def _parse_node(cls, node):
        if node is None:
            raise ValueError("Node must be specified when parsing a InterfaceDeclaration node.")
        return "InterfaceDeclaration", node.name


class ClassDeclaration(BaseTokenizer):
    """
    Tokenizer for ClassDeclaration.
    """

    @staticmethod
    def type():
        return type(javalang.tree.ClassDeclaration())

    @classmethod
    def _parse_node(cls, node):
        if node is None:
            raise ValueError("Node must be specified when parsing a ClassDeclaration node.")
        return "ClassDeclaration", node.name


class ConstructorDeclaration(BaseTokenizer):
    """
    Tokenizer for ConstructorDeclaration.
    """

    @staticmethod
    def type():
        return type(javalang.tree.ConstructorDeclaration())

    @classmethod
    def _parse_node(cls, node):
        if node is None:
            raise ValueError("Node must be specified when parsing a ConstructorDeclaration node.")
        return "ConstructorDeclaration", node.name

class BasicType(BaseTokenizer):
    """
    Tokenizer for BasicType.
    """

    @staticmethod
    def type():
        return type(javalang.tree.BasicType())

    @classmethod
    def _parse_node(cls, node):
        if node is None:
            raise ValueError("Node must be specified when parsing a BasicType node.")
        return "BasicType", node.name


class ReferenceType(BaseTokenizer):
    """
    Tokenizer for ReferenceType.
    """

    @staticmethod
    def type():
        return type(javalang.tree.ReferenceType())

    @classmethod
    def _parse_node(cls, node):
        if node is None:
            raise ValueError("Node must be specified when parsing a ReferenceType node.")
        return "ReferenceType", node.name


class MemberReference(BaseTokenizer):
    """
    Tokenizer for MemberReference
    """

    @staticmethod
    def type():
        return type(javalang.tree.MemberReference())

    @classmethod
    def _parse_node(cls, node):
        if node is None:
            raise ValueError("Node must be specified when parsing a MemberReference node.")
        return "MemberReference", node.member


class FormalParameter(BaseTokenizer):
    """
    Tokenizer for FormalParameter
    """

    @staticmethod
    def type():
        return type(javalang.tree.FormalParameter())

    @classmethod
    def _parse_node(cls, node):
        if node is None:
            raise ValueError("Node must be specified when parsing a FormalParameter node.")
        return "FormalParameter", node.name


class BaseNodeFilter:
    """
    Base Filter for nodes of abstract syntax tree.
    """

    def __init__(self):
        self._accepted_nodes = []

    def accepted_nodes(self):
        return self._accepted_nodes
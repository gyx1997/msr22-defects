from explanation.feature_representation import Instance


def build_atomic_features(token_sequence, atomic_unit_decider):
    """
    Get the atomic feature representation of a given token sequence.

    :param token_sequence: Required. A list of tokens. The tokens should be type of
        preprocess.java_parser.tokenizers.Token.
    :param atomic_unit_decider: Required. A function that determine which attribute
        of the token is the decider of atomic feature unit.
    :returns An explanation.feature_representation.Instance object.
    """
    if len(token_sequence) == 0:
        return []

    result = []
    lines = []
    tokens_in_atomic_unit = [token_sequence[0].str()]
    lines_in_atomic_unit = [token_sequence[0].line]
    flag_of_current_atomic_unit = atomic_unit_decider(token_sequence[0])
    for i in range(1, len(token_sequence)):
        token_object = token_sequence[i]
        if atomic_unit_decider(token_object) == flag_of_current_atomic_unit:
            tokens_in_atomic_unit.append(token_object.str(show_prefix=False))
            lines_in_atomic_unit.append(token_object.line)
        else:
            result.append(tokens_in_atomic_unit)
            lines.append(lines_in_atomic_unit)
            tokens_in_atomic_unit = [token_object.str()]
            lines_in_atomic_unit = [token_object.line]
            flag_of_current_atomic_unit = atomic_unit_decider(token_object)

    # Special treatment for the last piece.
    result.append(tokens_in_atomic_unit)
    lines.append(lines_in_atomic_unit)

    # Build the instance object.
    instance = Instance(result, lines)
    return instance

def atomic2seq(seq_of_atomic):
    # from collections import Iterable
    seq = []
    for atomic_element in seq_of_atomic:
        if isinstance(atomic_element, list):
            seq.extend(atomic2seq(atomic_element))
        else:
            seq.append(atomic_element)
    return seq
def merge_dict(*dicts):
    result = {}
    for sub_dict in dicts:
        result.update(sub_dict)
    return result
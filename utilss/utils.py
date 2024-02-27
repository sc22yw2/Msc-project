def add_prefix(dct, prefix):
    return {f'{prefix}-{key}': val for key, val in dct.items()}
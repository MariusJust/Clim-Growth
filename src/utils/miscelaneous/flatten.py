import yaml


def flatten_dict(d, parent_key='', sep='_'):
    """
    Flattens a nested dictionary.
    Args:
        d (dict): The dictionary to flatten.
        parent_key (str): The base key to use for the current level of recursion.
        sep (str): The separator to use between keys.
    Returns:
        dict: A flattened dictionary.
    """
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + str(k) if parent_key else str(k)
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

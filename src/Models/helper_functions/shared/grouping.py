import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Layer


def fid_country_map():
    """
    Build a fid -> iso3 country-code mapping from the ee data file.

    Only meaningful for the 'ee' (admin-1 sub-national) data source, where the
    model's units are fids. Each fid belongs to exactly one country.

    RETURNS
        dict mapping int fid -> str iso3.
    """
    from utils.miscelaneous.find_data_file import Find_data_file

    data = pd.read_csv(Find_data_file('ee_data.csv'), sep=';', usecols=['fid', 'iso3'])
    data = data.dropna(subset=['fid', 'iso3'])
    return {int(f): str(c) for f, c in zip(data['fid'], data['iso3'])}


def build_country_grouping(individuals, fid_to_country):
    """
    Build a 0/1 grouping matrix that collapses per-unit (fid) columns into
    per-country columns.

    ARGUMENTS
        * individuals:     ordered iterable of unit ids (fids) for this model/region,
                           in the same column order as the trend design matrix.
        * fid_to_country:  dict mapping fid -> country code.

    RETURNS
        * G:            (n_units, n_countries) float32 matrix; G[k, c] = 1 iff
                        unit k belongs to country c.
        * country_list: list of country codes, length n_countries, giving the
                        column order of G (and hence of the grouped trend layer).
    """
    units = [int(u) for u in individuals]
    countries = [fid_to_country[u] for u in units]

    # deterministic column order: first appearance, tied to the unit order
    country_list = list(dict.fromkeys(countries))
    country_index = {c: j for j, c in enumerate(country_list)}

    G = np.zeros((len(units), len(country_list)), dtype=np.float32)
    for k, c in enumerate(countries):
        G[k, country_index[c]] = 1.0

    return G, country_list


class GroupColumns(Layer):
    """
    Collapse the per-unit columns of a trend design matrix into per-group
    (per-country) columns via a fixed 0/1 grouping matrix.

    Input  shape: (1, noObs, n_units)
    Output shape: (1, noObs, n_groups)

    The layer holds the grouping matrix as a constant and has no trainable
    parameters, so it does not affect the model's parameter count.
    """

    def __init__(self, grouping_matrix, **kwargs):
        super().__init__(**kwargs)
        G = np.asarray(grouping_matrix, dtype=np.float32)
        if G.ndim == 2:
            G = G[None, :, :]
        self.grouping_matrix = tf.constant(G, dtype=tf.float32)

    def call(self, x):
        return tf.matmul(x, tf.cast(self.grouping_matrix, x.dtype))

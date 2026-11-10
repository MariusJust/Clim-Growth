from tensorflow.keras.layers import Dense
from tensorflow.keras.initializers import Zeros

def create_fixed_effects(self, Delta1, Delta2):
    """ Create country and time fixed effect layers. """
    self.country_FE_layer = Dense(1, activation='linear', use_bias=False, kernel_initializer=Zeros())
    self.time_FE_layer = Dense(1, activation='linear', use_bias=False, kernel_initializer=Zeros())
    country_FE = self.country_FE_layer(Delta1)
    time_FE = self.time_FE_layer(Delta2)
    
    return country_FE, time_FE, self.country_FE_layer, self.time_FE_layer

def create_country_trends(self, linear_trend, quadratic_trend, use_quadratic=True, grouping_matrix=None):
    """Create country-specific linear and (optionally) quadratic trend layers.

    When ``grouping_matrix`` is provided, the per-unit (fid) trend design
    columns are first collapsed into per-country columns, so the trend layers
    estimate one slope per country instead of one per unit. When it is None the
    original per-unit behaviour is preserved exactly.
    """
    if grouping_matrix is not None:
        from .grouping import GroupColumns
        linear_trend = GroupColumns(grouping_matrix)(linear_trend)
        if use_quadratic:
            quadratic_trend = GroupColumns(grouping_matrix)(quadratic_trend)

    self.linear_trend_layer = Dense(1, activation='linear', use_bias=False, kernel_initializer=Zeros())
    linear_trend = self.linear_trend_layer(linear_trend)

    if use_quadratic:
        self.quadratic_trend_layer = Dense(1, activation='linear', use_bias=False, kernel_initializer=Zeros())
        quadratic_trend = self.quadratic_trend_layer(quadratic_trend)
    else:
        self.quadratic_trend_layer = None
        quadratic_trend = None

    return linear_trend, quadratic_trend, self.linear_trend_layer, self.quadratic_trend_layer



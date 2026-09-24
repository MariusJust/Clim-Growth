import os
import numpy as np
import plotly.graph_objects as go
from utils.miscelaneous import Find_data_file
import pandas as pd


def simulate(seed, specification, add_noise, sample_data, dynamic, run_dir, save_effects=True, rep_id=None, fixed_effects=None, country_trends=False):
  
    """
    Simulate a synthetic panel dataset.

    For each country and time period:
      - Country fixed effect drawn from N(0, 0.1)
      - A cubic time trend
      - precipitation and temperature values as inputs
      - change in logGDP as the target variable
    """
    
    # 1. Reproducibility
    np.random.seed(seed)
    
      #if sample data is true, we use the real data from our analysis, ie we take the true values of precipitation and temperature from the dataset and use that to simulate the growth. 
    data_path = Find_data_file('MainData.xlsx')
    data=pd.read_excel(data_path)
    
    years=data['Year'].values

    countries  = data['CountryCode'].values

    if sample_data:

        temperature = data['TempPopWeight']
        precipitation = data['PrecipPopWeight'] / 1000
    else:
        # Full-grid robustness check (sample_data=False): draw temperature and
        # precipitation uniformly over the climate grid, keeping the same panel
        # and the same fixed-effect / trend structure as the sample branch, so
        # that the (T, P) space is covered uniformly and only the climate inputs
        # differ from the sample-based design.
        temperature = np.random.uniform(0, 30, size=len(data))
        precipitation = np.random.uniform(0.012, 5.435, size=len(data))

    # The block below (fixed effects, trends, simulated growth) is identical for
    # the sample-based and full-grid designs; only the climate inputs above differ.

    # Use the same ordering as the model input pivots so FE vectors align by key.
    # (pivot() column/index ordering can differ from np.unique order across environments.)
    country_order = data.pivot(index='Year', columns='CountryCode', values='TempPopWeight').columns
    year_order = data.pivot(index='Year', columns='CountryCode', values='TempPopWeight').index

    unique_countries = country_order.to_numpy()
    unique_years = year_order.to_numpy()

    # Keep the reference category consistent with the omitted dummy in model training.
    base_country = unique_countries[0]
    base_year = unique_years[0]

    if fixed_effects is not None:
        true_country_FE = fixed_effects["country"]
        true_time_FE = fixed_effects["time"]
        true_linear_trend = fixed_effects.get("linear_trend")
        true_quadratic_trend = fixed_effects.get("quadratic_trend")
    else:
        true_country_FE = {
            c: np.random.normal(0, 0.025)
            for c in unique_countries
        }

        true_time_FE = {
            t: 0.1 * np.log(t - 1960) + np.random.normal(0, 0.01)
            for t in unique_years
        }

        true_linear_trend = {
            c: np.random.normal(0, 0.001)
            for c in unique_countries
        }
        true_quadratic_trend = {
            c: np.random.normal(0, 0.00001)
            for c in unique_countries
        }

    country_effect = np.array([true_country_FE[c] for c in countries])
    time_trend = np.array([true_time_FE[t] for t in years])
    if country_trends:
        year_to_position = {year: idx for idx, year in enumerate(unique_years)}
        trend_idx = np.array([year_to_position[year] for year in years])
        country_time_trend = np.array([
            true_linear_trend[c] * t + true_quadratic_trend[c] * t**2
            for c, t in zip(countries, trend_idx)
        ])
    else:
        country_time_trend = 0

    true_country_FE_rel = {
        c: true_country_FE[c] - true_country_FE[base_country]
        for c in unique_countries if c != base_country
    }

    true_time_FE_rel = {
        t: true_time_FE[t] - true_time_FE[base_year]
        for t in unique_years if t != base_year
    }

    if save_effects:
        true_param_dir = os.path.join(run_dir, "true_parameters")
        os.makedirs(true_param_dir, exist_ok=True)
        np.save(os.path.join(true_param_dir, "country_effect_absolute.npy"), true_country_FE)
        np.save(os.path.join(true_param_dir, "country_effect_relative.npy"), true_country_FE_rel)
        np.save(os.path.join(true_param_dir, "time_trend_absolute.npy"), true_time_FE)
        np.save(os.path.join(true_param_dir, "time_trend_relative.npy"), true_time_FE_rel)
        if country_trends:
            np.save(os.path.join(true_param_dir, "linear_trend_true.npy"), true_linear_trend)
            np.save(os.path.join(true_param_dir, "quadratic_trend_true.npy"), true_quadratic_trend)

    growth = calculate_growth(
        specification,
        temperature,
        precipitation,
        country_effect,
        time_trend + country_time_trend,
        add_noise,
        dynamic=dynamic,
        year=years
    )

    final_dataset = pd.DataFrame({
        'CountryCode': data['CountryCode'],
        'Year': years,
        'delta_logGDP': growth,
        'precipitation': precipitation,
        'temperature': temperature
    })

    return final_dataset


def simulate_weak_exog(
    seed,
    specification="Burke",
    n_years=64,
    n_countries=170,
    country_means=None,       # per-country mean temperature (deg C); drawn if None
    lambda_T=0.5,             # AR(1) persistence of temperature
    feedback_r2=0.0,          # target share of Var(T) explained by growth-shock feedback
    sigma_eps=0.02,           # growth innovation sd (log points)
    sigma_vT=1.0,             # temperature innovation sd (deg C)
    precip_mean=1.0,          # precipitation mean (m)
    precip_sd=0.4,            # precipitation stationary sd (m)
    lambda_P=0.3,             # AR(1) persistence of precipitation (strictly exogenous)
    country_trends=True,
    fixed_effects=None,       # reuse one true FE/trend draw across replications
    burn_in=100,
):
    """Simulate a *static-in-growth* panel in which TEMPERATURE is weakly
    exogenous, to stress-test the Nickell (1981) / Chudik et al. (2018)
    weak-exogeneity bias:

        y_it = g(T_it, P_it) + mu_i + theta1_i * t + theta2_i * t^2 + eps_it
        T_it = c_i + lambda_T * T_{i,t-1} + v_it + delta * eps_{i,t-1}      (feedback)
        P_it = mP  + lambda_P * (P_{i,t-1} - mP) + w_it                     (strictly exogenous)

    There is NO lagged dependent variable in the outcome equation: the only
    dynamic link is the feedback delta from this year's growth innovation into
    next year's temperature, which is exactly the "weakly exogenous regressor"
    case of Chudik et al. (2018) -- the bias that arises "regardless of whether
    lags of the dependent variable are included".

    ``delta`` is backed out from an interpretable target feedback R^2 so the
    sweep is reported on an economically meaningful scale:

        delta = sqrt( feedback_r2 * Var(T_base) / sigma_eps^2 ),
        Var(T_base) = sigma_vT^2 / (1 - lambda_T^2).

    At ``feedback_r2 = 0`` temperature is strictly exogenous (delta = 0).

    Returns
    -------
    data : pandas.DataFrame
        Columns [CountryCode, Year, delta_logGDP, precipitation, temperature],
        the same schema consumed by ``mc_worker`` and ``bench_models.fit_burke``.
    truth : dict
        True fixed effects/trends, the realized ``delta``/``feedback_r2``, the
        DGP ``specification``, and ``surface(T, P)`` -- a callable giving the
        true climate response g(T, P) for computing bias and the size/coverage
        estimand.
    """
    rng = np.random.default_rng(seed)

    if country_means is None:
        country_means = rng.uniform(3.0, 28.0, size=n_countries)
    else:
        country_means = np.asarray(country_means, dtype=float)
    N = len(country_means)

    if fixed_effects is not None:
        mu = np.asarray(fixed_effects["country"], float)
        th1 = np.asarray(fixed_effects["linear_trend"], float)
        th2 = np.asarray(fixed_effects["quadratic_trend"], float)
    else:
        mu = rng.normal(0.0, 0.025, size=N)
        th1 = rng.normal(0.0, 0.001, size=N)
        th2 = rng.normal(0.0, 1e-5, size=N)

    var_T_base = sigma_vT ** 2 / (1.0 - lambda_T ** 2)
    delta = float(np.sqrt(feedback_r2 * var_T_base / (sigma_eps ** 2))) if feedback_r2 > 0 else 0.0

    c_i = country_means * (1.0 - lambda_T)          # so the stationary mean of T_it is country_means
    T_prev = country_means + rng.normal(0.0, np.sqrt(var_T_base), size=N)
    P_prev = np.full(N, precip_mean)
    eps_prev = np.zeros(N)

    codes_l, years_l, growth_l, precip_l, temp_l = [], [], [], [], []
    T_tot = n_years + burn_in
    for t in range(T_tot):
        v = rng.normal(0.0, sigma_vT, size=N)
        T = c_i + lambda_T * T_prev + v + delta * eps_prev
        w = rng.normal(0.0, precip_sd * np.sqrt(1.0 - lambda_P ** 2), size=N)
        P = np.maximum(precip_mean + lambda_P * (P_prev - precip_mean) + w, 0.0)
        eps = rng.normal(0.0, sigma_eps, size=N)

        if t >= burn_in:
            yr = t - burn_in + 1
            g = calculate_growth(specification, T, P, 0.0, 0.0,
                                 add_noise=False, dynamic=False, year=None)
            trend = (th1 * yr + th2 * yr ** 2) if country_trends else 0.0
            y = g + mu + trend + eps
            codes_l.append(np.arange(N))
            years_l.append(np.full(N, yr))
            growth_l.append(y)
            precip_l.append(P)
            temp_l.append(T)

        T_prev, P_prev, eps_prev = T, P, eps

    data = pd.DataFrame({
        "CountryCode": np.concatenate(codes_l).astype(int),
        "Year": np.concatenate(years_l).astype(int),
        "delta_logGDP": np.concatenate(growth_l),
        "precipitation": np.concatenate(precip_l),
        "temperature": np.concatenate(temp_l),
    })

    truth = {
        "country": mu,
        "linear_trend": th1,
        "quadratic_trend": th2,
        "delta": delta,
        "feedback_r2": float(feedback_r2),
        "specification": specification,
        "surface": lambda T, P: calculate_growth(
            specification, T, P, 0.0, 0.0, add_noise=False, dynamic=False, year=None
        ),
    }
    return data, truth


def calculate_growth(specification, temp, precip, country_effect, time_trend, add_noise, dynamic, year):
     
            if dynamic: 
                time_periods=year.max()-year.min()+1
                t=year - year.min() + 1  # t goes from 1 to time_periods'
            
                if specification=='Burke':
                    true_y = (
                          (time_periods - t + 1) / time_periods * 0.0127 * temp
                        + (time_periods - t + 1) / time_periods * 0.145 * precip
                        - (t - 1) / time_periods * 0.0005 * temp**2
                        - (t - 1) / time_periods * 0.047 * precip**2
                        + country_effect
                    )
                    
                    
                elif specification=='Leirvik':
                        true_y = (
                            (time_periods - t + 1) / time_periods * 0.0127 * temp
                            + (time_periods - t + 1) / time_periods * 0.145 * precip
                            - (t - 1) / time_periods * 0.0005 * temp**2
                            - (t - 1) / time_periods * 0.047 * precip**2
                            - (time_periods - t + 1)/time_periods * 0.0125 * temp * precip
                            + (t-1)/time_periods * 0.00029 * precip * temp**2
                            + (t-1)/time_periods * 0.007 * temp * precip**2
                            - (t-1)/time_periods * 0.00013 * temp**2 * precip**2
                            + country_effect
                        )
                      

                else:
                    raise ValueError(f"Unknown specification: {specification}")
             
                    

            else:
                if specification == 'linear':
                    true_y = (
                        0.0008 * temp
                    + 0.007 * precip
                    + country_effect
                    + time_trend
                    )           

                elif specification == 'Burke':
                    true_y = (
                        0.0127 * temp
                    + 0.145 * precip
                    -0.0005 * temp**2
                    -0.047* precip**2
                    + country_effect
                    + time_trend
                    )
                    
                
                elif specification == 'Leirvik':
                    true_y = (
                        0.01 * temp
                    + 0.105 * precip
                    -0.00048 * temp**2
                    -0.07* precip**2
                    -0.0125*temp*precip
                    +0.00029*precip*temp**2
                    +0.007*temp*precip**2
                    -0.00013*temp**2*precip**2
                    + country_effect
                    + time_trend
                    
                    )
            
                elif specification == 'Trig':
                # periodic structure in precip and mild modulation by temp
                # If precip is in meters across 0..5, use period ~5 to get one cycle across range:
                # sin(2*pi*precip/5). If precip was standardized, change the frequency accordingly.
                    true_y = (
                        0.001     * temp
                    + 0.0025    * precip
                    + 0.02      * np.sin(2 * np.pi * precip / 5.0)           # wave across precipitation
                    + 0.01      * np.cos(2 * np.pi * temp / 15.0)            # gentle seasonal-like temp cycle
                    - 0.005     * temp * np.sin(2 * np.pi * precip / 5.0)   # interaction: wave amplitude depends on temp
                    + country_effect
                    + time_trend
                    )
                    

                else:
                    raise ValueError(f"Unknown specification: {specification}")

            if add_noise:
                # 9. Add noise
                noise = np.random.normal(0, 0.001)
                y     = true_y + noise
            else:
                y = true_y
                
            return y
        


def Pivot(data):
    growth=data[['CountryCode', 'Year', 'delta_logGDP']]
    precip=data[['CountryCode', 'Year', 'precipitation']]
    temp=data[['CountryCode', 'Year', 'temperature']]

    
    #make dictionaries
    
    growth_dict={}
    precip_dict={}
    temp_dict={}

    dict_and_vars = [(growth_dict, growth),
    (precip_dict, precip),
    (temp_dict, temp)]

    
    for dict, var in dict_and_vars:

        pivot_data = var.pivot(index='Year', columns='CountryCode', values=var.columns[-1])

        mean = np.nanmean(pivot_data.values)
    
        std = np.nanstd(pivot_data.values)
       
        #we do not standardise the growth data 
        if var is growth:
            dict['global'] = pivot_data
        else:
            standardised_data = (pivot_data - mean) / std
            dict['global'] = standardised_data
    
    return growth_dict, precip_dict, temp_dict



def Surface(temp, precip, specification, dynamic, time_periods):
    """ Calculate the surface of growth based on temperature and precipitation.

    Args:
        temp (np.ndarray): 1D array of temperature values.
        precip (np.ndarray): 1D array of precipitation values.
        specification (str): Specification for the surface calculation ('linear', 'q_Leirvik', or 'interaction').
    Returns:
        np.ndarray: 1D array of growth values based on the specified surface.
    """
    if dynamic: 
        if specification=='Burke':
                  for t in range (1, time_periods+1):
                      
                    true_y= (
                          (time_periods - t + 1) / time_periods * 0.0127 * temp
                        + (time_periods - t + 1) / time_periods * 0.145 * precip
                        - (t - 1) / time_periods * 0.0005 * temp**2
                        - (t - 1) / time_periods * 0.047 * precip**2
                    
                    )
                    
                    return true_y
                    
        elif specification=='Leirvik':
                   true_y = (
                         (time_periods - t + 1) / time_periods * 0.0127 * temp
                        + (time_periods - t + 1) / time_periods * 0.145 * precip
                        - (t - 1) / time_periods * 0.0005 * temp**2
                        - (t - 1) / time_periods * 0.047 * precip**2
                        - (time_periods - t + 1)/time_periods * 0.0125 * temp * precip
                        + (t-1)/time_periods * 0.00029 * precip * temp**2
                        + (t-1)/time_periods * 0.007 * temp * precip**2
                        - (t-1)/time_periods * 0.00013 * temp**2 * precip**2
                    )
                   return true_y

        
    else: 
        if specification == 'linear':
            return 0.0008 * temp + 0.007 * precip
        
        elif specification == 'Burke':
                    return (
                        0.0127 * temp
                    + 0.145 * precip
                    -0.0005 * temp**2
                    -0.047* precip**2
                    )
                
                
        elif specification == 'Leirvik':
                    return (
                        0.01 * temp
                    + 0.105 * precip
                    -0.00048 * temp**2
                    -0.07* precip**2
                    -0.0125*temp*precip
                    +0.00029*precip*temp**2
                    +0.007*temp*precip**2
                    -0.00013*temp**2*precip**2
                    )

        elif specification == 'Trig':
            # periodic structure in precip and mild modulation by temp
            # If precip is in meters across 0..5, use period ~5 to get one cycle across range:
            # sin(2*pi*precip/5). If precip was standardized, change the frequency accordingly.
                return (
                    0.001     * temp
                + 0.0025    * precip
                + 0.02      * np.sin(2 * np.pi * precip / 5.0)           # wave across precipitation
                + 0.01      * np.cos(2 * np.pi * temp / 15.0)            # gentle seasonal-like temp cycle
                - 0.005     * temp * np.sin(2 * np.pi * precip / 5.0)   # interaction: wave amplitude depends on temp
                )

def dynamic_surface(spec, temp_grid, precip_grid, T_total):

    # --- build frames ---
    for t in range(1, T_total + 1):
        # growth formula from your function (broadcasting over the 2D grids)
        if spec=='Leirvik':
            growth = (
                (T_total - t + 1) / T_total * 0.0127 * temp_grid
                + (T_total - t + 1) / T_total * 0.145 * precip_grid
                - (T_total - t + 1)/T_total * 0.0125 * temp_grid * precip_grid
                + (t-1)/T_total * 0.00029 * precip_grid * temp_grid**2
                + (t-1)/T_total * 0.007 * temp_grid * precip_grid**2
                - (t-1)/T_total * 0.00013 * temp_grid**2 * precip_grid**2
                - (t - 1) / T_total * 0.0005 * temp_grid**2
                - (t - 1) / T_total * 0.047 * precip_grid**2
            )
        else:
            growth =(
                 (T_total - t + 1) / T_total * 0.0127 * temp_grid
                + (T_total - t + 1) / T_total * 0.145 * precip_grid
                - (t - 1) / T_total * 0.0005 * temp_grid**2
                - (t - 1) / T_total * 0.047 * precip_grid**2
            )
        z_arr[:, :, t-1] = growth
        
    

    return z_arr

                
def illustrate_synthetic_data(x,y,z):
    
    fig = go.Figure(
    data=go.Scatter3d(
    x=x,
    y=y,
    z=z,
    mode='markers',
    marker=dict(size=2, opacity=0.6)
    )
    )

    fig.update_layout(
    scene=dict(
        xaxis_title='Temperature',
        yaxis_title='precipitation',
        zaxis_title='Δ logGDP')
    )

    return fig

def illustate_surface(temp, precip, growth):
    """ Illustrate the 3d surface of growth based on temperature and precipitation.
    Args:
        temp (np.ndarray): 1D array of temperature values.
        precip (np.ndarray): 1D array of precipitation values.
        growth (np.ndarray): 1D array of growth values.
    """
    plot_data= go.Surface(
        x=temp, y=precip, z=growth.reshape(temp.shape),
        colorscale='Cividis',
        opacity=0.85,
        showscale=False,
        name='mean_surface'
        )


    fig = go.Figure(data=[plot_data])
    fig.update_layout(
        scene=dict(
            xaxis_title='Temperature (°C)',
            yaxis_title='Precipitation (m)',
            zaxis_title='Δ ln(Growth)',
            camera=dict(eye=dict(x=2.11, y=0.12, z=0.38))
        ),
        legend=dict(
            bgcolor='rgba(255,255,255,0.7)',
            bordercolor='black',
            borderwidth=1
        )
    )

    fig.show()    

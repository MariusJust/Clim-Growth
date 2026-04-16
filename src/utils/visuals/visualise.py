import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.io as pio


##############################################  Create the prediction input  ########################################################



def create_pred_input(mc, mean_T, std_T, mean_P, std_P, time_periods=None, two_dim=True, precip_capped=False, only_dims=False):
   
    """
    Create the input for predictions by standardizing temperature and precipitation.
    
    Parameters:
    - mc: bool either true or false 
    - mean_T: mean of temperature
    - std_T: standard deviation of temperature 
    - mean_p: mean of precipitation
    - std_p: standard deviation of precipitation
    Returns:
    - pred_input: Standardized input for predictions.
    """
    
    temp_vals = np.linspace(0, 30, 90) 
   
    if mc: #we use meters in mc 
        precip_vals=np.linspace(0.012,5.435,90)
    else:
        if precip_capped:
            #capping precipitation at 3000 mm
            precip_vals= np.linspace(12.03731002, 3000, 90)
        else:
            precip_vals= np.linspace(12.03731002, 5435.30011, 90)


    if time_periods is not None:
        
        T, P, time = np.meshgrid(temp_vals, precip_vals, np.arange(0,time_periods+1))  
        
        if only_dims:
            return T, P, time
        P_std=(P-mean_P)/std_P  
        T_std=(T - mean_T) / std_T
    
        flat_T_std = T_std.ravel()  
        flat_P_std = P_std.ravel()  
        time_input=time.ravel()
        
        pred_input = np.stack([flat_T_std, flat_P_std, time_input], axis=-1)  # shape (900, 3)
        return pred_input.reshape((1, 1, -1, 3)), T, P
    else:
        
        #if not 2d then we only have 1 dimension, ie only temperature is passed as input, so we create a grid of temperature values and standardize them
        if not two_dim:
        
            T = temp_vals
            T_std=(T - mean_T) / std_T
            flat_T_std = T_std.ravel()  
            return flat_T_std.reshape((1, 1, -1, 1)), T
        
        T, P = np.meshgrid(temp_vals, precip_vals)  
        
        if only_dims:
            return T, P
        P_std=(P-mean_P)/std_P  
        T_std=(T - mean_T) / std_T
    
        flat_T_std = T_std.ravel()  
        flat_P_std = P_std.ravel()  
        

        pred_input = np.stack([flat_T_std, flat_P_std], axis=-1)  # shape (900, 2)
        return pred_input.reshape((1, 1, -1, 2)), T, P




##############################################  Compare fixed effects with benchmark models  ########################################################

                                  
def compare_fixed_effects(model):
    """
    Compare fixed effects between the model and benchmark data.
    
    Parameters:
    - model: The trained model instance.

    Returns:
    - None
    """
    
    time_fe_bench=pd.read_csv('data/Benchmark/time_fixed_effects_Burke.csv')
    country_fe_bench=pd.read_csv('data/Benchmark/country_fixed_effects_Burke.csv')

    country_fe_diff = np.abs(model.alpha.values.flatten() - country_fe_bench['x'].values.flatten())

    # Create a DataFrame for plotting
    diff_df = pd.DataFrame({
        'Country_Index': np.arange(1, len(country_fe_diff) + 1),
        'FE_Diff': country_fe_diff
    })

    # Sort by difference for nicer visualization (optional)
    diff_df = diff_df.sort_values(by='FE_Diff', ascending=False)

    # Plot country fixed effects against the benchmark data 
    plt.figure(figsize=(12, 6))
    plt.bar(diff_df['Country_Index'], diff_df['FE_Diff'], color='skyblue')
    plt.xticks(rotation=90)
    plt.xlabel('Country')
    plt.ylabel('Fixed Effect Difference')
    plt.title('Country Fixed Effect Differences')
    plt.tight_layout()
    plt.show()
        
    #plot the time fixed effects against the benchmark data 
    time_fe_data = pd.DataFrame({
            'Time': model.time_periods[model.time_periods_not_na['global']][0:62],
            'Model': model.beta.values.flatten(),
            'Benchmark': pd.array(time_fe_bench['x'])[:-1]

        })  
    
    plt.plot(time_fe_data['Time'], time_fe_data['Model'], label='Model', color='red', linewidth=2)
    plt.plot(time_fe_data['Time'], time_fe_data['Benchmark'], label='Benchmark', color='blue', linewidth=2)
    plt.scatter(time_fe_data['Time'], time_fe_data['Model'], color='red', marker='o', s=10)
    plt.scatter(time_fe_data['Time'], time_fe_data['Benchmark'], color='blue', marker='o', s=10)
    plt.xlabel('Time')
    plt.ylabel('Time Fixed Effects')
    plt.title('Time Fixed Effects: Model vs. Benchmark')
    plt.legend()
    plt.show()




##############################################  plotting comparison of 3d models ########################################################




def compare_models_3d(model1, model2, pred_input, save_as_html, node1, node2):
    """
    Compare two models by plotting their predictions in 3D.
    
    Parameters:
    - model1: The first model instance.
    - model2: The second model instance.
    - save_as_html: Boolean to save the plot as an HTML file.
    - pred_input: Input data for predictions (standardized temperature and precipitation).
    - node1: Configuration of the first model (e.g., (4,)).
    - node2: Configuration of the second model (e.g., (8,2,2)).
    Returns:
    - None
    """
    
 

     #predictions   
    growth_pred_flat = model1.model_pred.predict(pred_input)
    growth_pred_flat = np.reshape(growth_pred_flat, (-1,))   # shape (900,)
    # Reshape predictions back to (30, 30) for surface plotting
    Growth1 = growth_pred_flat.reshape(T.shape)  # shape (30, 30)

    #predictions
    growth_pred_flat = model2.model_pred.predict(pred_input)
    growth_pred_flat = np.reshape(growth_pred_flat, (-1,))   # shape (900,)
    # Reshape predictions back to (30, 30) for surface plotting
    Growth2 = growth_pred_flat.reshape(T.shape)  # shape (30, 30)
    
    
    #plot the models
    fig = go.Figure(data=[go.Surface(z=Growth1, x=T, y=P, colorscale='Viridis', opacity=0.8, name='First model', showscale=True),
                        go.Surface(z=Growth2, x=T, y=P, colorscale='Cividis', opacity=0.8, name='Second model', showscale=False)])
    # Update layout with labels, title, and other settings
    fig.update_layout(
        title=f'3D Surface Plot: comparison of model {node1} and {node2}',
        # showlegend=False,
        scene=dict(
            xaxis_title='Temperature (°C)',
            yaxis_title='Precipitation (mm)',
            zaxis_title='Growth',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)  # Adjust the camera angle
            )
        )
    )
    # Show the plot
    fig.show()
    if save_as_html: 
         # Save the figure as an HTML file
        pio.write_html(fig, file=f'images/interactive_3d_plot_comparison_{node1}_{selection_method}_and_{node2}_{selection_method2}.html', auto_open=False)

    return None



##############################################  Makes confidence plot of top n models ########################################################

## Load the results from the file


def confidence_plot(ref_model, results, n_models, save_as_html, pred_input, model, x_train, growth):
    

    # Function to create a confidence plot for the model predictions.   
    """
    Function to create a confidence plot for the model predictions.
    calculates the ensemble mean and standard deviation of the top n models.
    The plot shows the chosen model's prediction surface colored by the ensemble standard deviation.
    
    Parameters:
    - ref_model: The reference model configuration (e.g., (8,2,2)).
    - results: Dictionary containing model configurations and their performance metrics.
    - n_models: Number of top models to consider for the ensemble.
    - save_as_html: Boolean to save the plot as an HTML file.
    - pred_input: Input data for predictions (standardized temperature and precipitation).
    - model: The model class 

    Returns:
    - None
    """
    # Sort the keys (node configurations) by performance metric and select the top 10.
    top_models = sorted(results, key=lambda node: results[node])[:n_models]
    
    
    # Initialize a list to store the prediction surfaces for each model.
    prediction_surfaces = []

    for node1 in top_models:
    
        # Instantiate your model with the given node configuration.

        model_instance = Model(nodes=node1, x_train=x_train, y_train=growth, dropout=0.2)
        

    # Convert the list into a NumPy array: shape (10, 30, 30)
    prediction_surfaces = np.array(prediction_surfaces)

    # Calculate the pointwise mean and standard deviation across the 10 models.
    ensemble_mean = np.mean(prediction_surfaces, axis=0)
    ensemble_std = np.std(prediction_surfaces, axis=0)

    # Select the Chosen Model (e.g. configuration (8,2,2))
    chosen_index = top_models.index(ref_model)
    chosen_surface = prediction_surfaces[chosen_index]



def model_instance_pred(model, date_of_run, node, T, pred_input):
    weight_file = f'../results/Model Parameters/global/cv/{date_of_run}/{str(node)}.weights.h5'
    model.load_params(weight_file)  # Load the model weights
    
    # Use the model to predict on the standardized (T,P) grid.
    growth_pred_flat = model.model_pred.predict(pred_input)
    growth_pred_flat = np.reshape(growth_pred_flat, (-1,))   # shape: (900,)
    Growth = growth_pred_flat.reshape(T.shape)  # reshape to (30, 30)
    
    return Growth

def model_confidence_plot(surface, std, T, P):
      
    """
    Function to create a 3D surface plot of the chosen model's predictions colored by ensemble standard deviation.
    """
    fig = go.Figure()
    fig.add_trace(
        go.Surface(
            z=surface,
            x=T,
            y=P/1000,
            surfacecolor=std,
            colorscale='Viridis',
            opacity=0.8,
            colorbar=dict(
            x=0.95,   
            len=0.5, 
            title=dict(
                text='Ensemble standard deviation',
                side='top'
            )
    )
        )
    )



    fig.update_layout(
        
            autosize=True,

            margin=dict(
                l=0,
                r=0,
                b=0,
                t=0,
            ),

                scene=dict(
                    xaxis_title='Temperature (°C)',
                    yaxis_title='Precipitation (m)',
                    zaxis=dict(title=dict(text="Δ ln(GDP)"),range=[-0.3, 0.3]),
                    camera=dict(eye=dict(x=1.738, y=-1.780, z=0.589))

                ),

                legend=dict(
                    bgcolor='rgba(255,255,255,0.7)',
                    bordercolor='black',
                    borderwidth=1
                ),
                font=dict(
                size=10
            ),
                showlegend=False
            ),

    return fig


# ============================================================
# Helper: add one 3D bar (cuboid) as a Mesh3d trace
# ============================================================
def add_histogram(fig, data, legend, precip_capped=False):
    def add_bar3d(fig, x0, x1, y0, y1, z0, z1, color, opacity=1, showscale=False,
                coloraxis=None):
        """
        Add a rectangular cuboid [x0,x1] x [y0,y1] x [z0,z1] as a Mesh3d trace.
        """

        # 8 vertices of the cuboid
        x = [x0, x1, x1, x0, x0, x1, x1, x0]
        y = [y0, y0, y1, y1, y0, y0, y1, y1]
        z = [z0, z0, z0, z0, z1, z1, z1, z1]

        # 12 triangles composing the 6 faces
        i = [0, 0, 0, 1, 4, 4, 3, 3, 0, 0, 1, 1]
        j = [1, 2, 4, 2, 5, 6, 2, 6, 3, 4, 2, 5]
        k = [2, 3, 5, 5, 6, 7, 6, 7, 4, 7, 6, 6]

        fig.add_trace(
            go.Mesh3d(
                x=x,
                y=y,
                z=z,
                i=i,
                j=j,
                k=k,
                intensity=np.full(8, color),
                colorscale="YlOrRd",
                cmin=0,
                cmax=1,
                opacity=opacity,
                showscale=showscale,
                coloraxis=coloraxis,
                flatshading=True,
                hoverinfo="skip"
            )
        )



    # ============================================================
    # Observed data
    # ============================================================
    temp_obs = np.array(data["TempPopWeight"]).flatten()
    precip_obs = np.array(data["PrecipPopWeight"]).flatten() / 1000

    mask = np.isfinite(temp_obs) & np.isfinite(precip_obs)
    temp_obs = temp_obs[mask]
    precip_obs = precip_obs[mask]

    # ============================================================
    # Define histogram bins
    # Force temperature axis to include 0
    # ============================================================
    temp_min = 0
    temp_max = int(np.ceil(temp_obs.max()))

    precip_min = int(np.floor(precip_obs.min()))
    if precip_capped:
        #capping precipitation at 3000 mm
        precip_max = 3
    else:
        precip_max = int(np.ceil(precip_obs.max()))

    temp_edges = np.arange(temp_min, temp_max + 1, 1)         # 1°C bins
    precip_edges = np.arange(precip_min, precip_max + 1, 1/3)   # 1 m bins

    counts, _, _ = np.histogram2d(
        temp_obs,
        precip_obs,
        bins=[temp_edges, precip_edges]
    )

    # counts shape: (n_temp_bins, n_precip_bins)
    # We keep it like this and loop explicitly.

    # ============================================================
    # 3D histogram settings
    # ============================================================
    z_floor = -0.30                 # bottom of the plot
    bar_max_height = 0.12           # how tall the tallest histogram bar can become
    max_count = counts.max()

    if max_count > 0:
        counts_scaled = counts / max_count
    else:
        counts_scaled = counts.copy()

    # ============================================================
    # Add 3D bars
    # ============================================================
    for ix in range(len(temp_edges) - 1):
        for iy in range(len(precip_edges) - 1):
            c = counts[ix, iy]
            if c <= 0:
                continue

            x0 = temp_edges[ix]
            x1 = temp_edges[ix + 1]

            y0 = precip_edges[iy]
            y1 = precip_edges[iy + 1]

            z0 = z_floor
            z1 = z_floor + counts_scaled[ix, iy] * bar_max_height

            # normalize color to [0,1] for YlOrRd
            color_value = counts_scaled[ix, iy]

            add_bar3d(
                fig,
                x0=x0, x1=x1,
                y0=y0, y1=y1,
                z0=z0, z1=z1,
                color=color_value,
                opacity=0.95,
                showscale=legend
            )
    return fig







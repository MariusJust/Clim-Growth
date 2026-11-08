import pandas as pd
import numpy as np
import netCDF4 as nc
import fiona
import matplotlib.pyplot as plt
import geopandas as gpd
import shapely 

# first load the temp and and precip nc files and do a population-weighted average to get a single temp and precip time series for each country

temp_data = nc.Dataset("../data/projections/NN/in/diff_tas_mon_mod_ssp585_192_ave_2081-2100_minus_1986-2005_mon1_ave12_withsd (1).nc")
precip_data = nc.Dataset("../data/projections/NN/in/diff_pr_mon_mod_ssp585_192_ave_2081-2100_minus_1986-2005_mon1_ave12_withsd.nc")
population_density = nc.Dataset("../data/RawData/gpw_v4_population_density_rev11_30_min.nc")
shapefile=fiona.open("../data/RawData/national-identifier-grid/gpw_v4_national_identifier_grid_rev11_30_min.shp")




### load and visualise the CMIP temperature and precipitation projections

tmp= temp_data.variables["diff"][0,:,:]
pre=np.squeeze(precip_data.variables["diff"][:])
lon = np.asarray(temp_data.variables["lon"][:], dtype=float)
lat = np.asarray(temp_data.variables["lat"][:], dtype=float)

# Convert longitude from 0..360 to -180..180 - gap size is 1.9 degrees. 
lon = ((lon + 180) % 360) - 180

# Sort longitudes and reorder data columns to match
order = np.argsort(lon)
lon = lon[order]
tmp = tmp[:, order]


def plot_projection(tmp, lon, lat):

 

    fig, ax = plt.subplots(figsize=(12, 6))

    pcm = ax.pcolormesh(
        lon,
        lat,
        tmp,
        shading="auto",
        cmap="terrain"
    )

    plt.colorbar(pcm, ax=ax, label="Temperature change")

    world = gpd.read_file(
        "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip"
    )
    world.boundary.plot(ax=ax, color="black", linewidth=0.4)

    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    plt.show()

plot_projection(tmp, lon, lat)



### Load the population density data and the country shapefile, and calculate the population-weighted average temperature and precipitation change for each country


pop_density = np.squeeze(population_density.variables["Population Density, v4.11 (2000, 2005, 2010, 2015, 2020): 30 arc-minutes"][0,:,:])
lon_pop = np.asarray(population_density.variables["longitude"][:], dtype=float)
lat_pop = np.asarray(population_density.variables["latitude"][:], dtype=float)

pop_density.shape

#load the country shapefile and extract the country codes and geometries
country_codes= dict()

shapefile.schema

for feature in shapefile:
    iso = feature["properties"]["ISOCODE"]
    geometry = feature["geometry"]
    country_codes[iso] = geometry
    
    
#map the tmp and precip data to countries using the country codes and geometries

country_temp = dict()
country_precip = dict()






def find_closest(pop_density, mask, lat, lon, lat_pop, lon_pop):
    # indices in the target grid where mask is True
    masked_indices = np.where(mask)

    # output array
    closest_pop_density = np.zeros(len(masked_indices[0]), dtype=pop_density.dtype)

    # loop over masked target points
    for idx, (i, j) in enumerate(zip(masked_indices[0], masked_indices[1])):
        # 2D grid of squared distances
        distances = (lat_pop[:, None] - lat[i])**2 + (lon_pop[None, :] - lon[j])**2

        # 2D index of nearest point in population grid
        closest_index = np.unravel_index(np.argmin(distances), distances.shape)
        
        # extract matching population density value
        closest_pop_density[idx] = pop_density[closest_index]

    return closest_pop_density



for iso, geometry in country_codes.items():
    # Create a mask for the current country
    mask = np.zeros_like(tmp, dtype=bool)
    
    iso="AUS"
    geometry = country_codes[iso]
        
    geometry_min_lon, geometry_min_lat, geometry_max_lon, geometry_max_lat = shapely.geometry.shape(geometry).bounds
    
    # Find the indices of the grid points that fall within the bounding box of the country
    lat_indices = np.where((lat >= geometry_min_lat) & (lat <= geometry_max_lat))[0]
    lon_indices = np.where((lon >= geometry_min_lon) & (lon <= geometry_max_lon))[0]
    
    #only loop through the grid points that fall within the bounding box of the country to check if they are actually within the country geometry
    for i in lat_indices:
        for j in lon_indices:
            point = shapely.geometry.Point(lon[j], lat[i])
            if shapely.geometry.shape(geometry).contains(point):
                mask[i, j] = True
                
    np.where(mask==True)
    
    #tmp and pop are on different grids, so we need to find the closest point on the pop grid to each point on the tmp grid and use that pop density value for the population-weighted average
    
    if np.sum(mask) > 0:
        country_temp[iso] = np.nansum(tmp[mask] * find_closest(pop_density, mask, lat, lon, lat_pop, lon_pop)) / np.nansum(find_closest(pop_density, mask, lat, lon, lat_pop, lon_pop))
        country_precip[iso] = np.nansum(pre[mask] * find_closest(pop_density, mask, lat, lon, lat_pop, lon_pop)) / np.nansum(find_closest(pop_density, mask, lat, lon, lat_pop, lon_pop))
    else:
        country_temp[iso] = np.nan
        country_precip[iso] = np.nan
        
        
   #drop countries with no data
country_temp = {k: v for k, v in country_temp.items() if not np.isnan(v)}
country_precip = {k: v for k, v in country_precip.items() if not np.isnan(v)}

#write out as csv files
pd.DataFrame.from_dict(country_temp, orient="index", columns=["temp_change"]).to_csv("../data/projections/NN/out/country_temp_change.csv")
pd.DataFrame.from_dict(country_precip, orient="index", columns=["precip_change"]).to_csv("../data/projections/NN/out/country_precip_change.csv")    
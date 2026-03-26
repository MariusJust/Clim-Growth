import numpy as np
import pandas as pd
from panelsplit.cross_validation import PanelSplit



def Prepare(data, data_source='WB'):
       #the growth data should contain the following columns: year, county, and GrowthWDI
    time_periods = len(data['Year'].unique())

    if data_source.lower()=='wb':
        growth=data[['CountryCode', 'RegionCode', 'Year', 'GrowthWDI']]

        #precipitation data
        precip=data[['CountryCode', 'RegionCode', 'Year', 'PrecipPopWeight']]

        #temperature data
        temp=data[['CountryCode', 'RegionCode', 'Year', 'TempPopWeight']]
        
    elif data_source.lower()=='ee':
            growth=data[['iso3', 'fid', 'RegionCode', 'Year', 'growth (gdp per capita)']].rename(columns={'iso3':'CountryCode', 'growth (gdp per capita)':'GrowthWDI'})

            #precipitation data
            precip=data[['iso3', 'fid', 'RegionCode', 'Year', 'precipitation (mm)']].rename(columns={'iso3':'CountryCode', 'precipitation (mm)':'PrecipPopWeight'})

            #temperature data
            temp=data[['iso3', 'fid', 'RegionCode', 'Year', 'temperature (celsius)']].rename(columns={'iso3':'CountryCode','Year':'Year', 'temperature (celsius)':'TempPopWeight'})
            

    else:
        raise ValueError("data_source must be either 'WB' or 'ee'")
    #Now I make dictionaries, to capture the region dependent variables

    growth_dict={}
    precip_dict={}
    temp_dict={}
    
    dict_and_vars = [(growth_dict, growth),
        (precip_dict, precip),
        (temp_dict, temp)]


    for dict, var in dict_and_vars:
        
        if data_source.lower()=='wb':
                pivot_data = var.pivot(index='Year', columns='CountryCode', values=var.columns[-1])
        elif data_source.lower()=='ee':
            #we now pivot based on fid and drop the year 1990 as we don't have growth data for that year
                pivot_data = var.pivot(index='Year', columns='fid', values=var.columns[-1]).iloc[1:time_periods,:]
                
        mean = np.nanmean(pivot_data.values)
    
        std = np.nanstd(pivot_data.values)
        
#we do not standardise the growth data 
        if var is growth:
            dict['global'] = pivot_data
        else:
            standardised_data = (pivot_data - mean) / std
            dict['global'] = standardised_data
    
    return growth_dict, precip_dict, temp_dict

def load_data(model_selection, data_source, n_splits=None, growth=None):
    
    if model_selection == 'IC':
        if data_source.lower()=='wb':
            data = pd.read_excel('data/MainData.xlsx')
            growth, precip, temp = Prepare(data, data_source=data_source)
            return growth, precip, temp
        elif data_source.lower()=='ee':
            data = pd.read_csv("data/ee_data.csv", sep=";")
            growth, precip, temp = Prepare(data, data_source=data_source)
            return growth, precip, temp
        
    elif model_selection == 'CV':
        if growth is None:
            data = pd.read_excel('data/MainData.xlsx')
            growth, precip, temp = Prepare(data, data_source=data_source)
            panel_split = PanelSplit(periods=growth['global'].index, n_splits=n_splits, gap=0, test_size=1)

            return growth, precip, temp, panel_split
        else: #mc case
            
            # Create a PanelSplit object for cross-validation
            growth_global = growth['global'].reset_index()
            growth_global['Year'] = pd.to_datetime(growth_global['Year'], format='%Y')

            panel_split = PanelSplit(periods=growth_global['Year'], n_splits=n_splits, gap=0, test_size=1)
            
            return panel_split
       
    else:
        raise ValueError("Invalid model_selection argument. Use 'IC' or 'CV'.")



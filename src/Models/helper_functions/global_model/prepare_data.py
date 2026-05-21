import numpy as np
import pandas as pd

from utils.miscelaneous.find_data_file import Find_data_file


def Prepare(data, data_source='wb'):
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

def load_data(model_selection, data_source, n_splits=None, growth=None, end_year=None):
    
    if model_selection == 'IC':
        if data_source.lower()=='wb':
            data = pd.read_excel(Find_data_file('MainData.xlsx'))
            if end_year is not None:
                data = data[data['Year'] <= end_year]
            growth, precip, temp = Prepare(data, data_source=data_source)
            return growth, precip, temp
        elif data_source.lower()=='ee':
            data = pd.read_csv(Find_data_file('ee_data.csv'), sep=";")
            if end_year is not None:
                data = data[data['Year'] <= end_year]
            growth, precip, temp = Prepare(data, data_source=data_source)
            return growth, precip, temp
        

    else:
        raise ValueError("Invalid model_selection argument. Use 'IC'")



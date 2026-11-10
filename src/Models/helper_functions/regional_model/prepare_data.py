import numpy as np
import pandas as pd

from utils.miscelaneous.find_data_file import Find_data_file


def Prepare(data, formulation, data_source="wb", target_mode='growth', income_time_varying=True):
    if str(target_mode).lower() == 'levels':
        raise NotImplementedError(
            "target_mode='levels' is not implemented for the regional/income "
            "formulation (the income path is two-group High/Low and untested for levels). Use "
            "formulation='global' for the levels (log GDP per capita) target."
        )
    if formulation != 'regional' and str(data_source).lower() != 'wb':
        raise ValueError(
            "formulation='income' requires data_source='wb' (income quintiles are "
            "built from World Bank GDP per capita). Add instance.data_source=wb."
        )
    time_periods = len(data['Year'].unique())

    if data_source.lower()=='wb':
            growth=data[['CountryCode', 'RegionCode', 'Year', 'GrowthWDI']]

            precip=data[['CountryCode', 'RegionCode', 'Year', 'PrecipPopWeight']]

            temp=data[['CountryCode', 'RegionCode', 'Year', 'TempPopWeight']]

    elif data_source.lower()=='ee':
            growth=data[['iso3', 'fid', 'RegionCode', 'Year', 'growth (gdp per capita)']].rename(columns={'iso3':'CountryCode', 'growth (gdp per capita)':'GrowthWDI'})

            precip=data[['iso3', 'fid', 'RegionCode', 'Year', 'precipitation (mm)']].rename(columns={'iso3':'CountryCode', 'precipitation (mm)':'PrecipPopWeight'})

            temp=data[['iso3', 'fid', 'RegionCode', 'Year', 'temperature (celsius)']].rename(columns={'iso3':'CountryCode','Year':'Year', 'temperature (celsius)':'TempPopWeight'})

    else:
            raise ValueError("data_source must be either 'WB' or 'ee'")


    growth_dict={}
    precip_dict={}
    temp_dict={}

    if data_source.lower()=='wb':
        if formulation=='regional':
            regions = {'Asia': [142, "CHN"], 'Europe': [150, 'DEU'], 'Africa': [2, 'ZAF'], 'Americas': [19, 'USA'], 'Oceania': [9, 'AUS']}
        else:
          # Two income groups, Low and High, split at the median of GDP per
          # capita pooled over all country-years. Membership is therefore time
          # varying by construction: a country crosses the median as it grows.
          #
          # This is the grouping behind the paper's Figure 7
          # (Figures/Income/Low_CT.pdf and High_CT.pdf) from run
          # 2026-06-03_10-22-50_income_IC_country_trends.
          #
          # Rolled back from five quintiles on 2026-11-08. The quintile version
          # is kept at archive/2026-11-08-prepare_data-quintiles.py; restoring it
          # means regenerating Figure 7 and re-running the income estimation,
          # since Q1-Q5 weights and High/Low weights are not interchangeable.
          #
          # income_time_varying is accepted for call compatibility and ignored:
          # the median split is already time varying.
          gdp_capita = data[['CountryCode', 'Year', 'GDPCap']].dropna(subset=['GDPCap']).copy()
          gdp_capita['IncomeGroup'] = pd.qcut(gdp_capita['GDPCap'], q=2, labels=['Low', 'High'])
          key = gdp_capita.set_index(['CountryCode', 'Year'])['IncomeGroup']

          def _attach_income(df):
              # Keyed on (CountryCode, Year). The original used
              # .iloc[:, [0, 1, 2, 4, 3]], a positional reindex, which is the bug
              # class behind the Projections.py country misalignment, so the
              # keyed merge is kept from the quintile version.
              df = df.copy()
              valcol = df.columns[-1]
              df['IncomeGroup'] = key.reindex(list(zip(df['CountryCode'], df['Year']))).values
              return df[['CountryCode', 'RegionCode', 'Year', 'IncomeGroup', valcol]].dropna(subset=['IncomeGroup'])

          growth = _attach_income(growth)
          precip = _attach_income(precip)
          temp = _attach_income(temp)

          regions = {'Low': ['Low', 'IND'], 'High': ['High', 'USA']}
    else:
        regions = {'Asia': [142, 408], 'Europe': [150, 2364], 'Africa': [2, 2854], 'Americas': [19, 2709], 'Oceania': [9, 196]}


    dict_and_vars = [(growth_dict, growth),
        (precip_dict, precip),
        (temp_dict, temp)]

    for region, value in regions.items():
        regionCode, referenceCountry = value
        for dict, var in dict_and_vars:
            if formulation=='regional':
                region_data = var[var['RegionCode'] == regionCode]
            else:
                region_data = var[var['IncomeGroup'] == regionCode]

            if data_source == 'ee':
                pivot_data = region_data.pivot(index='Year', columns='fid', values=region_data.columns[-1]).iloc[1:time_periods, :]
            else:
                pivot_data = region_data.pivot(index='Year', columns='CountryCode', values=region_data.columns[-1])

            cols = pivot_data.columns.tolist()
            cols.insert(0, cols.pop(cols.index(referenceCountry)))
            pivot_data = pivot_data[cols]

            if var is growth:
                dict[region] = pivot_data
            else:
                mean = np.nanmean(pivot_data.values)
                std = np.nanstd(pivot_data.values)
                pivot_data = (pivot_data - mean) / std
                dict[region] = pivot_data

    return growth_dict, precip_dict, temp_dict


def load_data(model_selection, formulation, data_source="wb", end_year=None, target_mode='growth', income_time_varying=True):

    if model_selection == 'IC':
        if data_source.lower()=='wb':
            data = pd.read_excel(Find_data_file('MainData.xlsx'))
            if end_year is not None:
                data = data[data['Year'] <= end_year]
            growth, precip, temp = Prepare(data, formulation, data_source=data_source, target_mode=target_mode, income_time_varying=income_time_varying)
            return growth, precip, temp
        elif data_source.lower()=='ee':
            data = pd.read_csv(Find_data_file('ee_data.csv'), sep=";")
            if end_year is not None:
                data = data[data['Year'] <= end_year]
            growth, precip, temp= Prepare(data, formulation, data_source=data_source, target_mode=target_mode, income_time_varying=income_time_varying)
            return growth, precip, temp

    else:
        raise ValueError("Invalid model_selection argument. Use 'IC'")

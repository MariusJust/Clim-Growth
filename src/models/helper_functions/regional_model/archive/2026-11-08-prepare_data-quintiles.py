import numpy as np
import pandas as pd

from utils.miscelaneous.find_data_file import Find_data_file


def Prepare(data, formulation, data_source="wb", target_mode='growth', income_time_varying=True):
    if str(target_mode).lower() == 'levels':
        raise NotImplementedError(
            "target_mode='levels' is not implemented for the regional/income "
            "formulation (the income-group merge is positional). Use "
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
          # Income-quintile groups. Q1 = poorest, Q5 = richest.
          n_income_groups = 5
          labels = [f"Q{i}" for i in range(1, n_income_groups + 1)]
          if income_time_varying:
              # Time-varying membership: each country-year is assigned to a quintile
              # by its RELATIVE GDP-per-capita rank WITHIN that year, so a country
              # moves between quintiles as its income position changes over time.
              tmp = data[['CountryCode', 'Year', 'GDPCap']].dropna(subset=['GDPCap']).copy()
              cnt = tmp.groupby('Year')['GDPCap'].transform('count')
              tmp = tmp[cnt >= n_income_groups]
              tmp['IncomeGroup'] = tmp.groupby('Year')['GDPCap'].transform(
                  lambda s: pd.qcut(s.rank(method='first'), n_income_groups, labels=labels)
              )
              key = tmp.set_index(['CountryCode', 'Year'])['IncomeGroup']
              # Reference country per group = the most frequent member across years.
              ref = {qn: tmp.loc[tmp['IncomeGroup'] == qn, 'CountryCode'].value_counts().idxmax()
                     for qn in labels}

              def _attach_income(df):
                  df = df.copy()
                  valcol = df.columns[-1]
                  df['IncomeGroup'] = key.reindex(list(zip(df['CountryCode'], df['Year']))).values
                  return df[['CountryCode', 'RegionCode', 'Year', 'IncomeGroup', valcol]].dropna(subset=['IncomeGroup'])
          else:
              # Fixed membership: each country in ONE quintile by long-run mean income,
              # so its country fixed effect and trend stay within a single group.
              country_mean = data.groupby('CountryCode')['GDPCap'].mean()
              qgroups = pd.qcut(country_mean, n_income_groups, labels=labels)
              grp = qgroups.to_dict()
              ref = {qn: (country_mean[qgroups == qn] - country_mean[qgroups == qn].median()).abs().idxmin()
                     for qn in labels}

              def _attach_income(df):
                  df = df.copy()
                  valcol = df.columns[-1]
                  df['IncomeGroup'] = df['CountryCode'].map(grp)
                  return df[['CountryCode', 'RegionCode', 'Year', 'IncomeGroup', valcol]].dropna(subset=['IncomeGroup'])

          growth = _attach_income(growth)
          precip = _attach_income(precip)
          temp = _attach_income(temp)

          regions = {qn: [qn, ref[qn]] for qn in labels}
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

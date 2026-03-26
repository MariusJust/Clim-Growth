
###### this file does the following: 1. loads the best model from our neural network, 2. loads the temperature and precipitation changes, loads the SSP files for population and growth,  loads the baseline growth projections.
###### 3. interpolates projections to annual, 
###### 4. uses the model to make projections of growth under the SSPs, with and without climate change impacts. 6. plot the projections.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go   
from scipy.interpolate import griddata
import re
import yaml

#number of years to project
yrs = np.arange(2024, 2100)
n_yrs = len(yrs)

#1. load the best model from our neural network

def load_model(date_of_run, formulation):
    with open(f'../runs/estimation/{date_of_run}/config.yaml', "r") as file:
        cfg = yaml.safe_load(file)
    
    
    # instantiate and load your model
    if cfg.formulation == "global":
        from models import MultivariateModelGlobal as Model   
        from models.global_model.model_functions.helper_functions.prepare_data import Prepare
        
    else:
         from models import MultivariateModelRegional as Model   
         from models.regional_model.model_functions.helper_functions.prepare_data import Prepare
         
    data=pd.read_excel('../data/MainData.xlsx')
    growth, precip, temp = Prepare(data)
    x_train = {0:temp, 1:precip}
    
    #load results from simulations and retrieve best n models
    results = dict(np.load(f'../runs/estimation/{date_of_run}/results.npy', 
                        allow_pickle=True).item())

    results={k: v for k,v in results.items() if v is not None}
    top_models = sorted(results, key=lambda node: results[node][0])

    node=top_models[0]
    
    factory = Model(node, cfg, x_train=None, y_train=None)
    factory.Depth=len(node)
    model=factory.get_model()
    
    weight_file = f'../runs/estimation/{date_of_run}/parameters/{node}.weights.h5'
    
    model.load_params(weight_file)
    
    return model


date_of_run = '2025-09-23' # regional model







####################################### Data #################################################
iso_metadata =data.loc[:, ["CountryName", "CountryCode"]].drop_duplicates().set_index("CountryCode")

TEMP=pd.read_csv("../data/projections/NN/out/country_temp_change.csv").rename(columns={"Unnamed: 0": "CountryCode"})
TEMP=TEMP.merge(iso_metadata, left_on="CountryCode", right_on="CountryCode", how="left").dropna(subset=["CountryName"])

PRECIP=pd.read_csv("../data/projections/NN/out/country_precip_change.csv").rename(columns={"Unnamed: 0": "CountryCode"})
PRECIP=PRECIP.merge(iso_metadata, left_on="CountryCode", right_on="CountryCode", how="left").dropna(subset=["CountryName"])

SSP_POP=pd.read_csv("../data/projections/NN/in/SSP_POP.csv")
SSP_GROWTH=pd.read_csv("../data/projections/NN/in/SSP_GROWTH.csv")
SSP_GROWTH = SSP_GROWTH[SSP_GROWTH["unit"] == "USD_2010/yr"]

#SSP growth numbers are in total annual gdp/cap, so we need to take the first differerence to get growth rates


data_baseline = (
    data.loc[:, ["CountryName", "TempPopWeight", "PrecipPopWeight", "GDPCap", "GrowthWDI"]]
    .groupby("CountryName", as_index=False)
    .agg(
        meantemp=("TempPopWeight", "mean"),
        meanprecip=("PrecipPopWeight", "mean"),
        basegrowth=("GrowthWDI", "mean"),
        gdpCap=("GDPCap", "mean"),
        )
)

############################################################################################
###################################### Align iso across datasets ###########################
############################################################################################

#dictionary where the key is the country name and the value is the iso code. We take intersections of SSP data, because that is the constraining dataset.

common_names = set(SSP_GROWTH["region"]) & set(SSP_POP["region"]) & set(TEMP["CountryName"]) & set(PRECIP["CountryName"])

# Dictionary: ISO code -> country name
common_country = {
    iso: row["CountryName"]
    for iso, row in iso_metadata.iterrows()
    if row["CountryName"] in common_names
}


#subset to common iso
SSP_GROWTH = SSP_GROWTH[SSP_GROWTH["region"].isin(common_country.values())]
SSP_POP = SSP_POP[SSP_POP["region"].isin(common_country.values())]

temp_RCP=TEMP[TEMP["CountryName"].isin(common_country.values())]
precip_RCP=PRECIP[PRECIP["CountryName"].isin(common_country.values())]



############################################################################################
###################################### Interpolations of SSP ###############################
############################################################################################

 
def ipolate_like_r(df: pd.DataFrame, yrs: np.ndarray) -> pd.DataFrame:
    out = df.iloc[:, :4].copy()

    # detect either X2010-style or plain 2010-style columns
    year_map = {}
    for c in df.columns:
        c_str = str(c).strip()
        m1 = re.fullmatch(r"X(\d{4})", c_str)
        m2 = re.fullmatch(r"(\d{4})", c_str)
        if m1:
            year_map[int(m1.group(1))] = c
        elif m2:
            year_map[int(m2.group(1))] = c

    support_years = np.array(sorted(year_map.keys()), dtype=int)

    if len(support_years) == 0:
        raise ValueError("No year columns detected in SSP file.")

    annual = np.empty((len(df), len(yrs)), dtype=float)

    for i, y in enumerate(yrs):
        if y in year_map:
            annual[:, i] = pd.to_numeric(df[year_map[y]], errors="coerce").to_numpy()
        else:
            lower = support_years[support_years < y]
            upper = support_years[support_years > y]

            if len(lower) == 0:
                # if target year is before first support year, carry first backward
                annual[:, i] = pd.to_numeric(df[year_map[support_years.min()]], errors="coerce").to_numpy()
                continue

            yl = lower.max()

            if len(upper) == 0:
                # beyond max support: carry last forward
                annual[:, i] = pd.to_numeric(df[year_map[yl]], errors="coerce").to_numpy()
                continue

            yu = upper.min()
            el = pd.to_numeric(df[year_map[yl]], errors="coerce").to_numpy()
            eu = pd.to_numeric(df[year_map[yu]], errors="coerce").to_numpy()
            annual[:, i] = el + (eu - el) * (y - yl) / (yu - yl)

    annual_df = pd.DataFrame(annual, columns=[str(y) for y in yrs])
    out = pd.concat([out.reset_index(drop=True), annual_df], axis=1)

    if "region" in out.columns:
        out["region"] = out["region"].replace({"COD": "ZAR", "ROU": "ROM"})
    return out


for scen in range(1,6):
    popSSP = ipolate_like_r(SSP_POP[SSP_POP["scenario"] == f"SSP{scen}"], np.arange(2010, 2101))
    growthSSP = ipolate_like_r(SSP_GROWTH[SSP_GROWTH["scenario"] == f"SSP{scen}"], np.arange(2010, 2101))

popSSP = popSSP.merge(data_baseline[["CountryName", "meantemp", "meanprecip", "basegrowth", "gdpCap"]], left_on="region", right_on="CountryName", how="left")
growthSSP = growthSSP.merge(data_baseline[["CountryName", "meantemp", "meanprecip", "basegrowth", "gdpCap"]], left_on="region", right_on="CountryName", how="left")

#we use ssp5 in line with burke 
growthSSP5 = growthSSP[growthSSP["scenario"] == "SSP5"].copy()
popSSP5 = popSSP[popSSP["scenario"] == "SSP5"].copy()

year_cols = [str(y) for y in range(2010, 2101)]

growthSSP5.loc[:, year_cols] = (
    growthSSP5.loc[:, year_cols]
    .diff(axis=1)
    .div(growthSSP5.loc[:, year_cols].shift(axis=1))
    .fillna(0)
)


baseGDP=growthSSP5["gdpCap"].to_numpy()

baseTemp=growthSSP5["meantemp"].to_numpy()
basePrecip=growthSSP5["meanprecip"].to_numpy()


############################################################################################
###################################### RCP 8.5 data  #######################################
############################################################################################


temp_change=temp_RCP["temp_change"]/n_yrs
precip_change=precip_RCP["precip_change"]/n_yrs



############################################################################################
###################################### Calculate GDP projections ###########################
############################################################################################


def pred_input(temp, precip):
    flat_T_std = temp.ravel()  
    flat_P_std = precip.ravel()  
    pred_input = np.stack([flat_T_std, flat_P_std], axis=-1)  # shape (900, 2)
    return pred_input.reshape((1, 1, -1, 2))

n_cntry = len(baseGDP)


GDPcapCC= np.zeros((n_cntry, n_yrs), dtype=float)
GDPcapNoCC = np.zeros((n_cntry, n_yrs), dtype=float)
GDPcapCC[:, 0] = baseGDP[:]
GDPcapNoCC[:, 0] = baseGDP[:]
tots = np.zeros((n_yrs, 4), dtype=float)


mean_temp_org = np.nanmean(data["TempPopWeight"])
sd_temp_org = np.nanstd(data["TempPopWeight"])
mean_precip_org = np.nanmean(data["PrecipPopWeight"])
sd_precip_org = np.nanstd(data["PrecipPopWeight"])


baseline_predicted_growth = model.model_visual.predict([pred_input((baseTemp-mean_temp_org)/sd_temp_org, (basePrecip-mean_precip_org)/sd_precip_org)]).reshape(-1,)



for j in range(1, n_yrs):
        y = str(yrs[j])
        
        #check what it is 
        base_growth=growthSSP5[y].to_numpy()
        GDPcapNoCC[:, j] = GDPcapNoCC[:, j - 1] * (1.0 + base_growth)

        
        #find common iso 
        newtemp=baseTemp+j*temp_change
        newprecip=basePrecip+j*precip_change
        
        #standardize the new temp and precip using the same mean and std as the training data
        newtemp = (newtemp - mean_temp_org) / sd_temp_org
        newprecip = (newprecip - mean_precip_org) / sd_precip_org
        
        predicted_growth = model.model_visual.predict([pred_input(newtemp, newprecip)]).reshape(-1,)
        
        diff_growth = predicted_growth - baseline_predicted_growth
        GDPcapCC[:, j] = GDPcapCC[:, j - 1] * (1.0 + base_growth + diff_growth)
        
        
        weights=popSSP5[y].to_numpy()
        tots[j, 0] = np.average(GDPcapCC[:, j], weights=weights)
        tots[j, 1] = np.average(GDPcapNoCC[:, j], weights=weights)
        tots[j, 2] = np.sum(GDPcapCC[:, j] * weights * 1e6)
        tots[j, 3] = np.sum(GDPcapNoCC[:, j] * weights * 1e6)
        
        




#plot the projections

chgs = (tots[1:, 0] / tots[1:, 1] - 1.0) * 100.0



plt.figure(figsize=(10, 6))
plt.plot(yrs, chgs[:], label="With Climate Change", color="red")
plt.xlabel("Year")
plt.ylabel("Population-Weighted Average GDP per Capita (USD)")
plt.title("Projected GDP per Capita under SSP5 with and without Climate Change Impacts")
plt.legend()
plt.grid()
plt.show()
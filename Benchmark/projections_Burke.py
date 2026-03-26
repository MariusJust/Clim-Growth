import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from zipfile import ZipFile
from io import BytesIO
import re

def read_csv_from_zip(zip_path: str, member: str, **kwargs) -> pd.DataFrame:
    with ZipFile(zip_path) as zf:
        with zf.open(member) as f:
            return pd.read_csv(f, **kwargs)



def ipolate_like_r(df: pd.DataFrame, yrs: np.ndarray) -> pd.DataFrame:
    out = df.iloc[:, :3].copy()

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

    if "Region" in out.columns:
        out["Region"] = out["Region"].replace({"COD": "ZAR", "ROU": "ROM"})
    return out


def density_grid(mat: np.ndarray, yrs: np.ndarray, yz: np.ndarray,
                 lo: float = 0.05, hi: float = 0.95, bw_adjust: float = 0.6) -> np.ndarray:
    """
    Python analogue of weightProj():
    each year -> KDE over bootstrap impacts after trimming to [5th,95th] percentiles,
    normalized to [0,1].
    """
    rows = []
    for j in range(mat.shape[1]):
        vals = mat[:, j]
        vals = vals[np.isfinite(vals)]

        if vals.size == 0:
            rows.append(np.zeros_like(yz))
            continue

        ql, qh = np.quantile(vals, [lo, hi])
        keep = vals[(vals >= ql) & (vals <= qh)]

        if keep.size < 2 or np.allclose(np.std(keep), 0):
            dens = np.zeros_like(yz, dtype=float)
            dens[np.argmin(np.abs(yz - np.mean(keep)))] = 1.0
        else:
            kde = gaussian_kde(keep, bw_method=lambda s: s.scotts_factor() * bw_adjust)
            dens = kde(yz)

        dens = dens - dens.min()
        if dens.max() > 0:
            dens = dens / dens.max()

        rows.append(dens)

    return np.array(rows).T  # shape: len(yz) x len(yrs)


# ---------------------------------------------------------------------
# FILES
# ---------------------------------------------------------------------

MAIN_DATASET = "../data/projections/old/in/mainDataset.csv"
BOOTSTRAP = "../data/projections/old/in/bootstrap_noLag.csv"

# Inputs from the replication archive
WPP_POP = "../data/projections/old/in/WPP2012_DB02_POPULATIONS_ANNUAL.csv"
WPP_LOC = "../data/projections/old/in/WPP2012_F01_LOCATIONS.csv"
SSP_POP = "../data/projections/old/in/SSP/SSP_PopulationProjections.csv"
SSP_GROWTH = "../data/projections/old/in/SSP/SSP_GrowthProjections.csv"
TCHG_FILE = "../data/projections/old/out/CountryTempChange_RCP85.csv"

# ---------------------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------------------
dta = pd.read_csv(MAIN_DATASET)
prj = pd.read_csv(BOOTSTRAP)


def read_old_csv(path):
                for enc in ["utf-8", "cp1252", "latin1"]:
                    try:
                        return pd.read_csv(path, encoding=enc)
                    except UnicodeDecodeError:
                        pass
                raise UnicodeDecodeError(f"Could not decode file: {path}")


pop_un = read_old_csv(WPP_POP)
loc = read_old_csv(WPP_LOC)
ssp_pop_raw = read_old_csv(SSP_POP)
ssp_growth_raw = read_old_csv(SSP_GROWTH)
tchg_df = read_old_csv(TCHG_FILE)

yrs = np.arange(2010, 2100)

# ---------------------------------------------------------------------
# BASELINE COUNTRY DATA (1980+ with non-missing temp and growth)
# matches ComputeMainProjections.R
# ---------------------------------------------------------------------
dta = dta.copy()
dta["gdpCap"] = dta["TotGDP"] / dta["Pop"]

mt = (
    dta.loc[
        (dta["year"] >= 1980)
        & dta["UDel_temp_popweight"].notna()
        & dta["growthWDI"].notna(),
        ["iso", "UDel_temp_popweight", "growthWDI", "gdpCap"]
    ]
    .groupby("iso", as_index=False)
    .agg(
        meantemp=("UDel_temp_popweight", "mean"),
        basegrowth=("growthWDI", "mean"),
        gdpCap=("gdpCap", "mean"),
    )
)

# ---------------------------------------------------------------------
# BASE POPULATION PROJECTIONS (UN WPP) -> popProjections[[1]]
# old R reshape is fragile; keep the Medium variant if present
# ---------------------------------------------------------------------
pop_un = pop_un.loc[pop_un["Time"].between(2010, 2100)].copy()
print(WPP_LOC)
print(loc.columns.tolist())
print(loc.head())

variant_cols = [c for c in pop_un.columns if "Variant" in c or "variant" in c]
if variant_cols:
    # prefer Medium if present
    varcol = variant_cols[0]
    medium_mask = pop_un[varcol].astype(str).str.contains("Medium", case=False, na=False)
    if medium_mask.any():
        pop_un = pop_un.loc[medium_mask].copy()

# keep one row per LocID-Time
pop_un = pop_un.sort_values(["LocID", "Time"]).drop_duplicates(["LocID", "Time"], keep="first")

pop_un = pop_un[["LocID", "Time", "PopTotal"]]
pop_un_wide = pop_un.pivot(index="LocID", columns="Time", values="PopTotal").reset_index()
pop_un_wide.columns = ["LocID"] + [str(c) for c in pop_un_wide.columns[1:]]
pop_un_wide = pop_un_wide.loc[pop_un_wide["LocID"] < 900].copy()

loc = loc.loc[loc["LocID"] < 900, ["LocID", "ISO3_Code"]].copy()
loc["iso"] = loc["ISO3_Code"].replace({"COD": "ZAR", "ROU": "ROM"})
loc = loc.loc[loc["iso"] != "GRL", ["LocID", "iso"]]

poploc = pop_un_wide.merge(loc, on="LocID", how="inner")
popprojb = mt.merge(poploc, on="iso", how="inner").drop(columns=["LocID"])

year_cols_un = [str(y) for y in yrs]
popprojb[year_cols_un] = popprojb[year_cols_un] / 1000.0  # millions, as in R

basegrowth = popprojb[["iso", "meantemp", "basegrowth", "gdpCap"]].copy()
for y in yrs:
    basegrowth[str(y)] = basegrowth["basegrowth"]


#save the baseline growth projections for later use 
pd.DataFrame(basegrowth).to_csv("../data/projections/old/baseline_growth_projections.csv", index=False)

# ---------------------------------------------------------------------
# SSP PROJECTIONS - contains all 6 different scenarios  - interpolate to annual
# ---------------------------------------------------------------------
ssp_pop_raw = ssp_pop_raw.copy()
if "Scenario" in ssp_pop_raw.columns:
    ssp_pop_raw["Scenario"] = ssp_pop_raw["Scenario"].replace({"SSP4d_v9_130115": "SSP4_v9_130115"})

pop1 = ipolate_like_r(ssp_pop_raw, yrs)
growth1 = ipolate_like_r(ssp_growth_raw, yrs)
growth1[[str(y) for y in yrs]] = growth1[[str(y) for y in yrs]] / 100.0

popSSP = mt.merge(pop1, left_on="iso", right_on="Region", how="inner")
growthSSP = mt.merge(growth1, left_on="iso", right_on="Region", how="inner")

# Scenario list exactly like R: base, SSP1..SSP5
popProjections = [popprojb]
growthProjections = [basegrowth]

for scen in range(1, 6):
    pgrow = (
        (growthSSP["Model"] == "OECD Env-Growth")
        & (growthSSP["Scenario"] == f"SSP{scen}_v9_130325")
    )
    ppop = popSSP["Scenario"] == f"SSP{scen}_v9_130115"

    growthProjections.append(growthSSP.loc[pgrow].copy())
    popProjections.append(popSSP.loc[ppop].copy())


# ---------------------------------------------------------------------
# TEMPERATURE-CHANGE VECTOR
# merges with popProjections[[1]][,1:3] and then keeps Tchg
# ---------------------------------------------------------------------
tchg = popProjections[0][["iso", "meantemp", "basegrowth"]].merge(
    tchg_df, left_on="iso", right_on="GMI_CNTRY", how="inner"
)
drop_names = {"West Bank", "Gaza Strip", "Bouvet Island"}
if "CNTRY_NAME" in tchg.columns:
    tchg = tchg.loc[~tchg["CNTRY_NAME"].isin(drop_names)].copy()

dtm = tchg["Tchg"].to_numpy(dtype=float)
ccd = dtm / len(yrs)

# ---------------------------------------------------------------------
# POOLED / NO-LAG / SSP5
# In Burke's indexing: scens = c("base","SSP1","SSP2","SSP3","SSP4","SSP5")
# so SSP5 is entry 6 in R, index 5 here.
# ---------------------------------------------------------------------
scen_idx = 5  # Python index for SSP5

growthproj = growthProjections[scen_idx].copy()
popproj = popProjections[scen_idx].copy()

# Align countries by iso, to avoid row mismatches in modern reruns
common_iso = pd.Index(growthproj["iso"]).intersection(popproj["iso"])
growthproj = growthproj.set_index("iso").loc[common_iso].reset_index()
popproj = popproj.set_index("iso").loc[common_iso].reset_index()

# Align dtm/ccd to the same iso set via popProjections[[1]] merge
ccd_df = tchg[["iso", "Tchg"]].copy().drop_duplicates("iso")
ccd_df = ccd_df.set_index("iso").reindex(common_iso)
ccd = (ccd_df["Tchg"].to_numpy(dtype=float)) / len(yrs)

n_ctry = len(common_iso)
n_years = len(yrs)
n_boot = len(prj)

basegdp = popproj["gdpCap"].to_numpy(dtype=float)
temp = popproj["meantemp"].to_numpy(dtype=float)

GDPcapCC = np.zeros((n_ctry, n_years, n_boot), dtype=float)
GDPcapNoCC = np.zeros((n_ctry, n_years, n_boot), dtype=float)
GDPcapCC[:, 0, :] = basegdp[:, None]
GDPcapNoCC[:, 0, :] = basegdp[:, None]


# calculate the GDP/cap projections for each bootstrap iteration, with and without climate change impacts

tots = np.zeros((n_boot, n_years, 4), dtype=float)

for tt in range(n_boot):
    beta1 = float(prj.loc[tt, "temp"])
    beta2 = float(prj.loc[tt, "temp2"])

    bg = beta1 * temp + beta2 * temp * temp

    for i in range(1, n_years):
        j = i - 1
        y = str(yrs[i])

        basegrowth_y = growthproj[y].to_numpy(dtype=float)
        GDPcapNoCC[:, i, tt] = GDPcapNoCC[:, j, tt] * (1.0 + basegrowth_y)

        newtemp = temp + j * ccd
        dg = beta1 * newtemp + beta2 * newtemp * newtemp
        hot = newtemp > 30.0
        dg[hot] = beta1 * 30.0 + beta2 * 30.0 * 30.0

        diff = dg - bg
        GDPcapCC[:, i, tt] = GDPcapCC[:, j, tt] * (1.0 + basegrowth_y + diff)

        wt = popproj[y].to_numpy(dtype=float)
        tots[tt, i, 0] = np.average(GDPcapCC[:, i, tt], weights=wt)
        tots[tt, i, 1] = np.average(GDPcapNoCC[:, i, tt], weights=wt)
        tots[tt, i, 2] = np.sum(GDPcapCC[:, i, tt] * wt * 1e6)
        tots[tt, i, 3] = np.sum(GDPcapNoCC[:, i, tt] * wt * 1e6)




# ---------------------------------------------------------------------
# FIGURE 5A OBJECT
# same as MakeFigure5.R:
# chgs <- tots[,,1]/tots[,,2]-1 ; chgs <- chgs*100 ; chgs[,1] <- 0
# note: R indices 1,2 correspond to Python 0,1
# ---------------------------------------------------------------------
chgs = (tots[:, :, 0] / tots[:, :, 1] - 1.0) * 100.0
chgs[:, 0] = 0.0

point_estimate = chgs[0, :]
end_2099 = point_estimate[np.where(yrs == 2099)[0][0]]
prob_positive_2099 = float(np.mean(chgs[:, np.where(yrs == 2099)[0][0]] > 0.0))

print(f"Point estimate in 2099: {end_2099:.3f}%")
print(f"P(positive impact in 2099): {prob_positive_2099:.3f}")

# ---------------------------------------------------------------------
# FIGURE 5A-STYLE PLOT (Python analogue of weightProj)
# ---------------------------------------------------------------------
yz = np.arange(-100.0, 100.0 + 0.1, 0.1)
Z = density_grid(chgs, yrs, yz, lo=0.05, hi=0.95, bw_adjust=0.6)

fig, ax = plt.subplots(figsize=(10, 6.5))
X, Y = np.meshgrid(yrs, yz)

pcm = ax.pcolormesh(X, Y, Z, shading="auto", cmap="Reds", vmin=0, vmax=1)
ax.plot(yrs, point_estimate, color="black", linewidth=2)
ax.axhline(0, color="black", linewidth=0.6)

ax.set_xlim(2010, 2100)
ax.set_ylim(-75, 60)
ax.set_xticks(np.arange(2020, 2101, 20))
ax.set_yticks(np.arange(-75, 61, 25))
ax.set_xlabel("year")
ax.set_ylabel("% change in GDP/cap")

# theme_classic-like appearance
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.show()

# Optional export of the point-estimate path
out = pd.DataFrame({"year": yrs, "point_estimate_pct": point_estimate})
out.to_csv("../data/projections/old/figure5a_point_estimate_python.csv", index=False)
density_grid_out = pd.DataFrame(Z, columns=[str(y) for y in yrs])
density_grid_out.to_csv("../data/projections/old/figure5a_density_grid_python.csv", index=False)
print("Saved figure5a_point_estimate_python.csv")
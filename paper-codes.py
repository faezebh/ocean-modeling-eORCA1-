
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 23 20:42:54 2025

@author: bahmanif
"""

from netCDF4 import Dataset  
from mpl_toolkits.basemap import Basemap
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

d6=xr.open_dataset('/esi/project/niwa02764/faezeh/eORCA1/mean_eORCA1-C14006o_1m_20000101_20191231_grid_T.nc') #control

d4=xr.open_dataset('/esi/project/niwa02764/faezeh/eORCA1/mean_eORCA1-C14004o_1m_20000101_20191231_grid_T.nc') #quadratic

d3=xr.open_dataset('/esi/project/niwa02764/faezeh/eORCA1/mean_eORCA1-C14003o_1m_20000101_20191231_grid_T.nc') #uniform



time=d6['time_counter'] #1
depth=d6['deptht']             #75
lat=d6['nav_lat']           #(332, 362)
lon=d6['nav_lon']          #(332, 362)
lon = lon.where(lon >= 0, lon + 360)


d=xr.open_dataset('/esi/project/niwa02764/faezeh/eORCA1/mesh_mask.nc')

e1t=d['e1t'][0,:,:]
e2t=d['e2t'][0,:,:]
tmask=d['tmask'][0,0,:,:]


#%%

# ******************************************************
# ***             data-mean         ***
# ******************************************************

"""           sea surface temperature        """

SST_6=d6['votemper']              #(1, 75, 332, 362)
SST_4=d4['votemper'] 
SST_3=d3['votemper'] 


"""           MLD        """


MLD_6=d6['somxl010']          ##(1, 332, 362)
MLD_4=d4['somxl010']
MLD_3=d3['somxl010']

"""           Wind Spedd        """

wind_speed_6=d6['sowindsp']     
wind_speed_4=d4['sowindsp']
wind_speed_3=d3['sowindsp']

"""           DIC_C14_Concentration        """


DIC_C14_Concentration_6=d6['DIC_C14_Concentration']      #(1, 75, 332, 362)
DIC_C14_Concentration_4=d4['DIC_C14_Concentration']
DIC_C14_Concentration_3=d3['DIC_C14_Concentration']


"""           Air_sea_flux-of_C14        """

Air_sea_flux_of_C14_6=d6['Air_sea_flux_of_C14']     #(1, 332, 362)
Air_sea_flux_of_C14_4=d4['Air_sea_flux_of_C14'] 
Air_sea_flux_of_C14_3=d3['Air_sea_flux_of_C14'] 


"""           inorganic_carbon        """

inorganic_carbon_6=d6['inorganic_carbon']      #(1, 75, 332, 362)DIC_Concentration
inorganic_carbon_4=d4['inorganic_carbon']
inorganic_carbon_3=d3['inorganic_carbon']


"""          Air_sea_flux-of_CO2        """

Air_sea_flux_of_CO2_6=d6['Air_sea_flux_of_CO2']      #(1, 332, 362)
Air_sea_flux_of_CO2_4=d4['Air_sea_flux_of_CO2'] 
Air_sea_flux_of_CO2_3=d3['Air_sea_flux_of_CO2']



#%%
# ******************************************************
# ***             check-BASINS         ***
# ******************************************************
# ============================================================
#  Atlantic = (-70..20)  
#  Indian   = (20..150)
#  Pacific  = (150..290)
#  
# ============================================================



def make_basin_masks(lon, lat, tmask=None,
                     atl_west_deg=70,  
                     atl_east_deg=20, 
                     ind_east_deg=150  
                     ):
    
    
    try:
        lon360 = lon.where(lon >= 0, lon + 360)
    except Exception:
        lon360 = np.where(lon >= 0, lon, lon + 360)

    LON = np.asarray(lon360)
    LAT = np.asarray(lat)

    atl_west_360 = (360 - atl_west_deg) % 360  # e.g., 360-70 = 290

    A = (LON >= atl_west_360) | (LON < atl_east_deg)
    I = (LON >= atl_east_deg) & (LON < ind_east_deg)
    P = (LON >= ind_east_deg) & (LON < atl_west_360)

    if tmask is not None:
        TM = np.asarray(tmask)
        ocean = TM > 0
        A = A & ocean
        I = I & ocean
        P = P & ocean

    return lon360, A, I, P


def plot_basins_overlay(lon360, lat, mask_atlantic, mask_indian, mask_pacific,
                        title="Basin masks (overlay)", savepath=None):
    
    LON = np.asarray(lon360)
    LAT = np.asarray(lat)

    A = np.asarray(mask_atlantic)
    I = np.asarray(mask_indian)
    P = np.asarray(mask_pacific)

    Amap = np.where(A, 1.0, np.nan)
    Imap = np.where(I, 1.0, np.nan)
    Pmap = np.where(P, 1.0, np.nan)

    atl_blue   = '#2b83ba'
    ind_green  = '#4daf4a'
    pac_orange = '#fdae61'

    fig, ax = plt.subplots(figsize=(10, 6))
    m = Basemap(projection='spstere', boundinglat=-40, lon_0=180, resolution='l', ax=ax)

    # Draw in any order; zorder ensures Atlantic on top if overlap (should be none)
    m.contourf(LON, LAT, Pmap, levels=[0.5, 1.5], colors=[pac_orange], latlon=True, zorder=1)
    m.contourf(LON, LAT, Imap, levels=[0.5, 1.5], colors=[ind_green],  latlon=True, zorder=2)
    m.contourf(LON, LAT, Amap, levels=[0.5, 1.5], colors=[atl_blue],   latlon=True, zorder=3)

    # Optional basin boundary longitudes for Atlantic=-70..20, Indian=20..150, Pacific=150..290
    m.drawmeridians([20, 150, 290], labels=[1, 1, 1, 0], linewidth=1.2, color='k', zorder=10)
    m.drawparallels([-40, -60], linewidth=0.2, zorder=10)

    m.drawcoastlines(linewidth=0.6, zorder=20)
    m.fillcontinents(color='lightgray', lake_color='white', zorder=20)

    legend_patches = [
        Patch(color=atl_blue,   label='Atlantic'),
        Patch(color=ind_green,  label='Indian'),
        Patch(color=pac_orange, label='Pacific')
    ]
    plt.legend(handles=legend_patches, loc='center left', bbox_to_anchor=(1.05, 0.5), frameon=False)
    plt.title(title, fontsize=13)
    plt.tight_layout()

    if savepath is not None:
        plt.savefig(savepath, dpi=300, bbox_inches='tight')

    plt.show()

    # Diagnostics
    print("Ocean cells A/I/P:", int(np.sum(A)), int(np.sum(I)), int(np.sum(P)))
    print("Overlaps A&I / A&P / I&P:", int(np.sum(A & I)), int(np.sum(A & P)), int(np.sum(I & P)))
    print("None:", int(np.sum(~(A | I | P))))



lon360, mask_atlantic, mask_indian, mask_pacific = make_basin_masks(
    lon, lat, tmask=tmask,
    atl_west_deg=70,   # Atlantic west = -70
    atl_east_deg=20,   # Atlantic east = 20E
    ind_east_deg=150   # Indian east = 150E
)

plot_basins_overlay(
    lon360, lat,
    mask_atlantic, mask_indian, mask_pacific,
    title="Basin masks (Atlantic=-70..20 → 290..360 & 0..20)",
    savepath="/esi/project/niwa02764/faezeh/plot/basins_overlay.png"
)




# ============================================================
#   Basin-Region    
# ============================================================


# -------------------------
lon360 = lon.where(lon >= 0, lon + 360)

# -------------------------
# 1) BASINS 
# -------------------------
mask_atlantic = (lon360 >= 290) | (lon360 < 20)
mask_indian   = (lon360 >= 20) & (lon360 < 150)
mask_pacific  = (lon360 >= 150) & (lon360 < 290)


# ******************************************************
# ***             show 6 ZONES         ***
# ******************************************************
ds = xr.open_dataset('/esi/project/niwa02764/faezeh/eORCA1/fronts.nc')

def _prep_front_for_interp(lon_front, lat_front):
    
    lon_front = np.asarray(lon_front)
    lat_front = np.asarray(lat_front)

    valid = ~np.isnan(lon_front) & ~np.isnan(lat_front)
    lon_f = (lon_front[valid] % 360.0).astype(float)
    lat_f = lat_front[valid].astype(float)

    order = np.argsort(lon_f)
    lon_f = lon_f[order]
    lat_f = lat_f[order]

    lon_unique, idx = np.unique(lon_f, return_index=True)
    lat_unique = lat_f[idx]
    return lon_unique, lat_unique


def mask_between_fronts_by_lat(ds, front_north, front_south, lon360, lat, tmask=None):
    lon_n = ds[f"Lon{front_north}"].values
    lat_n = ds[f"Lat{front_north}"].values
    lon_s = ds[f"Lon{front_south}"].values
    lat_s = ds[f"Lat{front_south}"].values

    lon_n_s, lat_n_s = _prep_front_for_interp(lon_n, lat_n)
    lon_s_s, lat_s_s = _prep_front_for_interp(lon_s, lat_s)

    LON = np.asarray(lon360).astype(float) % 360.0
    LAT = np.asarray(lat).astype(float)

    lat_n_interp = np.interp(LON, lon_n_s, lat_n_s)
    lat_s_interp = np.interp(LON, lon_s_s, lat_s_s)

    mask = (LAT <= lat_n_interp) & (LAT > lat_s_interp)
    if tmask is not None:
        mask &= (np.asarray(tmask) > 0)

    return mask.astype(int)


# ------------------------------
#  zone mask (1..6)
# ------------------------------
LAT = np.asarray(lat)
LON = np.asarray(lon360) % 360.0
TM  = np.asarray(tmask) if (tmask is not None) else None

zone_mask = np.zeros_like(LAT, dtype=int)

# ---- Zone 1: 40S–NB (near NB points) ----
if "LatNB" in ds and "LonNB" in ds:
    lon_nb = np.asarray(ds["LonNB"].values)
    lat_nb = np.asarray(ds["LatNB"].values)

    valid = ~np.isnan(lon_nb) & ~np.isnan(lat_nb)
    lon_nb, lat_nb = lon_nb[valid], lat_nb[valid]

    lon_nb_mod = (lon_nb % 360.0).astype(float)

    for i in range(len(lat_nb)):
        lat_i = float(lat_nb[i])
        lon_i = float(lon_nb_mod[i])

        # circular distance (dateline safe)
        dist = np.abs(LON - lon_i)
        dist = np.minimum(dist, 360.0 - dist)
        mask_lon = dist < 2.0

        if lat_i > -40:
            mask_lat = (LAT >= -40) & (LAT <= lat_i)
        else:
            mask_lat = (LAT <= -40) & (LAT >= lat_i)

        mask = mask_lon & mask_lat
        if TM is not None:
            mask &= (TM > 0)

        zone_mask[mask] = 1

print("Zone 1 pixel count:", int(np.sum(zone_mask == 1)))

# ---- Zone 6: SB–COAST (south of SB) ----
if "LatSB" in ds and "LonSB" in ds:
    lon_sb = ds["LonSB"].values
    lat_sb = ds["LatSB"].values

    lon_sb_s, lat_sb_s = _prep_front_for_interp(lon_sb, lat_sb)
    lat_sb_interp = np.interp(LON, lon_sb_s, lat_sb_s)

    mask_sb_antarctic = (LAT < lat_sb_interp) & (LAT > -80)
    if TM is not None:
        mask_sb_antarctic &= (TM > 0)

    zone_mask = np.where(mask_sb_antarctic, 6, zone_mask)

# ---- Zones 2–5: between fronts ----
m2 = mask_between_fronts_by_lat(ds, "NB", "SAF", lon360, lat, tmask)
zone_mask = np.where((m2 == 1) & (zone_mask == 0), 2, zone_mask)

m3 = mask_between_fronts_by_lat(ds, "SAF", "PF", lon360, lat, tmask)
zone_mask = np.where((m3 == 1) & (zone_mask == 0), 3, zone_mask)

m4 = mask_between_fronts_by_lat(ds, "PF", "SACCF", lon360, lat, tmask)
zone_mask = np.where((m4 == 1) & (zone_mask == 0), 4, zone_mask)

m5 = mask_between_fronts_by_lat(ds, "SACCF", "SB", lon360, lat, tmask)
zone_mask = np.where((m5 == 1) & (zone_mask == 0), 5, zone_mask)

# ---- Zone boolean masks ----
zone_mask_40_nb     = (zone_mask == 1)
zone_mask_nb_saf    = (zone_mask == 2)
zone_mask_saf_pf    = (zone_mask == 3)
zone_mask_pf_saccf  = (zone_mask == 4)
zone_mask_saccf_sb  = (zone_mask == 5)
zone_mask_sb_ant    = (zone_mask == 6)

# -------------------------
# 2) Zone-only plot 
# -------------------------
def plot_zone_mask_spstere(lon360, lat, zone_mask, tmask=None, title="Zones 1..6"):
    LON = np.asarray(lon360) % 360.0
    LAT = np.asarray(lat)
    Z   = np.asarray(zone_mask).astype(float)

    if tmask is not None:
        Z = np.where(np.asarray(tmask) > 0, Z, np.nan)
    Z = np.where((Z >= 1) & (Z <= 6), Z, np.nan)

    fig, ax = plt.subplots(figsize=(10, 6))
    m = Basemap(projection='spstere', boundinglat=-40, lon_0=180, resolution='l', ax=ax)

    levels = np.arange(0.5, 7.5, 1.0)
    m.contourf(LON, LAT, Z, levels=levels, latlon=True, cmap='tab10')

    m.drawcoastlines(linewidth=0.6)
    m.fillcontinents(color='lightgray', lake_color='white')
    m.drawparallels([-40, -60], linewidth=0.2)
    m.drawmeridians([0, 60, 120, 180, 240, 300], labels=[1, 0, 0, 0], linewidth=0.2)

    plt.title(title, fontsize=13)
    plt.tight_layout()
    plt.show()

plot_zone_mask_spstere(lon360, lat, zone_mask, tmask=tmask, title="Zones 1..6 (spstere)")
#%%
# -------------------------
#  Combined Zone+Basin plot
# =========================

from matplotlib.colors import ListedColormap



def plot_zones_basins_onepass(lon360, lat, zone_mask,
                              mask_atlantic, mask_indian, mask_pacific,
                              tmask=None, savepath=None):
    LON = np.asarray(lon360) % 360.0
    LAT = np.asarray(lat)
    Z   = np.asarray(zone_mask).astype(int)

    ocean = (np.asarray(tmask) > 0) if (tmask is not None) else np.ones_like(Z, dtype=bool)

    # Basin index: Atlantic=0, Indian=1, Pacific=2
    basin_idx = np.full(Z.shape, -1, dtype=int)
    basin_idx[np.asarray(mask_atlantic) & ocean] = 0
    basin_idx[np.asarray(mask_indian)   & ocean] = 1
    basin_idx[np.asarray(mask_pacific)  & ocean] = 2


    combined = np.full(Z.shape, np.nan, dtype=float)

    ok = ocean & (basin_idx >= 0) & (Z >= 1) & (Z <= 6)
    combined[ok] = (basin_idx[ok] * 6 + (Z[ok] - 1) + 1).astype(float)

    # 18 colors
    atlantic_colors = ['#f3e5f5', '#e1bee7', '#ce93d8', '#ba68c8', '#ab47bc', '#9c27b0']
    indian_colors   = ['#d9d9d9', '#c2b8aa', '#a89e91', '#8f857a', '#756c63', '#5c544d']
    pacific_colors  = ['#fdae6b', '#fd8d3c', '#f16913', '#d94801', '#a63603', '#7f2704']
    cmap = ListedColormap(atlantic_colors + indian_colors + pacific_colors)

    fig, ax = plt.subplots(figsize=(11, 6))
    m = Basemap(projection='spstere', boundinglat=-40, lon_0=180, resolution='l', ax=ax)

    pm = m.pcolormesh(
        LON, LAT, combined,
        latlon=True,
        cmap=cmap,
        shading='nearest',
        rasterized=True,
        antialiased=False,  
        linewidth=0.0
    )


    Zdisp = np.where(ok, Z.astype(float), np.nan)
    front_levels = np.arange(1.5, 6.0, 1.0)
    m.contour(LON, LAT, Zdisp, levels=front_levels, latlon=True,
              colors='black', linewidths=0.8, zorder=10)


    bdisp = np.where(ocean & (basin_idx >= 0), basin_idx.astype(float), np.nan)
    m.contour(LON, LAT, bdisp, levels=[0.5, 1.5], latlon=True,
              colors='black', linewidths=1.2, zorder=10)

    m.drawcoastlines(linewidth=0.6, zorder=20)
    m.fillcontinents(color='lightgray', lake_color='white', zorder=20)
    m.drawparallels([-40, -60], linewidth=0.2, zorder=20)
    m.drawmeridians([20, 150, 290], labels=[1, 1, 1, 0], linewidth=1.2, color='k', zorder=20)

    # Legend (18 entries)
    zone_labels_dict = {
        1: "40°S–NB", 2: "NB–SAF", 3: "SAF–PF", 4: "PF–SACCF", 5: "SACCF–SB", 6: "SB–COAST"
    }
    basin_names = ["Atlantic", "Indian", "Pacific"]
    basin_cols  = [atlantic_colors, indian_colors, pacific_colors]

    legend_patches = []
    for basin, cols in zip(basin_names, basin_cols):
        for znum, col in zip(range(1, 7), cols):
            legend_patches.append(Patch(color=col, label=f"{basin} - {zone_labels_dict[znum]}"))

    plt.legend(handles=legend_patches, loc='center left', bbox_to_anchor=(1.02, 0.5),
               fontsize=9, frameon=False, title="Zones")

    plt.title("Zones - Basins", fontsize=13)
    plt.tight_layout()

    if savepath is not None:
        plt.savefig(savepath, dpi=300, bbox_inches='tight')

    plt.show()


plot_zones_basins_onepass(
    lon360, lat, zone_mask,
    mask_atlantic, mask_indian, mask_pacific,
    tmask=tmask,
    savepath="/esi/project/niwa02764/faezeh/plot/zones_basins_overlay_noseams.png"
)



#%%
# -------------------------
# 4) BAR CHARTS (MLD + Wind) -CONTROL
# -------------------------
zones = {
    "40°S–NB":   zone_mask_40_nb,
    "NB–SAF":    zone_mask_nb_saf,
    "SAF–PF":    zone_mask_saf_pf,
    "PF–SACCF":  zone_mask_pf_saccf,
    "SACCF–SB":  zone_mask_saccf_sb,
    "SB-COAST":  zone_mask_sb_ant
}
basins = {"Atlantic": mask_atlantic, "Indian": mask_indian, "Pacific": mask_pacific}

zone_labels = list(zones.keys())
basin_labels = ['Atlantic', 'Indian', 'Pacific']
n_time = MLD_6.shape[0]

# ---- area-avg MLD ----
area_avg_mld = {}
for zone_name, zmask in zones.items():
    for basin_name, bmask in basins.items():
        vals = np.zeros(n_time)
        region_mask = (np.asarray(zmask) & np.asarray(bmask))
        for t in range(n_time):
            num = np.sum(MLD_6[t] * e1t * e2t * region_mask * tmask)
            den = np.sum(e1t * e2t * region_mask * tmask)
            vals[t] = num / den if den > 0 else np.nan
        area_avg_mld[(zone_name, basin_name)] = vals

mld_means = np.array([
    [np.nanmean(area_avg_mld[(zone, basin)]) for basin in basin_labels]
    for zone in zone_labels
])

# ---- area-avg Wind ----
area_avg_wind = {}
for zone_name, zmask in zones.items():
    for basin_name, bmask in basins.items():
        vals = np.zeros(n_time)
        region_mask = (np.asarray(zmask) & np.asarray(bmask))
        for t in range(n_time):
            num = np.sum(wind_speed_6[t] * e1t * e2t * region_mask * tmask)
            den = np.sum(e1t * e2t * region_mask * tmask)
            vals[t] = num / den if den > 0 else np.nan
        area_avg_wind[(zone_name, basin_name)] = vals

wind_means = np.array([
    [np.nanmean(area_avg_wind[(zone, basin)]) for basin in basin_labels]
    for zone in zone_labels
])

# ---- Plot bars ----
x = np.arange(len(zone_labels))
bar_width = 0.25
colors_b = ['#dbae1d', '#2887a1', '#60b187']

fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=False)
plt.subplots_adjust(wspace=0.3)

# Wind
axW = axes[0]
for i, basin in enumerate(basin_labels):
    axW.bar(x + i * bar_width, wind_means[:, i], width=bar_width,
            label=basin, color=colors_b[i], edgecolor='black')
axW.set_xticks(x + bar_width)
axW.set_xticklabels(zone_labels, rotation=45, ha='right')
axW.set_ylabel("Mean Wind Speed (m/s)")
axW.set_title("Mean Wind Speed per Zone by Basin", y=1.02)
axW.legend()
axW.grid(axis='y', linestyle='--', alpha=0.4)

# MLD
axM = axes[1]
for i, basin in enumerate(basin_labels):
    axM.bar(x + i * bar_width, mld_means[:, i], width=bar_width,
            label=basin, color=colors_b[i], edgecolor='black')
axM.set_xticks(x + bar_width)
axM.set_xticklabels(zone_labels, rotation=45, ha='right')
axM.set_ylabel("Mean MLD (m)")
axM.set_title("Mean MLD per Zone by Basin", y=1.02)
axM.legend()
axM.grid(axis='y', linestyle='--', alpha=0.4)

plt.tight_layout()
plt.show()

# -------------------------
# Basin counts per Zone
# -------------------------
print("=== Basin counts per Zone ===")
for z in range(1, 7):
    zm = (zone_mask == z)
    print(
        f"Zone {z}:",
        "A", int(np.sum(zm & np.asarray(mask_atlantic))),
        "I", int(np.sum(zm & np.asarray(mask_indian))),
        "P", int(np.sum(zm & np.asarray(mask_pacific))),
        "Total", int(np.sum(zm))
    )
print("=============================")




#%%
atlantic_colors = ['#f3e5f5', '#e1bee7', '#ce93d8', '#ba68c8', '#ab47bc', '#9c27b0']
indian_colors   = ['#d9d9d9', '#c2b8aa', '#a89e91', '#8f857a', '#756c63', '#5c544d']
pacific_colors  = ['#fdae6b', '#fd8d3c', '#f16913', '#d94801', '#a63603', '#7f2704']
colors = [atlantic_colors[-1], indian_colors[-1], pacific_colors[-1]]


# ============================================
#  wind speed and mld for control 
# ============================================

plt.rcParams.update({
    "figure.dpi": 150,
    "axes.titlesize": 12,
    "axes.labelsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12
})




#%%

from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "figure.dpi": 150,
    "axes.titlesize": 12,
    "axes.labelsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12
})

# -------------------------
#  fields
# -------------------------
mean_wind_speed_6 = np.nanmean(wind_speed_6, axis=0)
mean_MLD_6        = np.nanmean(MLD_6, axis=0)

mean_wind_speed_6n = np.where(lat < -40, mean_wind_speed_6, np.nan)
mean_MLD_6n        = np.where(lat < -40, mean_MLD_6, np.nan)

# -------------------------
# Bar placement
# -------------------------
x = np.arange(len(zone_labels))
bar_width = 0.25

# -------------------------
# Color 
# -------------------------
atlantic_colors = ['#f3e5f5', '#e1bee7', '#ce93d8', '#ba68c8', '#ab47bc', '#9c27b0']
indian_colors   = ['#d9d9d9', '#c2b8aa', '#a89e91', '#8f857a', '#756c63', '#5c544d']
pacific_colors  = ['#fdae6b', '#fd8d3c', '#f16913', '#d94801', '#a63603', '#7f2704']

# -------------------------
# fronts
# -------------------------
def plot_front(lon_f, lat_f, label, color, m, ax):
    lon_f = np.asarray(lon_f)
    lat_f = np.asarray(lat_f)
    lon_f = lon_f % 360.0
    x_f, y_f = m(lon_f, lat_f)
    ax.plot(x_f, y_f, label=label, color=color, linewidth=1)

# -------------------------
# Figure 
# -------------------------
fig, axes = plt.subplots(
    2, 2,
    figsize=(16, 16),
    gridspec_kw={'height_ratios': [1.5, 0.5],
                 'width_ratios': [1, 1]}
)

labels = ['a', 'b', 'c', 'd']
positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
for label, (i, j) in zip(labels, positions):
    axes[i, j].text(
        0.001, 1.15,
        label + ':',
        transform=axes[i, j].transAxes,
        fontsize=16,
        fontweight='bold',
        va='top',
        ha='left'
    )


# =========================================================
# (1) Wind Speed
# =========================================================
ax1 = axes[1, 0]

for bi, basin in enumerate(basin_labels):
    for zi, zname in enumerate(zone_labels):
        if basin == "Atlantic":
            col = atlantic_colors[zi]
        elif basin == "Indian":
            col = indian_colors[zi]
        else:
            col = pacific_colors[zi]

        ax1.bar(
            x[zi] + bi * bar_width,
            wind_means[zi, bi],
            width=bar_width * 0.7,
            color=col,
            edgecolor='black',
            linewidth=0.6,
            alpha=0.95,
        )

ax1.set_xticks(x + (len(basin_labels)-1) * bar_width / 2)
ax1.set_xticklabels(zone_labels, rotation=25, ha='right', fontsize=12)
ax1.set_ylabel("Wind Speed (m s$^{-1}$)", fontsize=12, fontweight='bold')
ax1.set_ylim(0, 12)
ax1.grid(axis='y', linestyle='--', linewidth=0.8, alpha=0.8)
for spine in ["top", "right"]:
    ax1.spines[spine].set_visible(False)

# =========================================================
# (2) MLD Bar 
# =========================================================
ax2 = axes[1, 1]

for bi, basin in enumerate(basin_labels):
    for zi, zname in enumerate(zone_labels):
        if basin == "Atlantic":
            col = atlantic_colors[zi]
        elif basin == "Indian":
            col = indian_colors[zi]
        else:
            col = pacific_colors[zi]

        ax2.bar(
            x[zi] + bi * bar_width,
            mld_means[zi, bi],
            width=bar_width * 0.7,
            color=col,
            edgecolor='black',
            linewidth=0.6,
            alpha=0.95
        )

ax2.set_xticks(x + (len(basin_labels)-1) * bar_width / 2)
ax2.set_xticklabels(zone_labels, rotation=25, ha='right', fontsize=12)
ax2.set_ylabel("Mixed Layer Depth (m)", fontsize=12, fontweight='bold')
ax2.grid(axis='y', linestyle='--', linewidth=0.8, alpha=0.8)
for spine in ["top", "right"]:
    ax2.spines[spine].set_visible(False)

# Basin legend 
legend_handles = [
    Patch(facecolor=atlantic_colors[-1], edgecolor='black', label='Atlantic'),
    Patch(facecolor=indian_colors[-1],   edgecolor='black', label='Indian'),
    Patch(facecolor=pacific_colors[-1],  edgecolor='black', label='Pacific')
]
ax1.legend(handles=legend_handles, loc='upper left', bbox_to_anchor=(-0.37,1.1),
           frameon=True, fontsize=12, title="Basin")

# ===================
# (3) Mean Wind Map 
# ===================
ax3 = axes[0, 0]
m1 = Basemap(projection='spstere', boundinglat=-40, lon_0=180, resolution='l', ax=ax3)
m1.drawcoastlines(linewidth=0.5)
m1.drawmapboundary(fill_color='white')

wind_levels = np.linspace(4, 13, 19)
cax1 = m1.contourf(lon, lat, mean_wind_speed_6n, latlon=True,
                   levels=wind_levels, cmap='rainbow', extend='both')

m1.fillcontinents(color='#dddddd', lake_color='white')
m1.drawparallels([-40, -60], linestyle='solid', linewidth=1, color='black')
m1.drawmeridians([30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330],
                 labels=[True, True, True, True], fontsize=12)

# FRONTS 
plot_front(ds["LonNB"],    ds["LatNB"],    "NB",    "black",       m1, ax3)
plot_front(ds["LonSAF"],   ds["LatSAF"],   "SAF",   "darkmagenta", m1, ax3)
plot_front(ds["LonPF"],    ds["LatPF"],    "PF",    "deeppink",    m1, ax3)
plot_front(ds["LonSACCF"], ds["LatSACCF"], "SACCF", "blue",        m1, ax3)
plot_front(ds["LonSB"],    ds["LatSB"],    "SB",    "red",         m1, ax3)

# Latitude labels
x40, y40 = m1(0, -40)
x60, y60 = m1(0, -60)
ax3.text(x40, y40, '40°S', fontsize=12, ha='center', va='bottom', color='black')
ax3.text(x60, y60, '60°S', fontsize=12, ha='center', va='bottom', color='black')

# ==============
# (4) Mean MLD 
# ==============
ax4 = axes[0, 1]
m2 = Basemap(projection='spstere', boundinglat=-40, lon_0=180, resolution='l', ax=ax4)
m2.drawcoastlines(linewidth=0.5)
m2.drawmapboundary(fill_color='white')

mld_levels = np.linspace(50, 200, 31)
cax2 = m2.contourf(lon, lat, mean_MLD_6n, latlon=True,
                   levels=mld_levels, cmap='BuGn', extend='both')

m2.fillcontinents(color='#dddddd', lake_color='white')
m2.drawparallels([-40, -60], linestyle='solid', linewidth=1, color='black')
m2.drawmeridians([30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330],
                 labels=[True, True, True, True], fontsize=12)

# FRONTS
plot_front(ds["LonNB"],    ds["LatNB"],    "NB",    "black",       m2, ax4)
plot_front(ds["LonSAF"],   ds["LatSAF"],   "SAF",   "darkmagenta", m2, ax4)
plot_front(ds["LonPF"],    ds["LatPF"],    "PF",    "deeppink",    m2, ax4)
plot_front(ds["LonSACCF"], ds["LatSACCF"], "SACCF", "blue",        m2, ax4)
plot_front(ds["LonSB"],    ds["LatSB"],    "SB",    "red",         m2, ax4)

# Latitude labels
x40, y40 = m2(0, -40)
x60, y60 = m2(0, -60)
ax4.text(x40, y40, '40°S', fontsize=12, ha='center', va='bottom', color='black')
ax4.text(x60, y60, '60°S', fontsize=12, ha='center', va='bottom', color='black')

# ==========
# Colorbars
# ==========
for ax, cf, label in zip([ax3, ax4], [cax1, cax2],
                        ["Wind Speed (m s$^{-1}$)", "Mixed Layer Depth (m)"]):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("bottom", size="5%", pad=0.35)
    cb = fig.colorbar(cf, cax=cax, orientation='horizontal')
    cb.set_label(label, fontsize=12, fontweight='bold')
    cb.ax.tick_params(labelsize=12, labelrotation=0)

# ============
# Front legend 
# ============
front_legend = [
    Line2D([0], [0], color='black',       lw=1, label='NB'),
    Line2D([0], [0], color='darkmagenta', lw=1, label='SAF'),
    Line2D([0], [0], color='deeppink',    lw=1, label='PF'),
    Line2D([0], [0], color='blue',        lw=1, label='SACCF'),
    Line2D([0], [0], color='red',         lw=1, label='SB')
]

fig.legend(handles=front_legend, loc='upper left', bbox_to_anchor=(0.01, 0.79),
           ncol=1, frameon=True, fontsize=12)

plt.subplots_adjust(hspace=0.001, wspace=0.28)   # این خط رو بیار پایین‌تر
plt.savefig("/esi/project/niwa02764/faezeh/plot/abc40_MLD_WS_Control.png",
            dpi=400, bbox_inches='tight')
plt.show()






#%%


"""--------------------------------------------------------------
                  Season winter summer
--------------------------------------------------------------"""
summer=xr.open_dataset('/esi/project/niwa02764/faezeh/eORCA1/DJF-eORCA1-C14006o_1m_20000101_20191231_grid_T.nc')
winter=xr.open_dataset('/esi/project/niwa02764/faezeh/eORCA1/JJA-eORCA1-C14006o_1m_20000101_20191231_grid_T.nc')

summer_mld=summer['somxl010']
winter_mld=winter['somxl010']


summer_sst=summer['votemper']
winter_sst=winter.variables['votemper']

summer_wind_speed=summer['sowindsp']      #(240, 332, 362)
winter_wind_speed=winter['sowindsp']



#%%
# === Define zones ===
scenario_labels_pretty = {
    "Quadratic ": "Quadratic ",
    "Uniform ": "Uniform ",
    "Control (baseline)": "Control "
}
# Convert all zone masks to boolean for safety
zone_mask_40_nb   = zone_mask_40_nb.astype(bool)
zone_mask_nb_saf  = zone_mask_nb_saf.astype(bool)
zone_mask_saf_pf  = zone_mask_saf_pf.astype(bool)
zone_mask_pf_saccf = zone_mask_pf_saccf.astype(bool)
zone_mask_saccf_sb = zone_mask_saccf_sb.astype(bool)
zone_mask_sb_ant  = zone_mask_sb_ant.astype(bool)


zones = {
    "40°S–NB": zone_mask_40_nb,
    "NB–SAF": zone_mask_nb_saf,
    "SAF–PF": zone_mask_saf_pf,
    "PF–SACCF": zone_mask_pf_saccf,
    "SACCF–SB": zone_mask_saccf_sb,
    "SB-COAST": zone_mask_sb_ant
    
}


# Convert basin masks to boolean as well
mask_atlantic = mask_atlantic.astype(bool)
mask_indian   = mask_indian.astype(bool)
mask_pacific  = mask_pacific.astype(bool)
# === Define basins ===
basins = {
    "Atlantic": mask_atlantic,
    "Indian": mask_indian,
    "Pacific": mask_pacific
}



# === Define scenarios ===
scenarios_mld = {
    "Quadratic ": MLD_4,
    "Uniform ": MLD_3,
    "Control (baseline)": MLD_6
}


scenarios_ws = {
    "Quadratic ": wind_speed_4,
    "Uniform ": wind_speed_3,
    "Control (baseline)": wind_speed_6
}


# SST_6_100 = np.mean(SST_6[:,:23,:,:],axis=1)
# SST_4_100 = np.mean(SST_4[:,:23,:,:],axis=1)
# SST_3_100 = np.mean(SST_3[:,:23,:,:],axis=1)
SST_6_100 = SST_6.isel(deptht=slice(0, 23)).mean(dim="deptht", skipna=True)
SST_4_100 = SST_4.isel(deptht=slice(0, 23)).mean(dim="deptht", skipna=True)
SST_3_100 = SST_3.isel(deptht=slice(0, 23)).mean(dim="deptht", skipna=True)
scenario_sst={
    "Quadratic": SST_4_100,
    "Uniform": SST_3_100,
    "Control (baseline)": SST_6_100
}
scenario_CO2={
    "Quadratic ": Air_sea_flux_of_CO2_4,
    "Uniform ": Air_sea_flux_of_CO2_3,
    "Control (baseline)": Air_sea_flux_of_CO2_6
}
zone_labels = list(zones.keys())
basin_labels = list(basins.keys())
scenario_labels_mld = list(scenarios_mld.keys())
scenario_labels_ws = list(scenarios_ws.keys())
scenario_labels_sst = list(scenario_sst.keys())
scenario_labels_CO2 = list(scenario_CO2.keys())

#%%


# ===================
# 1) SCENARIO LABELS 
# ===================
scenario_labels_pretty = {
    "Quadratic": "Quadratic",
    "Uniform": "Uniform",
    "Control (baseline)": "Control"
}

# =========
# 2) ZONES 
# =========
zones = {
    "40°S–NB":   np.asarray(zone_mask_40_nb).astype(bool),
    "NB–SAF":    np.asarray(zone_mask_nb_saf).astype(bool),
    "SAF–PF":    np.asarray(zone_mask_saf_pf).astype(bool),
    "PF–SACCF":  np.asarray(zone_mask_pf_saccf).astype(bool),
    "SACCF–SB":  np.asarray(zone_mask_saccf_sb).astype(bool),
    "SB-COAST":  np.asarray(zone_mask_sb_ant).astype(bool),
}

# ==========
# 3) BASINS 
# ==========
basins = {
    "Atlantic": np.asarray(mask_atlantic).astype(bool),
    "Indian":   np.asarray(mask_indian).astype(bool),
    "Pacific":  np.asarray(mask_pacific).astype(bool),
}

# =====================
# 4) QUICK SHAPE CHECKS 
# ====================
def _assert_same_shape(name, arr, ref_shape):
    if np.asarray(arr).shape != ref_shape:
        raise ValueError(f"[Shape mismatch] {name} shape={np.asarray(arr).shape} != ref_shape={ref_shape}")

ref_shape = next(iter(zones.values())).shape

for k, v in zones.items():
    _assert_same_shape(f"zone '{k}'", v, ref_shape)

for k, v in basins.items():
    _assert_same_shape(f"basin '{k}'", v, ref_shape)

if 'tmask' in globals() and tmask is not None:
    _assert_same_shape("tmask", tmask, ref_shape)

# =============
# 5) SCENARIOS 
# =============
scenarios_mld = {
    "Quadratic": MLD_4,
    "Uniform": MLD_3,
    "Control (baseline)": MLD_6
}

scenarios_ws = {
    "Quadratic": wind_speed_4,
    "Uniform": wind_speed_3,
    "Control (baseline)": wind_speed_6
}

# SST_6_100 = np.nanmean(SST_6[:, :23, :, :], axis=1)
# SST_4_100 = np.nanmean(SST_4[:, :23, :, :], axis=1)
# SST_3_100 = np.nanmean(SST_3[:, :23, :, :], axis=1)
SST_6_100 = SST_6.isel(deptht=slice(0, 23)).mean(dim="deptht", skipna=True)
SST_4_100 = SST_4.isel(deptht=slice(0, 23)).mean(dim="deptht", skipna=True)
SST_3_100 = SST_3.isel(deptht=slice(0, 23)).mean(dim="deptht", skipna=True)
scenarios_sst = {
    "Quadratic": SST_4_100,
    "Uniform": SST_3_100,
    "Control (baseline)": SST_6_100
}

# ===================
# 6) ENSURE scenario 
# ===================
keys_mld = list(scenarios_mld.keys())
keys_ws  = list(scenarios_ws.keys())
keys_sst = list(scenarios_sst.keys())

if not (keys_mld == keys_ws == keys_sst):
    raise ValueError(
        "Scenario keys do not match!\n"
        f"MLD: {keys_mld}\nWS:  {keys_ws}\nSST: {keys_sst}\n"
        "Fix by using the same keys (no trailing spaces)."
    )

scenario_labels = keys_mld  

# ======================
# 7) OUTPUT LABEL LISTS 
# ======================
zone_labels     = list(zones.keys())
basin_labels    = list(basins.keys())
scenario_labels_mld = keys_mld
scenario_labels_ws  = keys_ws
scenario_labels_sst = keys_sst

print("OK: zones, basins, scenarios are aligned.")
print("Zones:", zone_labels)
print("Basins:", basin_labels)
print("Scenarios:", scenario_labels)
#%%
# ============================================
#   bar chart and plot WS Comparison (ANOMALY WS + basins)
# ============================================


from mpl_toolkits.basemap import Basemap
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.lines import Line2D

# ----------------------------
# Basin palettes 
# ----------------------------
atlantic_colors = ['#f3e5f5', '#e1bee7', '#ce93d8', '#ba68c8', '#ab47bc', '#9c27b0']
indian_colors   = ['#d9d9d9', '#c2b8aa', '#a89e91', '#8f857a', '#756c63', '#5c544d']
pacific_colors  = ['#fdae6b', '#fd8d3c', '#f16913', '#d94801', '#a63603', '#7f2704']
basin_colors = {"Atlantic": atlantic_colors, "Indian": indian_colors, "Pacific": pacific_colors}

# -----------
# Mean values 
# -----------
mean_wind_speed_6 = np.nanmean(wind_speed_6, axis=0)  # Control
mean_wind_speed_4 = np.nanmean(wind_speed_4, axis=0)  # Quadratic
mean_wind_speed_3 = np.nanmean(wind_speed_3, axis=0)  # Uniform

# ----------------------------
# Anomalies (south of 40S)
# ----------------------------
anom_wind_speed_3 = np.where(lat < -40, mean_wind_speed_3 - mean_wind_speed_6, np.nan)
anom_wind_speed_4 = np.where(lat < -40, mean_wind_speed_4 - mean_wind_speed_6, np.nan)

wind_levels = np.linspace(-1, 1, 21)

plt.rcParams.update({
    "figure.dpi": 150,
    "axes.titlesize": 12,
    "axes.labelsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12
})

# ----------------------------
#fig
# ----------------------------
fig = plt.figure(figsize=(16, 16))
gs = fig.add_gridspec(
    2, 2,
    height_ratios=[4, 2],
    hspace=0.15,
    wspace=0.38
)

ax3 = fig.add_subplot(gs[0, 0])
ax4 = fig.add_subplot(gs[0, 1])

gs_bottom = gs[1, :].subgridspec(1, 3, wspace=0.4)
ax_atl = fig.add_subplot(gs_bottom[0, 0])
ax_ind = fig.add_subplot(gs_bottom[0, 1])
ax_pac = fig.add_subplot(gs_bottom[0, 2])

# ======================================================
# Compute mean wind speed per basin/zone/scenario 
# ======================================================
TM = np.asarray(tmask) > 0
zones_bool  = {k: np.asarray(v).astype(bool) for k, v in zones.items()}
basins_bool = {k: np.asarray(v).astype(bool) for k, v in basins.items()}

zone_labels = list(zones_bool.keys())
basin_labels = list(basins_bool.keys())

scenario_order = list(scenario_labels_ws)

mean_ws = {basin: {sc: [] for sc in scenario_order} for basin in basin_labels}

for basin_name, bmask in basins_bool.items():
    for zone_name, zmask in zones_bool.items():

        region = zmask & bmask & TM
        w = e1t * e2t * region
        den = np.nansum(w)

        for sc_name in scenario_order:
            sc_data = scenarios_ws[sc_name]  # (time, y, x)

            ts = np.full(sc_data.shape[0], np.nan)
            if den > 0:
                for t in range(sc_data.shape[0]):
                    num = np.nansum(sc_data[t] * w)
                    ts[t] = num / den

            mean_ws[basin_name][sc_name].append(np.nanmean(ts))

# ======================================================
# Bar chart function 
# ======================================================
def plot_basin_bar(ax, basin_name, show_legend=False):
    x = np.arange(len(zone_labels))
    bar_width = 0.25

    hatches = ['///', '...', '']  # Quadratic, Uniform, Control

    colors_zone = basin_colors[basin_name]

    ax.set_ylim(0, 12)

    for j, sc_name in enumerate(scenario_order):
        heights = mean_ws[basin_name][sc_name]

        bar_colors = [colors_zone[i] for i in range(len(zone_labels))]

        ax.bar(
            x + j * bar_width - bar_width,
            heights,
            width=bar_width,
            label=scenario_labels_pretty.get(sc_name, sc_name) if show_legend else "",
            color=bar_colors,
            edgecolor='black',
            hatch=hatches[j],
            linewidth=0.6
        )

    ax.set_title(f"{basin_name} Ocean", fontsize=14, pad=4)
    ax.set_xticks(x)
    ax.set_xticklabels(zone_labels, rotation=35, ha='right', fontsize=12)
    ax.tick_params(axis='both', labelsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.4)

    if basin_name == "Atlantic":
        ax.set_ylabel("Wind Speed (m s$^{-1}$)", fontsize=12, fontweight='bold')
    if show_legend:
        ax.legend(loc='upper left', bbox_to_anchor=(-0.8, 1.1), frameon=True, fontsize=14)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

# Bars
plot_basin_bar(ax_atl, "Atlantic", show_legend=True)
plot_basin_bar(ax_ind, "Indian")
plot_basin_bar(ax_pac, "Pacific")


m1 = Basemap(projection='spstere', boundinglat=-40, lon_0=180, resolution='l', ax=ax3)
m1.drawcoastlines(linewidth=0.5)
m1.fillcontinents(color='#dddddd', lake_color='white')
m1.drawmapboundary(fill_color='white')
m1.drawparallels([-40, -60], linestyle='solid', linewidth=1, color='black')
m1.drawmeridians([30,60,90,120,150,180,210,240,270,300,330],
                 labels=[True, True, True, True], fontsize=12)

cax1 = m1.contourf(lon, lat, anom_wind_speed_4, latlon=True,
                   levels=wind_levels, cmap='RdBu_r', extend='both')


m2 = Basemap(projection='spstere', boundinglat=-40, lon_0=180, resolution='l', ax=ax4)
m2.drawcoastlines(linewidth=0.5)
m2.fillcontinents(color='lightgray', lake_color='white')
m2.drawmapboundary(fill_color='white')
m2.drawparallels([-40, -60], linestyle='solid', linewidth=1, color='black')
m2.drawmeridians(np.arange(30, 360, 30), labels=[True, True, True, True], fontsize=12)

cax2 = m2.contourf(lon, lat, anom_wind_speed_3, latlon=True,
                   levels=wind_levels, cmap='RdBu_r', extend='both')


cax = inset_axes(
    ax4,
    width="100%",
    height="3%",
    loc='lower center',
    bbox_to_anchor=(-0.75, -0.105, 1, 1),
    bbox_transform=ax4.transAxes,
    borderpad=0
)

cb1 = fig.colorbar(cax1, cax=cax, orientation="horizontal")
cb1.ax.tick_params(labelsize=12)
cb1.set_label("Wind Speed Anomaly (m s$^{-1}$)", fontsize=12, fontweight='bold')

for m, ax in [(m1, ax3), (m2, ax4)]:
    x40, y40 = m(0, -40)
    x60, y60 = m(0, -60)
    ax.text(x40, y40, '40°S', fontsize=12, ha='center', va='bottom', color='black')
    ax.text(x60, y60, '60°S', fontsize=12, ha='center', va='bottom', color='black')


for (ax, m) in [(ax3, m1), (ax4, m2)]:
    plot_front(ds["LonNB"],    ds["LatNB"],    "NB",    "black",       m, ax)
    plot_front(ds["LonSAF"],   ds["LatSAF"],   "SAF",   "darkmagenta", m, ax)
    plot_front(ds["LonPF"],    ds["LatPF"],    "PF",    "deeppink",    m, ax)
    plot_front(ds["LonSACCF"], ds["LatSACCF"], "SACCF", "blue",        m, ax)
    plot_front(ds["LonSB"],    ds["LatSB"],    "SB",    "red",         m, ax)

# Legend for fronts (kept)
fig.legend(
    handles=front_legend,
    loc='upper left',
    bbox_to_anchor=(-0.03, 0.81),
    ncol=1,
    frameon=True,
    fontsize=14
)


labels = ['a', 'b', 'c', 'd', 'e']
axes_list = [ax3, ax4, ax_atl, ax_ind, ax_pac]
for label, ax in zip(labels, axes_list):
    ax.text(0.01, 1.08, f"{label}:", transform=ax.transAxes,
            fontsize=16, fontweight='bold', va='bottom', ha='left')

plt.savefig("/esi/project/niwa02764/faezeh/plot/abc5_WindSpeed_anomaly_barchart_plot.png",
            dpi=400, bbox_inches='tight')
plt.show()





#%%
#   bar chart and plot MLD Comparison (ANOMALY MLD + basins)


from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.colors import ListedColormap

# ===  Colors ===
atlantic_colors = ['#f3e5f5', '#e1bee7', '#ce93d8', '#ba68c8', '#ab47bc', '#9c27b0']
indian_colors   = ['#d9d9d9', '#c2b8aa', '#a89e91', '#8f857a', '#756c63', '#5c544d']
pacific_colors  = ['#fdae6b', '#fd8d3c', '#f16913', '#d94801', '#a63603', '#7f2704']
basin_colors    = {"Atlantic": atlantic_colors, "Indian": indian_colors, "Pacific": pacific_colors}
cmap = ListedColormap(atlantic_colors + indian_colors + pacific_colors)

mean_MLD_6 = np.nanmean(MLD_6, axis=0)  # Control
mean_MLD_4 = np.nanmean(MLD_4, axis=0)  # Quadratic
mean_MLD_3 = np.nanmean(MLD_3, axis=0)  # Uniform

# Compute anomalies 
anom_MLD_3 = np.where(lat < -40, mean_MLD_3 - mean_MLD_6, np.nan)
anom_MLD_4 = np.where(lat < -40, mean_MLD_4 - mean_MLD_6, np.nan)

mld_levels = np.linspace(-50, 50, 21)

plt.rcParams.update({
    "figure.dpi": 150,
    "axes.titlesize": 12,
    "axes.labelsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12
})

fig = plt.figure(figsize=(16, 16))

gs = fig.add_gridspec(
    2, 2,
    height_ratios=[4, 2],
    hspace=0.15,
    wspace=0.38
)

ax3 = fig.add_subplot(gs[0, 0])
ax4 = fig.add_subplot(gs[0, 1])

gs_bottom = gs[1, :].subgridspec(1, 3, wspace=0.4)
ax_atl = fig.add_subplot(gs_bottom[0, 0])
ax_ind = fig.add_subplot(gs_bottom[0, 1])
ax_pac = fig.add_subplot(gs_bottom[0, 2])

# Mean MLD for zones & basins 
TM = np.asarray(tmask) > 0

zones_bool  = {k: np.asarray(v).astype(bool) for k, v in zones.items()}
basins_bool = {k: np.asarray(v).astype(bool) for k, v in basins.items()}

zone_labels = list(zones_bool.keys())
basin_labels = list(basins_bool.keys())

scenario_order = list(scenario_labels_mld)

mean_mld = {basin: {sc: [] for sc in scenario_order} for basin in basin_labels}

for basin_name, basin_mask in basins_bool.items():
    for zone_name, zone_mask in zones_bool.items():

        region = zone_mask & basin_mask & TM
        w = e1t * e2t * region
        den = np.nansum(w)

        for sc_name in scenario_order:
            sc_data = scenarios_mld[sc_name]  # (time, y, x)

            ts = np.full(sc_data.shape[0], np.nan)
            if den > 0:
                for t in range(sc_data.shape[0]):
                    num = np.nansum(sc_data[t] * w)
                    ts[t] = num / den

            mean_mld[basin_name][sc_name].append(np.nanmean(ts))


def plot_basin_bar(ax, basin_name, show_legend=False):
    x = np.arange(len(zone_labels))
    bar_width = 0.25

    hatches = ['///', '...', '']  # Quadratic, Uniform, Control (your original intent)

    colors_zone = basin_colors[basin_name]

    ax.set_ylim(0, 120)

    for j, sc_name in enumerate(scenario_order):
        heights = mean_mld[basin_name][sc_name]

        bar_colors = [colors_zone[i] for i in range(len(zone_labels))]

        ax.bar(
            x + j * bar_width - bar_width,
            heights,
            width=bar_width,
            label=scenario_labels_pretty.get(sc_name, sc_name) if show_legend else "",
            color=bar_colors,               # <<< FIXED
            edgecolor='black',
            hatch=hatches[j],
            linewidth=0.6
        )

    ax.set_title(f"{basin_name} Ocean", fontsize=14, pad=4)
    ax.set_xticks(x)
    ax.set_xticklabels(zone_labels, rotation=35, ha='right', fontsize=12)
    ax.tick_params(axis='both', labelsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.4)

    if basin_name == "Atlantic":
        ax.set_ylabel("Mixed Layer Depth (m)", fontsize=12, fontweight='bold')

    if show_legend:
        ax.legend(loc='upper left', bbox_to_anchor=(-0.8, 1.1), frameon=True, fontsize=14)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

# Bar plots for each basin
plot_basin_bar(ax_atl, "Atlantic", show_legend=True)
plot_basin_bar(ax_ind, "Indian")
plot_basin_bar(ax_pac, "Pacific")

# Maps

m1 = Basemap(projection='spstere', boundinglat=-40, lon_0=180, resolution='l', ax=ax3)
m1.drawcoastlines(linewidth=0.5)
m1.fillcontinents(color='lightgray', lake_color='white')
m1.drawmapboundary(fill_color='white')
m1.drawparallels([-40, -60], linestyle='solid', linewidth=1, color='black')
m1.drawmeridians([30,60,90,120,150,180,210,240,270,300,330],
                 labels=[True, True, True, True], fontsize=12)

cax1 = m1.contourf(lon, lat, anom_MLD_4, latlon=True,
                   levels=mld_levels, cmap='RdBu_r', extend='both')

m2 = Basemap(projection='spstere', boundinglat=-40, lon_0=180, resolution='l', ax=ax4)
m2.drawcoastlines(linewidth=0.5)
m2.fillcontinents(color='lightgray', lake_color='white')
m2.drawmapboundary(fill_color='white')
m2.drawparallels([-40, -60], linestyle='solid', linewidth=1, color='black')
m2.drawmeridians([30,60,90,120,150,180,210,240,270,300,330],
                 labels=[True, True, True, True], fontsize=12)

cax2 = m2.contourf(lon, lat, anom_MLD_3, latlon=True,
                   levels=mld_levels, cmap='RdBu_r', extend='both')

cax = inset_axes(
    ax4,
    width="100%",
    height="3%",
    loc='lower center',
    bbox_to_anchor=(-0.75, -0.105, 1, 1),
    bbox_transform=ax4.transAxes,
    borderpad=0
)
cb1 = fig.colorbar(cax1, cax=cax, orientation="horizontal")
cb1.ax.tick_params(labelsize=12)
cb1.set_label("Mixed Layer Depth Anomaly (m)", fontsize=12, fontweight='bold')

# Latitude Labels
for m, ax in [(m1, ax3), (m2, ax4)]:
    x40, y40 = m(0, -40)
    x60, y60 = m(0, -60)
    ax.text(x40, y40, '40°S', fontsize=12, ha='center', va='bottom', color='black')
    ax.text(x60, y60, '60°S', fontsize=12, ha='center', va='bottom', color='black')

# Fronts
for (ax, m) in [(ax3, m1), (ax4, m2)]:
    plot_front(ds["LonNB"],    ds["LatNB"],    "NB",    "black",       m, ax)
    plot_front(ds["LonSAF"],   ds["LatSAF"],   "SAF",   "darkmagenta", m, ax)
    plot_front(ds["LonPF"],    ds["LatPF"],    "PF",    "deeppink",    m, ax)
    plot_front(ds["LonSACCF"], ds["LatSACCF"], "SACCF", "blue",        m, ax)
    plot_front(ds["LonSB"],    ds["LatSB"],    "SB",    "red",         m, ax)

# Legend for fronts
fig.legend(
    handles=front_legend,
    loc='upper left',
    bbox_to_anchor=(-0.03, 0.81),
    ncol=1,
    frameon=True,
    fontsize=12
)

# Subplot Labels A–E
labels = ['a', 'b', 'c', 'd', 'e']
axes_list = [ax3, ax4, ax_atl, ax_ind, ax_pac]
for label, ax in zip(labels, axes_list):
    ax.text(0.01, 1.08, f"{label}:", transform=ax.transAxes,
            fontsize=16, fontweight='bold', va='bottom', ha='left')

# Save
plt.savefig("/esi/project/niwa02764/faezeh/plot/abc5_MLD_anomaly_barchart_plot.png",
            dpi=400, bbox_inches='tight')
plt.show()
#%%
#compare ws,mld

def make_anomaly_dict(mean_dict, control_name="Control (baseline)"):
    out = {}
    for basin in mean_dict:
        out[basin] = {}
        ctrl = np.array(mean_dict[basin][control_name])
        for sc in mean_dict[basin]:
            out[basin][sc] = (np.array(mean_dict[basin][sc]) - ctrl).tolist()
    return out

anom_ws  = make_anomaly_dict(mean_ws,  control_name="Control (baseline)")
anom_mld = make_anomaly_dict(mean_mld, control_name="Control (baseline)")

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["hatch.linewidth"] = 0.5

basin_list = ["Atlantic", "Indian", "Pacific"]
hatches = ['///', '\\\\\\', '']   # Quadratic, Uniform, Control

fig, axes = plt.subplots(2, 3, figsize=(18, 9), constrained_layout=True)

#  anomaly
def plot_anomaly(ax, basin, data_dict, ylabel=None, ylim=None, show_legend=False):

    x = np.arange(len(zone_labels))
    bw = 0.25
    colors_zone = basin_colors[basin]

    for j, sc in enumerate(scenario_order):

        heights = data_dict[basin][sc]
        bar_colors = [colors_zone[i] for i in range(len(zone_labels))]

        ax.bar(
            x + j*bw - bw,
            heights,
            width=bw,
            color=bar_colors,
            edgecolor="black",
            hatch=hatches[j],
            linewidth=0.6,
            label=sc if show_legend else None
        )

    ax.axhline(0, color="black", linewidth=1)  # zero line
    ax.set_title(f"{basin} Ocean")
    ax.set_xticks(x)
    ax.set_xticklabels(zone_labels, rotation=35, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if ylim is not None:
        ax.set_ylim(*ylim)

    if ylabel:
        ax.set_ylabel(ylabel)

    if show_legend:
        ax.legend(loc="upper left", frameon=True)


#  Wind Speed Anomaly
for col, basin in enumerate(basin_list):
    plot_anomaly(
        axes[0, col],
        basin,
        anom_ws,
        ylabel="Wind Speed Anomaly (m/s)" if col == 0 else None,
        ylim=(-1.0, 1.0),
        show_legend=(col == 0)
    )

# MLD Anomaly
for col, basin in enumerate(basin_list):
    plot_anomaly(
        axes[1, col],
        basin,
        anom_mld,
        ylabel="MLD Anomaly (m)" if col == 0 else None,
        ylim=(-50, 50)
    )

plt.savefig("/esi/project/niwa02764/faezeh/plot/WS_MLD_ANOMALY_bars.png",
            dpi=400, bbox_inches="tight")
plt.show()



#%%

#   ANOMALY SST + basins)

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# ===  Colors ===
atlantic_colors = ['#f3e5f5', '#e1bee7', '#ce93d8', '#ba68c8', '#ab47bc', '#9c27b0']
indian_colors   = ['#d9d9d9', '#c2b8aa', '#a89e91', '#8f857a', '#756c63', '#5c544d']
pacific_colors  = ['#fdae6b', '#fd8d3c', '#f16913', '#d94801', '#a63603', '#7f2704']
basin_colors    = [atlantic_colors, indian_colors, pacific_colors]
cmap = ListedColormap(atlantic_colors + indian_colors + pacific_colors)

basin_order = ["Atlantic", "Indian", "Pacific"]  # fixed order for indexing

# === Mean 
mean_SST_6 =SST_6_100[0,:,:] # Control
mean_SST_4 = SST_4_100[0,:,:]   # Quadratic
mean_SST_3 = SST_3_100[0,:,:]   # Uniform

anom_SST_3 = mean_SST_3 - mean_SST_6
anom_SST_4 = mean_SST_4 - mean_SST_6

values = np.linspace(-0.3, 0.3, 21)
SST_levels = np.round(values, 2)

plt.rcParams.update({
    "figure.dpi": 150,
    "axes.titlesize": 12,
    "axes.labelsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12
})

fig = plt.figure(figsize=(25, 25))
gs = fig.add_gridspec(2, 3, height_ratios=[4.5, 2], hspace=0.009, wspace=0.25)

ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[0, 2])

gs_bottom = gs[1, :].subgridspec(1, 3, wspace=0.3)
ax_atl = fig.add_subplot(gs_bottom[0, 0])
ax_ind = fig.add_subplot(gs_bottom[0, 1])
ax_pac = fig.add_subplot(gs_bottom[0, 2])


TM = np.asarray(tmask) > 0
zones_bool  = {k: np.asarray(v).astype(bool) for k, v in zones.items()}
basins_bool = {k: np.asarray(v).astype(bool) for k, v in basins.items()}

scenario_order = list(scenario_labels_sst)  #
zone_labels = list(zones_bool.keys())

mean_sst = {basin: {sc: [] for sc in scenario_order} for basin in basin_labels}

for basin_name, basin_mask in basins_bool.items():
    for zone_name, zone_mask in zones_bool.items():

        region = zone_mask & basin_mask & TM
        w = e1t * e2t * region
        den = np.nansum(w)

        for sc_name in scenario_order:
            sc_data = scenario_sst[sc_name]  # (time,y,x)

            ts = np.full(sc_data.shape[0], np.nan)
            if den > 0:
                for t in range(sc_data.shape[0]):
                    ts[t] = np.nansum(sc_data[t] * w) / den

            mean_sst[basin_name][sc_name].append(np.nanmean(ts))

# Bar chart function 
def plot_basin_bar(ax, basin_name, show_legend=False):
    x = np.arange(len(zone_labels))
    bar_width = 0.25
    hatches = ['///', '...', '']   # Quadratic, Uniform, Control (your style)

    zone_colors = basin_colors[basin_order.index(basin_name)]
    ax.set_ylim(-3, 12)

    for j, sc_name in enumerate(scenario_order):
        for i, value in enumerate(mean_sst[basin_name][sc_name]):
            ax.bar(
                x[i] + j * bar_width - bar_width,
                value,
                width=bar_width,
                color=zone_colors[i],
                edgecolor='black',
                hatch=hatches[j],
                linewidth=0.6,
                label=scenario_labels_pretty.get(sc_name, sc_name) if (i == 0 and show_legend) else None
            )

    ax.set_title(f"{basin_name} Ocean", fontsize=16, pad=4)
    ax.set_xticks(x)
    ax.set_xticklabels(zone_labels, rotation=45, ha='right', fontsize=16)
    ax.tick_params(axis='both', labelsize=16)
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    if basin_name == "Atlantic":
        ax.set_ylabel("Sea Surface Temperature (°C)", fontsize=14, fontweight='bold')

    if show_legend:
        ax.legend(loc='center left', bbox_to_anchor=(-0.5, 1.0), frameon=True, fontsize=16)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

# Plot bar charts
plot_basin_bar(ax_atl, "Atlantic", show_legend=True)
plot_basin_bar(ax_ind, "Indian")
plot_basin_bar(ax_pac, "Pacific")

# MAPS
m0 = Basemap(projection='spstere', boundinglat=-40, lon_0=180, resolution='l', ax=ax1)
m0.drawcoastlines(linewidth=0.5)
m0.fillcontinents(color='#dddddd', lake_color='white')
m0.drawmapboundary(fill_color='white')
m0.drawmeridians(np.arange(30, 360, 30), labels=[1, 1, 1, 1], fontsize=12)
m0.drawparallels([-40, -60], linestyle='solid', linewidth=1, color='black')

mean_SST_6_1 = np.where(lat < -40, mean_SST_6, np.nan)
cax0 = m0.contourf(lon, lat, mean_SST_6_1, latlon=True,
                   levels=np.linspace(-4, 4, 17), cmap='viridis', extend='both')

cax0_inset = inset_axes(
    ax1, width="100%", height="3%", loc='lower center',
    bbox_to_anchor=(0.0, -0.23, 1, 1), bbox_transform=ax1.transAxes, borderpad=0
)
cb0 = fig.colorbar(cax0, cax=cax0_inset, orientation="horizontal")
cb0.ax.tick_params(labelsize=12)
cb0.set_label("Control Sea Surface Temperature (°C)", fontsize=14, labelpad=4, fontweight='bold')

m1 = Basemap(projection='spstere', boundinglat=-40, lon_0=180, resolution='l', ax=ax2)
m1.drawcoastlines(linewidth=0.5)
m1.fillcontinents(color='#dddddd', lake_color='white')
m1.drawmapboundary(fill_color='white')
m1.drawmeridians(np.arange(30, 360, 30), labels=[1, 1, 1, 1], fontsize=12)
m1.drawparallels([-40, -60], linestyle='solid', linewidth=1, color='black')

anom_SST_4_1 = np.where(lat < -40, anom_SST_4, np.nan)
cax1 = m1.contourf(lon, lat, anom_SST_4_1, latlon=True,
                   levels=SST_levels, cmap='RdBu_r', extend='both', alpha=0.8)

cax_inset = inset_axes(
    ax2, width="100%", height="3%", loc='lower center',
    bbox_to_anchor=(0.6, -0.23, 1, 1), bbox_transform=ax2.transAxes, borderpad=0
)
cb1 = fig.colorbar(cax1, cax=cax_inset, orientation="horizontal")
cb1.set_label("Sea Surface Temperature Anomaly (°C)", fontsize=14, fontweight='bold')
cb1.ax.tick_params(labelsize=12)

m2 = Basemap(projection='spstere', boundinglat=-40, lon_0=180, resolution='l', ax=ax3)
m2.drawcoastlines(linewidth=0.5)
m2.fillcontinents(color='#dddddd', lake_color='gray')
m2.drawmapboundary(fill_color='white')
m2.drawparallels([-40, -60], linestyle='solid', linewidth=1, color='black')
m2.drawmeridians(np.arange(30, 360, 30), labels=[1, 1, 1, 1], fontsize=12)

anom_SST_3_1 = np.where(lat < -40, anom_SST_3, np.nan)
cax2 = m2.contourf(lon, lat, anom_SST_3_1, latlon=True,
                   levels=SST_levels, cmap='RdBu_r', extend='both', alpha=0.8)

#  labels
for m, ax in [(m0, ax1), (m1, ax2), (m2, ax3)]:
    x40, y40 = m(0, -40)
    x60, y60 = m(0, -60)
    ax.text(x40, y40, '40°S', fontsize=12, ha='center', va='bottom', color='black')
    ax.text(x60, y60, '60°S', fontsize=12, ha='center', va='top', color='black')

# Fronts
for ax, m in [(ax1, m0), (ax2, m1), (ax3, m2)]:
    plot_front(ds["LonNB"],    ds["LatNB"],    "NB",    "black",       m, ax)
    plot_front(ds["LonSAF"],   ds["LatSAF"],   "SAF",   "darkmagenta", m, ax)
    plot_front(ds["LonPF"],    ds["LatPF"],    "PF",    "deeppink",    m, ax)
    plot_front(ds["LonSACCF"], ds["LatSACCF"], "SACCF", "blue",        m, ax)
    plot_front(ds["LonSB"],    ds["LatSB"],    "SB",    "red",         m, ax)

fig.legend(
    handles=front_legend,
    loc='upper right',
    bbox_to_anchor=(0.1, 0.729),
    ncol=1,
    frameon=True,
    fontsize=16
)

#  labels
labels = ['a', 'b', 'c', 'd', 'e', 'f']
axes_list = [ax1, ax2, ax3, ax_atl, ax_ind, ax_pac]
for label, ax in zip(labels, axes_list):
    ax.text(0.01, 1.08, f"{label}:", transform=ax.transAxes,
            fontsize=16, fontweight='bold', va='bottom', ha='left')

plt.savefig("/esi/project/niwa02764/faezeh/plot/abc6_SST_plot_barchart.png",
            dpi=400, bbox_inches='tight')


plt.show()

#%%

# ANOMALY CO2 + basins)

from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.transforms import offset_copy
import matplotlib.ticker as mticker

# Mean 
mean_Air_sea_flux_of_CO2_6 = np.nanmean(Air_sea_flux_of_CO2_6, axis=0)  # Control
mean_Air_sea_flux_of_CO2_4 = np.nanmean(Air_sea_flux_of_CO2_4, axis=0)  # Quadratic
mean_Air_sea_flux_of_CO2_3 = np.nanmean(Air_sea_flux_of_CO2_3, axis=0)  # Uniform

anom_CO2_4 = mean_Air_sea_flux_of_CO2_4 - mean_Air_sea_flux_of_CO2_6
anom_CO2_3 = mean_Air_sea_flux_of_CO2_3 - mean_Air_sea_flux_of_CO2_6

#  south of 40S
mean_CO2_6_s = np.where(lat < -40, mean_Air_sea_flux_of_CO2_6, np.nan)
anom_CO2_4_s = np.where(lat < -40, anom_CO2_4, np.nan)
anom_CO2_3_s = np.where(lat < -40, anom_CO2_3, np.nan)

levels_ctr = np.linspace(-1e-07, 1e-07, 21)
levels_anm = np.linspace(-4.5e-08, 4.5e-08, 13)

plt.rcParams.update({
    "figure.dpi": 150,
    "axes.titlesize": 12,
    "axes.labelsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12
})

# fig
fig = plt.figure(figsize=(25, 25))
gs  = fig.add_gridspec(2, 3, height_ratios=[5, 3], hspace=0.009, wspace=0.25)

ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[0, 2])

gs_bottom = gs[1, :].subgridspec(1, 3, wspace=0.3)
ax_atl = fig.add_subplot(gs_bottom[0, 0])
ax_ind = fig.add_subplot(gs_bottom[0, 1])
ax_pac = fig.add_subplot(gs_bottom[0, 2])

# Compute mean CO2 
TM = np.asarray(tmask) > 0
zones_bool  = {k: np.asarray(v).astype(bool) for k, v in zones.items()}
basins_bool = {k: np.asarray(v).astype(bool) for k, v in basins.items()}

zone_labels = list(zones_bool.keys())
basin_labels = list(basins_bool.keys())

scenario_order = list(scenario_labels_CO2)   # e.g. ["Quadratic ", "Uniform ", "Control (baseline)"]

mean_CO2 = {basin: {sc: [] for sc in scenario_order} for basin in basin_labels}

for basin_name, bmask in basins_bool.items():
    for zone_name, zmask in zones_bool.items():

        region = zmask & bmask & TM
        w = e1t * e2t * region
        den = np.nansum(w)

        for sc_name in scenario_order:
            sc_data = scenario_CO2[sc_name]  # (time,y,x)

            ts = np.full(sc_data.shape[0], np.nan)
            if den > 0:
                for t in range(sc_data.shape[0]):
                    ts[t] = np.nansum(sc_data[t] * w) / den

            mean_CO2[basin_name][sc_name].append(np.nanmean(ts))

# Bar chart 
def plot_basin_bar(ax, basin_name, show_legend=False):
    x = np.arange(len(zone_labels))
    bar_width = 0.25

    hatches = ['///', '\\\\', '']  # Quadratic, Uniform, Control
    colors  = ['#d2fbd4', '#4477AA', '#e76a24']

    for j, sc_name in enumerate(scenario_order):
        ax.bar(
            x + j * bar_width - bar_width,
            mean_CO2[basin_name][sc_name],
            width=bar_width,
            label=scenario_labels_pretty.get(sc_name, sc_name) if show_legend else None,
            color=colors[j],
            edgecolor='black',
            hatch=hatches[j],
            linewidth=0.6
        )

    ax.set_title(f"{basin_name} Ocean", fontsize=16, pad=4)
    ax.set_xticks(x)
    ax.set_xticklabels(zone_labels, rotation=45, ha='right', fontsize=16)
    ax.tick_params(axis='both', labelsize=16)
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda val, pos: f"{val/1e-8:.1f}"))
    ax.text(-0.14, 1.04, r"$\times 10^{-8}$", transform=ax.transAxes,
            fontsize=14, ha='left', va='bottom')

    if basin_name == "Atlantic":
        ax.set_ylabel(r"Air-sea CO$_2$ flux (mol/m$^{2}$/s)", fontsize=14, fontweight='bold')



    if show_legend:
        ax.legend(loc='center left', bbox_to_anchor=(-0.5, 1.2), frameon=True, fontsize=16)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

# --- Bars
plot_basin_bar(ax_atl, "Atlantic", show_legend=True)
plot_basin_bar(ax_ind, "Indian")
plot_basin_bar(ax_pac, "Pacific")

# MAPS
mer_list = [30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]


m0 = Basemap(projection='spstere', boundinglat=-40, lon_0=180, resolution='l', ax=ax1)
m0.drawcoastlines(linewidth=0.5)
m0.fillcontinents(color='#dddddd', lake_color='white')
m0.drawmapboundary(fill_color='white')
m0.drawmeridians(mer_list, labels=[1, 1, 1, 1], fontsize=12)
m0.drawparallels([-40, -60], linestyle='solid', linewidth=1, color='black')

cax0 = m0.contourf(lon, lat, mean_CO2_6_s, latlon=True,
                   levels=levels_ctr, cmap='RdBu_r', extend='both')

cax0_inset = inset_axes(
    ax1, width="100%", height="3%", loc='lower center',
    bbox_to_anchor=(0.0, -0.23, 1, 1), bbox_transform=ax1.transAxes, borderpad=0
)
cb0 = fig.colorbar(cax0, cax=cax0_inset, orientation="horizontal")
cb0.ax.tick_params(labelsize=12)
cb0.set_label("Air-sea CO$_2$ flux (mol/m$^{2}$/s)", fontsize=14, labelpad=8, fontweight='bold')

# Uptake/Outgas labels 
cb0.ax.text(0.95, 3.0, 'Uptake', transform=cb0.ax.transAxes,
            fontsize=12, fontweight='bold', va='center', ha='left')
cb0.ax.text(-0.07, 3.0, 'Outgas', transform=cb0.ax.transAxes,
            fontsize=12, fontweight='bold', va='center', ha='left')

# Quadratic anomaly
m1 = Basemap(projection='spstere', boundinglat=-40, lon_0=180, resolution='l', ax=ax2)
m1.drawcoastlines(linewidth=0.5)
m1.fillcontinents(color='#dddddd', lake_color='white')
m1.drawmapboundary(fill_color='white')
m1.drawmeridians(mer_list, labels=[1, 1, 1, 1], fontsize=12)
m1.drawparallels([-40, -60], linestyle='solid', linewidth=1, color='black')

cax1 = m1.contourf(lon, lat, anom_CO2_4_s, latlon=True,
                   levels=levels_anm, cmap='RdBu_r', extend='both')

cax1_inset = inset_axes(
    ax2, width="100%", height="3%", loc='lower center',
    bbox_to_anchor=(0.0, -0.23, 1, 1), bbox_transform=ax2.transAxes, borderpad=0
)
cb1 = fig.colorbar(cax1, cax=cax1_inset, orientation="horizontal")
cb1.ax.tick_params(labelsize=12)
cb1.set_label("Air-sea CO$_2$ flux Anomaly (mol/m$^{2}$/s)", fontsize=14, fontweight='bold')

offset_text = cb1.ax.xaxis.get_offset_text()
offset_text.set_fontsize(12)
offset_text.set_transform(offset_copy(offset_text.get_transform(), fig=fig, x=10, y=-30, units='points'))

# Uniform anomaly
m2 = Basemap(projection='spstere', boundinglat=-40, lon_0=180, resolution='l', ax=ax3)
m2.drawcoastlines(linewidth=0.5)
m2.fillcontinents(color='#dddddd', lake_color='gray')
m2.drawmapboundary(fill_color='white')
m2.drawparallels([-40, -60], linestyle='solid', linewidth=1, color='black')
m2.drawmeridians(mer_list, labels=[1, 1, 1, 1], fontsize=12)

cax2 = m2.contourf(lon, lat, anom_CO2_3_s, latlon=True,
                   levels=levels_anm, cmap='RdBu_r', extend='both')


for m, ax in zip([m0, m1, m2], [ax1, ax2, ax3]):
    x40, y40 = m(0, -40)
    x60, y60 = m(0, -60)
    ax.text(x40, y40, '40°S', fontsize=14, ha='center', va='bottom', color='black')
    ax.text(x60, y60, '60°S', fontsize=14, ha='center', va='top', color='black')

for mapper, axis in zip([m0, m1, m2], [ax1, ax2, ax3]):
    plot_front(ds["LonNB"],    ds["LatNB"],    "NB",    "black",       mapper, axis)
    plot_front(ds["LonSAF"],   ds["LatSAF"],   "SAF",   "darkmagenta", mapper, axis)
    plot_front(ds["LonPF"],    ds["LatPF"],    "PF",    "deeppink",    mapper, axis)
    plot_front(ds["LonSACCF"], ds["LatSACCF"], "SACCF", "blue",        mapper, axis)
    plot_front(ds["LonSB"],    ds["LatSB"],    "SB",    "red",         mapper, axis)

# Legend 
fig.legend(
    handles=front_legend,
    loc='upper right',
    bbox_to_anchor=(0.1, 0.75),
    ncol=1,
    frameon=True,
    fontsize=16
)

# Subplot labels A–F
labels = ['A', 'B', 'C', 'D', 'E', 'F']
axes_list = [ax1, ax2, ax3, ax_atl, ax_ind, ax_pac]
for lab, ax in zip(labels, axes_list):
    ax.text(0.01, 1.08, f"{lab}:", transform=ax.transAxes,
            fontsize=16, fontweight='bold', va='bottom', ha='left')

plt.savefig("/esi/project/niwa02764/faezeh/plot/6_CO2_plot_barchart_1.png",
            dpi=400, bbox_inches='tight')

plt.show()



#%%
#mol_s
# ============================================
#ANOMALY CO2 + basins)
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.basemap import Basemap
import numpy as np
import matplotlib.pyplot as plt


mean_Air_sea_flux_of_CO2_6 = np.nanmean(Air_sea_flux_of_CO2_6, axis=0)  # Control
mean_Air_sea_flux_of_CO2_4 = np.nanmean(Air_sea_flux_of_CO2_4, axis=0)  # Quadratic
mean_Air_sea_flux_of_CO2_3 = np.nanmean(Air_sea_flux_of_CO2_3, axis=0)  # Uniform

anom_CO2_3 = mean_Air_sea_flux_of_CO2_3 - mean_Air_sea_flux_of_CO2_6
anom_CO2_4 = mean_Air_sea_flux_of_CO2_4 - mean_Air_sea_flux_of_CO2_6

levels_ctr = np.linspace(-1e-07, 1e-07, 21)
levels_anm = np.linspace(-4e-08, 4e-08, 13)

plt.rcParams.update({
    "figure.dpi": 150,
    "axes.titlesize": 12,
    "axes.labelsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12
})


fig = plt.figure(figsize=(25, 25))
gs  = fig.add_gridspec(2, 3, height_ratios=[5, 3], hspace=0.009, wspace=0.25)

ax1 = fig.add_subplot(gs[0,0])
ax2 = fig.add_subplot(gs[0,1])
ax3 = fig.add_subplot(gs[0,2])

gs_bottom = gs[1,:].subgridspec(1, 3, wspace=0.3)
ax_atl = fig.add_subplot(gs_bottom[0,0])
ax_ind = fig.add_subplot(gs_bottom[0,1])
ax_pac = fig.add_subplot(gs_bottom[0,2])


zones_bool  = {k: np.asarray(v).astype(bool) for k, v in zones.items()}
basins_bool = {k: np.asarray(v).astype(bool) for k, v in basins.items()}
tmask_bool  = (np.asarray(tmask) > 0)

scenario_order = list(scenario_labels_CO2)   # e.g. ["Quadratic ", "Uniform ", "Control (baseline)"]


mean_CO2 = {basin: {sc: [] for sc in scenario_order} for basin in basin_labels}

for basin_name, basin_mask in basins_bool.items():
    for zone_name, zone_mask in zones_bool.items():

        region_mask = zone_mask & basin_mask & tmask_bool
        cell_area   = e1t * e2t * region_mask  

        for sc_name in scenario_order:
            sc_data = scenario_CO2[sc_name]     # (time, y, x)

            series = np.full(sc_data.shape[0], np.nan)
            for t in range(sc_data.shape[0]):
                series[t] = np.nansum(sc_data[t] * cell_area)  # mol/s

            mean_CO2[basin_name][sc_name].append(np.nanmean(series))


atlantic_colors = ['#f3e5f5', '#e1bee7', '#ce93d8', '#ba68c8', '#ab47bc', '#9c27b0']
indian_colors   = ['#d9d9d9', '#c2b8aa', '#a89e91', '#8f857a', '#756c63', '#5c544d']
pacific_colors  = ['#fdae6b', '#fd8d3c', '#f16913', '#d94801', '#a63603', '#7f2704']
basin_colors    = [atlantic_colors, indian_colors, pacific_colors]


def plot_basin_bar(ax, basin_name, show_legend=False):
    x = np.arange(len(zone_labels))
    bar_width = 0.25
    hatches   = ['///', 'x', '']  # Quadratic, Uniform, Control 

    zone_colors = basin_colors[["Atlantic", "Indian", "Pacific"].index(basin_name)]

    for j, sc_name in enumerate(scenario_order):
        for i in range(len(zone_labels)):
            ax.bar(
                x[i] + j*bar_width - bar_width,
                mean_CO2[basin_name][sc_name][i],
                width=bar_width,
                color=zone_colors[i],        
                edgecolor='black',
                hatch=hatches[j],            
                linewidth=0.6,
                label=scenario_labels_pretty.get(sc_name, sc_name) if (show_legend and i == 0) else None
            )

    formatter = mticker.FuncFormatter(lambda val, pos: f"{val / 1e6:.2f}")
    ax.yaxis.set_major_formatter(formatter)
    ax.text(0.0, 1.02, r"$\times 10^{6}$", transform=ax.transAxes,
            fontsize=14, ha='left', va='bottom', clip_on=False)

    ax.set_title(f"{basin_name} Ocean", fontsize=16, pad=4)
    ax.set_xticks(x)
    ax.set_xticklabels(zone_labels, rotation=45, ha='right', fontsize=16)
    ax.tick_params(axis='both', labelsize=16)
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    if basin_name == "Atlantic":
        ax.set_ylabel(r"Air-sea CO$_2$ flux (mol/s)", fontsize=14, fontweight='bold')

    basin_values = []
    for sc in scenario_order:
        basin_values.extend(mean_CO2[basin_name][sc])
    ymax = np.nanmax(basin_values) * 1.2 if np.isfinite(np.nanmax(basin_values)) else 1.0
    ax.set_ylim(0, ymax)

    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.1 * 1e6))

    if show_legend:
        ax.legend(loc='center left', bbox_to_anchor=(-0.5, 1.2), frameon=True, fontsize=16)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

plot_basin_bar(ax_atl, "Atlantic", show_legend=True)
plot_basin_bar(ax_ind, "Indian")
plot_basin_bar(ax_pac, "Pacific")


mer_list = [30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]

m0 = Basemap(projection='spstere', boundinglat=-40, lon_0=180, resolution='l', ax=ax1)
m0.drawcoastlines(linewidth=0.5)
m0.fillcontinents(color='#dddddd', lake_color='white')
m0.drawmapboundary(fill_color='white')
m0.drawmeridians(mer_list, labels=[True, True, True, True], fontsize=12)
m0.drawparallels([-40, -60], linestyle='solid', linewidth=1, color='black')

mean_Air_sea_flux_of_CO2_6_1 = np.where(lat < -40, mean_Air_sea_flux_of_CO2_6, np.nan)
cax0 = m0.contourf(lon, lat, mean_Air_sea_flux_of_CO2_6_1, latlon=True,
                   levels=levels_ctr, cmap='RdBu_r', extend='both')

cax0_inset = inset_axes(ax1, width="100%", height="3%", loc='lower center',
                        bbox_to_anchor=(0.0, -0.23, 1, 1),
                        bbox_transform=ax1.transAxes, borderpad=0)
cb0 = fig.colorbar(cax0, cax=cax0_inset, orientation="horizontal")
cb0.ax.tick_params(labelsize=12)
cb0.set_label("Air-sea CO$_2$ flux (mol/m$^{2}$/s)", fontsize=14, fontweight='bold', labelpad=8)
cb0.ax.text(0.95, 3.0, 'Uptake', transform=cb0.ax.transAxes, fontsize=12, fontweight='bold', va='center', ha='left')
cb0.ax.text(-0.07, 3.0, 'Outgas', transform=cb0.ax.transAxes, fontsize=12, fontweight='bold', va='center', ha='left')

m1 = Basemap(projection='spstere', boundinglat=-40, lon_0=180, resolution='l', ax=ax2)
m1.drawcoastlines(linewidth=0.5)
m1.fillcontinents(color='#dddddd', lake_color='white')
m1.drawmapboundary(fill_color='white')
m1.drawmeridians(mer_list, labels=[True, True, True, True], fontsize=12)
m1.drawparallels([-40, -60], linestyle='solid', linewidth=1, color='black')

anom_CO2_4_1 = np.where(lat < -40, anom_CO2_4, np.nan)
cax1 = m1.contourf(lon, lat, anom_CO2_4_1, latlon=True,
                   levels=levels_anm, cmap='RdBu_r', extend='both')

cax = inset_axes(ax2, width="100%", height="3%", loc='lower center',
                 bbox_to_anchor=(0.6, -0.23, 1, 1),
                 bbox_transform=ax2.transAxes, borderpad=0)
cb1 = fig.colorbar(cax1, cax=cax, orientation="horizontal")
cb1.ax.tick_params(labelsize=12)
cb1.set_label("Air-sea CO$_2$ flux Anomaly (mol/m$^{2}$/s)", fontsize=14, fontweight='bold')

m2 = Basemap(projection='spstere', boundinglat=-40, lon_0=180, resolution='l', ax=ax3)
m2.drawcoastlines(linewidth=0.5)
m2.fillcontinents(color='#dddddd', lake_color='gray')
m2.drawmapboundary(fill_color='white')
m2.drawmeridians(mer_list, labels=[True, True, True, True], fontsize=12)
m2.drawparallels([-40, -60], linestyle='solid', linewidth=1, color='black')

anom_CO2_3_1 = np.where(lat < -40, anom_CO2_3, np.nan)
cax2 = m2.contourf(lon, lat, anom_CO2_3_1, latlon=True,
                   levels=levels_anm, cmap='RdBu_r', extend='both')

for m, ax in zip([m0, m1, m2], [ax1, ax2, ax3]):
    x40, y40 = m(0, -40)
    x60, y60 = m(0, -60)
    ax.text(x40, y40, '40°S', fontsize=14, ha='center', va='bottom', color='black')
    ax.text(x60, y60, '60°S', fontsize=14, ha='center', va='top', color='black')

# fronts
for mapper, axis in zip([m0, m1, m2], [ax1, ax2, ax3]):
    plot_front(ds["LonNB"],    ds["LatNB"],    "NB",    "black",       mapper, axis)
    plot_front(ds["LonSAF"],   ds["LatSAF"],   "SAF",   "darkmagenta", mapper, axis)
    plot_front(ds["LonPF"],    ds["LatPF"],    "PF",    "deeppink",    mapper, axis)
    plot_front(ds["LonSACCF"], ds["LatSACCF"], "SACCF", "blue",        mapper, axis)
    plot_front(ds["LonSB"],    ds["LatSB"],    "SB",    "red",         mapper, axis)
# Legend 
fig.legend(
    handles=front_legend,
    loc='upper right',
    bbox_to_anchor=(0.1, 0.75),
    ncol=1,
    frameon=True,
    fontsize=16
)

labels = ['A', 'B', 'C', 'D', 'E', 'F']
axes_list = [ax1, ax2, ax3, ax_atl, ax_ind, ax_pac]
for lab, ax in zip(labels, axes_list):
    ax.text(0.01, 1.08, f"{lab}:", transform=ax.transAxes,
            fontsize=16, fontweight='bold', va='bottom', ha='left')
plt.savefig("/esi/project/niwa02764/faezeh/plot/6_CO2_plot_barchart_mol_s_FIXED.png",
            dpi=400, bbox_inches='tight')
plt.subplots_adjust(hspace=0.4)
plt.show()


#%%



#fluxes

from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.basemap import Basemap
import numpy as np
import matplotlib.pyplot as plt



mean_Air_sea_flux_of_CO2_6 = np.nanmean(Air_sea_flux_of_CO2_6, axis=0)  # Control
mean_Air_sea_flux_of_CO2_4 = np.nanmean(Air_sea_flux_of_CO2_4, axis=0)  # Quadratic
mean_Air_sea_flux_of_CO2_3 = np.nanmean(Air_sea_flux_of_CO2_3, axis=0)  # Uniform

anom_CO2_3 = mean_Air_sea_flux_of_CO2_3 - mean_Air_sea_flux_of_CO2_6
anom_CO2_4 = mean_Air_sea_flux_of_CO2_4 - mean_Air_sea_flux_of_CO2_6

mean_Air_sea_flux_of_C14_6 = np.nanmean(Air_sea_flux_of_C14_6, axis=0)  # Control
mean_Air_sea_flux_of_C14_4 = np.nanmean(Air_sea_flux_of_C14_4, axis=0)  # Quadratic
mean_Air_sea_flux_of_C14_3 = np.nanmean(Air_sea_flux_of_C14_3, axis=0)  # Uniform

anom_C14_3 = mean_Air_sea_flux_of_C14_3 - mean_Air_sea_flux_of_C14_6
anom_C14_4 = mean_Air_sea_flux_of_C14_4 - mean_Air_sea_flux_of_C14_6


levels_ctr = np.linspace(-1e-07, 1e-07, 21)
levels_anm = np.linspace(-4e-08, 4e-08, 13)

def _sym_levels(field2d, n=21, q=98):
    m = np.nanpercentile(np.abs(field2d), q)
    if not np.isfinite(m) or m == 0:
        m = 1.0
    return np.linspace(-m, m, n)

levels_ctr_C14 =np.linspace(-1e-07, 1e-07, 21) 
levels_anm_C14 = np.linspace(-4e-08, 4e-08, 13)

plt.rcParams.update({
    "figure.dpi": 150,
    "axes.titlesize": 12,
    "axes.labelsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12
})


fig = plt.figure(figsize=(25, 32))
gs  = fig.add_gridspec(3, 3, height_ratios=[4.5, 4.5, 3], hspace=0.05, wspace=0.25)

#  CO2 maps
ax1 = fig.add_subplot(gs[0,0])
ax2 = fig.add_subplot(gs[0,1])
ax3 = fig.add_subplot(gs[0,2])

#  C14 maps
ax4 = fig.add_subplot(gs[1,0])
ax5 = fig.add_subplot(gs[1,1])
ax6 = fig.add_subplot(gs[1,2])

gs_bottom = gs[2,:].subgridspec(1, 3, wspace=0.3)
ax_atl = fig.add_subplot(gs_bottom[0,0])
ax_ind = fig.add_subplot(gs_bottom[0,1])
ax_pac = fig.add_subplot(gs_bottom[0,2])


zones_bool  = {k: np.asarray(v).astype(bool) for k, v in zones.items()}
basins_bool = {k: np.asarray(v).astype(bool) for k, v in basins.items()}
tmask_bool  = (np.asarray(tmask) > 0)

scenario_order = list(scenario_labels_CO2)   # e.g. ["Quadratic ", "Uniform ", "Control (baseline)"]

# Compute total CO2 flux per basin/zone/scenario (mol/s)
mean_CO2 = {basin: {sc: [] for sc in scenario_order} for basin in basin_labels}

for basin_name, basin_mask in basins_bool.items():
    for zone_name, zone_mask in zones_bool.items():

        region_mask = zone_mask & basin_mask & tmask_bool
        cell_area   = e1t * e2t * region_mask  # [m^2] masked

        for sc_name in scenario_order:
            sc_data = scenario_CO2[sc_name]     # (time, y, x)

            series = np.full(sc_data.shape[0], np.nan)
            for t in range(sc_data.shape[0]):
                series[t] = np.nansum(sc_data[t] * cell_area)  # mol/s

            mean_CO2[basin_name][sc_name].append(np.nanmean(series))

# --------
# Colors 
# --------
atlantic_colors = ['#f3e5f5', '#e1bee7', '#ce93d8', '#ba68c8', '#ab47bc', '#9c27b0']
indian_colors   = ['#d9d9d9', '#c2b8aa', '#a89e91', '#8f857a', '#756c63', '#5c544d']
pacific_colors  = ['#fdae6b', '#fd8d3c', '#f16913', '#d94801', '#a63603', '#7f2704']
basin_colors    = [atlantic_colors, indian_colors, pacific_colors]

# Bar chart function

def plot_basin_bar(ax, basin_name, show_legend=False):
    x = np.arange(len(zone_labels))
    bar_width = 0.25
    hatches   = ['///', 'x', '']  # Quadratic, Uniform, Control 

    zone_colors = basin_colors[["Atlantic", "Indian", "Pacific"].index(basin_name)]

    for j, sc_name in enumerate(scenario_order):
        for i in range(len(zone_labels)):
            ax.bar(
                x[i] + j*bar_width - bar_width,
                mean_CO2[basin_name][sc_name][i],
                width=bar_width,
                color=zone_colors[i],
                edgecolor='black',
                hatch=hatches[j],
                linewidth=0.6,
                label=scenario_labels_pretty.get(sc_name, sc_name) if (show_legend and i == 0) else None
            )

    formatter = mticker.FuncFormatter(lambda val, pos: f"{val / 1e6:.2f}")
    ax.yaxis.set_major_formatter(formatter)
    ax.text(0.0, 1.02, r"$\times 10^{6}$", transform=ax.transAxes,
            fontsize=14, ha='left', va='bottom', clip_on=False)

    ax.set_title(f"{basin_name} Ocean", fontsize=16, pad=4)
    ax.set_xticks(x)
    ax.set_xticklabels(zone_labels, rotation=45, ha='right', fontsize=16)
    ax.tick_params(axis='both', labelsize=16)
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    if basin_name == "Atlantic":
        ax.set_ylabel(r"Air-sea CO$_2$ flux (mol/s)", fontsize=14, fontweight='bold')

    basin_values = []
    for sc in scenario_order:
        basin_values.extend(mean_CO2[basin_name][sc])
    ymax = np.nanmax(basin_values) * 1.2 if np.isfinite(np.nanmax(basin_values)) else 1.0
    ax.set_ylim(0, ymax)

    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.1 * 1e6))

    if show_legend:
        ax.legend(loc='center left', bbox_to_anchor=(-0.5, 1.2), frameon=True, fontsize=16)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

# === Plot basin bars ===
plot_basin_bar(ax_atl, "Atlantic", show_legend=True)
plot_basin_bar(ax_ind, "Indian")
plot_basin_bar(ax_pac, "Pacific")

# CO2 Control + anomalies
mer_list = [30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]

m0 = Basemap(projection='spstere', boundinglat=-40, lon_0=180, resolution='l', ax=ax1)
m0.drawcoastlines(linewidth=0.5)
m0.fillcontinents(color='#dddddd', lake_color='white')
m0.drawmapboundary(fill_color='white')
m0.drawmeridians(mer_list, labels=[True, True, True, True], fontsize=12)
m0.drawparallels([-40, -60], linestyle='solid', linewidth=1, color='black')

mean_Air_sea_flux_of_CO2_6_1 = np.where(lat < -40, mean_Air_sea_flux_of_CO2_6, np.nan)
cax0 = m0.contourf(lon, lat, mean_Air_sea_flux_of_CO2_6_1, latlon=True,
                   levels=levels_ctr, cmap='RdBu_r', extend='both')

cax0_inset = inset_axes(ax1, width="100%", height="3%", loc='lower center',
                        bbox_to_anchor=(0.6, -0.23, 1, 1),
                        bbox_transform=ax1.transAxes, borderpad=0)
cb0 = fig.colorbar(cax0, cax=cax0_inset, orientation="horizontal")
cb0.ax.tick_params(labelsize=12)
cb0.set_label("Air-sea CO$_2$ flux (mol/m$^{2}$/s)", fontsize=14, fontweight='bold', labelpad=8)
cb0.ax.text(0.95, 3.0, 'Uptake', transform=cb0.ax.transAxes, fontsize=12, fontweight='bold', va='center', ha='left')
cb0.ax.text(-0.07, 3.0, 'Outgas', transform=cb0.ax.transAxes, fontsize=12, fontweight='bold', va='center', ha='left')

m1 = Basemap(projection='spstere', boundinglat=-40, lon_0=180, resolution='l', ax=ax2)
m1.drawcoastlines(linewidth=0.5)
m1.fillcontinents(color='#dddddd', lake_color='white')
m1.drawmapboundary(fill_color='white')
m1.drawmeridians(mer_list, labels=[True, True, True, True], fontsize=12)
m1.drawparallels([-40, -60], linestyle='solid', linewidth=1, color='black')

anom_CO2_4_1 = np.where(lat < -40, anom_CO2_4, np.nan)
cax1 = m1.contourf(lon, lat, anom_CO2_4_1, latlon=True,
                   levels=levels_anm, cmap='RdBu_r', extend='both')

cax1_inset = inset_axes(ax2, width="100%", height="3%", loc='lower center',
                        bbox_to_anchor=(0.6, -0.23, 1, 1),
                        bbox_transform=ax2.transAxes, borderpad=0)
cb1 = fig.colorbar(cax1, cax=cax1_inset, orientation="horizontal")
cb1.ax.tick_params(labelsize=12)
cb1.set_label("Air-sea CO$_2$ flux Anomaly (mol/m$^{2}$/s)", fontsize=14, fontweight='bold')

m2 = Basemap(projection='spstere', boundinglat=-40, lon_0=180, resolution='l', ax=ax3)
m2.drawcoastlines(linewidth=0.5)
m2.fillcontinents(color='#dddddd', lake_color='gray')
m2.drawmapboundary(fill_color='white')
m2.drawmeridians(mer_list, labels=[True, True, True, True], fontsize=12)
m2.drawparallels([-40, -60], linestyle='solid', linewidth=1, color='black')

anom_CO2_3_1 = np.where(lat < -40, anom_CO2_3, np.nan)
cax2 = m2.contourf(lon, lat, anom_CO2_3_1, latlon=True,
                   levels=levels_anm, cmap='RdBu_r', extend='both')

# C14 Control + anomalies 
m3 = Basemap(projection='spstere', boundinglat=-40, lon_0=180, resolution='l', ax=ax4)
m3.drawcoastlines(linewidth=0.5)
m3.fillcontinents(color='#dddddd', lake_color='white')
m3.drawmapboundary(fill_color='white')
m3.drawmeridians(mer_list, labels=[True, True, True, True], fontsize=12)
m3.drawparallels([-40, -60], linestyle='solid', linewidth=1, color='black')

mean_Air_sea_flux_of_C14_6_1 = np.where(lat < -40, mean_Air_sea_flux_of_C14_6, np.nan)
cax3 = m3.contourf(lon, lat, mean_Air_sea_flux_of_C14_6_1, latlon=True,
                   levels=levels_ctr_C14, cmap='RdBu_r', extend='both')

cax3_inset = inset_axes(ax4, width="100%", height="3%", loc='lower center',
                        bbox_to_anchor=(0.6, -0.23, 1, 1),
                        bbox_transform=ax4.transAxes, borderpad=0)
cb3 = fig.colorbar(cax3, cax=cax3_inset, orientation="horizontal")
cb3.ax.tick_params(labelsize=12)
cb3.set_label("Air-sea C$^{14}$ flux (mol/m$^{2}$/s)", fontsize=14, fontweight='bold', labelpad=8)
cb3.ax.text(0.95, 3.0, 'Uptake', transform=cb3.ax.transAxes, fontsize=12, fontweight='bold', va='center', ha='left')
cb3.ax.text(-0.07, 3.0, 'Outgas', transform=cb3.ax.transAxes, fontsize=12, fontweight='bold', va='center', ha='left')

m4 = Basemap(projection='spstere', boundinglat=-40, lon_0=180, resolution='l', ax=ax5)
m4.drawcoastlines(linewidth=0.5)
m4.fillcontinents(color='#dddddd', lake_color='white')
m4.drawmapboundary(fill_color='white')
m4.drawmeridians(mer_list, labels=[True, True, True, True], fontsize=12)
m4.drawparallels([-40, -60], linestyle='solid', linewidth=1, color='black')

anom_C14_4_1 = np.where(lat < -40, anom_C14_4, np.nan)
cax4 = m4.contourf(lon, lat, anom_C14_4_1, latlon=True,
                   levels=levels_anm_C14, cmap='RdBu_r', extend='both')

cax4_inset = inset_axes(ax5, width="100%", height="3%", loc='lower center',
                        bbox_to_anchor=(0.6, -0.23, 1, 1),
                        bbox_transform=ax5.transAxes, borderpad=0)
cb4 = fig.colorbar(cax4, cax=cax4_inset, orientation="horizontal")
cb4.ax.tick_params(labelsize=12)
cb4.set_label("Air-sea C$^{14}$ flux Anomaly (mol/m$^{2}$/s)", fontsize=14, fontweight='bold')

m5 = Basemap(projection='spstere', boundinglat=-40, lon_0=180, resolution='l', ax=ax6)
m5.drawcoastlines(linewidth=0.5)
m5.fillcontinents(color='#dddddd', lake_color='gray')
m5.drawmapboundary(fill_color='white')
m5.drawmeridians(mer_list, labels=[True, True, True, True], fontsize=12)
m5.drawparallels([-40, -60], linestyle='solid', linewidth=1, color='black')

anom_C14_3_1 = np.where(lat < -40, anom_C14_3, np.nan)
cax5 = m5.contourf(lon, lat, anom_C14_3_1, latlon=True,
                   levels=levels_anm_C14, cmap='RdBu_r', extend='both')


# Latitude labels on ALL maps
for m, ax in zip([m0, m1, m2, m3, m4, m5], [ax1, ax2, ax3, ax4, ax5, ax6]):
    x40, y40 = m(0, -40)
    x60, y60 = m(0, -60)
    ax.text(x40, y40, '40°S', fontsize=14, ha='center', va='bottom', color='black')
    ax.text(x60, y60, '60°S', fontsize=14, ha='center', va='top', color='black')

# Fronts on ALL maps
for mapper, axis in zip([m0, m1, m2, m3, m4, m5], [ax1, ax2, ax3, ax4, ax5, ax6]):
    plot_front(ds["LonNB"],    ds["LatNB"],    "NB",    "black",       mapper, axis)
    plot_front(ds["LonSAF"],   ds["LatSAF"],   "SAF",   "darkmagenta", mapper, axis)
    plot_front(ds["LonPF"],    ds["LatPF"],    "PF",    "deeppink",    mapper, axis)
    plot_front(ds["LonSACCF"], ds["LatSACCF"], "SACCF", "blue",        mapper, axis)
    plot_front(ds["LonSB"],    ds["LatSB"],    "SB",    "red",         mapper, axis)

# Legend for fronts
fig.legend(
    handles=front_legend,
    loc='upper right',
    bbox_to_anchor=(0.1, 0.86),
    ncol=1,
    frameon=True,
    fontsize=16
)


labels = ['A','B','C','D','E','F','G','H','I']
axes_list = [ax1, ax2, ax3, ax4, ax5, ax6, ax_atl, ax_ind, ax_pac]
for lab, ax in zip(labels, axes_list):
    ax.text(0.01, 1.08, f"{lab}:", transform=ax.transAxes,
            fontsize=16, fontweight='bold', va='bottom', ha='left')


plt.savefig("/esi/project/niwa02764/faezeh/plot/7_CO2_C14_plot_barchart_mol_s.png",
            dpi=400, bbox_inches='tight')

plt.show()
#%%
#just C14 and CO2 flux



from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.basemap import Basemap
import numpy as np
import matplotlib.pyplot as plt


mean_Air_sea_flux_of_CO2_6 = np.nanmean(Air_sea_flux_of_CO2_6, axis=0)  # Control
mean_Air_sea_flux_of_CO2_4 = np.nanmean(Air_sea_flux_of_CO2_4, axis=0)  # Quadratic
mean_Air_sea_flux_of_CO2_3 = np.nanmean(Air_sea_flux_of_CO2_3, axis=0)  # Uniform

anom_CO2_3 = mean_Air_sea_flux_of_CO2_3 - mean_Air_sea_flux_of_CO2_6
anom_CO2_4 = mean_Air_sea_flux_of_CO2_4 - mean_Air_sea_flux_of_CO2_6

# C14 mean + anomalies
mean_Air_sea_flux_of_C14_6 = np.nanmean(Air_sea_flux_of_C14_6, axis=0)  # Control
mean_Air_sea_flux_of_C14_4 = np.nanmean(Air_sea_flux_of_C14_4, axis=0)  # Quadratic
mean_Air_sea_flux_of_C14_3 = np.nanmean(Air_sea_flux_of_C14_3, axis=0)  # Uniform

anom_C14_3 = mean_Air_sea_flux_of_C14_3 - mean_Air_sea_flux_of_C14_6
anom_C14_4 = mean_Air_sea_flux_of_C14_4 - mean_Air_sea_flux_of_C14_6


levels_ctr = np.linspace(-1e-07, 1e-07, 21)
levels_anm = np.linspace(-4e-08, 4e-08, 13)

levels_ctr_C14 = np.linspace(-5e-07, 5e-07, 21)
levels_anm_C14 = np.linspace(-10e-08, 10e-08, 13)

plt.rcParams.update({
    "figure.dpi": 150,
    "axes.titlesize": 12,
    "axes.labelsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12
})


fig = plt.figure(figsize=(25, 25))
gs  = fig.add_gridspec(2, 3, height_ratios=[4.5, 4.5], hspace=-0.25, wspace=0.25)

#  CO2 maps
ax1 = fig.add_subplot(gs[0,0])
ax2 = fig.add_subplot(gs[0,1])
ax3 = fig.add_subplot(gs[0,2])

#  C14 maps
ax4 = fig.add_subplot(gs[1,0])
ax5 = fig.add_subplot(gs[1,1])
ax6 = fig.add_subplot(gs[1,2])

#  CO2 Control + anomalies
mer_list = [30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]

m0 = Basemap(projection='spstere', boundinglat=-40, lon_0=180, resolution='l', ax=ax1)
m0.drawcoastlines(linewidth=0.5)
m0.fillcontinents(color='#dddddd', lake_color='white')
m0.drawmapboundary(fill_color='white')
m0.drawmeridians(mer_list, labels=[True, True, True, True], fontsize=12)
m0.drawparallels([-40, -60], linestyle='solid', linewidth=1, color='black')

mean_Air_sea_flux_of_CO2_6_1 = np.where(lat < -40, mean_Air_sea_flux_of_CO2_6, np.nan)
cax0 = m0.contourf(lon, lat, mean_Air_sea_flux_of_CO2_6_1, latlon=True,
                   levels=levels_ctr, cmap='viridis', extend='both')

    
    
cax0_inset = inset_axes(ax1, width="100%", height="3%", loc='lower center',
                        bbox_to_anchor=(0.03, -0.23, 1, 1),
                        bbox_transform=ax1.transAxes, borderpad=0)






cb0 = fig.colorbar(cax0, cax=cax0_inset, orientation="horizontal")
cb0.ax.tick_params(labelsize=12)
cb0.set_label(
    "Control Air-sea CO$_2$ flux (mol m$^{-2}$ s$^{-1}$)",
    fontsize=14,
    fontweight='bold',
    labelpad=8
)




cb0.ax.text(0.95, 3.0, 'Uptake', transform=cb0.ax.transAxes, fontsize=12, fontweight='bold', va='center', ha='left')
cb0.ax.text(-0.07, 3.0, 'Outgas', transform=cb0.ax.transAxes, fontsize=12, fontweight='bold', va='center', ha='left')



m1 = Basemap(projection='spstere', boundinglat=-40, lon_0=180, resolution='l', ax=ax2)
m1.drawcoastlines(linewidth=0.5)
m1.fillcontinents(color='#dddddd', lake_color='white')
m1.drawmapboundary(fill_color='white')
m1.drawmeridians(mer_list, labels=[True, True, True, True], fontsize=12)
m1.drawparallels([-40, -60], linestyle='solid', linewidth=1, color='black')

anom_CO2_4_1 = np.where(lat < -40, anom_CO2_4, np.nan)
cax1 = m1.contourf(lon, lat, anom_CO2_4_1, latlon=True,
                   levels=levels_anm, cmap='RdBu_r', extend='both')


cax1_inset = inset_axes(ax2, width="100%", height="3%", loc='lower center',
                        bbox_to_anchor=(0.6, -0.23, 1, 1),
                        bbox_transform=ax2.transAxes, borderpad=0)


cb1 = fig.colorbar(cax1, cax=cax1_inset, orientation="horizontal")
cb1.ax.tick_params(labelsize=12)
cb1.set_label(
    "Air-sea CO$_2$ flux Anomaly (mol m$^{-2}$ s$^{-1}$)",
    fontsize=14,
    fontweight='bold'
)




m2 = Basemap(projection='spstere', boundinglat=-40, lon_0=180, resolution='l', ax=ax3)
m2.drawcoastlines(linewidth=0.5)
m2.fillcontinents(color='#dddddd', lake_color='gray')
m2.drawmapboundary(fill_color='white')
m2.drawmeridians(mer_list, labels=[True, True, True, True], fontsize=12)
m2.drawparallels([-40, -60], linestyle='solid', linewidth=1, color='black')

anom_CO2_3_1 = np.where(lat < -40, anom_CO2_3, np.nan)
cax2 = m2.contourf(lon, lat, anom_CO2_3_1, latlon=True,
                   levels=levels_anm, cmap='RdBu_r', extend='both')

#  C14 Control + anomalies 
m3 = Basemap(projection='spstere', boundinglat=-40, lon_0=180, resolution='l', ax=ax4)
m3.drawcoastlines(linewidth=0.5)
m3.fillcontinents(color='#dddddd', lake_color='white')
m3.drawmapboundary(fill_color='white')
m3.drawmeridians(mer_list, labels=[True, True, True, True], fontsize=12)
m3.drawparallels([-40, -60], linestyle='solid', linewidth=1, color='black')

mean_Air_sea_flux_of_C14_6_1 = np.where(lat < -40, mean_Air_sea_flux_of_C14_6, np.nan)
cax3 = m3.contourf(lon, lat, mean_Air_sea_flux_of_C14_6_1, latlon=True,
                   levels=levels_ctr_C14, cmap='viridis', extend='both')

cax3_inset = inset_axes(ax4, width="100%", height="3%", loc='lower center',
                        bbox_to_anchor=(0.03, -0.23, 1, 1),
                        bbox_transform=ax4.transAxes, borderpad=0)




cb3 = fig.colorbar(cax3, cax=cax3_inset, orientation="horizontal")
cb3.ax.tick_params(labelsize=12)
cb3.set_label(
    "Control Air-sea $^{14}$C flux (mol m$^{-2}$ s$^{-1}$)",
    fontsize=14,
    fontweight='bold',
    labelpad=8
)




cb3.ax.text(0.95, 3.0, 'Uptake', transform=cb3.ax.transAxes, fontsize=12, fontweight='bold', va='center', ha='left')
cb3.ax.text(-0.07, 3.0, 'Outgas', transform=cb3.ax.transAxes, fontsize=12, fontweight='bold', va='center', ha='left')







m4 = Basemap(projection='spstere', boundinglat=-40, lon_0=180, resolution='l', ax=ax5)
m4.drawcoastlines(linewidth=0.5)
m4.fillcontinents(color='#dddddd', lake_color='white')
m4.drawmapboundary(fill_color='white')
m4.drawmeridians(mer_list, labels=[True, True, True, True], fontsize=12)
m4.drawparallels([-40, -60], linestyle='solid', linewidth=1, color='black')

anom_C14_4_1 = np.where(lat < -40, anom_C14_4, np.nan)
cax4 = m4.contourf(lon, lat, anom_C14_4_1, latlon=True,
                   levels=levels_anm_C14, cmap='RdBu_r', extend='both')

cax4_inset = inset_axes(ax5, width="100%", height="3%", loc='lower center',
                        bbox_to_anchor=(0.6, -0.23, 1, 1),
                        bbox_transform=ax5.transAxes, borderpad=0)



cb4 = fig.colorbar(cax4, cax=cax4_inset, orientation="horizontal")
cb4.ax.tick_params(labelsize=12)

cb4.set_label(
    'Air-sea $^{14}$C flux Anomaly (mol m$^{-2}$ s$^{-1}$)',
    fontsize=14,
    fontweight='bold'
)

m5 = Basemap(projection='spstere', boundinglat=-40, lon_0=180, resolution='l', ax=ax6)
m5.drawcoastlines(linewidth=0.5)
m5.fillcontinents(color='#dddddd', lake_color='gray')
m5.drawmapboundary(fill_color='white')
m5.drawmeridians(mer_list, labels=[True, True, True, True], fontsize=12)
m5.drawparallels([-40, -60], linestyle='solid', linewidth=1, color='black')

anom_C14_3_1 = np.where(lat < -40, anom_C14_3, np.nan)
cax5 = m5.contourf(lon, lat, anom_C14_3_1, latlon=True,
                   levels=levels_anm_C14, cmap='RdBu_r', extend='both')



cb1.ax.text(0.95, 3.0, 'Uptake', transform=cb1.ax.transAxes, fontsize=12, fontweight='bold', va='center', ha='left')
cb1.ax.text(-0.07, 3.0, 'Outgas', transform=cb1.ax.transAxes, fontsize=12, fontweight='bold', va='center', ha='left')


cb4.ax.text(0.95, 3.0, 'Uptake', transform=cb4.ax.transAxes, fontsize=12, fontweight='bold', va='center', ha='left')
cb4.ax.text(-0.07, 3.0, 'Outgas', transform=cb4.ax.transAxes, fontsize=12, fontweight='bold', va='center', ha='left')




for m, ax in zip([m0, m1, m2, m3, m4, m5], [ax1, ax2, ax3, ax4, ax5, ax6]):
    x40, y40 = m(0, -40)
    x60, y60 = m(0, -60)
    ax.text(x40, y40, '40°S', fontsize=14, ha='center', va='bottom', color='black')
    ax.text(x60, y60, '60°S', fontsize=14, ha='center', va='top', color='black')


for mapper, axis in zip([m0, m1, m2, m3, m4, m5], [ax1, ax2, ax3, ax4, ax5, ax6]):
    plot_front(ds["LonNB"],    ds["LatNB"],    "NB",    "black",       mapper, axis)
    plot_front(ds["LonSAF"],   ds["LatSAF"],   "SAF",   "darkmagenta", mapper, axis)
    plot_front(ds["LonPF"],    ds["LatPF"],    "PF",    "deeppink",    mapper, axis)
    plot_front(ds["LonSACCF"], ds["LatSACCF"], "SACCF", "blue",        mapper, axis)
    plot_front(ds["LonSB"],    ds["LatSB"],    "SB",    "red",         mapper, axis)



fig.legend(
    handles=front_legend,
    loc='upper right',
    bbox_to_anchor=(0.1, 0.778),
    ncol=1,
    frameon=True,
    fontsize=16
)


labels = ['a','b','c','d','e','f']
axes_list = [ax1, ax2, ax3, ax4, ax5, ax6]
for lab, ax in zip(labels, axes_list):
    ax.text(0.01, 1.08, f"{lab}:", transform=ax.transAxes,
            fontsize=16, fontweight='bold', va='bottom', ha='left')


plt.savefig("/esi/project/niwa02764/faezeh/plot/abc7_CO2_C14_plot_maps_only.png",
            dpi=400, bbox_inches='tight')

plt.show()



#%%
#mean-flux
import numpy as np

mask = (lat < -40) & (tmask > 0)

def mean_flux_simple(flux_time_yx, mask):

    net_series = []
    up_series  = []
    out_series = []

    for t in range(flux_time_yx.shape[0]):

        f = flux_time_yx[t]
        f_masked = f.where(mask)

        net_series.append(f_masked.mean().values)

        # uptake is POSITIVE 
        up_series.append(
            f_masked.where(f_masked > 0).mean().values
        )

        # outgas is NEGATIVE 
        out_series.append(
            f_masked.where(f_masked < 0).mean().values
        )

    return (np.nanmean(net_series),
            np.nanmean(up_series),
            np.nanmean(out_series))

co2_net, co2_uptake, co2_outgas = mean_flux_simple(Air_sea_flux_of_CO2_3, mask)
c14_net, c14_uptake, c14_outgas = mean_flux_simple(Air_sea_flux_of_C14_3, mask)

print("=== CONTROL MEAN FLUX (mol/m^2/s), lat < -40 ===")
print(f"CO2  net    : {co2_net:.6e}")
print(f"CO2  uptake : {co2_uptake:.6e}")
print(f"CO2  outgas : {co2_outgas:.6e}")
print("")
print(f"C14  net    : {c14_net:.6e}")
print(f"C14  uptake : {c14_uptake:.6e}")
print(f"C14  outgas : {c14_outgas:.6e}")


#%%
# barchart ( mol m-2 s-1)

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "figure.dpi": 150,
    "axes.titlesize": 14,
    "axes.labelsize": 14,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14
})


zones_bool  = {k: np.asarray(v).astype(bool) for k, v in zones.items()}
basins_bool = {k: np.asarray(v).astype(bool) for k, v in basins.items()}
tmask_bool  = (np.asarray(tmask) > 0)

scenario_order = list(scenario_labels_CO2)

# Compute AREA-WEIGHTED mean CO2 flux (keeps unit mol m-2 s-1)
mean_CO2 = {basin: {sc: [] for sc in scenario_order} for basin in basin_labels}

for basin_name, basin_mask in basins_bool.items():
    for zone_name, zone_mask in zones_bool.items():

        region_mask = zone_mask & basin_mask & tmask_bool
        cell_area   = (e1t * e2t) * region_mask  # m^2 in region, 0 outside
        area_sum    = np.nansum(cell_area)

        for sc_name in scenario_order:
            sc_data = scenario_CO2[sc_name]  # expected unit: mol m-2 s-1

            series = np.full(sc_data.shape[0], np.nan)
            for t in range(sc_data.shape[0]):
                num = np.nansum(sc_data[t] * cell_area)
                series[t] = num / area_sum if area_sum > 0 else np.nan

            mean_CO2[basin_name][sc_name].append(np.nanmean(series))


atlantic_colors = ['#f3e5f5', '#e1bee7', '#ce93d8', '#ba68c8', '#ab47bc', '#9c27b0']
indian_colors   = ['#d9d9d9', '#c2b8aa', '#a89e91', '#8f857a', '#756c63', '#5c544d']
pacific_colors  = ['#fdae6b', '#fd8d3c', '#f16913', '#d94801', '#a63603', '#7f2704']
basin_colors    = [atlantic_colors, indian_colors, pacific_colors]


fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

def plot_basin_bar(ax, basin_name, show_legend=False):

    x = np.arange(len(zone_labels))
    bar_width = 0.25
    hatches   = ['///', '...', '']

    zone_colors = basin_colors[["Atlantic", "Indian", "Pacific"].index(basin_name)]

    for j, sc_name in enumerate(scenario_order):
        for i in range(len(zone_labels)):
            ax.bar(
                x[i] + j*bar_width - bar_width,
                mean_CO2[basin_name][sc_name][i],
                width=bar_width,
                color=zone_colors[i],
                edgecolor='black',
                hatch=hatches[j],
                linewidth=0.6,
                label=scenario_labels_pretty.get(sc_name, sc_name)
                if (show_legend and i == 0) else None
            )

    ax.set_title(f"{basin_name} Ocean")
    ax.set_xticks(x)
    ax.set_xticklabels(zone_labels, rotation=35, ha='right')
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    basin_values = []
    for sc in scenario_order:
        basin_values.extend(mean_CO2[basin_name][sc])

    ymax = np.nanmax(basin_values) * 1.2
    ax.set_ylim(0, ymax)

    if show_legend:
        ax.legend(loc='upper left', frameon=True)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

plot_basin_bar(axes[0], "Atlantic", show_legend=True)
plot_basin_bar(axes[1], "Indian")
plot_basin_bar(axes[2], "Pacific")

axes[0].set_ylabel(
    r"Air-sea CO$_2$ flux (mol m$^{-2}$ s$^{-1}$)",
    fontweight='bold'
)

plt.tight_layout()
plt.savefig("/esi/project/niwa02764/faezeh/plot/abcCO2_flux-mol-m2-s.png",
            dpi=400, bbox_inches='tight')
plt.show()
#%%
#table

import pandas as pd

mean_Air_sea_flux_of_CO2_6 = np.mean(Air_sea_flux_of_CO2_6, axis=0)  # Control
mean_Air_sea_flux_of_CO2_4 = np.mean(Air_sea_flux_of_CO2_4, axis=0)  # Quadratic
mean_Air_sea_flux_of_CO2_3 = np.mean(Air_sea_flux_of_CO2_3, axis=0)  # Uniform

anom_CO2_3 = mean_Air_sea_flux_of_CO2_3 - mean_Air_sea_flux_of_CO2_6
anom_CO2_4 = mean_Air_sea_flux_of_CO2_4 - mean_Air_sea_flux_of_CO2_6


mean_CO2 = {basin: {sc: [] for sc in scenario_labels_CO2} for basin in basin_labels}

for basin_name, basin_mask in basins.items():
    for zone_name, zone_mask in zones.items():
        region_mask = zone_mask * basin_mask * tmask
        cell_area = e1t * e2t * region_mask  # [m²]
        for sc_name, sc_data in scenario_CO2.items():
            series = []
            for t in range(sc_data.shape[0]):
                total_flux = np.nansum(sc_data[t] * cell_area)  # mol/s
                series.append(total_flux)
            mean_CO2[basin_name][sc_name].append(np.nanmean(series))


print("\n===============================")
print("  Total Air–Sea CO₂ Flux by Basin (mol/s)")
print("===============================")

summary_rows = []
for basin_name in basin_labels:
    for sc_name in scenario_labels_CO2:
        total_flux = np.nansum(mean_CO2[basin_name][sc_name])
        summary_rows.append({
            "Basin": basin_name,
            "Scenario": sc_name,
            "Total Flux (mol/s)": total_flux,
            "Flux (×10⁶ mol/s)": total_flux / 1e6
        })

df_summary = pd.DataFrame(summary_rows)
print(df_summary.to_string(index=False))

df_summary.to_csv("CO2_flux_summary.csv", index=False)
print("\n Results saved as 'CO2_flux_summary.csv'")
#%%



# ==========================================================
# 2. Compute grid-cell area (m²)
# ==========================================================
cell_area = e1t * e2t * tmask

# ==========================================================
# 3. Southern Ocean mask (lat < -40°S)
# ==========================================================
lat = d["nav_lat"]
mask_SO = xr.where(lat < -40, 1, 0)

# ==========================================================
# 4. Total air–sea CO₂ flux (mol/s)
# ==========================================================
# Convert mol/m²/s → mol/s per grid cell
flux_SO = Air_sea_flux_of_CO2_6 * cell_area * mask_SO

# Integrate over the ocean area
total_flux_SO = flux_SO.sum(dim=["y", "x"])

# ==========================================================
# 5. Time-mean total flux (mol/s)
# ==========================================================
mean_flux_SO = total_flux_SO.mean(dim="time_counter")
print(f"\nSouthern Ocean total CO₂ flux (south of 40°S): {mean_flux_SO.values:.3e} mol/s")

# ==========================================================
# 6. Separate uptake and outgassing (uptake is positive)
# ==========================================================
# uptake → flux > 0 (atmosphere → ocean)
# outgassing → flux < 0 (ocean → atmosphere)
uptake = flux_SO.where(flux_SO > 0).sum(dim=["y", "x"]).mean(dim="time_counter")
outgassing = flux_SO.where(flux_SO < 0).sum(dim=["y", "x"]).mean(dim="time_counter")

print(f"Uptake (mol/s):     {uptake.values:.3e}")
print(f"Outgassing (mol/s): {outgassing.values:.3e}")

#%%
import numpy as np

# mask جنوب 40 درجه
south40_mask = (lat < -40)

# مساحت سلول
cell_area = e1t * e2t

# فلکس کنترل
flux_control = Air_sea_flux_of_CO2_6[0,:,:]

# محاسبه mol/s
flux_south40 = np.nansum(flux_control * cell_area * tmask * south40_mask)

print("CO2 flux south of 40S (Control):", flux_south40, "mol/s")
print("CO2 flux south of 40S (×10^6 mol/s):", flux_south40/1e6)

flux_pgCyr = flux_south40 * 12 * 3.1536e7 / 1e15
print("CO2 flux south of 40S (Control):", flux_pgCyr, "pgCyr")

#%%
"""--------------------------------------------------------------
                    Surface DI C14 
--------------------------------------------------------------"""

lat2=d6.variables['nav_lat'][:]             #(332, 362)
lon2=d6.variables['nav_lon'][:]          #(332, 362)
lon2 = lon2.where(lon2 >= 0, lon2 + 360)

from matplotlib.lines import Line2D

def plot_front(lon2, lat2, label, color, m, ax):
    x, y = m(lon2, lat2)
    ax.plot(x, y, label=label, color=color, linewidth=1.5)


if (np.ndim(lon2) == 1) and (np.ndim(lat2) == 1):
    lon22, lat22 = np.meshgrid(lon2, lat2)
else:
    lon22, lat22 = lon2, lat2

anom_levels = np.linspace(-0.022, 0.022, 23)  
cmap_dic = 'RdBu_r'

mean_DIC_C14_Concentration_3_1=np.mean(DIC_C14_Concentration_3[:,0,:,:],axis=0)

mean_DIC_C14_Concentration_4_1=np.mean(DIC_C14_Concentration_4[:,0,:,:],axis=0)
mean_DIC_C14_Concentration_6_1=np.mean(DIC_C14_Concentration_6[:,0,:,:],axis=0)

mean_DIC_C14_Concentration_3 = np.where(lat2 < -40, mean_DIC_C14_Concentration_3_1, np.nan)
mean_DIC_C14_Concentration_4 = np.where(lat2 < -40, mean_DIC_C14_Concentration_4_1, np.nan)
mean_DIC_C14_Concentration_6 = np.where(lat2 < -40, mean_DIC_C14_Concentration_6_1, np.nan)

anom_4_6 = (mean_DIC_C14_Concentration_4 - mean_DIC_C14_Concentration_6)
anom_3_6 = (mean_DIC_C14_Concentration_3 - mean_DIC_C14_Concentration_6)



ds = xr.open_dataset('/esi/project/niwa02764/faezeh/eORCA1/fronts.nc')
front_colors = {"NB": "black", "SAF": "darkmagenta", "PF": "deeppink", "SACCF": "blue", "SB": "red"}

fig, axes = plt.subplots(1, 2, figsize=(20, 20))
fig.subplots_adjust(wspace=0.2, top=0.88, left=0.05, right=0.88, bottom=0.18)

axL = axes[0]
mL = Basemap(projection='spstere', boundinglat=-40, lon_0=180, resolution='l', ax=axL)


mL.drawcoastlines(linewidth=0.5)
mL.fillcontinents(color='#dddddd', lake_color='white')
mL.drawmapboundary(fill_color='white')
mL.drawparallels([-40, -60], linestyle='solid', linewidth=1, color='black')
mL.drawmeridians([ 30,  60,  90, 120, 150, 180, 210, 240, 270, 300, 330], labels=[True, True, True, True], fontsize=12)

cfL = mL.contourf(lon22, lat22, anom_4_6, latlon=True,
                  levels=anom_levels, cmap=cmap_dic, extend='both')

plot_front(ds["LonNB"],    ds["LatNB"],    "NB",    front_colors["NB"],    mL, axL)
plot_front(ds["LonSAF"],   ds["LatSAF"],   "SAF",   front_colors["SAF"],   mL, axL)
plot_front(ds["LonPF"],    ds["LatPF"],    "PF",    front_colors["PF"],    mL, axL)
plot_front(ds["LonSACCF"], ds["LatSACCF"], "SACCF", front_colors["SACCF"], mL, axL)
plot_front(ds["LonSB"],    ds["LatSB"],    "SB",    front_colors["SB"],    mL, axL)

x40, y40 = mL(0, -40); x60, y60 = mL(0, -60)
axL.text(x40, y40, '40°S', fontsize=12, ha='center', va='bottom', color='k')
axL.text(x60, y60, '60°S', fontsize=12, ha='center', va='bottom', color='k')
axL.set_title('a:', fontsize=16,y=1.05,fontweight='bold',loc='left')

axR = axes[1]

mR = Basemap(projection='spstere', boundinglat=-40, lon_0=180, resolution='l', ax=axR)


mR.drawcoastlines(linewidth=0.5)
mR.fillcontinents(color='#dddddd', lake_color='white')
mR.drawmapboundary(fill_color='white')
mR.drawparallels([-40, -60], linestyle='solid', linewidth=1, color='black')
mR.drawmeridians([ 30,  60,  90, 120, 150, 180, 210, 240, 270, 300, 330], labels=[True, True, True, True], fontsize=12)

cfR = mR.contourf(lon22, lat22, anom_3_6, latlon=True,
                  levels=anom_levels, cmap=cmap_dic, extend='both')

plot_front(ds["LonNB"],    ds["LatNB"],    "NB",    front_colors["NB"],    mR, axR)
plot_front(ds["LonSAF"],   ds["LatSAF"],   "SAF",   front_colors["SAF"],   mR, axR)
plot_front(ds["LonPF"],    ds["LatPF"],    "PF",    front_colors["PF"],    mR, axR)
plot_front(ds["LonSACCF"], ds["LatSACCF"], "SACCF", front_colors["SACCF"], mR, axR)
plot_front(ds["LonSB"],    ds["LatSB"],    "SB",    front_colors["SB"],    mR, axR)

x40r, y40r = mR(0, -40); x60r, y60r = mR(0, -60)
axR.text(x40r, y40r, '40°S', fontsize=12, ha='center', va='bottom', color='k')
axR.text(x60r, y60r, '60°S', fontsize=12, ha='center', va='bottom', color='k')
axR.set_title('b:', fontsize=16,y=1.05,fontweight='bold',loc='left')


from mpl_toolkits.axes_grid1.inset_locator import inset_axes

cax = inset_axes(axR, 
                 width="100%",    # پهن‌تر تا هر دو نقشه رو پوشش بده
                 height="3%",     # باریک‌تر برای افقی
                 loc='lower center',
                 bbox_to_anchor=(-0.6, -0.105, 1, 1),  # پایین دو نقشه
                 bbox_transform=axR.transAxes,
                 borderpad=0)

cb = fig.colorbar(cfR, cax=cax, orientation="horizontal")
cb.set_label(
    '$^{14}$C Concentration (mol m$^{-3}$) Anomaly',
    fontsize=12,
    fontweight='bold'
)
cb.ax.tick_params(labelsize=12)
cb.outline.set_linewidth(0.8)
front_legend = [
    Line2D([0], [0], color='black',   lw=2, label='NB'),
    Line2D([0], [0], color='darkmagenta',lw=2, label='SAF'),
    Line2D([0], [0], color='deeppink', lw=2, label='PF'),
    Line2D([0], [0], color='blue',  lw=2, label='SACCF'),
    Line2D([0], [0], color='red', lw=2, label='SB')
]

fig.legend(handles=front_legend, loc='upper right', bbox_to_anchor=(-0.005, 0.73),
           ncol=1, frameon=True, fontsize=16)
plt.savefig("/esi/project/niwa02764/faezeh/plot/abC14.png", dpi=400, bbox_inches='tight')

plt.show()
#%%


"""--------------------------------------------------------------
                    Meridional_Overt.Cell_Global 
------------------------------------------------------------------------------------------------"""




dmoc6=xr.open_dataset('/esi/project/niwa02764/faezeh/mocsig/mocsig-monthly-eORCA1-C14006o_1m_20000101_20191231.nc')
dmoc3=xr.open_dataset('/esi/project/niwa02764/faezeh/mocsig/mocsig-eORCA1-C14003o_1m_20000101_20191231_grid_.nc')
dmoc4=xr.open_dataset('/esi/project/niwa02764/faezeh/mocsig/mocsig-eORCA1-C14004o_1m_20000101_20191231_grid_T.nc')
#%%

lat_1=dmoc6['nav_lat']
lon_1=dmoc6['nav_lon']
lon_1 = (dmoc6["nav_lon"] % 360)

sigma6=dmoc6['sigma']
sigma3=dmoc3['sigma']
sigma4=dmoc4['sigma']

Meridional_Overt_Cell_Global_6=dmoc6['zomsfglo']

Meridional_Overt_Cell_Global_4=dmoc4['zomsfglo']
Meridional_Overt_Cell_Global_3=dmoc3['zomsfglo']

#%%
mean_Meridional_Overt_Cell_Global_6=np.mean(Meridional_Overt_Cell_Global_6[:,:,:,0],axis=(0))
mean_Meridional_Overt_Cell_Global_4=np.mean(Meridional_Overt_Cell_Global_4[:,:,:,0],axis=(0))
mean_Meridional_Overt_Cell_Global_3=np.mean(Meridional_Overt_Cell_Global_3[:,:,:,0],axis=(0))
#%%
"""--------------------------------------------------------------
                    Meridional_Overt.Cell_Global 
------------------------------------------------------------------------------------------------"""


# === Latitude and density slices ===
lat_section = lat_1[10:130, 0]
sigma_section = sigma6[80:]
lat_slice = slice(10, 130)
sigma_slice = slice(80, None)

fig, ax = plt.subplots(figsize=(8, 5))

cf = ax.contourf(
    lat_section,
    sigma_section,
    mean_Meridional_Overt_Cell_Global_6[sigma_slice, lat_slice],
    levels=np.linspace(-15, 15, 15),
    cmap='coolwarm',
    extend='both'
)

cs = ax.contour(
    lat_section,
    sigma_section,
    mean_Meridional_Overt_Cell_Global_6[sigma_slice, lat_slice],
    levels=np.linspace(-15, 15, 15),
    colors='black',
    linewidths=0.5
)
ax.clabel(cs, fmt="%.0f", fontsize=10)

ax.set_ylim(sigma_section.max(), sigma_section.min())

ax.set_xlabel('Latitude', fontsize=12)
ax.set_ylabel('σ$_{2000}$ (kg m$^{-3}$)', fontsize=12)
ax.set_title('Meridional Overturning Circulation (Control Simulation),over 2000-2019', fontsize=13)

cbar = fig.colorbar(cf, ax=ax, orientation='vertical', pad=0.05, label='Streamfunction (Sv)')

# === Water mass labels ===
for label, y in zip(["AABW", "LCDW", "NADW", "UCDW", "AAIW", "SAMW"], [37.1, 37.0, 36.7, 36.5, 35.2, 34.25]):
    ax.text(-38, y, label, fontsize=9, va='center')

ax.tick_params(labelsize=10)
ax.grid(True, linestyle='--', alpha=0.3)

plt.tight_layout()
plt.show()


#%%

"""--------------------------------------------------------------
                    all Meridional_Overt.Cell_Global
--------------------------------------------------------------"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

def nan_gaussian_filter(arr, sigma):
    a = np.array(arr, dtype=float)
    mask = np.isfinite(a).astype(float)
    a_filled = np.where(np.isfinite(a), a, 0.0)
    num = gaussian_filter(a_filled, sigma=sigma, mode='nearest')
    den = gaussian_filter(mask,    sigma=sigma, mode='nearest')
    return np.where(den > 0, num/den, np.nan)

anomaly_4 = mean_Meridional_Overt_Cell_Global_4 - mean_Meridional_Overt_Cell_Global_6
anomaly_3 = mean_Meridional_Overt_Cell_Global_3 - mean_Meridional_Overt_Cell_Global_6
smoothed_anomaly_4 = nan_gaussian_filter(anomaly_4, sigma=2.0)
smoothed_anomaly_3 = nan_gaussian_filter(anomaly_3, sigma=2.0)

levels_anom_f = np.linspace(-3,  3, 13)      # anomaly fill
levels_ctrl   = np.linspace(-25, 25, 51)     # control fill

levels_6 = np.concatenate([
    np.arange(-9, 0, 2),
    np.arange(0, 20, 2)
])

levels_4 = np.concatenate([
    np.arange(-6, 0, 0.2),
    np.arange(0, 6.1, 0.8)
])

levels_3 = np.concatenate([
    np.arange(-3, 0, 0.4),
    np.arange(0, 3.1, 0.7)
])

fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(
    nrows=2, ncols=3,
    height_ratios=[1.0, 0.035],          
    left=0.2, right=0.88, top=0.7, bottom=0.2,
    wspace=0.25, hspace=0.35           
)

axA = fig.add_subplot(gs[0, 0])
axB = fig.add_subplot(gs[0, 1])
axC = fig.add_subplot(gs[0, 2])
axes = [axA, axB, axC]

# Colorbar axes
caxA  = fig.add_subplot(gs[1, 0])     # under A
caxBC = fig.add_subplot(gs[1, 1:3])   # under B and C

ctrl_field = mean_Meridional_Overt_Cell_Global_6[sigma_slice, lat_slice]

cf1 = axA.contourf(
    lat_section, sigma_section, ctrl_field,
    levels=levels_ctrl, cmap='viridis', extend='both'
)

c1 = axA.contour(
    lat_section, sigma_section, ctrl_field,
    levels=levels_6, colors='black', linewidths=0.45
)
axA.clabel(c1, fmt="%.0f", fontsize=10, inline=True, inline_spacing=5)

axA.invert_yaxis()
axA.set_ylabel('σ$_{2000}$ (kg m$^{-3}$)', fontsize=12, fontweight='bold')
axA.set_xlabel('Latitude', fontsize=12)

# ===================== B) Anomaly C4−C6 =====================
fld_4 = smoothed_anomaly_4[sigma_slice, lat_slice]

cf2 = axB.contourf(
    lat_section, sigma_section, fld_4,
    levels=levels_anom_f, cmap='coolwarm', extend='both'
)

c2 = axB.contour(
    lat_section, sigma_section, fld_4,
    levels=levels_4, colors='black', linewidths=0.5
)
axB.clabel(c2, levels=c2.levels, fmt="%.2f", fontsize=10, inline=True, inline_spacing=5)
axB.contour(lat_section, sigma_section, fld_4, levels=[0],
            colors='k', linewidths=0.9, linestyles='--')

axB.invert_yaxis()
axB.set_xlabel('Latitude', fontsize=12)

# ===================== C) Anomaly C3−C6 =====================
fld_3 = smoothed_anomaly_3[sigma_slice, lat_slice]

cf3 = axC.contourf(
    lat_section, sigma_section, fld_3,
    levels=levels_anom_f, cmap='coolwarm', extend='both'
)

c3 = axC.contour(
    lat_section, sigma_section, fld_3,
    levels=levels_3, colors='black', linewidths=0.5
)
axC.clabel(c3, levels=c3.levels, fmt="%.2f", fontsize=10, inline=True, inline_spacing=5)
axC.contour(lat_section, sigma_section, fld_3, levels=[0],
            colors='k', linewidths=0.9, linestyles='--')

axC.invert_yaxis()
axC.set_xlabel('Latitude', fontsize=12)

cb1 = fig.colorbar(cf1, cax=caxA, orientation='horizontal')
cb1.set_label('Control Meridional Overturning Circulation (Sv)', fontsize=10, fontweight='bold')
cb1.ax.tick_params(labelsize=12)

cb2 = fig.colorbar(cf2, cax=caxBC, orientation='horizontal')
cb2.set_label('Meridional Overturning Circulation Anomaly (Sv)', fontsize=10, fontweight='bold')
cb2.ax.tick_params(labelsize=12)

# ===================== Titles, ticks, grids =====================
axA.set_title('a:', fontsize=16, fontweight='bold', loc='left')
axB.set_title('b:', fontsize=16, fontweight='bold', loc='left')
axC.set_title('c:', fontsize=16, fontweight='bold', loc='left')

for ax in axes:
    ax.tick_params(labelsize=10)
    ax.grid(True, linestyle='--', alpha=0.3)

plt.savefig("/esi/project/niwa02764/faezeh/plot/abcMOC.png", dpi=400, bbox_inches='tight')
plt.show()



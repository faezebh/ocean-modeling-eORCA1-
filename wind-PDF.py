
import xarray as xr
import matplotlib.pyplot as plt

import numpy as np
v_o=xr.open_dataset(r"C:\Users\bahmanif\OneDrive - NIWA\Documents\Screenshots\paper\paper\Presentation\wind\ERA5_3h_atmospheric_forcing_10m_v_component_of_wind_y2000.nc")
u_o=xr.open_dataset(r"C:\Users\bahmanif\OneDrive - NIWA\Documents\Screenshots\paper\paper\Presentation\wind\ERA5_3h_atmospheric_forcing_10m_u_component_of_wind_y2000.nc")

v_u=xr.open_dataset(r"C:\Users\bahmanif\OneDrive - NIWA\Documents\Screenshots\paper\paper\Presentation\wind\ERA5_3h_atmospheric_forcing_10m_v_component_of_wind_second_increase_abs_uni_y2000.nc")
u_u=xr.open_dataset(r"C:\Users\bahmanif\OneDrive - NIWA\Documents\Screenshots\paper\paper\Presentation\wind\ERA5_3h_atmospheric_forcing_10m_u_component_of_wind_second_increase_abs_uni_y2000.nc")


v_q=xr.open_dataset(r"C:\Users\bahmanif\OneDrive - NIWA\Documents\Screenshots\paper\paper\Presentation\wind\ERA5_3h_atmospheric_forcing_10m_v_component_of_wind_second_increase_abs_y2000.nc")
u_q=xr.open_dataset(r"C:\Users\bahmanif\OneDrive - NIWA\Documents\Screenshots\paper\paper\Presentation\wind\ERA5_3h_atmospheric_forcing_10m_u_component_of_wind_second_increase_abs_y2000.nc")
d=xr.open_dataset(r"C:\Users\bahmanif\OneDrive - NIWA\Documents\Screenshots\paper\paper\Presentation\wind\mesh_mask.nc")


e1t=d['e1t'][0,:,:]
e2t=d['e2t'][0,:,:]
tmask=d['tmask'][0,0,:,:]


u_org=u_o['u10']    
v_org=v_o['v10'] 


v_uni=v_u['v10']    
u_uni=u_u['u10'] 

v_qua=v_q['v10']    
u_qua=u_q['u10'] 


lat=u_o['latitude']
lon=u_o['longitude']

#%%

u__org_south40 = u_org.sel(latitude=slice(-40, -90))

v__org_south40 = v_org.sel(latitude=slice(-40, -90))
#%%
u__uni_south40 = u_uni.sel(latitude=slice(-40, -90))

v__uni_south40 = v_uni.sel(latitude=slice(-40, -90))
#%%
u__qua_south40 = u_qua.sel(latitude=slice(-40, -90))

v__qua_south40 = v_qua.sel(latitude=slice(-40, -90))


#%%
wind_speed_org= np.sqrt(u__org_south40**2 + v__org_south40**2)
#%%
wind_speed_uni= np.sqrt(u__uni_south40**2 + v__uni_south40**2)
#%%
wind_speed_qua= np.sqrt(u__qua_south40**2 + v__qua_south40**2)



#%%

threshold = 10

mask_org = wind_speed_org >= threshold
mask_uni = wind_speed_uni >= threshold
mask_qua = wind_speed_qua >= threshold



pct_org = 100 * np.sum(mask_org) / mask_org.size
pct_uni = 100 * np.sum(mask_uni) / mask_uni.size
pct_qua = 100 * np.sum(mask_qua) / mask_qua.size

print("Control strong winds:", pct_org, "%")
print("Uniform strong winds:", pct_uni, "%")
print("Quadratic strong winds:", pct_qua, "%")
#%%
thresholds = np.arange(8,40,1)

pct_org = []
pct_uni = []
pct_qua = []

for th in thresholds:
    
    pct_org.append((wind_speed_org >= thresholds).mean().item()*100)
    pct_uni.append((wind_speed_uni >= thresholds).mean().item()*100)
    pct_qua.append((wind_speed_qua >= thresholds).mean().item()*100)
#%%

plt.figure(figsize=(6,5))

plt.plot(pct_org, thresholds,label="Control")
plt.plot(pct_uni,thresholds, label="Uniform")
plt.plot(pct_qua,thresholds, label="Quadratic")

plt.xlabel("Wind Speed Threshold (m/s)")
plt.ylabel("Percentage of winds ≥ threshold (%)")

plt.grid()
plt.legend()

plt.show()





#%%
np.nanpercentile(wind_speed_org, 90)
#%%
np.nanpercentile(wind_speed_uni, 90)
#%%
np.nanpercentile(wind_speed_qua, 90)

#%%


org_flat = wind_speed_org.where(np.isfinite(wind_speed_org), drop=True).values.ravel()
#%%
qua_flat = wind_speed_qua.where(np.isfinite(wind_speed_qua), drop=True).values.ravel()
#%%
uni_flat = wind_speed_uni.where(np.isfinite(wind_speed_uni), drop=True).values.ravel()





#%%


import matplotlib.pyplot as plt
import numpy as np

# -------------------------
# Global font settings (paper-ready)
# -------------------------
plt.rcParams.update({
    "font.size":14,
    "axes.titlesize": 14,
    "axes.labelsize": 14,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
})

# -------------------------
# Histogram → PDF
# -------------------------
bins = 70
bin_edges = np.arange(0, 70, 0.5)

counts_org, bin_edges = np.histogram(org_flat, bins=bins)
counts_qua, _ = np.histogram(qua_flat, bins=bin_edges)
counts_uni, _ = np.histogram(uni_flat, bins=bin_edges)

bin_width = np.diff(bin_edges)[0]
pdf_org = counts_org / (counts_org.sum() * bin_width)
pdf_qua = counts_qua / (counts_qua.sum() * bin_width)
pdf_uni = counts_uni / (counts_uni.sum() * bin_width)

bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

# -------------------------
# Plot
# -------------------------
plt.figure(figsize=(6, 4))

plt.plot(
    bin_centers, pdf_org,
    color='#00b6e3', lw=2.2, label='CONTROL'
)
plt.plot(
    bin_centers, pdf_qua,
    color='green', lw=2.4, label='QUADRATIC'
)
plt.plot(
    bin_centers, pdf_uni,
    color='#ff8833', lw=2.0, label='UNIFROM'
)

plt.axvline(
    x=8, color='black', linestyle='--',
    linewidth=1.8, label='8 m/s'
)

plt.xlabel('Wind speed (m/s)')
plt.ylabel('PDF')
plt.title('')

plt.legend(frameon=False)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# -------------------------
# Sanity check
# -------------------------
print("Area org:", np.sum(pdf_org * bin_width))
print("Area qua:", np.sum(pdf_qua * bin_width))
print("Area uni:", np.sum(pdf_uni * bin_width))


#%%
#Percentile

import numpy as np

p = np.arange(0, 101, 1)   # percentiles

p_org = np.nanpercentile(wind_speed_org.values, p)
p_uni = np.nanpercentile(wind_speed_uni.values, p)
p_qua = np.nanpercentile(wind_speed_qua.values, p)


plt.figure(figsize=(6,5))

plt.plot(p_org,p, label="Control")
plt.plot(p_uni,p , label="Uniform")
plt.plot(p_qua, p, label="Quadratic")

plt.axhline(8, color='k', linestyle='--', label='8 m/s')

plt.xlabel("Percentile (%)")
plt.ylabel("Wind Speed (m/s)")

plt.legend()
plt.grid()

plt.show()











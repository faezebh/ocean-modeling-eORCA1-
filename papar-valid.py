"""

@author: bahmanif
"""


from netCDF4 import Dataset  

from mpl_toolkits.basemap import Basemap


import numpy as np

import matplotlib.pyplot as plt


import matplotlib.gridspec as gridspec

#  model

ds_model = Dataset('/esi/project/niwa02764/faezeh/eORCA1/model_SSH.nc')

lon = ds_model.variables['nav_lon'][:]         # 2D
lat = ds_model.variables['nav_lat'][:]         # 2D
ssh_model = ds_model.variables['sossheig'][0, :, :]

#  AVISO
ds_aviso = Dataset('/esi/project/niwa02764/faezeh/eORCA1/aviso_1_degree.nc')

ssh_aviso = ds_aviso.variables['adt'][0, :, :]


lon = np.array(lon, dtype=float)
lat = np.array(lat, dtype=float)
ssh_model = np.array(ssh_model, dtype=float)
ssh_aviso = np.array(ssh_aviso, dtype=float)

ssh_model[np.abs(ssh_model) > 1e10] = np.nan
ssh_aviso[np.abs(ssh_aviso) > 1e10] = np.nan

# remove GLOBAL mean 
ssh_model_anom = ssh_model - np.nanmean(ssh_model)
ssh_aviso_anom = ssh_aviso - np.nanmean(ssh_aviso)


so_mask = lat <= -40

ssh_model_anom = np.where(so_mask, ssh_model_anom, np.nan)
ssh_aviso_anom = np.where(so_mask, ssh_aviso_anom, np.nan)


valid_diff = np.isfinite(ssh_model_anom) & np.isfinite(ssh_aviso_anom)
ssh_diff = np.where(valid_diff, ssh_model_anom - ssh_aviso_anom, np.nan)


allvals = np.concatenate([
    ssh_aviso_anom[np.isfinite(ssh_aviso_anom)],
    ssh_model_anom[np.isfinite(ssh_model_anom)]
])

vabs = np.nanpercentile(np.abs(allvals), 98)
levels = np.linspace(-1.65, 1.65, 23)

diffvals = ssh_diff[np.isfinite(ssh_diff)]
dabs = np.nanpercentile(np.abs(diffvals), 98)
levels_diff = np.linspace(-0.5, 0.5, 21)


fig = plt.figure(figsize=(18, 8))
gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1], wspace=0.35)

ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[0, 2])

meridians = [30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]
parallels = [-60, -40]

# A: AVISO
map1 = Basemap(
    projection='spstere',
    boundinglat=-40,
    lon_0=180,
    resolution='l',
    ax=ax1
)

cs1 = map1.contourf(
    lon, lat, ssh_aviso_anom,
    levels=levels,
    latlon=True,
    extend='both',
    cmap='RdBu_r'
)

map1.drawmapboundary(fill_color='white')
map1.fillcontinents(color='white', lake_color='white')
map1.drawparallels(parallels, labels=[0, 0, 0, 0], fontsize=12, linewidth=1, color='black')
map1.drawmeridians(meridians, labels=[True, True, True, True], fontsize=12)

cb1 = plt.colorbar(cs1, ax=ax1, orientation='horizontal', pad=0.08, aspect=50, shrink=0.95)
cb1.set_label('Sea Surface Height anomaly (m)', fontsize=12, fontweight='bold')

ax1.text(0.02, 1.08, 'a:', transform=ax1.transAxes, fontsize=16,
         fontweight='bold', va='top', ha='left')

# B: Model
map2 = Basemap(
    projection='spstere',
    boundinglat=-40,
    lon_0=180,
    resolution='l',
    ax=ax2
)

cs2 = map2.contourf(
    lon, lat, ssh_model_anom,
    levels=levels,
    latlon=True,
    extend='both',
    cmap='RdBu_r'
)

map2.drawmapboundary(fill_color='white')
map2.fillcontinents(color='white', lake_color='white')
map2.drawparallels(parallels, labels=[0, 0, 0, 0], fontsize=12, linewidth=1, color='black')
map2.drawmeridians(meridians, labels=[True, True, True, True], fontsize=12)

cb2 = plt.colorbar(cs2, ax=ax2, orientation='horizontal', pad=0.08, aspect=50, shrink=0.95)
cb2.set_label('Sea Surface Height anomaly (m)', fontsize=12, fontweight='bold')

# ax2.set_title('Model mean SSH anomaly', fontsize=13, pad=40)
ax2.text(0.02, 1.08, 'b:', transform=ax2.transAxes, fontsize=16,
         fontweight='bold', va='top', ha='left')

# C: Difference
map3 = Basemap(
    projection='spstere',
    boundinglat=-40,
    lon_0=180,
    resolution='l',
    ax=ax3
)

cs3 = map3.contourf(
    lon, lat, ssh_diff,
    levels=levels_diff,
    latlon=True,
    extend='both',
    cmap='RdBu_r'
)

map3.drawmapboundary(fill_color='white')
map3.fillcontinents(color='white', lake_color='white')
map3.drawparallels(parallels, labels=[0, 0, 0, 0], fontsize=12, linewidth=1, color='black')
map3.drawmeridians(meridians, labels=[True, True, True, True], fontsize=12)

cb3 = plt.colorbar(cs3, ax=ax3, orientation='horizontal', pad=0.08, aspect=50, shrink=0.95)
cb3.set_label('SSH anomaly difference (m)', fontsize=12, fontweight='bold')

# ax3.set_title('Model - AVISO', fontsize=13, pad=40)
ax3.text(0.02, 1.08, 'c:', transform=ax3.transAxes, fontsize=16,
         fontweight='bold', va='top', ha='left')

# latitude labels
for m, ax in [(map1, ax1), (map2, ax2), (map3, ax3)]:
    x40, y40 = m(0, -40)
    x60, y60 = m(0, -60)
    ax.text(x40, y40, '40°S', fontsize=12, ha='center', va='bottom', color='black')
    ax.text(x60, y60, '60°S', fontsize=12, ha='center', va='bottom', color='black')

plt.subplots_adjust(wspace=0.12, top=0.92, bottom=0.1)

plt.savefig(
    "/esi/project/niwa02764/faezeh/plot/abcAVISO_model_anomaly.png",
    dpi=400,
    bbox_inches='tight'
)

plt.show()

#%%%


from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt

d2=Dataset('/esi/project/niwa02764/faezeh/eORCA1/GLODAPv2.2016b.TCO2_1x1.nc')

lon_o=d2.variables['lon'][:]
lat_o=d2.variables['lat'][:]
dic_o=d2.variables['TCO2'][:]
#%%
d3=Dataset('/esi/project/niwa02764/faezeh/eORCA1/C14_2_1x1.nc')

c14_o=d3.variables['C14'][:]
depth_o=d3.variables['depth'][:]
d1=Dataset('/esi/project/niwa02764/faezeh/eORCA1/remap_eORCA1-C14006o_1m_20000101_20191231_grid_T_33l.nc')#MODEl

lon_m=d1.variables['lon'][:]
lat_m=d1.variables['lat'][:]

dic_m=d1.variables['inorganic_carbon'][:]
c14_m=d1.variables['DIC_C14_Concentration'][:]

depth_m=d1.variables['deptht'][:]


c14_m_pacifi=np.nanmean(c14_m[0,:,:,150:280],axis=2)
dic_m_pacifi=np.nanmean(dic_m[0,:,:,150:280],axis=2)



c14_o_pacifi=np.nanmean(c14_o[:,:,150:280],axis=2) #unit permil or Δ¹⁴C
dic_o_pacifi=np.nanmean(dic_o[:,:,150:280],axis=2) #µmol/kg 
#%%
fig = plt.figure(figsize=(14,8))

ax=fig.add_subplot(312)
plt.subplots_adjust(hspace=0.5, wspace=0.3)


plt.pcolormesh(lat_m[:],depth_m,c14_m_pacifi[:,:],cmap='RdBu_r')


plt.ylim(5000,0)
plt.xlim(-80,0)
plt.grid()
plt.xticks(range(-80,10,10),['80$\degree$S','70$\degree$S','60$\degree$S','50$\degree$S','40$\degree$S','30$\degree$S','20$\degree$S','10$\degree$S','0$\degree$'],fontsize=12)
plt.title('Modelled $^{14}$C concentration (mol/m$^3$), Pacific')
#plt.colorbar()
plt.ylabel('Depth [m]',fontweight='bold')
plt.clim(1.85,2.35)
plt.text(-80,-200,'a:', fontsize=20,fontweight='bold')

ax=fig.add_subplot(311)



c=plt.pcolormesh(lat_o,depth_o[:],((c14_o_pacifi[:,:]/1000)+1)*dic_o_pacifi[:,:]/1e6*1025,cmap='RdBu_r')
plt.ylim(5000,0)
plt.xlim(-80,0)
plt.xticks(range(-80,10,10),['80$\degree$S','70$\degree$S','60$\degree$S','50$\degree$S','40$\degree$S','30$\degree$S','20$\degree$S','10$\degree$S','0$\degree$'],fontsize=12)

plt.grid()
plt.title('Observed $^{14}$C concentration (mol/m$^3$) (GLODAP), Pacific')
plt.ylabel('Depth [m]',fontweight='bold')
plt.clim(1.85,2.35)
plt.text(-80,-200,'b:', fontsize=20,fontweight='bold')

ax1=fig.add_axes([.835,.32,.1,.6])
plt.axis('off')
from mpl_toolkits.axes_grid1 import make_axes_locatable

# cbar_ax = fig.add_axes([0.92, 0.4, 0.02, 0.5])  # [left, bottom, width, height]
c1=plt.colorbar(c, pad=0.05, aspect=40, shrink=0.9,ax=ax1,label='$^{14}$C concentration (mol/m$^3$) ')
c1.ax.yaxis.label.set_size(12)
ax=fig.add_subplot(313)

cax=plt.pcolormesh(lat_o,depth_o[:32],c14_m_pacifi[:32,:]-((c14_o_pacifi[:32,:]/1000)+1)*dic_o_pacifi[:32,:]/1e6*1025,cmap='RdBu_r',shading='auto')



plt.ylim(5000,0)
plt.xlim(-80,0)
plt.xticks(range(-80,10,10),['80$\degree$S','70$\degree$S','60$\degree$S','50$\degree$S','40$\degree$S','30$\degree$S','20$\degree$S','10$\degree$S','0$\degree$'],fontsize=12)
plt.ylabel('Depth [m]',fontweight='bold')
plt.grid()
plt.title('Difference model-observation (mol/m$^3$), Pacific')
plt.text(-80,-200,'c:', fontsize=20,fontweight='bold')

plt.clim(-.1,.1)

ax1=fig.add_axes([.835,.12,.1,.2])
plt.axis('off')
cb=plt.colorbar(cax, ax=ax1, shrink=1, pad=0.05, aspect=20,label='$^{14}$C concentration (mol/m$^3$)')
cb.ax.yaxis.label.set_size(12)  # or any size you prefer
plt.savefig("/esi/project/niwa02764/faezeh/plot/glodap.png", dpi=400, bbox_inches='tight')
plt.show()




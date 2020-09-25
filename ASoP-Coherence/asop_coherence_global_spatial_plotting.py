#!/usr/bin/env python
# coding: utf-8

# In[1]:


import iris
import iris.quickplot as qplt
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import sys
sys.path.append('/home/nick/python/ASoP_global/ASoP-Coherence')
from asop_coherence_global import get_asop_dict
import matplotlib.cm as mpl_cm
import numpy as np
import numpy.ma as ma
import pandas as pd
from iris.experimental.equalise_cubes import equalise_attributes
from iris.util import unify_time_units


# In[2]:


def load_summary_metric(filename,constraint,new_long_name,new_units):
    cube = iris.load_cube(filename,constraint)
    cube.long_name = new_long_name
    cube.units = new_units
    return(cube)


# In[3]:


def load_all_summary_metrics(asop_dict,wet_season_threshold='1d24'):
    constraints_longnames_units = [
        ('spatial intermittency on-off metric based on 4 divisions (weighted mean of all months in wet season)','spatial coh metric, 4divs, wet seas','1'),
        ('Probability of upper division neighbouring upper division (weighted mean of all months in wet season)','p(on|on), 4divs, wet seas','1'),
        ('Probability of upper division neighbouring lower division (weighted mean of all months in wet season)','p(on|off), 4divs, wet seas','1'),
        ('Probability of lower division neighbouring upper division (weighted mean of all months in wet season)','p(off|on), 4divs, wet seas','1'),
        ('Probability of lower division neighbouring lower division (weighted mean of all months in wet season)','p(off|off), 4divs, wet seas','1')
    ]
    out_cubelist = []
    summary_file = asop_dict['desc']+'_asop_'+asop_dict['year_range']+'_spatial_summary_wetseason'+wet_season_threshold+'.nc'
    for constraint,long_name,units in constraints_longnames_units:
        print(constraint)
        cube = load_summary_metric(str(asop_dict['dir']/summary_file),constraint,long_name,units)
        out_cubelist.append(cube)
    return(out_cubelist)


# In[4]:


def plot_summary_metric(model_cube,obs_cube,model_dict,obs_dict,raw_levs,diff_levs,raw_cmap,diff_cmap):
    fig = plt.figure()
    raw_cmap = mpl_cm.get_cmap(raw_cmap)
    diff_cmap = mpl_cm.get_cmap(diff_cmap)
    ax = plt.axes(projection=ccrs.Mollweide(central_longitude=0))
    qplt.contourf(model_cube,raw_levs,cmap=raw_cmap,axes=ax)
    #qplt.contourf(model_cube,raw_levs)
    plt.gca().coastlines(color='grey')
    plt.savefig('plots/asop_coherence_global_spatial_'+model_dict['name']+'_'+model_cube.var_name+'.png',dpi=200)

    if obs_cube is not None:
        fig = plt.figure()
        ax = plt.axes(projection=ccrs.Mollweide(central_longitude=0))
        qplt.contourf(obs_cube,raw_levs,cmap=raw_cmap,axes=ax)
        #qplt.contourf(model_cube,raw_levs)
        plt.gca().coastlines(color='grey')
        plt.savefig('plots/asop_coherence_global_spatial_'+obs_dict['name']+'_'+obs_cube.var_name+'.png',dpi=200)

        fig = plt.figure()
        ax = plt.axes(projection=ccrs.Mollweide(central_longitude=0))
        diff_cube = model_cube.copy()
        diff_cube.data = model_cube.data-obs_cube.data
        diff_cube.long_name=model_cube.long_name+' diff '+obs_dict['name']
    #    qplt.contourf(diff_cube,diff_levs,cmap=diff_cmap)
        qplt.contourf(diff_cube,diff_levs,cmap=diff_cmap,axes=ax)
        plt.gca().coastlines(color='grey')
        plt.savefig('plots/asop_coherence_global_spatial_'+model_dict['name']+'-minus-'+obs_dict['name']+'_'+model_cube.var_name+'.png',dpi=200)


# In[5]:


def find_distcorr_threshold(cube,threshold=0.5,long_name=None,units=None):
    import numpy as np
    #max_autocorr = cube.collapsed('lag',iris.analysis.MEAN).copy()
    lon = cube.coord('longitude')
    lat = cube.coord('latitude')
    dist = cube.coord('distance')
    nlon = len(lon.points)
    nlat = len(lat.points)
#    max_autocorr = iris.cube.Cube(np.ma.empty((nlat,nlon)),dim_coords_and_dims=[(lat,0),(lon,1)])
    max_autocorr = np.zeros((nlat,nlon))
    print(cube)
    for y in range(nlat):
        for x in range(nlon):
            below_threshold = np.where(cube.data[:,y,x] <= threshold)
            if np.sum(below_threshold) >= 1:
                max_autocorr[y,x] = dist.points[np.amin(below_threshold)]-1
            else:
                max_autocorr[y,x] = np.nan #np.amax(dist.points)
    max_autocorr_cube = iris.cube.Cube(data=max_autocorr,dim_coords_and_dims=[(lat,0),(lon,1)],long_name=long_name,units=units,var_name='distcorr_threshold'+str(threshold))
    return(max_autocorr_cube)

def mask_model_by_gpm(model_cube,gpm_cube):
    model_mask = np.isnan(model_cube.data)
    gpm_mask = np.isnan(gpm_cube.data)
    my_mask = np.ma.mask_or(model_mask,gpm_mask)
    new_cube = model_cube.copy(data=np.ma.masked_array(model_cube.data,my_mask))
    return(new_cube)


# In[6]:


time_type='day'
if time_type == 'day':
    models=['BCC','AWI','CanESM5','ACCESS','CESM2','CMCC','CNRM','FGOALS','GFDL','HadGEM3','INM','IPSL','KACE','MIROC','MPI','MRI','NESM','NorESM','SAM0-UNICON','TaiESM','UKESM1']
elif time_type == '3hr':
    models=['AWI','BCC','ACCESS','CMCC','CNRM','FGOALS','KACE','MRI','SAM0-UNICON']
grid_type='2x2'
threshold=0.2
ponoff_raw_levs=np.arange(0.0,0.21,0.02) #.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9]
ponoff_diff_levs=np.arange(-0.10,0.11,0.02)
poffon_raw_levs=np.arange(0.0,0.41,0.04) #.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9]
poffon_diff_levs=np.arange(-0.20,0.21,0.04)
poffoff_raw_levs=np.arange(0.30,0.9,0.06) #.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9]
poffoff_diff_levs=np.arange(-0.2,0.21,0.04)
summary_raw_levs=np.arange(0.0,0.81,0.05)
summary_diff_levs=np.arange(-0.3,0.31,0.05)
autocorr_raw_levels=[200,400,600,800,1000,1200]
autocorr_diff_levels=np.arange(-1000,1001,200)


# In[7]:


def fix_zerolon(cube):
    cube.data[:,-2] = cube.data[:,-3]
    cube.data[:,-1] = cube.data[:,-2]
    cube.data[:,0] = cube.data[:,1]
    cube.data[:,1] = cube.data[:,2]
    #cube.data[:,0] = cube.data[:,-2]
    return(cube)


# In[8]:


gpm_dict = get_asop_dict('GPM_IMERG',time=time_type,grid=grid_type)
print(gpm_dict['desc'])
ponon_raw_levs=np.arange(0.40,0.80,0.04) #.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9]
ponon_diff_levs=np.arange(-0.2,0.21,0.04)
gpm_spatial_summary,gpm_ponon,gpm_ponoff,gpm_poffon,gpm_poffoff = load_all_summary_metrics(gpm_dict)
gpm_autocorr_filename = str(gpm_dict['dir'])+'/'+gpm_dict['desc']+'_asop_'+gpm_dict['year_range']+'_spatial_corr_wetseason1d24.nc'
gpm_autocorr = load_summary_metric(gpm_autocorr_filename,'distance_correlations_mean',None,'km')
gpm_threshold = find_distcorr_threshold(gpm_autocorr,threshold=threshold,long_name='Dist corr, >'+str(threshold)+', wet seas '+gpm_dict['name'],units='km')
gpm_threshold = fix_zerolon(gpm_threshold)
gpm_spatial_summary = fix_zerolon(gpm_spatial_summary)


# In[9]:


def add_model_coord(cube,number):
    model_coord = iris.coords.AuxCoord(m,long_name='model number',var_name='model_number')
    cube.add_aux_coord(model_coord)


# In[10]:


threshold_cl = iris.cube.CubeList()
summary_cl = iris.cube.CubeList()
for m,model in enumerate(models):
    model_dict = get_asop_dict(model,time=time_type,grid=grid_type)
    model_spatial_summary,model_ponon,model_ponoff,model_poffon,model_poffoff = load_all_summary_metrics(model_dict)
    model_autocorr_filename = str(model_dict['dir'])+'/'+model_dict['desc']+'_asop_'+model_dict['year_range']+'_spatial_corr_wetseason1d24.nc'
    model_autocorr = load_summary_metric(model_autocorr_filename,'distance_correlations_mean',None,'km')
    model_threshold = find_distcorr_threshold(model_autocorr,threshold=threshold,long_name='Distance corr >'+str(threshold)+', wet seas',units='km')
    model_threshold = fix_zerolon(model_threshold)
    model_spatial_summary = fix_zerolon(model_spatial_summary)
    model_threshold = mask_model_by_gpm(model_threshold,gpm_threshold)
    model_spatial_summary = mask_model_by_gpm(model_spatial_summary,gpm_spatial_summary)
    add_model_coord(model_spatial_summary,m)
    add_model_coord(model_threshold,m)
    summary_cl.append(model_spatial_summary)
    threshold_cl.append(model_threshold)
    plot_summary_metric(model_threshold,gpm_threshold,model_dict,gpm_dict,autocorr_raw_levels,autocorr_diff_levels,'Oranges','brewer_PRGn_11')
#    plot_summary_metric(model_ponon,gpm_ponon,model_dict,gpm_dict,ponon_raw_levs,ponon_diff_levs,'Oranges','brewer_PRGn_11')
#    plot_summary_metric(model_ponoff,gpm_ponoff,model_dict,gpm_dict,ponoff_raw_levs,ponoff_diff_levs,'Oranges','brewer_PRGn_11')
#    plot_summary_metric(model_poffon,gpm_poffon,model_dict,gpm_dict,poffon_raw_levs,poffon_diff_levs,'Oranges','brewer_PRGn_11')
#    plot_summary_metric(model_poffoff,gpm_poffoff,model_dict,gpm_dict,poffoff_raw_levs,poffoff_diff_levs,'Oranges','brewer_PRGn_11')
    plot_summary_metric(model_spatial_summary,gpm_spatial_summary,model_dict,gpm_dict,summary_raw_levs,summary_diff_levs,'Oranges','brewer_PRGn_11')


# In[11]:


equalise_attributes(threshold_cl)
unify_time_units(threshold_cl)
threshold_cube = threshold_cl.merge_cube()
summary_cube = summary_cl.merge_cube()


# In[12]:


threshold_mmm = threshold_cube.collapsed('model number',iris.analysis.MEAN)
threshold_mmm = mask_model_by_gpm(threshold_mmm,gpm_threshold)
threshold_sdev = threshold_cube.collapsed('model number',iris.analysis.STD_DEV)
threshold_sdev = mask_model_by_gpm(threshold_sdev,gpm_threshold)
threshold_mmm.long_name = 'multi-model mean'
threshold_sdev.long_name = 'multi-model sdev'
summary_mmm = summary_cube.collapsed('model number',iris.analysis.MEAN)
summary_mmm = mask_model_by_gpm(summary_mmm,gpm_spatial_summary)
summary_sdev = summary_cube.collapsed('model number',iris.analysis.STD_DEV)
summary_sdev = mask_model_by_gpm(summary_sdev,gpm_spatial_summary)
summary_mmm.long_name = 'multi-model mean'
summary_sdev.long_name = 'multi-model sdev'
mmm_dict = {
    'name': 'multi_model_mean_'+time_type
}
mms_dict = {
    'name': 'multi_model_sdev_'+time_type
}


# In[13]:


autocorr_mmm_raw_levels=[200,300,400,500,600,700,800,900,1000]
autocorr_mmm_diff_levels=[-350,-250,-150,-50,50,150,250,350]
plot_summary_metric(threshold_mmm,gpm_threshold,mmm_dict,gpm_dict,autocorr_mmm_raw_levels,autocorr_mmm_diff_levels,'Oranges','brewer_PRGn_11')
autocorr_sdev_levels = [50,100,150,200,250,300,350,400]
plot_summary_metric(threshold_sdev,None,mms_dict,None,autocorr_sdev_levels,None,'Oranges','brewer_PRGn_11')


# In[14]:


summary_mmm_diff_levs = [-0.22,-0.18,-0.14,-0.10,-0.06,-0.02,0.02,0.06,0.10,0.14,0.18,0.22]
plot_summary_metric(summary_mmm,gpm_spatial_summary,mmm_dict,gpm_dict,summary_raw_levs,summary_mmm_diff_levs,'Oranges','brewer_PRGn_11')
summary_sdev_levs=np.arange(0.0,0.21,0.02)
plot_summary_metric(summary_sdev,None,mms_dict,None,summary_sdev_levs,None,'Oranges','brewer_PRGn_11')


# In[ ]:





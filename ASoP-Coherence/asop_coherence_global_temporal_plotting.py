#!/usr/bin/env python
# coding: utf-8

# In[22]:


import iris
import iris.plot as iplt
import iris.quickplot as qplt
import matplotlib.pyplot as plt
import sys
sys.path.append('/home/nick/python/ASoP_global/ASoP-Coherence')
from asop_coherence_global import get_asop_dict
import matplotlib.cm as mpl_cm
import numpy as np
import numpy.ma as ma
import pandas as pd
import cartopy.crs as ccrs
from iris.experimental.equalise_cubes import equalise_attributes
from iris.util import unify_time_units


# In[ ]:


def load_summary_metric(filename,constraint,new_long_name,new_units):
    cube = iris.load_cube(filename,constraint)
    cube.long_name = new_long_name
    cube.units = new_units
    return(cube)


# In[ ]:


def add_model_coord(cube,number):
    model_coord = iris.coords.AuxCoord(m,long_name='model number',var_name='model_number')
    cube.add_aux_coord(model_coord)


# In[ ]:


def load_all_summary_metrics(asop_dict,wet_season_threshold='1d24'):
    constraints_longnames_units = [
        ('Temporal intermittency on-off metric based on 4 divisions (weighted mean of all months in wet season)','Temporal coh, 4 divs, wet seas, 3hr','1'),
        ('Probability of upper division followed by upper division (weighted mean of all months in wet season)','p(on|on), 4 divs, wet seas, 3hr','1'),
        ('Probability of upper division followed by lower division (weighted mean of all months in wet season)','p(on|off), 4 divs, wet seas, 3hr','1'),
        ('Probability of lower division followed by upper division (weighted mean of all months in wet season)','p(off|on), 4 divs, wet seas, 3hr','1'),
        ('Probability of lower division followed by lower division (weighted mean of all months in wet season)','p(off|off), 4 divs, wet seas, 3hr','1')
    ]
    out_cubelist = []
    summary_file = asop_dict['desc']+'_asop_'+asop_dict['year_range']+'_temporal_summary_wetseason'+wet_season_threshold+'.nc'
    print(asop_dict['dir']/summary_file)
    for constraint,long_name,units in constraints_longnames_units:
        cube = load_summary_metric(str(asop_dict['dir']/summary_file),constraint,long_name,units)
        out_cubelist.append(cube)
    return(out_cubelist)


# In[ ]:


def find_autocorr_threshold(cube,lag_length,threshold=0.5,long_name=None,units=None):
    import numpy as np
    #max_autocorr = cube.collapsed('lag',iris.analysis.MEAN).copy()
    lon = cube.coord('longitude')
    lat = cube.coord('latitude')
    nlon = len(lon.points)
    nlat = len(lat.points)
    nlag = len(cube.coord('lag').points)
#    max_autocorr = iris.cube.Cube(np.ma.empty((nlat,nlon)),dim_coords_and_dims=[(lat,0),(lon,1)])
    max_autocorr = np.zeros((nlat,nlon))
    for y in range(nlat):
        for x in range(nlon):
            below_threshold = np.where(cube.data[:,y,x] <= threshold)
            if np.sum(below_threshold) >= 1:
                max_autocorr[y,x] = np.amin(below_threshold)*lag_length+0.5
            else:
                max_autocorr[y,x] = np.nan #nlag*lag_length+0.5
    max_autocorr_cube = iris.cube.Cube(data=max_autocorr,dim_coords_and_dims=[(lat,0),(lon,1)],long_name=long_name,units=units,var_name='autocorr_threshold'+str(threshold))
    return(max_autocorr_cube) 


# In[26]:


def plot_summary_metric(model_cube,obs_cube,model_dict,obs_dict,raw_levs,diff_levs,raw_cmap,diff_cmap):
    fig = plt.figure()
    raw_cmap = mpl_cm.get_cmap(raw_cmap)
    diff_cmap = mpl_cm.get_cmap(diff_cmap)
    ax = plt.axes(projection=ccrs.Mollweide(central_longitude=0))
    qplt.contourf(model_cube,raw_levs,cmap=raw_cmap,axes=ax)
    #qplt.contourf(model_cube,raw_levs)
    plt.gca().coastlines(color='grey')
    fig.savefig('plots/asop_coherence_global_temporal_'+model_dict['name']+'_'+model_cube.var_name+'.png',dpi=200)
    plt.close()

    if obs_cube is not None:
        fig = plt.figure()
        ax = plt.axes(projection=ccrs.Mollweide(central_longitude=0))
        qplt.contourf(obs_cube,raw_levs,cmap=raw_cmap,axes=ax)
        #qplt.contourf(model_cube,raw_levs)
        plt.gca().coastlines(color='grey')
        fig.savefig('plots/asop_coherence_global_temporal_'+obs_dict['name']+'_'+obs_cube.var_name+'.png',dpi=200)
        plt.close()

        fig = plt.figure()
        ax = plt.axes(projection=ccrs.Mollweide(central_longitude=0))
        diff_cube = model_cube.copy()
        diff_cube.data = model_cube.data-obs_cube.data
        diff_cube.long_name=model_cube.long_name+' diff '+obs_dict['name']
    #    qplt.contourf(diff_cube,diff_levs,cmap=diff_cmap)
        qplt.contourf(diff_cube,diff_levs,cmap=diff_cmap,axes=ax)
        plt.gca().coastlines(color='grey')
        fig.savefig('plots/asop_coherence_global_temporal_'+model_dict['name']+'-minus-'+obs_dict['name']+'_'+model_cube.var_name+'.png',dpi=200)
        plt.close()


# In[ ]:


def compute_summary_stats(model_cube,obs_cube,region,region_type,region_name,diag_name,model_dict,mask=None):
    import pandas as pd
    # Average space-time summary metrics over a given region.
    # Mask for land-only if requested.
    import iris.analysis.stats as istats
    grid_constraint = iris.Constraint(latitude=lambda cell: region[0] <= cell <= region[1],
                                      longitude=lambda cell: region[2] <= cell <= region[3])
    model_region = model_cube.extract(grid_constraint)
    obs_region = obs_cube.extract(grid_constraint)
    if region_type == 'land' or region_type == 'ocean':
        if mask is None:
            raise Exception('Computing summary stats over '+region_type+' requires a mask, but mask is None.')
        mask_region = mask.extract(grid_constraint)
        mask_region.coord('longitude').guess_bounds()
        mask_region.coord('latitude').guess_bounds()
        mask_region = mask_region.regrid(model_region,iris.analysis.AreaWeighted())
        if region_type == 'land':
            model_region = model_region.copy(data=np.ma.array(model_region.data,mask=np.where(mask_region.data > 0.5,False,True)))
            obs_region = obs_region.copy(data=np.ma.array(obs_region.data,mask=np.where(mask_region.data > 0.5,False,True)))
        if region_type == 'ocean':
            model_region = model_region.copy(data=np.ma.array(model_region.data,mask=np.where(mask_region.data < 0.5,False,True)))
            obs_region = obs_region.copy(data=np.ma.array(obs_region.data,mask=np.where(mask_region.data < 0.5,False,True)))
    weights = iris.analysis.cartography.area_weights(model_region)
    model_avg = model_region.collapsed(['longitude','latitude'],iris.analysis.MEAN,weights=weights)
    obs_avg = obs_region.collapsed(['longitude','latitude'],iris.analysis.MEAN,weights=weights)
    bias = model_avg-obs_avg
    diff = model_region-obs_region
    rmse = diff.collapsed(('longitude','latitude'),iris.analysis.RMS,weights=weights)
    pcorr = istats.pearsonr(model_region,obs_region) #,corr_coords=('latitude','longitude'))
    metric_dict = pd.Series({
        'model_name': model_dict['name'],
        'diag_name': diag_name,
        'region_name': region_name,
        'bias': bias.data,
        'rmse': rmse.data,
        'pattern_corr': pcorr.data
    },name=region_name)
    return(metric_dict)

def fix_zerolon(cube):
    cube.data[:,-2] = cube.data[:,-3]
    cube.data[:,-1] = cube.data[:,-2]
    cube.data[:,0] = cube.data[:,1]
    cube.data[:,1] = cube.data[:,2]
    #cube.data[:,0] = cube.data[:,-2]
    return(cube)

def mask_model_by_gpm(model_cube,gpm_cube):
    model_mask = np.isnan(model_cube.data)
    gpm_mask = np.isnan(gpm_cube.data)
    my_mask = np.ma.mask_or(model_mask,gpm_mask)
    new_cube = model_cube.copy(data=np.ma.masked_array(model_cube.data,my_mask))
    return(new_cube)

# In[ ]:


regions = [
    ([-30,30,0,360],'both','trop'),
    ([-60,-30,0,360],'both','sh'),
    ([30,60,0,360],'both','nh'),
    ([-60,60,0,360],'both','glob'),
    ([-30,30,0,360],'land','trop_land'),
    ([-30,30,0,360],'ocean','trop_ocean'),
    ([-60,-30,0,360],'land','sh_land'),
    ([-60,-30,0,360],'ocean','sh_ocean'),
    ([30,60,0,360],'land','nh_land'),
    ([30,60,0,360],'ocean','nh_ocean'),
    ([-60,60,0,360],'land','glob_land'),
    ([-60,60,0,360],'ocean','glob_ocean')
]
mask_file='/media/nick/lacie_tb3/datasets/land_sea_mask/landfrac_n216e_hadgem3-10.3.nc'
mask = iris.load_cube(mask_file,'land_area_fraction')
metrics = pd.DataFrame()


# In[28]:

threshold = 0.2
time_type='3hr'
if time_type == 'day':
    models=['BCC','AWI','CanESM5','ACCESS','CESM2','CMCC','CNRM','FGOALS','GFDL','HadGEM3','INM','IPSL','KACE','MIROC','MPI','MRI','NESM','NorESM','SAM0-UNICON','TaiESM','UKESM1']
    units='Days'
    lag_length=1
    if threshold <= 0.3:
        autocorr_raw_levels=[1,2,3,4,5]
        autocorr_mmm_raw_levels=autocorr_raw_levels
        autocorr_diff_levels=[-3.5,-2.5,-1.5,-0.5,0.5,1.5,2.5,3.5]
        autocorr_mmm_diff_levels=autocorr_diff_levels
    else:
        autocorr_raw_levels=[3,6,9,12,15,18,21,24,27]
        autocorr_diff_levels=[-15,-13,-11,-9,-7,-5,-3,-1,1,3,5,7,9,11,13,15]
elif time_type == '3hr':
    models=['AWI','BCC','ACCESS','CMCC','CNRM','FGOALS','KACE','MRI','SAM0-UNICON']
    units='Hours'
    lag_length=3
    if threshold <= 0.3:
        autocorr_raw_levels=[3,6,9,12,15,18,21,24,27,30,33,36,39,42,45,48]
        autocorr_diff_levels=[-25,-20,-15,-11,-8,-5,-3,-1,1,3,5,8,11,15,20,25]
        autocorr_mmm_raw_levels = autocorr_raw_levels
        autocorr_mmm_diff_levels = autocorr_diff_levels
    else:
        autocorr_raw_levels=[3,6,9,12,15,18,21,24,27]
        autocorr_diff_levels=[-15,-13,-11,-9,-7,-5,-3,-1,1,3,5,7,9,11,13,15]
prob_raw_levs=[0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9]
prob_diff_levs=[-0.45,-0.35,-0.25,-0.15,-0.05,0.05,0.15,0.25,0.35,0.45]
grid_type='2x2'


# In[29]:


gpm_dict = get_asop_dict('GPM_IMERG',time=time_type,grid=grid_type)
gpm_temporal_summary,gpm_ponon,gpm_ponoff,gpm_poffon,gpm_poffoff = load_all_summary_metrics(gpm_dict)
gpm_autocorr_filename = str(gpm_dict['dir'])+'/'+gpm_dict['desc']+'_asop_'+gpm_dict['year_range']+'_temporal_autocorr_wetseason1d24.nc'
print(gpm_autocorr_filename)
gpm_autocorr = load_summary_metric(gpm_autocorr_filename,'autocorr_wetseason_precip_mean',None,'Hours')
gpm_autocorr_threshold = find_autocorr_threshold(gpm_autocorr,lag_length,threshold=threshold,long_name='Time corr >'+str(threshold)+', wet seas, '+gpm_dict['name']+' '+time_type+' '+grid_type,units=units)
gpm_temporal_summary = fix_zerolon(gpm_temporal_summary)
gpm_autocorr_threshold = fix_zerolon(gpm_autocorr_threshold)

# In[30]:


threshold_cl = iris.cube.CubeList()
summary_cl = iris.cube.CubeList()
for m,model in enumerate(models):
    model_dict = get_asop_dict(model,time=time_type,grid=grid_type)
    model_temporal_summary,model_ponon,model_ponoff,model_poffon,model_poffoff = load_all_summary_metrics(model_dict)
    autocorr_filename = str(model_dict['dir'])+'/'+model_dict['desc']+'_asop_'+model_dict['year_range']+'_temporal_autocorr_wetseason1d24.nc'
    model_autocorr = load_summary_metric(autocorr_filename,'autocorr_wetseason_precip_mean',None,'Hours')
    model_autocorr_threshold = find_autocorr_threshold(model_autocorr,lag_length,threshold=threshold,long_name='Time corr >'+str(threshold)+', wet seas, '+time_type+' '+grid_type,units=units)
    model_autocorr_threshold = mask_model_by_gpm(model_autocorr_threshold,gpm_autocorr_threshold)
    model_temporal_summary = mask_model_by_gpm(model_temporal_summary,gpm_temporal_summary)
    plot_summary_metric(model_autocorr_threshold,gpm_autocorr_threshold,model_dict,gpm_dict,autocorr_raw_levels,autocorr_diff_levels,'Oranges','brewer_PRGn_11')
    plot_summary_metric(model_temporal_summary,gpm_temporal_summary,model_dict,gpm_dict,prob_raw_levs,prob_diff_levs,'Oranges','brewer_PRGn_11')
    add_model_coord(model_autocorr_threshold,m)
    add_model_coord(model_temporal_summary,m)
    threshold_cl.append(model_autocorr_threshold)
    summary_cl.append(model_temporal_summary)

# In[ ]:

equalise_attributes(threshold_cl)
unify_time_units(threshold_cl)
threshold_cube = threshold_cl.merge_cube()
summary_cube = summary_cl.merge_cube()

threshold_mmm = threshold_cube.collapsed('model number',iris.analysis.MEAN)
threshold_mmm = mask_model_by_gpm(threshold_mmm,gpm_autocorr_threshold)
threshold_sdev = threshold_cube.collapsed('model number',iris.analysis.STD_DEV)
threshold_sdev = mask_model_by_gpm(threshold_sdev,gpm_autocorr_threshold)
threshold_mmm.long_name = 'multi-model mean'
threshold_sdev.long_name = 'multi-model sdev'
summary_mmm = summary_cube.collapsed('model number',iris.analysis.MEAN)
summary_mmm = mask_model_by_gpm(summary_mmm,gpm_temporal_summary)
summary_sdev = summary_cube.collapsed('model number',iris.analysis.STD_DEV)
summary_sdev = mask_model_by_gpm(summary_sdev,gpm_temporal_summary)
summary_mmm.long_name = 'multi-model mean'
summary_sdev.long_name = 'multi-model sdev'
mmm_dict = {
    'name': 'multi_model_mean_'+time_type
}
mms_dict = {
    'name': 'multi_model_sdev_'+time_type
}

summary_mmm_diff_levs = [-0.22,-0.18,-0.14,-0.10,-0.06,-0.02,0.02,0.06,0.10,0.14,0.18,0.22]
plot_summary_metric(summary_mmm,gpm_temporal_summary,mmm_dict,gpm_dict,prob_raw_levs,prob_diff_levs,'Oranges','brewer_PRGn_11')
summary_sdev_levs=np.arange(0.0,0.21,0.02)
plot_summary_metric(summary_sdev,None,mms_dict,None,summary_sdev_levs,None,'Oranges','brewer_PRGn_11')

plot_summary_metric(threshold_mmm,gpm_autocorr_threshold,mmm_dict,gpm_dict,autocorr_mmm_raw_levels,autocorr_mmm_diff_levels,'Oranges','brewer_PRGn_11')
autocorr_sdev_levels = [1,3,5,7,9,11,13,15,17]
plot_summary_metric(threshold_sdev,None,mms_dict,None,autocorr_sdev_levels,None,'Oranges','brewer_PRGn_11')


# In[ ]:
'''

prob_raw_levs=[0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9]
prob_diff_levs=[-0.45,-0.35,-0.25,-0.15,-0.05,0.05,0.15,0.25,0.35,0.45]
plot_summary_metric(awi_temporal_summary,gpm_temporal_summary,awi_dict,gpm_dict,prob_raw_levs,prob_diff_levs,'Oranges','brewer_PRGn_11')
plot_summary_metric(awi_ponon,gpm_ponon,awi_dict,gpm_dict,prob_raw_levs,prob_diff_levs,'Oranges','brewer_PRGn_11')
plot_summary_metric(awi_ponoff,gpm_ponoff,awi_dict,gpm_dict,prob_raw_levs,prob_diff_levs,'Oranges','brewer_PRGn_11')
plot_summary_metric(awi_poffon,gpm_poffon,awi_dict,gpm_dict,prob_raw_levs,prob_diff_levs,'Oranges','brewer_PRGn_11')
plot_summary_metric(awi_poffoff,gpm_poffoff,awi_dict,gpm_dict,prob_raw_levs,prob_diff_levs,'Oranges','brewer_PRGn_11')


# In[ ]:


test = metrics[(metrics['diag_name'] == 'temporal_summary')]
from tabulate import tabulate
print(tabulate(test[['bias','rmse','pattern_corr']]))


# In[ ]:


test = metrics[(metrics['diag_name'] == 'autocorr_threshold0.5')]
print(tabulate(test[['bias','rmse','pattern_corr']]))


# In[ ]:


test = metrics[(metrics['diag_name'] == 'autocorr_threshold0.2')]
print(tabulate(test[['bias','rmse','pattern_corr']]))


# In[ ]:




    for region,region_type,region_name in regions:
        metric_dict = compute_summary_stats(awi_autocorr_threshold,gpm_autocorr_threshold,region,region_type,region_name,'autocorr_threshold'+str(threshold),awi_dict,mask=mask)
        metrics = metrics.append(metric_dict)

'''
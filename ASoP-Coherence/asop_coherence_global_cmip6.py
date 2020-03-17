# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import sys
sys.path.append('/home/nick/python/asop_global/ASoP-Coherence')
import asop_coherence as asop
import iris
from pathlib import Path
import numpy as np
import warnings
warnings.filterwarnings("ignore")
# %%
def load_cmip6(asop_dict):
    from iris.util import unify_time_units
    from iris.experimental.equalise_cubes import equalise_attributes
    from iris.time import PartialDateTime

    cubelist = iris.load(str(asop_dict['dir'])+'/pr_3hr*.nc') # Use NetCDF3 data to save compute time
    unify_time_units(cubelist)
    equalise_attributes(cubelist)
    cube = cubelist.concatenate_cube()
    cube.coord('time').bounds = None
    constraint = iris.Constraint(time = lambda cell: PartialDateTime(year=asop_dict['start_year']) <= cell <= PartialDateTime(year=asop_dict['stop_year']),longitude = lambda cell: 60 <= cell <= 90, latitude = lambda cell: 10 <= cell <= 40)
    cube = cube.extract(constraint)
    return(cube)


# %%
def get_asop_dict(key):
    datapath=Path('/media/nick/lacie_tb31/data_from_gill/CMIP6')
    if key == 'AWI':
        asop_dict={
            'dir': datapath/'AWI-CM-1-1-MR',
            'name': 'AWI',
            'start_year': 1990,
            'stop_year': 1991,
            'dt': 10800,
            'legend_name': 'AWI',
            'region': [-90,90,0,360],
            'color': 'red',
            'symbol': '<'
        }
    elif key == 'BCC':
        asop_dict={
            'dir': datapath/'BCC-CSM2-MR',
            'name': 'BCC',
            'dt': 10800,
            'legend_name': 'BCC',
            'region': [-90,90,0,360],
            'color': 'blue',
            'symbol': '8'
        }
    else:
        raise Exception('No dictionary for '+key)
    return(asop_dict)


# %%
regions = [ ([-30,30,0,360],'land','trop_land'),
            ([-30,30,0,360],'ocean','trop_ocean'),
            ([-90,-30,0,360],'land','sh_land'),
            ([-90,-30,0,360],'ocean','sh_ocean'),
            ([30,90,0,360],'land','nh_land'),
            ([30,90,0,360],'ocean','nh_ocean'),
            ([-90,90,0,360],'land','glob_land'),
            ([-90,90,0,360],'ocean','glob_ocean')]
datasets=['AWI'] #,'BCC']
n_datasets=len(datasets)
n_regions = len(regions)
space_metrics_plot = np.empty((n_datasets,n_regions))
time_metrics_plot = np.empty((n_datasets,n_regions))
all_datasets = []
all_colors = []
all_symbols = []
all_regions = []
for box,mask_type,region_name in regions:
	all_regions.append(region_name)


# %%
#for model in datasets:
model = 'AWI'
asop_dict = get_asop_dict(model)
precip = load_cmip6(asop_dict)
#precip.convert_units('mm day-1')


# %%
def find_wet_season(precip,threshold=1.0/24.0):
    import iris.coord_categorisation
    from iris.time import PartialDateTime
    from iris.experimental.equalise_cubes import equalise_attributes
    from iris.util import unify_time_units
    
    iris.coord_categorisation.add_month_number(precip,'time')
    month_clim = precip.aggregated_by('month_number',iris.analysis.SUM)
    ann_clim = month_clim.collapsed('time',iris.analysis.SUM)
    month_frac = month_clim/ann_clim
    month_list = []
    for m,month in enumerate(month_clim.coord('month_number').points):
        if month_frac.data[m] > threshold:
            month_list.append(month)
    month_constraint = iris.Constraint(month_number=month_list) #PartialDateTime(month=month_list))
    wet_season_cube = precip.extract(month_constraint)
    return(wet_season_cube)


# %%
def compute_temporal_summary(precip,ndivs,twod=False,cyclic_lon=False,min_precip_threshold=1/86400.0):
    from tqdm import tqdm

    # Compute temporal summary metric only
    lon_coord = precip.coord('longitude')
    lat_coord = precip.coord('latitude')
    nlon = len(lon_coord.points)
    nlat = len(lat_coord.points)
    lower_thresh = iris.cube.Cube(data=np.empty((nlat,nlon)),dim_coords_and_dims=[(lat_coord,0),(lon_coord,1)])
    upper_thresh = lower_thresh.copy()
    if twod:
        time_inter = lower_thresh.copy()
        onon_freq = lower_thresh.copy()
        onoff_freq = lower_thresh.copy()
        offon_freq = lower_thresh.copy()
        offoff_freq = lower_thresh.copy()

    # Use cube slices to avoid loading all data into memory
    for t,t_slice in tqdm(enumerate(precip.slices(['time']))):
        lat = t // nlon
        lon = t % nlon
        wet_season = find_wet_season(t_slice)
        this_precip = wet_season.data[np.where(wet_season.data > min_precip_threshold)]
        nt = np.size(this_precip)
        if nt > ndivs:
            lower_thresh.data[lat,lon] = np.percentile(this_precip,25)
            upper_thresh.data[lat,lon] = np.percentile(this_precip,75)
            upper_mask = np.where(wet_season.data >= upper_thresh.data[lat,lon],1,0)
            lower_mask = np.where(wet_season.data <= lower_thresh.data[lat,lon],1,0)
            non = np.sum(upper_mask)
            noff = np.sum(lower_mask)
            onon = upper_mask + np.roll(upper_mask,1)
            onon_count = np.count_nonzero(np.where(onon == 2,1,0))
            onoff = upper_mask + np.roll(lower_mask,1)
            onoff_count = np.count_nonzero(np.where(onoff == 2,1,0))
            offon = lower_mask + np.roll(upper_mask,1)
            offon_count = np.count_nonzero(np.where(offon == 2,1,0))
            offoff = lower_mask + np.roll(lower_mask,1)
            offoff_count = np.count_nonzero(np.where(offoff == 2,1,0))
            onon_freq.data[lat,lon] = onon_count/float(non)
            onoff_freq.data[lat,lon] = onoff_count/float(non)
            offon_freq.data[lat,lon] = offon_count/float(noff)
            offoff_freq.data[lat,lon] = offoff_count/float(noff)
        else:
            lower_thresh.data[lat,lon] = np.nan
            upper_thresh.data[lat,lon] = np.nan
    
    time_inter = 0.5*((onon_freq+offoff_count)-(onoff_count+offon_count))
    out_cubelist = iris.cube.CubeList()
    out_cubelist.append(time_inter,onon_freq,onoff_freq,offon_freq,offoff_freq,lower_thresh,upper_thresh)
    return(out_cubelist)


# %%
time_inter = compute_temporal_summary(precip,4,twod=True,cyclic_lon=True)
iris.save(time_inter,'asop_coherence_global_cmip6_timeinter.nc')


# %%



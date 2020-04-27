import iris
from pathlib import Path
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import iris.coord_categorisation
from iris.experimental.equalise_cubes import equalise_attributes
from iris.util import unify_time_units
import dask
from dask.distributed import Client,progress
import os

def load_cmip6(asop_dict):
    from iris.util import unify_time_units
    from iris.experimental.equalise_cubes import equalise_attributes
    from iris.time import PartialDateTime

    constraint = iris.Constraint(time = lambda cell: PartialDateTime(year=asop_dict['start_year']) <= cell <= PartialDateTime(year=asop_dict['stop_year']),latitude = lambda cell: -60 <= cell <= 60)
    cubelist = iris.load(str(asop_dict['dir'])+'/'+asop_dict['file_pattern']) # Use NetCDF3 data to save compute time
    unify_time_units(cubelist)
    equalise_attributes(cubelist)
    cube = cubelist.concatenate_cube()
    cube.coord('time').bounds = None
    out_cube = cube.extract(constraint)
    return(out_cube)

def get_asop_dict(key):
    cmip6_path=Path('/media/nick/lacie_tb3/data_from_gill/CMIP6')
    obs_path=Path('/media/nick/lacie_tb3/datasets')
    if key == 'AWI':
        asop_dict={
            'desc': 'AWI-CM-1-1-MR_historical_r1i1p1f1_gn_3hr_3x3',
            'dir': cmip6_path/'AWI-CM-1-1-MR',
            'file_pattern': 'pr_3hr*.3x3.nc',
            'name': 'AWI',
            'start_year': 1990,
            'stop_year': 2014,
            'dt': 10800,
            'legend_name': 'AWI',
            'region': [-90,90,0,360],
            'color': 'red',
            'symbol': '<'
        }
    elif key == 'BCC':
        asop_dict={
            'dir': cmip6_path/'BCC-CSM2-MR',
            'desc': 'BCC-CSM2-MR_historical_r1i1p1f1_gn_3hr_3x3',
            'name': 'BCC',
            'file_pattern': 'pr_3hr*.3x3.nc',
            'dt': 10800,
            'start_year': 1980,
            'stop_year': 2014,
            'legend_name': 'BCC',
            'region': [-90,90,0,360],
            'color': 'blue',
            'symbol': '8'
        }
    elif key == 'GPM_IMERG':
        asop_dict={
            'desc': '3B-HHR.MS.MRG.3IMERG.V06B.3hr_means_3x3',
            'dir': obs_path/'GPM_IMERG',
            'file_pattern': '3B-HHR.MS.MRG.3IMERG.*.3hr_means_3x3.V06B.nc',
            'name': 'IMERG-3B-V06',
            'start_year': 2001,
            'stop_year': 2018,
            'dt': 10800,
            'legend_name': 'IMERG',
            'region': [-60,60,0,360],
            'color': 'black',
            'symbol': '>'
        }
    elif key == 'ACCESS':
        asop_dict={
            'desc': 'ACCESS-CM2_historical_r1i1p1f1_gn_3hr_3x3',
            'dir': cmip6_path/'ACCESS-CM2',
            'name': 'ACCESS',
            'file_pattern': 'pr_3hr*.3x3.nc',
            'dt': 10800,
            'start_year': 1990,
            'stop_year': 2014,
            'legend_name': 'ACCESS',
            'region': [-90,90,0,360],
            'color': 'purple',
            'symbol': '>'
        }
    elif key == 'FGOALS':
        asop_dict={
            'desc': 'FGOALS-g3_historical_r1i1p1f1_gn_3hr_3x3',
            'dir': cmip6_path/'FGOALS-g3',
            'name': 'FGOALS',
            'dt': 10800,
            'file_pattern': 'pr_3hr*.3x3.nc',
            'start_year': 1990,
            'stop_year': 2016,
            'legend_name': 'FGOALS',
            'region': [-90,90,0,360],
            'color': 'brown',
            'symbol': '<'
        }
    elif key == 'GISS':
        asop_dict={
            'dir': cmip6_path/'GISS-E2-1-G',
            'name': 'GISS',
            'desc': 'GISS-E2-1-G_historical_r1i1p1f1_gn_3hr_3x3',
            'dt': 10800,
            'file_pattern': 'pr_3hr*.3x3.nc',
            'start_year': 1990,
            'stop_year': 2014,
            'legend_name': 'GISS',
            'region': [-90,90,0,360],
            'color': 'brown',
            'symbol': '<'
        }
    elif key == 'MIROC':
        asop_dict={
            'dir': cmip6_path/'MIROC6',
            'name': 'MIROC',
            'desc': 'MIROC6_historical_r1i1p1f1_gn_3hr_3x3',
            'dt': 10800,
            'file_pattern': 'pr_3hr*.3x3.nc',
            'start_year': 1990,
            'stop_year': 2014,
            'legend_name': 'MIROC6',
            'region': [-90,90,0,360],
            'color': 'brown',
            'symbol': '<'
        }
    elif key == 'MPI-ESM1':
        asop_dict={
            'dir': cmip6_path/'MPI-ESM1-2-HR',
            'name': 'MPI-ESM',
            'desc': 'MPI-ESM1-2-HR_historical_r1i1p1f1_gn_3hr_3x3',
            'dt': 10800,
            'file_pattern': 'pr_3hr*.3x3.nc',
            'start_year': 1990,
            'stop_year': 2014,
            'legend_name': 'MPI-ESM',
            'region': [-90,90,0,360],
            'color': 'brown',
            'symbol': '<'
        }
    elif key == 'SAM0-UNICON':
        asop_dict={
            'dir': cmip6_path/'SAM0-UNICON',
            'name': 'SAM',
            'desc': 'SAM0-UNICON_historical_r1i1p1f1_gn_3hr_3x3',
            'dt': 10800,
            'file_pattern': 'pr_3hr*.3x3.nc',
            'start_year': 1990,
            'stop_year': 2014,
            'legend_name': 'SAM',
            'region': [-90,90,0,360],
            'color': 'brown',
            'symbol': '<'
        }
    else:
        raise Exception('No dictionary for '+key)
    return(asop_dict)

def new_cube_copy(cube,var_name,long_name):
    new_cube = cube.copy()
    new_cube.var_name=var_name
    new_cube.long_name=long_name
    return(new_cube)

def mask_wet_season(precip,wet_season_threshold=1.0/24.0):
    import numpy.ma as ma

    if not 'month_number' in [coord.name() for coord in precip.coords()]:
        iris.coord_categorisation.add_month_number(precip,'time')
    if not 'year' in [coord.name() for coord in precip.coords()]:
        iris.coord_categorisation.add_year(precip,'time')
    ann_total = precip.aggregated_by('year',iris.analysis.SUM)
    ann_clim = ann_total.collapsed('time',iris.analysis.MEAN)

    cubelist = iris.cube.CubeList()
    nt = len(precip.coord('time').points)
    nlon = len(precip.coord('longitude').points)
    nlat = len(precip.coord('latitude').points)

    month_total = precip.aggregated_by(['month_number','year'],iris.analysis.SUM)
    month_clim = month_total.aggregated_by(['month_number'],iris.analysis.MEAN)
    precip_mask = precip.copy(data=np.empty((nt,nlat,nlon)))
    for m,month in enumerate(set(precip.coord('month_number').points)):
        month_frac = month_clim[m,:,:]/ann_clim
        month_mask = np.where(month_frac.data < wet_season_threshold,True,False)
        ntm = len(precip[np.where(precip.coord('month_number').points == month)].data)
        month_mask_repeat = np.broadcast_to(month_mask,(ntm,nlat,nlon))
        precip_mask.data[np.where(precip.coord('month_number').points == month)] = month_mask_repeat
    masked_precip = precip.copy(data=(ma.array(precip.data,mask=precip_mask.data)))
    return(masked_precip)

def mask_min_precip(precip,min_precip_threshold=1.0):
    import numpy.ma as ma
    from cfunits import Units

    if precip.units == 'mm/hr':
        input_units = 'mm/h'
    elif precip.units == 'kg m-2 s-1':
        input_units = 'mm/s'
    else:
        input_units = precip.units
    threshold_units = Units.conform(min_precip_threshold,Units('mm/day'),Units(input_units)) #precip.units))
    nt = len(precip.coord('time').points)
    nlon = len(precip.coord('longitude').points)
    nlat = len(precip.coord('latitude').points)
    precip_mask = precip.copy(data=np.empty((nt,nlat,nlon)))
    precip_mean = precip.collapsed('time',iris.analysis.MEAN)
    for y in range(nlat):
        for x in range(nlon):
            if precip_mean.data[y,x] < threshold_units:
                precip_mask.data[:,y,x] = 1
            else:
                precip_mask.data[:,y,x] = 0
    masked_precip = precip.copy(data=ma.array(precip.data,mask=precip_mask.data))
    return(masked_precip)

    print(threshold_units)
#    ann_mean

def compute_autocorr_grid(precip,lag):
    import iris.analysis.stats as istats
    import numpy as np
    lagged_slice = precip.copy(data=np.roll(precip.data,lag,0))
    output = istats.pearsonr(precip,lagged_slice,corr_coords='time')
    return(output.data)

def compute_temporal_autocorr(precip,max_lag):
    import numpy.ma as ma
     # Compute temporal summary metric only
    if not 'month_number' in [coord.name() for coord in precip.coords()]:
        iris.coord_categorisation.add_month_number(precip,'time')
    if not 'year' in [coord.name() for coord in precip.coords()]:
        iris.coord_categorisation.add_year(precip,'time')
    lon_coord = precip.coord('longitude')
    lat_coord = precip.coord('latitude')
    nlon = len(lon_coord.points)
    nlat = len(lat_coord.points)
    months = sorted(set(precip.coord('month_number').points))
    month_coord = iris.coords.DimCoord(months,var_name='month_number')
    lag_coord = iris.coords.DimCoord(range(max_lag),var_name='lag')
    nmonths = len(month_coord.points)

    temporal_autocorr = iris.cube.Cube(data=np.empty((nmonths,max_lag,nlat,nlon)),dim_coords_and_dims=[(month_coord,0),(lag_coord,1),(lat_coord,2),(lon_coord,3)])
    temporal_autocorr = temporal_autocorr.copy(data=temporal_autocorr.data.fill(np.nan)) #temporal_autocorr.data[:,:,:,:].fill(np.nan)
    temporal_autocorr.var_name='autocorr_wetseason_precip'
    temporal_autocorr.long_name='Auto-correlation of precipitation masked for the wet season'
    for m,month in enumerate(months):
        dask_autocorr = []
        print('-->-->--> Month '+str(month))
        month_constraint = iris.Constraint(month_number=month)
        this_month = precip.extract(month_constraint)
        years = set(this_month.coord('year').points)
        nyears = len(years)
        for year in years:
            year_constraint = iris.Constraint(year=year)
            this_monthyear = dask.delayed(this_month.extract(year_constraint))
            this_autocorr = [dask.delayed(compute_autocorr_grid)(this_monthyear,lag) for lag in range(max_lag)]
            dask_autocorr.append(this_autocorr)
        result = dask.compute(*dask_autocorr)
        result = np.ma.asarray(result)
        result[np.where(result == 0.0)].mask = True
        result[np.where(result == 0.0)] = np.nan
        temporal_autocorr.data[m,:,:,:] = np.nanmean(result[:,:,:,:],axis=0)
    temporal_autocorr_mean = temporal_autocorr.collapsed('month_number',iris.analysis.MEAN,mdtol=0)
    temporal_autocorr_mean.var_name='autocorr_wetseason_precip_mean'
    temporal_autocorr_mean.long_name='Auto-correlation of precipitation masked for the wet season (mean of all months in wet season)'
    temporal_autocorr_mean.data = np.nanmean(temporal_autocorr.data,axis=0)
    out_cubelist = [temporal_autocorr,temporal_autocorr_mean]
    return(out_cubelist)

def compute_temporal_summary(precip,ndivs,min_precip_threshold=1/86400.0,wet_season_threshold=1.0/24.0):
    import numpy.ma as ma
    import dask

    # Compute temporal summary metric only
    if not 'month_number' in [coord.name() for coord in precip.coords()]:
        iris.coord_categorisation.add_month_number(precip,'time')
    lon_coord = precip.coord('longitude')
    lat_coord = precip.coord('latitude')
    nlon = len(lon_coord.points)
    nlat = len(lat_coord.points)

    months = sorted(set(precip.coord('month_number').points))
    month_coord = iris.coords.DimCoord(months,var_name='month_number')
    nmonths = len(months)
    
    lower_thresh = iris.cube.Cube(data=np.ma.zeros((nmonths,nlat,nlon)),dim_coords_and_dims=[(month_coord,0),(lat_coord,1),(lon_coord,2)])
    lower_thresh.var_name='lower_threshold'
    lower_thresh.long_name='Lower (off) threshold based on '+str(ndivs)+' divisions'
    upper_thresh = new_cube_copy(lower_thresh,'upper_threshold','Upper (on) threshold based on '+str(ndivs)+' divisions')
    time_inter = new_cube_copy(lower_thresh,'temporal_onoff_metric','Temporal intermittency on-off metric based on '+str(ndivs)+' divisions')
    onon_freq = new_cube_copy(lower_thresh,'prob_onon','Probability of upper division followed by upper division')
    onoff_freq = new_cube_copy(lower_thresh,'prob_onoff','Probability of upper division followed by lower division')
    offon_freq = new_cube_copy(lower_thresh,'prob_offon','Probability of lower division followed by upper division')
    offoff_freq = new_cube_copy(lower_thresh,'prob_offoff','Probability of lower division followed by lower division')

    for m,month in enumerate(months):
        print('-->-->--> Month '+str(month))
        month_summaries=[]
        month_constraint = iris.Constraint(month_number=month)
        this_month = precip.extract(month_constraint)
        lower_thresh.data[m,:,:] = this_month.collapsed('time',iris.analysis.PERCENTILE,percent=100.0/ndivs).data
        upper_thresh.data[m,:,:] = this_month.collapsed('time',iris.analysis.PERCENTILE,percent=100.0*(1.0-1.0/ndivs)).data
        years = set(this_month.coord('year').points)
        nyears = len(years)
        for year in years:
            this_monthyear = dask.delayed(this_month.extract(iris.Constraint(year=year)))
            monthyear_summary = dask.delayed(compute_onoff_metric_grid)(this_monthyear,lower_thresh[m,:,:],upper_thresh[m,:,:])
            month_summaries.append(monthyear_summary)
        result = dask.compute(*month_summaries)
        result = np.ma.asarray(result)
        print(np.shape(result))

        onon_freq.data[m,:,:] = np.nanmean(result[:,0,:,:],axis=0)
        onoff_freq.data[m,:,:] = np.nanmean(result[:,1,:,:],axis=0)
        offon_freq.data[m,:,:] = np.nanmean(result[:,2,:,:],axis=0)
        offoff_freq.data[m,:,:] = np.nanmean(result[:,3,:,:],axis=0)
    
    onon_freq.data.mask = upper_thresh.data.mask
    offon_freq.data.mask = upper_thresh.data.mask
    onoff_freq.data.mask = upper_thresh.data.mask
    offoff_freq.data.mask = upper_thresh.data.mask

    time_inter.data = 0.5*((onon_freq.data+offoff_freq.data)-(onoff_freq.data+offon_freq.data))
    time_inter.data.mask = upper_thresh.data.mask
    time_inter_mean = time_inter.collapsed('month_number',iris.analysis.MEAN,mdtol=0)  
    time_inter_mean.data = np.nanmean(time_inter.data,axis=0)
    time_inter_mean.var_name='temporal_onoff_metric_mean'
    time_inter_mean.long_name='Temporal intermittency on-off metric based on '+str(ndivs)+' divisions (mean of all months in wet season)'
    onon_freq_mean = onon_freq.collapsed('month_number',iris.analysis.MEAN)
    onon_freq_mean.data = np.nanmean(onon_freq.data,axis=0)
    onon_freq_mean.var_name='prob_onon_mean'
    onon_freq_mean.long_name='Probability of upper division followed by upper division (mean of all months in wet season)'
    onoff_freq_mean = onoff_freq.collapsed('month_number',iris.analysis.MEAN)
    onoff_freq_mean.data = np.nanmean(onoff_freq.data,axis=0)
    onoff_freq_mean.var_name='prob_onoff_mean'
    onoff_freq_mean.long_name='Probability of upper division followed by lower division (mean of all months in wet season)'
    offon_freq_mean = offon_freq.collapsed('month_number',iris.analysis.MEAN)
    offon_freq_mean.data = np.nanmean(offon_freq.data,axis=0)
    offon_freq_mean.var_name='prob_offon_mean'
    offon_freq_mean.long_name='Probability of lower division followed by upper division (mean of all months in wet season)'
    offoff_freq_mean = offoff_freq.collapsed('month_number',iris.analysis.MEAN)
    offoff_freq_mean.data = np.nanmean(offoff_freq.data,axis=0)
    offoff_freq_mean.var_name='prob_offoff_mean'
    offoff_freq_mean.long_name='Probability of lower division followed by lower division (mean of all months in wet season)'
    out_cubelist = [time_inter,onon_freq,onoff_freq,offon_freq,offoff_freq,lower_thresh,upper_thresh,time_inter_mean,onon_freq_mean,onoff_freq_mean,offon_freq_mean,offoff_freq_mean]
    return(out_cubelist)

def compute_onoff_metric_grid(this_monthyear,lower_thresh,upper_thresh):
    import numpy as np

    upper_mask = this_monthyear.copy(data=np.where(this_monthyear.data >= upper_thresh.data,1,0))
    lower_mask = this_monthyear.copy(data=np.where(this_monthyear.data <= lower_thresh.data,1,0))
    upper_roll = upper_mask.copy(data=np.roll(upper_mask.data,1,axis=0))
    lower_roll = lower_mask.copy(data=np.roll(lower_mask.data,1,axis=0))
    non = upper_mask.collapsed('time',iris.analysis.SUM)
    noff = lower_mask.collapsed('time',iris.analysis.SUM)

    onon = upper_mask + upper_roll
    onon_count = onon.collapsed('time',iris.analysis.COUNT,function=lambda values: values == 2) / non
    onon_count.var_name = 'onon_count'
    onoff = upper_mask + lower_roll
    onoff_count = onoff.collapsed('time',iris.analysis.COUNT,function=lambda values: values == 2) / non
    onoff_count.var_name = 'onoff_count'
    offon = lower_mask + upper_roll
    offon_count = offon.collapsed('time',iris.analysis.COUNT,function=lambda values: values == 2) / noff
    offon_count.var_name = 'offon_count'
    offoff = lower_mask + lower_roll
    offoff_count = offoff.collapsed('time',iris.analysis.COUNT,function=lambda values: values == 2) / noff
    offoff_count.var_name = 'offoff_count'

    output = np.stack([onon_count.data,onoff_count.data,offon_count.data,offoff_count.data],axis=0)
    return(output)

if __name__ == '__main__':
    client = Client()
#    datasets=['BCC','GPM_IMERG','AWI']
    datasets=['ACCESS','FGOALS','GISS','MIROC','MPI-ESM1','SAM0-UNICON']
    n_datasets=len(datasets)
    wet_season_threshold = 1.0/24.0
    wet_season_threshold_str='1d24'
    min_precip_threshold = 1.0 # mm/day
    min_precip_threshold_str='1mmpd'

    masked_overwrite=True
    for model in datasets:
        print('--> '+model)
        asop_dict = get_asop_dict(model)

        masked_precip_file=str(asop_dict['dir'])+'/'+asop_dict['desc']+'_asop_masked_precip_wetseason'+wet_season_threshold_str+'.nc'
        masked_min_precip_file=str(asop_dict['dir'])+'/'+asop_dict['desc']+'_asop_masked_precip_wetseason'+wet_season_threshold_str+'_minprecip'+min_precip_threshold_str+'.nc'
        temporal_summary_file=str(asop_dict['dir'])+'/'+str(asop_dict['desc'])+'_asop_temporal_summary_wetseason'+wet_season_threshold_str+'.nc'
        temporal_autocorr_file=str(asop_dict['dir'])+'/'+str(asop_dict['desc'])+'_asop_temporal_autocorr_wetseason'+wet_season_threshold_str+'.nc'

        if os.path.exists(masked_min_precip_file) and not masked_overwrite:
            print('-->--> Reading masked precipitation (by wet season and by min precip) from file')
            masked_min_precip = iris.load_cube(masked_min_precip_file)
        elif os.path.exists(masked_precip_file) and not masked_overwrite:
            print('-->--> Reading masked precipitation (by wet season) from file')
            masked_precip = iris.load_cube(masked_precip_file)
            print('-->--> Masking precip for minimum rain rate')
            masked_min_precip = mask_min_precip(masked_precip,min_precip_threshold=min_precip_threshold)
            masked_min_precip.var_name='precipitation_flux_masked'
            masked_min_precip.long_name='Masked precipitation for wet season (threshold '+wet_season_threshold_str+' of annual total) and min mean precip (threshold '+min_precip_threshold_str+' mm/day)'
            with dask.config.set(scheduler='synchronous'):
                iris.save(masked_min_precip,masked_min_precip_file)
        else:
            print('-->--> Reading precipitation ')
            precip = load_cmip6(asop_dict)
            print('-->--> Masking precipitation for wet season')
            masked_precip = mask_wet_season(precip)
            masked_precip.var_name='precipitation_flux_masked'
            masked_precip.long_name='Masked precipitation for wet season (threshold '+wet_season_threshold_str+' of annual total)'
            with dask.config.set(scheduler='synchronous'):
                iris.save(masked_precip,masked_precip_file)
            print('-->--> Masking precipitation for minimum rain rate')
            masked_min_precip = mask_min_precip(masked_precip,min_precip_threshold=min_precip_threshold)
            masked_min_precip.var_name='precipitation_flux_masked'
            masked_min_precip.long_name='Masked precipitation for wet season (threshold '+wet_season_threshold_str+' of annual total) and min mean precip (threshold '+min_precip_threshold_str+' mm/day)'
            with dask.config.set(scheduler='synchronous'):
                iris.save(masked_min_precip,masked_min_precip_file)
        print('-->--> Computing temporal autocorrelation metrics')
        temporal_autocorr = compute_temporal_autocorr(masked_min_precip,17)
        print('-->--> Computing temporal summary metrics')
        temporal_summary = compute_temporal_summary(masked_min_precip,4)
        with dask.config.set(scheduler='synchronous'):
            iris.save(temporal_summary,temporal_summary_file)
            iris.save(temporal_autocorr,temporal_autocorr_file)
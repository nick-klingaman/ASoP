from numba import jit
import numpy as np
import iris

def get_asop_dict(key,time=None,grid=''):
    from pathlib import Path
    cmip6_path=Path('/media/nick/lacie_tb3/cmip6')
    obs_path=Path('/media/nick/lacie_tb3/datasets')
    print(key)
    if grid is not '':
        grid_str=grid+'.'
    else:   
        grid_str=''
    if time == '3hr':
        last_digit = '30'
    elif time == 'day':
        last_digit = '31'
    if key == 'GPM_IMERG':
        asop_dict={
            'desc': '3B-HHR.MS.MRG.3IMERG.V06B.'+time,
            'dir': obs_path/'GPM_IMERG'/time,
            'file_pattern': '3B-HHR.MS.3IMERG.*_means_'+grid_str+'.V06*.nc',
            'name': 'GPM_IMERG_'+time,
            'start_year': 2001,
            'stop_year': 2019,
            'legend_name': 'GPM_IMERG_'+time,
            'region': [-60,60,0,360]
        }
        if time == '3hr':
            asop_dict['file_pattern'] = '3B-HHR.MS.MRG.3IMERG.*_means_'+grid_str+'V06*.nc'
        elif time == 'day':
            asop_dict['file_pattern'] = '3B-DAY.MS.MRG.3IMERG.*.V06.'+grid_str+'nc'
    elif key == 'AWI':
        asop_dict={
            'desc': 'AWI-CM-1-1-MR_historical_r1i1p1f1_gn_'+time,
            'dir': cmip6_path/'AWI-CM-1-1-MR',
            'file_pattern': 'pr_'+time+'*0.'+grid_str+'nc',
            'name': 'AWI_CM-1-1-MR_'+time,
            'start_year': 1980,
            'stop_year': 2014,
            'legend_name': 'AWI_'+time,
            'region': [-60,60,0,360]
        }
    elif key == 'BCC':
        asop_dict={
            'dir': cmip6_path/'BCC-CSM2-MR',
            'desc': 'BCC-CSM2-MR_historical_r1i1p1f1_gn_'+time,
            'name': 'BCC-CSM2-MR_'+time,
            'file_pattern': 'pr_'+time+'*'+last_digit+'.'+grid_str+'nc',
            'start_year': 1980,
            'stop_year': 2014,
            'legend_name': 'BCC_'+time,
            'region': [-60,60,0,360]
        }
    elif key == 'CanESM5':
        asop_dict={
            'dir': cmip6_path/'CanESM5',
            'desc': 'CanESM5_historical_r1i1p1f1_gn_'+time,
            'name': 'CanESM5_'+time,
            'file_pattern': 'pr_'+time+'*'+last_digit+'.'+grid_str+'nc',
            'start_year': 1980,
            'stop_year': 2014,
            'legend_name': 'CanESM5_'+time,
            'region': [-60,60,0,360]
        }
    elif key == 'ACCESS-CM2':
        asop_dict={
            'desc': 'ACCESS-CM2_historical_r1i1p1f1_gn_'+time,
            'dir': cmip6_path/'ACCESS-CM2',
            'name': 'ACCESS-CM2_'+time,
            'file_pattern': 'pr_'+time+'*30.'+grid_str+'nc',
            'start_year': 1980,
            'stop_year': 2014,
            'legend_name': 'ACCESS-CM2_'+time,
            'region': [-60,60,0,360]
        }
    elif key == 'CESM2':
        asop_dict={
            'desc': 'CESM2_historical_r1i1p1f1_gn_'+time,
            'dir': cmip6_path/'CESM2',
            'name': 'CESM2_'+time,
            'file_pattern': 'pr_'+time+'*'+last_digit+'.'+grid_str+'nc',
            'start_year': 1980,
            'stop_year': 2014,
            'legend_name': 'CESM2_'+time,
            'region': [-60,60,0,360]
        }
    elif key == 'CMCC':
        asop_dict={
            'desc': 'CMCC-CM2-SR5_historical_r1i1p1f1_gn_'+time,
            'dir': cmip6_path/'CMCC-CM2-SR5',
            'name': 'CMCC-CM2-SR5_'+time,
            'file_pattern': 'pr_'+time+'*'+last_digit+'.'+grid_str+'nc',
            'start_year': 1980,
            'stop_year': 2014,
            'legend_name': 'CMCC-CM2-SR5_'+time,
            'region': [-60,60,0,360]
        }
    elif key == 'CNRM':
        asop_dict={
            'desc': 'CNRM-CM6-1-HR_historical_r1i1p1f1_gn_'+time,
            'dir': cmip6_path/'CNRM-CM6-1-HR',
            'name': 'CNRM-CM6-1-HR_'+time,
            'file_pattern': 'pr_'+time+'*'+last_digit+'.'+grid_str+'nc',
            'start_year': 1980,
            'stop_year': 2014,
            'legend_name': 'CNRM-CM6-1-HR_'+time,
            'region': [-60,60,0,360]
        }
    elif key == 'FGOALS':
        asop_dict={
            'desc': 'FGOALS-g3_historical_r1i1p1f1_gn_'+time,
            'dir': cmip6_path/'FGOALS-g3',
            'name': 'FGOALS-g3_'+time,
            'file_pattern': 'pr_'+time+'*'+last_digit+'.'+grid_str+'nc',
            'start_year': 1980,
            'stop_year': 2014,
            'legend_name': 'FGOALS-g3_'+time,
            'region': [-60,60,0,360]
        }
    elif key == 'GFDL-CM4':
        asop_dict={
            'desc': 'GFDL-CM4_historical_r1i1p1f1_gr1_'+time,
            'dir': cmip6_path/'GFDL-CM4',
            'name': 'GFDL-CM4_'+time,
            'file_pattern': 'pr_'+time+'*_gr1_*'+last_digit+'.'+grid_str+'nc',
            'start_year': 1980,
            'stop_year': 2014,
            'legend_name': 'GFDL-CM4_'+time,
            'region': [-60,60,0,360]
        }
    elif key == 'GFDL-ESM4':
        asop_dict={
            'desc': 'GFDL-ESM4_historical_r1i1p1f1_gr1_'+time,
            'dir': cmip6_path/'GFDL-ESM4',
            'name': 'GFDL-ESM4_'+time,
            'file_pattern': 'pr_'+time+'*_gr1_*'+last_digit+'.'+grid_str+'nc',
            'start_year': 1980,
            'stop_year': 2014,
            'legend_name': 'GFDL-ESM4_'+time,
            'region': [-60,60,0,360]
        }
    elif key == 'GISS':
        asop_dict={
            'dir': cmip6_path/'GISS-E2-1-G',
            'name': 'GISS-E2-1-G_'+time,
            'desc': 'GISS-E2-1-G_historical_r1i1p1f1_gn_'+time,
            'file_pattern': 'pr_'+time+'*30.'+grid_str+'nc',
            'start_year': 1980,
            'stop_year': 2014,
            'legend_name': 'GISS-E2-1-G_'+time,
            'region': [-60,60,0,360]
        }
    elif key == 'HadGEM3':
        asop_dict={
            'desc': 'HadGEM3-GC31-MM_historical_r1i1p1f3_gn_'+time,
            'dir': cmip6_path/'HadGEM3-GC31-MM',
            'name': 'HadGEM3-GC31-MM_'+time,
            'file_pattern': 'pr_'+time+'*30.'+grid_str+'nc',
            'start_year': 1980,
            'stop_year': 2014,
            'legend_name': 'HadGEM3-GC31-MM_'+time,
            'region': [-60,60,0,360]
        }
    elif key == 'INM':
        asop_dict={
            'desc': 'INM-CM5-0_historical_r1i1p1f1_gr1_'+time,
            'dir': cmip6_path/'INM-CM5-0',
            'name': 'INM-CM-5-0_'+time,
            'file_pattern': 'pr_'+time+'*'+last_digit+'.'+grid_str+'nc',
            'start_year': 1980,
            'stop_year': 2014,
            'legend_name': 'INM-CM5_'+time,
            'region': [-60,60,0,360]
        }
    elif key == 'IPSL':
        asop_dict={
            'desc': 'IPSL-CM6A-LR_historical_r1i1p1f1_gr1_'+time,
            'dir': cmip6_path/'IPSL-CM6A-LR',
            'name': 'IPSL-CM6A-LR_'+time,
            'file_pattern': 'pr_'+time+'*'+last_digit+'.'+grid_str+'nc',
            'start_year': 1980,
            'stop_year': 2014,
            'legend_name': 'IPSL-CM6A-LR_'+time,
            'region': [-60,60,0,360]
        }
    elif key == 'KACE':
        asop_dict={
            'desc': 'KACE-1-0-G_historical_r1i1p1f1_gr_'+time,
            'dir': cmip6_path/'KACE-1-0-G',
            'name': 'KACE-1-0-G_'+time,
            'file_pattern': 'pr_'+time+'*30.'+grid_str+'nc',
            'start_year': 1980,
            'stop_year': 2014,
            'legend_name': 'KACE-1-0-G_'+time,
            'region': [-60,60,0,360]
        }
    elif key == 'MIROC':
        asop_dict={
            'dir': cmip6_path/'MIROC6',
            'name': 'MIROC6_'+time,
            'desc': 'MIROC6_historical_r1i1p1f1_gn_'+time,
            'file_pattern': 'pr_'+time+'*'+last_digit+'.'+grid_str+'nc',
            'start_year': 1980,
            'stop_year': 2014,
            'legend_name': 'MIROC6_'+time,
            'region': [-60,60,0,360]
        }
    elif key == 'MPI':
        asop_dict={
            'dir': cmip6_path/'MPI-ESM1-2-HR',
            'name': 'MPI-ESM1-2-HR_'+time,
            'desc': 'MPI-ESM1-2-HR_historical_r1i1p1f1_gn_'+time,
            'file_pattern': 'pr_'+time+'*'+last_digit+'.'+grid_str+'nc',
            'start_year': 1980,
            'stop_year': 2014,
            'legend_name': 'MPI-ESM1-2-HR_'+time,
            'region': [-60,60,0,360]
        }
    elif key == 'MRI':
        asop_dict={
            'dir': cmip6_path/'MRI-ESM2-0',
            'name': 'MRI-ESM2-0_'+time,
            'desc': 'MPI-ESM2-0_historical_r1i1p1f1_gn_'+time,
            'file_pattern': 'pr_'+time+'*'+last_digit+'.'+grid_str+'nc',
            'start_year': 1980,
            'stop_year': 2014,
            'legend_name': 'MRI-ESM2-0_'+time,
            'region': [-60,60,0,360]
        }
    elif key == 'NESM':
        asop_dict={
            'dir': cmip6_path/'NESM3',
            'name': 'NESM3_'+time,
            'desc': 'NESM3_historical_r1i1p1f1_gn_'+time,
            'file_pattern': 'pr_'+time+'*'+last_digit+'.'+grid_str+'nc',
            'start_year': 1980,
            'stop_year': 2014,
            'legend_name': 'NESM3_'+time,
            'region': [-60,60,0,360]
        }
    elif key == 'NorESM':
        asop_dict={
            'dir': cmip6_path/'NorESM2-MM',
            'name': 'NorESM2-MM_'+time,
            'desc': 'NorESM2-MM_historical_r1i1p1f1_gn_'+time,
            'file_pattern': 'pr_'+time+'*'+last_digit+'.'+grid_str+'nc',
            'start_year': 1980,
            'stop_year': 2014,
            'legend_name': 'NorESM2-MM_'+time,
            'region': [-60,60,0,360]
        }
    elif key == 'SAM0-UNICON':
        asop_dict={
            'dir': cmip6_path/'SAM0-UNICON',
            'name': 'SAM_'+time,
            'desc': 'SAM0-UNICON_historical_r1i1p1f1_gn_'+time,
            'file_pattern': 'pr_'+time+'*'+last_digit+'.'+grid_str+'nc',
            'start_year': 1980,
            'stop_year': 2014,
            'legend_name': 'SAM0-UNICON_'+time,
            'region': [-60,60,0,360]
        }
    elif key == 'TaiESM':
        asop_dict={
            'dir': cmip6_path/'TaiESM1',
            'name': 'TaiESM1_'+time,
            'desc': 'TaiESM1_historical_r1i1p1f1_gn_'+time,
            'file_pattern': 'pr_'+time+'*'+last_digit+'.'+grid_str+'nc',
            'start_year': 1980,
            'stop_year': 2014,
            'legend_name': 'TaiESM1_'+time,
            'region': [-60,60,0,360]
        }
    elif key == 'UKESM1':
        asop_dict={
            'dir': cmip6_path/'UKESM1-0-LL',
            'name': 'UKESM1-0-LL_'+time,
            'desc': 'UKESM1-0-LL_historical_r1i1p1f2_gn_'+time,
            'file_pattern': 'pr_'+time+'*30.'+grid_str+'nc',
            'start_year': 1980,
            'stop_year': 2014,
            'legend_name': 'UKESM1-0-LL_'+time,
            'region': [-60,60,0,360]
        }
    elif key == 'ACCESS-ESM':
        asop_dict={
            'desc': 'ACCESS-ESM1-5_historical_r1i1p1f1_gn_'+time,
            'dir': cmip6_path/'ACCESS-ESM1-5',
            'name': 'ACCESS-ESM1-5_'+time,
            'file_pattern': 'pr_'+time+'*30.'+grid_str+'nc',
            'start_year': 1980,
            'stop_year': 2014,
            'legend_name': 'ACCESS-ESM_'+time,
            'region': [-60,60,0,360]
        }
    else:
        raise Exception('No dictionary for '+key)
    asop_dict['year_range'] = str(asop_dict['start_year'])+'-'+str(asop_dict['stop_year'])
    if grid is not '':
        asop_dict['desc'] = asop_dict['desc']+'_'+grid
        asop_dict['name'] = asop_dict['name']+'_'+grid
    return(asop_dict)

def load_cmip6(asop_dict):
    import iris
    from iris.util import unify_time_units
    from iris.experimental.equalise_cubes import equalise_attributes
    from iris.time import PartialDateTime

    constraint = iris.Constraint(time = lambda cell: PartialDateTime(year=asop_dict['start_year']) <= cell <= PartialDateTime(year=asop_dict['stop_year']),latitude = lambda cell: -60 <= cell <= 60)
    cubelist = iris.load(str(asop_dict['dir'])+'/'+asop_dict['file_pattern']) # Use NetCDF3 data to save compute time
    print(cubelist)
    unify_time_units(cubelist)
    equalise_attributes(cubelist)
    cube = cubelist.concatenate_cube()
    cube.coord('time').bounds = None
    out_cube = cube.extract(constraint)
    return(out_cube)

def new_cube_copy(cube,var_name,long_name):
    new_cube = cube.copy()
    new_cube.var_name=var_name
    new_cube.long_name=long_name
    return(new_cube)

def mask_wet_season(precip,wet_season_threshold=1.0/24.0):
    import numpy.ma as ma
    import iris.coord_categorisation
    import numpy as np

    if not 'month_number' in [coord.name() for coord in precip.coords()]:
        iris.coord_categorisation.add_month_number(precip,'time')
    if not 'year' in [coord.name() for coord in precip.coords()]:
        iris.coord_categorisation.add_year(precip,'time')
    ann_total = precip.aggregated_by('year',iris.analysis.SUM)
    ann_clim = ann_total.collapsed('time',iris.analysis.MEAN)

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

def conform_precip_threshold(precip,threshold):
    from cfunits import Units
    if precip.units == 'mm/hr':
        input_units = 'mm/h'
    elif precip.units == 'kg m-2 s-1':
        input_units = 'mm/s'
    elif precip.units == 'mm':
        input_units = 'mm/day'
    else:
        input_units = precip.units
    if input_units == 'mm/day':
        threshold_units = threshold
    else:
        threshold_units = Units.conform(threshold,Units('mm/day'),Units(input_units)) #precip.units)
    return(threshold_units)

def mask_min_precip(precip,min_precip_threshold=1.0):
    import numpy.ma as ma
    import numpy as np
    import iris

    threshold_units = conform_precip_threshold(precip,min_precip_threshold)
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

def compute_autocorr_grid(precip,lags):
    import iris.analysis.stats as istats
    import numpy as np

    lon_coord = precip.coord('longitude')
    lat_coord = precip.coord('latitude')
    lag_coord = iris.coords.DimCoord(lags,var_name='lag')
    autocorr = iris.cube.Cube(data=np.empty((len(lags),len(lat_coord.points),len(lon_coord.points))),dim_coords_and_dims = [(lag_coord,0),(lat_coord,1),(lon_coord,2)])

    for l,lag in enumerate(lags):
        lagged_slice = precip.copy(data=np.roll(precip.data,lag,0))
        autocorr.data[l,:,:] = istats.pearsonr(precip,lagged_slice,corr_coords='time').data
    return(autocorr.data)

def compute_temporal_autocorr(precip,lags,min_precip_threshold=1):
    import numpy.ma as ma
    import numpy as np
    import iris.coord_categorisation
    import dask

     # Compute temporal summary metric only
    if not 'month_number' in [coord.name() for coord in precip.coords()]:
        iris.coord_categorisation.add_month_number(precip,'time')
    if not 'year' in [coord.name() for coord in precip.coords()]:
        iris.coord_categorisation.add_year(precip,'time')
    threshold_units = conform_precip_threshold(precip,min_precip_threshold)

    lon_coord = precip.coord('longitude')
    lat_coord = precip.coord('latitude')
    nlon = len(lon_coord.points)
    nlat = len(lat_coord.points)
    nlags = len(lags)
    months = sorted(set(precip.coord('month_number').points))
    month_coord = iris.coords.DimCoord(months,var_name='month_number')
    lag_coord = iris.coords.DimCoord(lags,var_name='lag')
    nmonths = len(month_coord.points)

    temporal_autocorr = iris.cube.Cube(data=np.empty((nmonths,nlags,nlat,nlon)),dim_coords_and_dims=[(month_coord,0),(lag_coord,1),(lat_coord,2),(lon_coord,3)])
    temporal_autocorr = temporal_autocorr.copy(data=temporal_autocorr.data.fill(np.nan)) #temporal_autocorr.data[:,:,:,:].fill(np.nan)
    temporal_autocorr.var_name='autocorr_wetseason_precip'
    temporal_autocorr.long_name='Auto-correlation of precipitation masked for the wet season'

    precip_weights = iris.cube.Cube(data=np.empty((nmonths,nlat,nlon)),dim_coords_and_dims=[(month_coord,0),(lat_coord,1),(lon_coord,2)])
    precip_weights.var_name='precip_weights'
    precip_weights.long_name='Weights for computing mean metric over wet season (count of precip values above '+str(min_precip_threshold)+' mm/day'

    for m,month in enumerate(months):
        dask_autocorr = []
        print('-->-->--> Month '+str(month))
        month_constraint = iris.Constraint(month_number=month)
        this_month = precip.extract(month_constraint)
        years = set(this_month.coord('year').points)
        weights = np.zeros((len(years),nlags,nlat,nlon))
        for y,year in enumerate(years):
            year_constraint = iris.Constraint(year=year)
            this_monthyear = this_month.extract(year_constraint)
            this_autocorr = dask.delayed(compute_autocorr_grid)(this_monthyear,lags)
            weights[y,:,:,:] = this_monthyear.collapsed('time',iris.analysis.COUNT,function=lambda values: values >= threshold_units).data
            dask_autocorr.append(this_autocorr)
        result = dask.compute(*dask_autocorr)
        result = np.ma.asarray(result)
        result[np.where(result == 0.0)].mask = True
        result[np.where(result == 0.0)] = np.nan
        temporal_autocorr.data[m,:,:,:] = np.ma.average(result[:,:,:,:],axis=0,weights=weights)
        # Compute number of valid precipitation data points (> 1 mm/day)
        precip_weights.data[m,:,:] = this_month.collapsed('time',iris.analysis.COUNT,function=lambda values: values >= threshold_units).data
    temporal_autocorr_masked = temporal_autocorr.copy(data=np.ma.masked_array(temporal_autocorr.data,np.isnan(temporal_autocorr.data)))
    temporal_autocorr_mean = temporal_autocorr.collapsed('month_number',iris.analysis.MEAN,mdtol=0)
    temporal_autocorr_mean.var_name='autocorr_wetseason_precip_mean'
    temporal_autocorr_mean.long_name='Auto-correlation of precipitation masked for the wet season (weighted mean of all months in wet season)'
    for lag in range(nlags):
        temporal_autocorr_mean.data[lag,:,:] = np.ma.average(temporal_autocorr_masked.data[:,lag,:,:],axis=0,weights=precip_weights.data)
    out_cubelist = [temporal_autocorr,temporal_autocorr_mean,precip_weights]
    return(out_cubelist)

def compute_temporal_summary(precip,ndivs,min_precip_threshold=1):
    import numpy.ma as ma
    import numpy as np
    import dask
    import iris.coord_categorisation

    # Compute temporal summary metric only
    if not 'month_number' in [coord.name() for coord in precip.coords()]:
        iris.coord_categorisation.add_month_number(precip,'time')
    lon_coord = precip.coord('longitude')
    lat_coord = precip.coord('latitude')
    nlon = len(lon_coord.points)
    nlat = len(lat_coord.points)
    threshold_units = conform_precip_threshold(precip,min_precip_threshold)

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

    precip_weights = iris.cube.Cube(data=np.empty((nmonths,nlat,nlon)),dim_coords_and_dims=[(month_coord,0),(lat_coord,1),(lon_coord,2)])
    precip_weights.var_name='precip_weights'
    precip_weights.long_name='Weights for computing mean metric over wet season (count of precip values above '+str(min_precip_threshold)+' mm/day'

    for m,month in enumerate(months):
        print('-->-->--> Month '+str(month))
        month_summaries=[]
        month_constraint = iris.Constraint(month_number=month)
        this_month = precip.extract(month_constraint)
        lower_thresh.data[m,:,:] = this_month.collapsed('time',iris.analysis.PERCENTILE,percent=100.0/ndivs).data
        upper_thresh.data[m,:,:] = this_month.collapsed('time',iris.analysis.PERCENTILE,percent=100.0*(1.0-1.0/ndivs)).data
        years = set(this_month.coord('year').points)
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
        # Compute number of valid precipitation data points (> 1 mm/day)
        precip_weights.data[m,:,:] = this_month.collapsed('time',iris.analysis.COUNT,function=lambda values: values >= threshold_units).data
    
    #onon_freq.data.mask = upper_thresh.data.mask
    #offon_freq.data.mask = upper_thresh.data.mask
    #offoff_freq.data.mask = upper_thresh.data.mask
    ##onoff_freq.data.mask = upper_thresh.data.mask

    time_inter.data = 0.5*((onon_freq.data+offoff_freq.data)-(onoff_freq.data+offon_freq.data))
    #time_inter.data.mask = upper_thresh.data.mask

    time_inter_mean = time_inter.collapsed('month_number',iris.analysis.MEAN)  
    time_inter_mean.data = np.ma.average(time_inter.data,axis=0,weights=precip_weights.data)
    time_inter_mean.var_name='temporal_onoff_metric_mean'
    time_inter_mean.long_name='Temporal intermittency on-off metric based on '+str(ndivs)+' divisions (weighted mean of all months in wet season)'
    onon_freq_mean = onon_freq.collapsed('month_number',iris.analysis.MEAN)
    onon_freq_mean.data = np.ma.average(onon_freq.data,axis=0,weights=precip_weights.data)
    onon_freq_mean.var_name='prob_onon_mean'
    onon_freq_mean.long_name='Probability of upper division followed by upper division (weighted mean of all months in wet season)'
    onoff_freq_mean = onoff_freq.collapsed('month_number',iris.analysis.MEAN)
    onoff_freq_mean.data = np.ma.average(onoff_freq.data,axis=0,weights=precip_weights.data)
    onoff_freq_mean.var_name='prob_onoff_mean'
    onoff_freq_mean.long_name='Probability of upper division followed by lower division (weighted mean of all months in wet season)'
    offon_freq_mean = offon_freq.collapsed('month_number',iris.analysis.MEAN)
    offon_freq_mean.data = np.ma.average(offon_freq.data,axis=0,weights=precip_weights.data)
    offon_freq_mean.var_name='prob_offon_mean'
    offon_freq_mean.long_name='Probability of lower division followed by upper division (weighted mean of all months in wet season)'
    offoff_freq_mean = offoff_freq.collapsed('month_number',iris.analysis.MEAN)
    offoff_freq_mean.data = np.ma.average(offoff_freq.data,axis=0,weights=precip_weights.data)
    offoff_freq_mean.var_name='prob_offoff_mean'
    offoff_freq_mean.long_name='Probability of lower division followed by lower division (weighted mean of all months in wet season)'
    out_cubelist = [time_inter,onon_freq,onoff_freq,offon_freq,offoff_freq,lower_thresh,upper_thresh,time_inter_mean,onon_freq_mean,onoff_freq_mean,offon_freq_mean,offoff_freq_mean,precip_weights]
    return(out_cubelist)

def compute_onoff_metric_grid(this_monthyear,lower_thresh,upper_thresh):
    import numpy as np
    import iris

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

def haversine(origin, destination):
    import math

    lat1, lon1 = origin
    lat2, lon2 = destination
    radius = 6371 # km

    dlat = math.radians(lat2-lat1)
    dlon = math.radians(lon2-lon1)
    a = math.sin(dlat/2) * math.sin(dlat/2) + math.cos(math.radians(lat1)) \
        * math.cos(math.radians(lat2)) * math.sin(dlon/2) * math.sin(dlon/2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    d = radius * c
    return d

def compute_equalgrid_corr_global(precip,haversine_map,distance_bins,min_precip_threshold=1):
    import iris
    import numpy as np
    import dask

    longitude = precip.coord('longitude')
    nlon=len(longitude.points)
    latitude = precip.coord('latitude')
    nlat=len(latitude.points)
    nbins = len(distance_bins)-1
    threshold_units = conform_precip_threshold(precip,min_precip_threshold)

    dist_centre = np.zeros(nbins)
    min_dist = np.zeros(nbins)
    max_dist = np.zeros(nbins)
    bounds = np.zeros((nbins,2))
    for b,left in enumerate(distance_bins[0:-1]):
        min_dist[b] = left
        max_dist[b] = distance_bins[b+1]
        dist_centre[b] = (max_dist[b]+min_dist[b])/2.0
        bounds[b,:] = np.asarray((min_dist[b],max_dist[b]))
  
    months = sorted(set(precip.coord('month_number').points))
    month_coord = iris.coords.DimCoord(months,var_name='month_number')
    nmonths = len(months)
    distance = iris.coords.DimCoord(dist_centre,var_name='distance',bounds=bounds)
    distance_corrs = iris.cube.Cube(np.zeros((nmonths,nbins,nlat,nlon)),var_name='distance_correlations',\
        long_name='Spatial correlation of precipitation masked for the wet season',\
        dim_coords_and_dims=[(month_coord,0),(distance,1),(latitude,2),(longitude,3)])
    npts = iris.cube.Cube(np.zeros((nmonths,nbins,nlat,nlon)),var_name='distance_npts',\
        long_name='Number of points considered in distance bin',\
        dim_coords_and_dims=[(month_coord,0),(distance,1),(latitude,2),(longitude,3)])

    precip_weights = iris.cube.Cube(data=np.empty((nmonths,nlat,nlon)),dim_coords_and_dims=[(month_coord,0),(latitude,1),(longitude,2)])
    precip_weights.var_name='precip_weights'
    precip_weights.long_name='Weights for computing mean metric over wet season (count of precip values above '+str(min_precip_threshold)+' mm/day'#

    for m,month in enumerate(months):
        month_constraint = iris.Constraint(month_number=month)
        this_month = precip.extract(month_constraint)
        # Compute number of valid precipitation data points (> 1 mm/day)
        precip_weights.data[m,:,:] = this_month.collapsed('time',iris.analysis.COUNT,function=lambda values: values >= threshold_units).data

    haversine_map = dask.delayed(haversine_map)
    for m,month in enumerate(months):
        print('-->-->--> Month '+str(month))
        month_constraint = iris.Constraint(month_number=month)
        this_month = precip.extract(month_constraint)
        this_month = dask.delayed(this_month)
        dask_distcorr = []
        for y in range(nlat):
            this_distcorr = dask.delayed(compute_equalgrid_corr_row)(this_month,haversine_map.data[y,:,:,:],y,min_dist,max_dist)
            dask_distcorr.append(this_distcorr)
        result = dask.compute(*dask_distcorr)
        result = np.asarray(result)
        result = np.reshape(result,(nlat,2,nlon,nbins))
        for b in range(nbins):
            distance_corrs.data[m,b,:,:] = result[:,0,:,b]
            npts.data[m,b,:,:] = result[:,1,:,b]
            print(npts.data[m,b,:,0])

    distance_corrs_masked = distance_corrs.copy(data=np.ma.masked_array(distance_corrs.data,np.isnan(distance_corrs.data)))
    distance_corrs_mean = iris.cube.Cube(np.zeros((nbins,nlat,nlon)),var_name='distance_correlations_mean',\
        long_name='Spatial correlation of precipitation masked for the wet season (weighted mean of all months in wet season)',\
        dim_coords_and_dims=[(distance,0),(latitude,1),(longitude,2)])
    for b in range(nbins):
        distance_corrs_mean.data[b,:,:] = np.ma.average(distance_corrs_masked.data[:,b,:,:],axis=0,weights=precip_weights.data)
    out_cubelist=[npts,distance_corrs,distance_corrs_mean,precip_weights]
    return(out_cubelist)

@jit
def compute_equalgrid_corr_row(precip,haversine_row,myrow,min_dist,max_dist):
    longitude = precip.coord('longitude')
    nlon = len(longitude.points)
    nbins = len(min_dist)
    row_distcorr = np.zeros((nlon,nbins))
    npts = np.zeros((nlon,nbins))
    for x,lonpt in enumerate(longitude.points):
        precip_centre = precip[:,myrow,x]
        haversine_centre = haversine_row[x,:,:]
        for b in range(nbins):
            row_distcorr[x,b],npts[x,b] = compute_equalgrid_corr_pt(precip,precip_centre,haversine_centre,min_dist[b],max_dist[b])
    return(np.stack([row_distcorr,npts],axis=0))

@jit
def compute_equalgrid_corr_pt(precip,precip_centre,pt_dist,dist_min,dist_max):
    latitude = precip.coord('latitude')
    longitude = precip.coord('longitude')
    nlon=len(longitude.points)
    nlat=len(latitude.points)
    
    corr_cube = iris.cube.Cube(data=np.zeros((nlat,nlon)),dim_coords_and_dims=[(latitude,0),(longitude,1)])
    corr_cube.data[:,:] = np.nan
    pts = np.where(np.logical_and(np.greater_equal(pt_dist,dist_min),np.less_equal(pt_dist,dist_max)))
    indices = list(zip(pts[0],pts[1]))
    temp = numba_corrs(precip.data,precip_centre.data,indices)
    for count,(yy,xx) in enumerate(indices):
        corr_cube.data[yy,xx] = temp[count]
    corr_cube = corr_cube.copy(data=np.ma.masked_array(corr_cube.data,np.isnan(corr_cube.data)))
    weights = iris.analysis.cartography.area_weights(corr_cube)
    output = corr_cube.collapsed(['longitude','latitude'],iris.analysis.MEAN,weights=weights)
    npts = len(pts[0])
    return(output.data,npts)

@jit(nopython=True)
def numba_corrs(precip_grid,precip_centre,indices):
    output = np.zeros(len(indices))
    for count,(yy,xx) in enumerate(indices):
        output[count] = np.corrcoef(precip_grid[:,yy,xx],precip_centre)[1,0]
    return(output)

def compute_haversine_single(latitude,longitude,centre_lat,centre_lon):
    import dask
    import numpy as np

    nlon = len(longitude.points)
    nlat = len(latitude.points)

    my_map = np.zeros((nlat,nlon))
    for yy,target_lat in enumerate(latitude.points):
        for xx,target_lon in enumerate(longitude.points):
            my_map[yy,xx] = haversine((centre_lat,centre_lon),(target_lat,target_lon))
    return(my_map)

def compute_haversine_map(cube):
    import dask
    import numpy as np
    import iris

    latitude = cube.coord('latitude')
    nlat = len(latitude.points)
    longitude = cube.coord('longitude')
    nlon = len(longitude.points)
    target_latitude = iris.coords.DimCoord(latitude.points,var_name='target_lat',long_name='Target latitude')
    target_longitude = iris.coords.DimCoord(longitude.points,var_name='target_lon',long_name='Target longitude')
    haversine_map = iris.cube.Cube(data=np.zeros((nlat,nlon,nlat,nlon)),dim_coords_and_dims=[(latitude,0),(longitude,1),(target_latitude,2),(target_longitude,3)],\
        var_name='haversine_map',long_name='Map of point-to-point distances')
    haversines = []
    for lat in latitude.points:
        for lon in longitude.points:
            my_map = dask.delayed(compute_haversine_single)(latitude,longitude,lat,lon)
            haversines.append(my_map)
    result = dask.compute(*haversines)
    result = np.asarray(result)
    print(result.shape)
    haversine_map.data = np.reshape(result,(nlat,nlon,nlat,nlon))
    return(haversine_map)

def compute_spatial_summary(precip,ndivs,min_precip_threshold=1):
    import numpy.ma as ma
    import numpy as np
    import iris.coord_categorisation

    # Compute spatial summary metric only
    if not 'month_number' in [coord.name() for coord in precip.coords()]:
        iris.coord_categorisation.add_month_number(precip,'time')
    lon_coord = precip.coord('longitude')
    lat_coord = precip.coord('latitude')
    nlon = len(lon_coord.points)
    nlat = len(lat_coord.points)
    threshold_units = conform_precip_threshold(precip,min_precip_threshold)

    months = sorted(set(precip.coord('month_number').points))
    month_coord = iris.coords.DimCoord(months,var_name='month_number')
    nmonths = len(months)
    
    lower_thresh = iris.cube.Cube(data=ma.zeros((nmonths,nlat,nlon)),dim_coords_and_dims=[(month_coord,0),(lat_coord,1),(lon_coord,2)])
    lower_thresh.var_name='lower_threshold'
    lower_thresh.long_name='Lower (off) threshold based on '+str(ndivs)+' divisions'
    upper_thresh = new_cube_copy(lower_thresh,'upper_threshold','Upper (on) threshold based on '+str(ndivs)+' divisions')
    space_inter = new_cube_copy(lower_thresh,'spatial_onoff_metric','Spatial intermittency on-off metric based on '+str(ndivs)+' divisions')
    onon_freq = new_cube_copy(lower_thresh,'prob_onon','Probability of upper division neighbouring upper division')
    onoff_freq = new_cube_copy(lower_thresh,'prob_onoff','Probability of upper division neighbouring lower division')
    offon_freq = new_cube_copy(lower_thresh,'prob_offon','Probability of lower division neighbouring upper division')
    offoff_freq = new_cube_copy(lower_thresh,'prob_offoff','Probability of lower division neighbouring lower division')

    precip_weights = iris.cube.Cube(data=np.empty((nmonths,nlat,nlon)),dim_coords_and_dims=[(month_coord,0),(lat_coord,1),(lon_coord,2)])
    precip_weights.var_name='precip_weights'
    precip_weights.long_name='Weights for computing mean metric over wet season (count of precip values above '+str(min_precip_threshold)+' mm/day'

    for m,month in enumerate(months):
        print('-->-->--> Month '+str(month))
        month_constraint = iris.Constraint(month_number=month)
        this_month = precip.extract(month_constraint)
        lower_thresh.data[m,:,:] = this_month.collapsed('time',iris.analysis.PERCENTILE,percent=100.0/ndivs).data
        upper_thresh.data[m,:,:] = this_month.collapsed('time',iris.analysis.PERCENTILE,percent=100.0*(1.0-1.0/ndivs)).data
        month_summary = compute_spatial_onoff_metric_grid(this_month,lower_thresh[m,:,:],upper_thresh[m,:,:])
        onon_freq.data[m,:,:] = month_summary[0,:,:]
        onoff_freq.data[m,:,:] =  month_summary[1,:,:]
        offon_freq.data[m,:,:] =  month_summary[2,:,:]
        offoff_freq.data[m,:,:] = month_summary[3,:,:]
        # Compute number of valid precipitation data points (> 1 mm/day)
        precip_weights.data[m,:,:] = this_month.collapsed('time',iris.analysis.COUNT,function=lambda values: values >= threshold_units).data
    

    onon_freq.data.mask = upper_thresh.data.mask
    offon_freq.data.mask = upper_thresh.data.mask
    onoff_freq.data.mask = upper_thresh.data.mask
    offoff_freq.data.mask = upper_thresh.data.mask

    space_inter.data = 0.5*((onon_freq.data+offoff_freq.data)-(onoff_freq.data+offon_freq.data))
    space_inter.data.mask = upper_thresh.data.mask
    space_inter_mean = space_inter.collapsed('month_number',iris.analysis.MEAN,mdtol=0)  
    space_inter_mean.data = np.ma.average(space_inter.data,axis=0,weights=precip_weights.data)
    space_inter_mean.var_name='spatial_onoff_metric_mean'
    space_inter_mean.long_name='spatial intermittency on-off metric based on '+str(ndivs)+' divisions (weighted mean of all months in wet season)'
    onon_freq_mean = onon_freq.collapsed('month_number',iris.analysis.MEAN)
    onon_freq_mean.data = np.ma.average(onon_freq.data,axis=0,weights=precip_weights.data)
    onon_freq_mean.var_name='prob_onon_mean'
    onon_freq_mean.long_name='Probability of upper division neighbouring upper division (weighted mean of all months in wet season)'
    onoff_freq_mean = onoff_freq.collapsed('month_number',iris.analysis.MEAN)
    onoff_freq_mean.data = np.ma.average(onoff_freq.data,axis=0,weights=precip_weights.data)
    onoff_freq_mean.var_name='prob_onoff_mean'
    onoff_freq_mean.long_name='Probability of upper division neighbouring lower division (weighted mean of all months in wet season)'
    offon_freq_mean = offon_freq.collapsed('month_number',iris.analysis.MEAN)
    offon_freq_mean.data = np.ma.average(offon_freq.data,axis=0,weights=precip_weights.data)
    offon_freq_mean.var_name='prob_offon_mean'
    offon_freq_mean.long_name='Probability of lower division neighbouring upper division (weighted mean of all months in wet season)'
    offoff_freq_mean = offoff_freq.collapsed('month_number',iris.analysis.MEAN)
    offoff_freq_mean.data = np.ma.average(offoff_freq.data,axis=0,weights=precip_weights.data)
    offoff_freq_mean.var_name='prob_offoff_mean'
    offoff_freq_mean.long_name='Probability of lower division neighbouring lower division (weighted mean of all months in wet season)'
    out_cubelist = [space_inter,onon_freq,onoff_freq,offon_freq,offoff_freq,lower_thresh,upper_thresh,space_inter_mean,onon_freq_mean,onoff_freq_mean,offon_freq_mean,offoff_freq_mean,precip_weights]
    return(out_cubelist)

def compute_spatial_onoff_metric_grid(precip,lower_thresh,upper_thresh,cyclic=True):
    import numpy as np
    import dask
    upper_mask = precip.copy(data=np.where(precip.data >= upper_thresh.data,1,0)) 
    lower_mask = precip.copy(data=np.where(precip.data <= lower_thresh.data,1,0))
    
    lon_coord = precip.coord('longitude')
    lat_coord = precip.coord('latitude')
    nlon = len(lon_coord.points)
    nlat = len(lat_coord.points)

    onon = np.zeros((nlat,nlon),dtype=np.float32) ; onoff = np.zeros((nlat,nlon),dtype=np.float32) ; offon = np.zeros((nlat,nlon),dtype=np.float32) ; offoff = np.zeros((nlat,nlon),dtype=np.float32)
    precip = dask.delayed(precip)
    upper_thresh = dask.delayed(upper_thresh)
    lower_mask = dask.delayed(lower_mask)
    upper_mask = dask.delayed(upper_mask)
    row_metrics = []
    for lat in range(1,nlat-1):
        row_metric = dask.delayed(compute_spatial_onoff_metric_row)(precip,lat,upper_thresh,lower_mask,upper_mask) 
        row_metrics.append(row_metric)
    result = dask.compute(*row_metrics)
    result = np.ma.asarray(result)
    result = np.reshape(result,(nlat-2,nlon,4))
    onon[1:nlat-1,:] = result[:,:,0] ; onoff[1:nlat-1,:] = result[:,:,1] ; offon[1:nlat-1,:] = result[:,:,2] ; offoff[1:nlat-1,:] = result[:,:,3]
    output = np.stack([onon,onoff,offon,offoff],axis=0)
    return(output)

def compute_spatial_onoff_metric_row(precip,row,upper_thresh,lower_mask,upper_mask):
    lon_coord = precip.coord('longitude')
    lat_coord = precip.coord('latitude')
    nlon = len(precip.coord('longitude').points)
    row_metric = np.zeros((nlon,4))
    for lon in range(nlon):
        if upper_thresh.data.mask[row,lon]:
            point_metric = compute_spatial_onoff_metric_point(None,None,None,None)
        else:
            if lon == 0:
                min_lon = lon_coord.points[-1]-360
            else:
                min_lon = lon_coord.points[lon-1]
            if lon == nlon-1:
                max_lon = lon_coord.points[0]+360
            else:
                max_lon = lon_coord.points[lon+1]
            upper_neighbors = upper_mask.intersection(longitude = (min_lon,max_lon), latitude = (lat_coord.points[row-1],lat_coord.points[row+1]))
            lower_neighbors = lower_mask.intersection(longitude = (min_lon,max_lon), latitude = (lat_coord.points[row-1],lat_coord.points[row+1]))
            on_neighbors_mask = upper_neighbors.copy() 
            off_neighbors_mask = lower_neighbors.copy()
            for y in range(len(upper_neighbors.coord('latitude').points)):
                for x in range(len(upper_neighbors.coord('longitude').points)):
                    on_neighbors_mask.data[:,y,x] = 1-upper_mask.data[:,row,lon] # Mask neighbors where central point is above upper threshold
                    off_neighbors_mask.data[:,y,x] = 1-lower_mask.data[:,row,lon] # Mask neighbors where central point is below lower threshold
            on_neighbors_mask.data[:,1,1] = 1 # Mask values at central point
            off_neighbors_mask.data[:,1,1] = 1 # Mask values at central point
            point_metric = compute_spatial_onoff_metric_point(on_neighbors_mask,off_neighbors_mask,upper_neighbors,lower_neighbors)
        row_metric[lon,:] = point_metric
    return(row_metric)

def compute_spatial_onoff_metric_point(on_neighbors_mask=None,off_neighbors_mask=None,upper_neighbors=None,lower_neighbors=None):
    import numpy as np
    import iris
    if on_neighbors_mask is None:
        onon=np.ma.masked
        onoff=np.ma.masked
        offon=np.ma.masked
        offoff=np.ma.masked
    else:
        onon_neighbors_masked = upper_neighbors.copy(data=np.ma.array(upper_neighbors.data,mask=on_neighbors_mask.data)) # Neighbors on and centre on
        onoff_neighbors_masked = lower_neighbors.copy(data=np.ma.array(lower_neighbors.data,mask=on_neighbors_mask.data)) # Neighbors off and centre on
        offon_neighbors_masked = upper_neighbors.copy(data=np.ma.array(upper_neighbors.data,mask=off_neighbors_mask.data)) # Neighbors on and centre off
        offoff_neighbors_masked = lower_neighbors.copy(data=np.ma.array(lower_neighbors.data,mask=off_neighbors_mask.data)) # Neighbors off and centre off
        onon = onon_neighbors_masked.collapsed(['time','latitude','longitude'],iris.analysis.MEAN).data
        onoff = onoff_neighbors_masked.collapsed(['time','latitude','longitude'],iris.analysis.MEAN).data
        offon = offon_neighbors_masked.collapsed(['time','latitude','longitude'],iris.analysis.MEAN).data
        offoff = offoff_neighbors_masked.collapsed(['time','latitude','longitude'],iris.analysis.MEAN).data
    output=np.stack([onon,onoff,offon,offoff],axis=0)
    return(output)

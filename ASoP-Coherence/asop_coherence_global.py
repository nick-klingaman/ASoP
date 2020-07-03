def get_asop_dict(key,time=None,grid=''):
    from pathlib import Path
#    cmip6_path=Path('/media/nick/lacie_tb3/data_from_gill/CMIP6')
    cmip6_path=Path('/media/nick/lacie_tb3/cmip6')
    obs_path=Path('/media/nick/lacie_tb3/datasets')
    print(key)
    if grid is not '':
        grid_str=grid+'.'
    else:   
        grid_str=''
    if key == 'AWI':
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
            'file_pattern': 'pr_'+time+'*1.'+grid_str+'nc',
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
            'file_pattern': 'pr_'+time+'*1.'+grid_str+'nc',
            'start_year': 1980,
            'stop_year': 2014,
            'legend_name': 'CanESM5_'+time,
            'region': [-60,60,0,360]
        }
    elif key == 'ACCESS':
        asop_dict={
            'desc': 'ACCESS-CM2_historical_r1i1p1f1_gn_'+time,
            'dir': cmip6_path/'ACCESS-CM2',
            'name': 'ACCESS-CM2_'+time,
            'file_pattern': 'pr_'+time+'*1.'+grid_str+'nc',
            'start_year': 1980,
            'stop_year': 2014,
            'legend_name': 'ACCESS_'+time,
            'region': [-60,60,0,360]
        }
    elif key == 'CESM2':
        asop_dict={
            'desc': 'CESM2_historical_r1i1p1f1_gn_'+time,
            'dir': cmip6_path/'CESM2',
            'name': 'CESM2_'+time,
            'file_pattern': 'pr_'+time+'*1.'+grid_str+'nc',
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
            'file_pattern': 'pr_'+time+'*1.'+grid_str+'nc',
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
            'file_pattern': 'pr_'+time+'*1.'+grid_str+'nc',
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
            'file_pattern': 'pr_'+time+'*1.'+grid_str+'nc',
            'start_year': 1980,
            'stop_year': 2014,
            'legend_name': 'FGOALS-g3_'+time,
            'region': [-60,60,0,360]
        }
    elif key == 'GFDL':
        asop_dict={
            'desc': 'GFDL-CM4_historical_r1i1p1f1_gr1_'+time,
            'dir': cmip6_path/'GFDL-CM4',
            'name': 'GFDL-CM4_'+time,
            'file_pattern': 'pr_'+time+'*_gr1_*1.'+grid_str+'nc',
            'start_year': 1980,
            'stop_year': 2014,
            'legend_name': 'GFDL-CM4_'+time,
            'region': [-60,60,0,360]
        }
    elif key == 'GISS':
        asop_dict={
            'dir': cmip6_path/'GISS-E2-1-G',
            'name': 'GISS-E2-1-G_'+time,
            'desc': 'GISS-E2-1-G_historical_r1i1p1f1_gn_'+time,
            'file_pattern': 'pr_'+time+'*0.'+grid_str+'.nc',
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
            'file_pattern': 'pr_'+time+'*0.'+grid_str+'nc',
            'start_year': 1980,
            'stop_year': 2014,
            'legend_name': 'HadGEM3-GC31-MM_'+time,
            'region': [-60,60,0,360]
        }
    elif key == 'INM':
        asop_dict={
            'desc': 'INM-CM5-0_historical_r1i1p1f1_gr1_'+time,
            'dir': cmip6_path/'INM-CM5',
            'name': 'INM-CM-5-0_'+time,
            'file_pattern': 'pr_'+time+'*1.'+grid_str+'nc',
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
            'file_pattern': 'pr_'+time+'*1.'+grid_str+'nc',
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
            'file_pattern': 'pr_'+time+'*0.'+grid_str+'nc',
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
            'file_pattern': 'pr_'+time+'*1.nc',
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
            'file_pattern': 'pr_'+time+'*1.nc',
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
            'file_pattern': 'pr_'+time+'*1.nc',
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
            'file_pattern': 'pr_'+time+'*1.nc',
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
            'file_pattern': 'pr_'+time+'*1.nc',
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
            'file_pattern': 'pr_'+time+'*1.nc',
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
            'file_pattern': 'pr_'+time+'*1.nc',
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
            'file_pattern': 'pr_'+time+'*0.nc',
            'start_year': 1980,
            'stop_year': 2014,
            'legend_name': 'UKESM1-0-LL_'+time,
            'region': [-60,60,0,360]
        }
    else:
        raise Exception('No dictionary for '+key)
    if grid is not '':
        asop_dict['desc'] = asop_dict['desc']+'_'+grid
        asop_dict['name'] = asop_dict['name']+'_'+grid
    return(asop_dict)
    '''
    elif key == 'GPM_IMERG_daily_3x3':
        asop_dict={
            'desc': '3B-DAY.MS.MRG.3IMERG.3x3',
            'dir': obs_path/'GPM_IMERG'/'daily',
            'file_pattern': '3B-DAY.MS.MRG.3IMERG.*.V06.3x3.nc',
            'name': 'IMERG-3B-V06_daily_3x3',
            'start_year': 2001,
            'stop_year': 2019,
            'legend_name': 'IMERG',
            'region': [-60,60,0,360],
        }
    elif key == 'GPM_IMERG_daily_2x2':
        asop_dict={
            'desc': '3B-DAY.MS.MRG.3IMERG.2x2',
            'dir': obs_path/'GPM_IMERG'/'daily',
            'file_pattern': '3B-DAY.MS.MRG.3IMERG.*.V06.2x2.nc',
            'name': 'IMERG-3B-V06_daily_2x2',
            'start_year': 2001,
            'stop_year': 2019,
            'legend_name': 'IMERG',
            'region': [-60,60,0,360]
        }
    elif key == 'GPM_IMERG_daily_1x1':
        asop_dict={
            'desc': '3B-DAY.MS.MRG.3IMERG.1x1',
            'dir': obs_path/'GPM_IMERG'/'daily',
            'file_pattern': '3B-DAY.MS.MRG.3IMERG.*.V06.1x1.nc',
            'name': 'IMERG-3B-V06_daily_1x1',
            'start_year': 2001,
            'stop_year': 2019,
            'legend_name': 'IMERG',
            'region': [-59,59,0,360]
        }
    elif key == 'GPM_IMERG_3hr_3x3':
        asop_dict={
            'desc': '3B-HHR.MS.MRG.3IMERG.V06B.3hr_means_3x3',
            'dir': obs_path/'GPM_IMERG',
            'file_pattern': '3B-HHR.MS.MRG.3IMERG.*.3hr_means_3x3.V06B.nc',
            'name': 'IMERG-3B-V06_3hr_3x3',
            'start_year': 2001,
            'stop_year': 2018,
            'legend_name': 'IMERG',
            'region': [-60,60,0,360]
        }
    elif key == 'GPM_IMERG_3hr_2x2':
        asop_dict={
            'desc': '3B-HHR.MS.MRG.3IMERG.V06B.3hr_means_2x2',
            'dir': obs_path/'GPM_IMERG',
            'file_pattern': '3B-HHR.MS.MRG.3IMERG.*.3hr_means_2x2.V06B.nc',
            'name': 'IMERG-3B-V06_3hr_2x2',
            'start_year': 2001,
            'stop_year': 2018,
            'legend_name': 'IMERG',
            'region': [-60,60,0,360]
        }
    elif key == 'GPM_IMERG_3hr_1x1':
        asop_dict={
            'desc': '3B-HHR.MS.MRG.3IMERG.V06B.3hr_means_1x1',
            'dir': obs_path/'GPM_IMERG',
            'file_pattern': '3B-HHR.MS.MRG.3IMERG.*.3hr_means_1x1.V06B.nc',
            'name': 'IMERG-3B-V06',
            'start_year': 2001,
            'stop_year': 2018,
            'legend_name': 'IMERG',
            'region': [-60,60,0,360]
        }
    elif key == 'ACCESS_daily_3x3':
        asop_dict={
            'desc': 'ACCESS-CM2_historical_r1i1p1f1_gn_daily_3x3',
            'dir': cmip6_path/'ACCESS-CM2',
            'name': 'ACCESS_daily_3x3',
            'file_pattern': 'pr_3hr*.daily.3x3.nc',
            'start_year': 1990,
            'stop_year': 2014,
            'legend_name': 'ACCESS',
            'region': [-60,60,0,360]
        }
    elif key == 'FGOALS_3hr_3x3':
        asop_dict={
            'desc': 'FGOALS-g3_historical_r1i1p1f1_gn_3hr_3x3',
            'dir': cmip6_path/'FGOALS-g3',
            'name': 'FGOALS_3hr_3x3',
            'file_pattern': 'pr_3hr*0.3x3.nc',
            'start_year': 1990,
            'stop_year': 2016,
            'legend_name': 'FGOALS',
            'region': [-60,60,0,360]
        }
    elif key == 'FGOALS_daily_3x3':
        asop_dict={
            'desc': 'FGOALS-g3_historical_r1i1p1f1_gn_daily_3x3',
            'dir': cmip6_path/'FGOALS-g3',
            'name': 'FGOALS_daily_3x3',
            'file_pattern': 'pr_3hr*.daily.3x3.nc',
            'start_year': 1990,
            'stop_year': 2016,
            'legend_name': 'FGOALS',
            'region': [-60,60,0,360]
        }
    elif key == 'GISS_3hr_3x3':
        asop_dict={
            'dir': cmip6_path/'GISS-E2-1-G',
            'name': 'GISS_3hr_3x3',
            'desc': 'GISS-E2-1-G_historical_r1i1p1f1_gn_3hr_3x3',
            'file_pattern': 'pr_3hr*0.3x3.nc',
            'start_year': 1990,
            'stop_year': 2014,
            'legend_name': 'GISS',
            'region': [-60,60,0,360]
        }
    elif key == 'GISS_daily_3x3':
        asop_dict={
            'dir': cmip6_path/'GISS-E2-1-G',
            'name': 'GISS_daily_3x3',
            'desc': 'GISS-E2-1-G_historical_r1i1p1f1_gn_daily_3x3',
            'file_pattern': 'pr_3hr*.daily.3x3.nc',
            'start_year': 1990,
            'stop_year': 2014,
            'legend_name': 'GISS',
            'region': [-60,60,0,360]
        }
    elif key == 'MIROC_3hr_3x3':
        asop_dict={
            'dir': cmip6_path/'MIROC6',
            'name': 'MIROC_3hr_3x3',
            'desc': 'MIROC6_historical_r1i1p1f1_gn_3hr_3x3',
            'file_pattern': 'pr_3hr*0.3x3.nc',
            'start_year': 1990,
            'stop_year': 2014,
            'legend_name': 'MIROC6',
            'region': [-60,60,0,360]
        }
    elif key == 'MIROC_daily_3x3':
        asop_dict={
            'dir': cmip6_path/'MIROC6',
            'name': 'MIROC_daily_3x3',
            'desc': 'MIROC6_historical_r1i1p1f1_gn_daily_3x3',
            'file_pattern': 'pr_3hr*.daily.3x3.nc',
            'start_year': 1990,
            'stop_year': 2014,
            'legend_name': 'MIROC6',
            'region': [-60,60,0,360]
        }
    elif key == 'MPI-ESM1_3hr_3x3':
        asop_dict={
            'dir': cmip6_path/'MPI-ESM1-2-HR',
            'name': 'MPI-ESM_3hr_3x3',
            'desc': 'MPI-ESM1-2-HR_historical_r1i1p1f1_gn_3hr_3x3',
            'file_pattern': 'pr_3hr*0.3x3.nc',
            'start_year': 1990,
            'stop_year': 2014,
            'legend_name': 'MPI-ESM',
            'region': [-60,60,0,360]
        }
    elif key == 'MPI-ESM1_daily_3x3':
        asop_dict={
            'dir': cmip6_path/'MPI-ESM1-2-HR',
            'name': 'MPI-ESM_daily_3x3',
            'desc': 'MPI-ESM1-2-HR_historical_r1i1p1f1_gn_daily_3x3',
            'file_pattern': 'pr_3hr*.daily.3x3.nc',
            'start_year': 1990,
            'stop_year': 2014,
            'legend_name': 'MPI-ESM',
            'region': [-60,60,0,360]
        }
    elif key == 'SAM0-UNICON_3hr_3x3':
        asop_dict={
            'dir': cmip6_path/'SAM0-UNICON',
            'name': 'SAM_3hr_3x3',
            'desc': 'SAM0-UNICON_historical_r1i1p1f1_gn_3hr_3x3',
            'file_pattern': 'pr_3hr*0.3x3.nc',
            'start_year': 1990,
            'stop_year': 2014,
            'legend_name': 'SAM',
            'region': [-60,60,0,360]
        }
    elif key == 'SAM0-UNICON_daily_3x3':
        asop_dict={
            'dir': cmip6_path/'SAM0-UNICON',
            'name': 'SAM_daily_3x3',
            'desc': 'SAM0-UNICON_historical_r1i1p1f1_gn_daily_3x3',
            'file_pattern': 'pr_3hr*.daily.3x3.nc',
            'start_year': 1990,
            'stop_year': 2014,
            'legend_name': 'SAM',
            'region': [-60,60,0,360]
        }
    '''

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

def compute_autocorr_grid(precip,lag):
    import iris.analysis.stats as istats
    import numpy as np
    lagged_slice = precip.copy(data=np.roll(precip.data,lag,0))
    output = istats.pearsonr(precip,lagged_slice,corr_coords='time')
    return(output.data)

def compute_temporal_autocorr(precip,max_lag,min_precip_threshold=1):
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
    months = sorted(set(precip.coord('month_number').points))
    month_coord = iris.coords.DimCoord(months,var_name='month_number')
    lag_coord = iris.coords.DimCoord(range(max_lag),var_name='lag')
    nmonths = len(month_coord.points)

    temporal_autocorr = iris.cube.Cube(data=np.empty((nmonths,max_lag,nlat,nlon)),dim_coords_and_dims=[(month_coord,0),(lag_coord,1),(lat_coord,2),(lon_coord,3)])
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
        # Compute number of valid precipitation data points (> 1 mm/day)
        precip_weights.data[m,:,:] = this_month.collapsed('time',iris.analysis.COUNT,function=lambda values: values >= threshold_units).data
    temporal_autocorr_masked = temporal_autocorr.copy(data=np.ma.masked_array(temporal_autocorr.data,np.isnan(temporal_autocorr.data)))
    temporal_autocorr_mean = temporal_autocorr.collapsed('month_number',iris.analysis.MEAN,mdtol=0)
    temporal_autocorr_mean.var_name='autocorr_wetseason_precip_mean'
    temporal_autocorr_mean.long_name='Auto-correlation of precipitation masked for the wet season (weighted mean of all months in wet season)'
    for lag in range(max_lag):
        temporal_autocorr_mean.data[lag,:,:] = np.ma.average(temporal_autocorr_masked.data[:,lag,:,:],axis=0,weights=precip_weights.data)
    out_cubelist = [temporal_autocorr,temporal_autocorr_mean,precip_weights]
    return(out_cubelist)

def compute_temporal_summary(precip,ndivs):
    import numpy.ma as ma
    import numpy as np
    import dask
    import iris

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
    import iris

    upper_mask = this_monthyear.copy(data=np.where(this_monthyear.data >= upper_thresh.data,1,0))
    lower_mask = this_monthyear.copy(data=np.where(this_monthyear.data < lower_thresh.data,1,0))
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

def compute_equalgrid_corr_global(precip,distance_bins):
    import iris
    import numpy as np
    import dask

    longitude = precip.coord('longitude')
    nlon=len(longitude.points)
    latitude = precip.coord('latitude')
    nlat=len(latitude.points)
    time = precip.coord('time')
    ntime=len(time.points)
    nbins = len(distance_bins)-1
    dist_centre = np.zeros(nbins)
    min_dist = np.zeros(nbins)
    max_dist = np.zeros(nbins)
    bounds = np.zeros((nbins,2))
    for b,left in enumerate(distance_bins[0:-1]):
        min_dist[b] = left
        max_dist[b] = distance_bins[b+1]
        dist_centre[b] = (max_dist[b]+min_dist[b])/2.0
        bounds[b,:] = np.asarray((min_dist[b],max_dist[b]))
    distance = iris.coords.DimCoord(dist_centre,var_name='distance',bounds=bounds)
    distance_corrs = iris.cube.Cube(np.zeros((nbins,nlat,nlon)),var_name='distance_correlations',dim_coords_and_dims=[(distance,0),(latitude,1),(longitude,2)])
    for b in range(nbins):
        for y,latpt in enumerate(latitude.points):
            dask_distcorr=[]
            for x,lonpt in enumerate(longitude.points):
                precip_mask = extract_mask_region(precip,latpt,lonpt,min_dist[b],max_dist[b])
                this_distcorr = dask.delayed(compute_gridcorr_grid)(precip[:,y,x],precip_mask)
                dask_distcorr.append(this_distcorr)
            result = dask.compute(*dask_distcorr)
            result = np.ma.asarray(result)
        #result = np.reshape(result,(nlon,nbins))
            distance_corrs.data[b,y,:] = result
    return(distance_corrs)

def extract_mask_region(precip,centre_lat,centre_lon,dist_min,dist_max):
    import numpy as np

    longitude = precip.coord('longitude')
    nlon=len(longitude.points)
    latitude = precip.coord('latitude')
    nlat=len(latitude.points)
    time = precip.coord('time')
    ntime=len(time.points)
    pt_dist = np.ones((nlat,nlon))
    dN = 0 ; dS = 0 ; dlon = 0 
#    print(centre_lat,centre_lon)
    for yy,target_lat in enumerate(latitude.points):
        for xx,target_lon in enumerate(longitude.points):
            pt_dist[yy,xx] = haversine((centre_lat,centre_lon),(target_lat,target_lon))
            if pt_dist[yy,xx] <= dist_max:
                dN = np.amax([dN,target_lat-centre_lat])
                dS = np.amin([dS,target_lat-centre_lat])
                if np.abs(target_lon-centre_lon) >= 180:
                    dlon = np.amax([np.abs(dlon),np.abs(np.abs(target_lon-centre_lon)-360)])
                else:
                    dlon = np.amax([dlon,np.abs(target_lon-centre_lon)])
    minlon = centre_lon-dlon
    subset = precip.intersection(longitude = (minlon,centre_lon+dlon),latitude=(centre_lat+dS,centre_lat+dN))
    dist_mask = np.ones_like(subset.data)
    for yy,target_lat in enumerate(subset.coord('latitude').points):
        for xx,target_lon in enumerate(subset.coord('longitude').points):
            pt_dist = haversine((centre_lat,centre_lon),(target_lat,target_lon))
            if pt_dist >= dist_min and pt_dist <= dist_max:
                dist_mask[:,yy,xx] = 0
    subset_mask = subset.copy(data=np.ma.array(subset.data,mask=dist_mask))
    return(subset_mask)

def compute_spatial_summary(precip,ndivs):
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
    
    onon_freq.data.mask = upper_thresh.data.mask
    offon_freq.data.mask = upper_thresh.data.mask
    onoff_freq.data.mask = upper_thresh.data.mask
    offoff_freq.data.mask = upper_thresh.data.mask

    space_inter.data = 0.5*((onon_freq.data+offoff_freq.data)-(onoff_freq.data+offon_freq.data))
    space_inter.data.mask = upper_thresh.data.mask
    space_inter_mean = space_inter.collapsed('month_number',iris.analysis.MEAN,mdtol=0)  
    space_inter_mean.data = np.nanmean(space_inter.data,axis=0)
    space_inter_mean.var_name='spatial_onoff_metric_mean'
    space_inter_mean.long_name='spatial intermittency on-off metric based on '+str(ndivs)+' divisions (mean of all months in wet season)'
    onon_freq_mean = onon_freq.collapsed('month_number',iris.analysis.MEAN)
    onon_freq_mean.data = np.nanmean(onon_freq.data,axis=0)
    onon_freq_mean.var_name='prob_onon_mean'
    onon_freq_mean.long_name='Probability of upper division neighbouring upper division (mean of all months in wet season)'
    onoff_freq_mean = onoff_freq.collapsed('month_number',iris.analysis.MEAN)
    onoff_freq_mean.data = np.nanmean(onoff_freq.data,axis=0)
    onoff_freq_mean.var_name='prob_onoff_mean'
    onoff_freq_mean.long_name='Probability of upper division neighbouring lower division (mean of all months in wet season)'
    offon_freq_mean = offon_freq.collapsed('month_number',iris.analysis.MEAN)
    offon_freq_mean.data = np.nanmean(offon_freq.data,axis=0)
    offon_freq_mean.var_name='prob_offon_mean'
    offon_freq_mean.long_name='Probability of lower division neighbouring upper division (mean of all months in wet season)'
    offoff_freq_mean = offoff_freq.collapsed('month_number',iris.analysis.MEAN)
    offoff_freq_mean.data = np.nanmean(offoff_freq.data,axis=0)
    offoff_freq_mean.var_name='prob_offoff_mean'
    offoff_freq_mean.long_name='Probability of lower division neighbouring lower division (mean of all months in wet season)'
    out_cubelist = [space_inter,onon_freq,onoff_freq,offon_freq,offoff_freq,lower_thresh,upper_thresh,space_inter_mean,onon_freq_mean,onoff_freq_mean,offon_freq_mean,offoff_freq_mean]
    return(out_cubelist)

def compute_gridcorr_grid(precip,grid):
    import iris
    import iris.analysis.stats as istats
    corr_map = istats.pearsonr(precip,grid,corr_coords='time')
    weights = iris.analysis.cartography.area_weights(corr_map)
    output=corr_map.collapsed(['longitude','latitude'],iris.analysis.MEAN,weights=weights)
    return(output.data)

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

    for lat in range(1,nlat-1):
        point_metrics = []
        for lon in range(nlon):
            if upper_thresh.data.mask[lat,lon]:
                point_metric = dask.delayed(compute_spatial_onoff_metric_point)(None,None,None,None)
            else:
                if lon == 0:
                    min_lon = lon_coord.points[-1]-360
                else:
                    min_lon = lon_coord.points[lon-1]
                if lon == nlon-1:
                    max_lon = lon_coord.points[0]+360
                else:
                    max_lon = lon_coord.points[lon+1]

                upper_neighbors = upper_mask.intersection(longitude = (min_lon,max_lon), latitude = (lat_coord.points[lat-1],lat_coord.points[lat+1]))
                lower_neighbors = lower_mask.intersection(longitude = (min_lon,max_lon), latitude = (lat_coord.points[lat-1],lat_coord.points[lat+1]))
                on_neighbors_mask = upper_neighbors.copy() 
                off_neighbors_mask = lower_neighbors.copy()
                for y in range(len(upper_neighbors.coord('latitude').points)):
                    for x in range(len(upper_neighbors.coord('longitude').points)):
                        on_neighbors_mask.data[:,y,x] = 1-upper_mask.data[:,lat,lon] # Mask neighbors where central point is above upper threshold
                        off_neighbors_mask.data[:,y,x] = 1-lower_mask.data[:,lat,lon] # Mask neighbors where central point is below lower threshold
                on_neighbors_mask.data[:,1,1] = 1 # Mask values at central point
                off_neighbors_mask.data[:,1,1] = 1 # Mask values at central point
                point_metric = dask.delayed(compute_spatial_onoff_metric_point)(on_neighbors_mask,off_neighbors_mask,upper_neighbors,lower_neighbors)
            point_metrics.append(point_metric)
        result = dask.compute(*point_metrics)
        result = np.ma.asarray(result)
        result = np.reshape(result,(nlon,4))
        onon[lat,:] = result[:,0] ; onoff[lat,:] = result[:,1] ; offon[lat,:] = result[:,2] ; offoff[lat,:] = result[:,3]
    output = np.stack([onon,onoff,offon,offoff],axis=0)
    return(output)

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

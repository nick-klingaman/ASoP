def get_key_dict(set_name):
    if set_name == '3hr_all':
        key_dict = {
            'key_list': ['GPM_IMERG','AWI','BCC','CanESM5','ACCESS-CM','ACCESS-ESM','CMCC','CNRM-LR','CNRM-HR','CNRM-ESM',\
                'EC-Earth','EC-Earth-Veg','EC-Earth-Veg-LR','FGOALS','GFDL-CM','GFDL-ESM','GISS','HadGEM','IITM-ESM',\
                'IPSL','KACE','MIROC','MPI-HAM','MPI-LR','MPI-HR','MRI','NESM','SAM','UKESM'],
            'control': 'GPM_IMERG'
        }
    elif set_name == 'day_all':
        key_dict = {
            'key_list': ['GPM_IMERG','AWI','BCC','CanESM5','ACCESS-CM','ACCESS-ESM','CESM','CMCC','CNRM-LR','CNRM-HR','CNRM-ESM',\
                'EC-Earth','EC-Earth-Veg','EC-Earth-Veg-LR','FGOALS','GFDL-CM','GFDL-ESM','GISS','HadGEM','IITM-ESM','INM',\
                'IPSL','KACE','MIROC','MPI-HAM','MPI-LR','MPI-HR','MRI','NESM','NorESM','SAM','TaiESM','UKESM'],
            'control': 'GPM_IMERG'
        }
    elif set_name == '3hr_day_2x2':
        # Combined set of models with both 3hr and daily data that can be analysed at 2x2 resolution.
        # Used for work with Gill on CMIP6.
        key_dict = {
            'key_list': ['ACCESS-CM','ACCESS-ESM','AWI','BCC','CMCC','CNRM-LR','CNRM-HR','CNRM-ESM','EC-Earth',\
                'EC-Earth-Veg','EC-Earth-Veg-LR','GFDL-ESM','HadGEM','IPSL','KACE','MIROC','MPI-HAM','MPI-LR',\
                'MPI-HR','MRI','NESM','SAM','UKESM'],
            'control': 'GPM_IMERG'
        }
    elif set_name == '3hr_day_2x2_mmm':
        key_dict = {
            'key_list': ['ACCESS-CM','AWI','BCC','CMCC','CNRM-HR','EC-Earth','GFDL-ESM','HadGEM','IITM-ESM','IPSL',\
                'KACE','MIROC','MPI-HR','MRI','NESM','SAM'],
            'control': 'GPM_IMERG'
        }
    elif set_name == 'metum_ga7-ga8_diff_gpm':
        key_dict = {
            'key_list': ['bs774','bs819'],
            'control': 'GPM_IMERG'
        }
    elif set_name == 'metum_ga8_diff_ga7':
        key_dict = {
            'key_list': ['bs819',],
            'control': 'bs774'
        }

    else:
        raise Exception("I do not know about set "+set_name)
    return(key_dict)

def get_asop_dict(key,time=None,grid='',months=None):
    from pathlib import Path
    cmip6_path=Path('/media/nick/lacie_tb3/cmip6')
    metum_path=Path('/media/nick/lacie_tb3/metum')
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
        }
        if time == '3hr':
            asop_dict['file_pattern'] = '3B-HHR.MS.MRG.3IMERG.*_means_'+grid_str+'V06*.nc'
        elif time == 'day':
            asop_dict['file_pattern'] = '3B-DAY.MS.MRG.3IMERG.*.V06.'+grid_str+'nc'
    elif key == 'bs774':
        asop_dict={
            'desc': 'bs774_'+time,
            'dir': metum_path/'u-bs774',
            'file_pattern': 'bs774_'+time+'_*'+grid_str+'nc',
            'name': 'GA7-N216_'+time,
            'start_year': 1982,
            'stop_year': 2008,
            'legend_name': 'GA7-N216_'+time,
            'months': [6,7,8,9]
        }
    elif key == 'bs819':
        asop_dict={
            'desc': 'bs819_'+time,
            'dir': metum_path/'u-bs819',
            'file_pattern': 'bs819_'+time+'_*'+grid_str+'nc',
            'name': 'GA8-N216_'+time,
            'start_year': 1982,
            'stop_year': 2008,
            'legend_name': 'GA8-N216_'+time,
            'months': [6,7,8,9]
        }
    elif key == 'AWI':
        asop_dict={
            'desc': 'AWI-CM-1-1-MR_historical_r1i1p1f1_gn_'+time,
            'dir': cmip6_path/'AWI-CM-1-1-MR',
            'file_pattern': 'pr_'+time+'*0.'+grid_str+'nc',
            'name': 'AWI_CM-1-1-MR_'+time,
            'legend_name': 'AWI_'+time,
        }
    elif key == 'BCC':
        asop_dict={
            'dir': cmip6_path/'BCC-CSM2-MR',
            'desc': 'BCC-CSM2-MR_historical_r1i1p1f1_gn_'+time,
            'name': 'BCC-CSM2-MR_'+time,
            'file_pattern': 'pr_'+time+'*'+last_digit+'.'+grid_str+'nc',
            'legend_name': 'BCC_'+time,
        }
    elif key == 'CanESM5':
        asop_dict={
            'dir': cmip6_path/'CanESM5',
            'desc': 'CanESM5_historical_r1i1p1f1_gn_'+time,
            'name': 'CanESM5_'+time,
            'file_pattern': 'pr_'+time+'*'+last_digit+'.'+grid_str+'nc',
            'legend_name': 'CanESM5_'+time,
        }
    elif key == 'ACCESS-CM':
        asop_dict={
            'desc': 'ACCESS-CM2_historical_r1i1p1f1_gn_'+time,
            'dir': cmip6_path/'ACCESS-CM2',
            'name': 'ACCESS-CM2_'+time,
            'file_pattern': 'pr_'+time+'*'+last_digit+'.'+grid_str+'nc',
            'legend_name': 'ACCESS-CM2_'+time,
        }
    elif key == 'ACCESS-ESM':
        asop_dict={
            'desc': 'ACCESS-ESM1-5_historical_r1i1p1f1_gn_'+time,
            'dir': cmip6_path/'ACCESS-ESM1-5',
            'name': 'ACCESS-ESM1-5_'+time,
            'file_pattern': 'pr_'+time+'*'+last_digit+'.'+grid_str+'nc',
            'legend_name': 'ACCESS-ESM_'+time,
        }
    elif key == 'CESM':
        asop_dict={
            'desc': 'CESM2_historical_r1i1p1f1_gn_'+time,
            'dir': cmip6_path/'CESM2',
            'name': 'CESM2_'+time,
            'file_pattern': 'pr_'+time+'*'+last_digit+'.'+grid_str+'nc',
            'legend_name': 'CESM2_'+time,
        }
    elif key == 'CMCC':
        asop_dict={
            'desc': 'CMCC-CM2-SR5_historical_r1i1p1f1_gn_'+time,
            'dir': cmip6_path/'CMCC-CM2-SR5',
            'name': 'CMCC-CM2-SR5_'+time,
            'file_pattern': 'pr_'+time+'*'+last_digit+'.'+grid_str+'nc',
            'legend_name': 'CMCC-CM2-SR5_'+time,
        }
    elif key == 'CNRM-LR':
        asop_dict={
            'desc': 'CNRM-CM6-1_historical_r1i1p1f1_gn_'+time,
            'dir': cmip6_path/'CNRM-CM6-1',
            'name': 'CNRM-CM6-1_'+time,
            'file_pattern': 'pr_'+time+'*'+last_digit+'.'+grid_str+'nc',
            'legend_name': 'CNRM-CM6-1_'+time,
        }
    elif key == 'CNRM-HR':
        asop_dict={
            'desc': 'CNRM-CM6-1-HR_historical_r1i1p1f1_gn_'+time,
            'dir': cmip6_path/'CNRM-CM6-1-HR',
            'name': 'CNRM-CM6-1-HR_'+time,
            'file_pattern': 'pr_'+time+'*'+last_digit+'.'+grid_str+'nc',
            'legend_name': 'CNRM-CM6-1-HR_'+time,
        }
    elif key == 'CNRM-ESM':
        asop_dict={
            'desc': 'CNRM-ESM2-1_historical_r1i1p1f1_gn_'+time,
            'dir': cmip6_path/'CNRM-ESM2-1',
            'name': 'CNRM-ESM2-1_'+time,
            'file_pattern': 'pr_'+time+'*'+last_digit+'.'+grid_str+'nc',
            'legend_name': 'CNRM-ESM2-1_'+time,
        }
    elif key == 'EC-Earth':
        asop_dict={
            'desc': 'EC-Earth3_historical_r1i1p1f1_gr_'+time,
            'dir': cmip6_path/'EC-Earth3',
            'name': 'EC-Earth3_'+time,
            'file_pattern': 'pr_'+time+'*'+last_digit+'.'+grid_str+'nc',
            'legend_name': 'EC-Earth3'
        }
    elif key == 'EC-Earth-Veg':
        asop_dict={
            'desc': 'EC-Earth3-Veg_historical_r1i1p1f1_gr_'+time,
            'dir': cmip6_path/'EC-Earth3-Veg',
            'name': 'EC-Earth3-Veg_'+time,
            'file_pattern': 'pr_'+time+'*'+last_digit+'.'+grid_str+'nc',
            'legend_name': 'EC-Earth3-Veg'
        }
    elif key == 'EC-Earth-Veg-LR':
        asop_dict={
            'desc': 'EC-Earth3-Veg-LR_historical_r1i1p1f1_gr_'+time,
            'dir': cmip6_path/'EC-Earth3-Veg-LR',
            'name': 'EC-Earth3-Veg-LR_'+time,
            'file_pattern': 'pr_'+time+'*'+last_digit+'.'+grid_str+'nc',
            'legend_name': 'EC-Earth3-Veg-LR'
        }
    elif key == 'FGOALS':
        asop_dict={
            'desc': 'FGOALS-g3_historical_r1i1p1f1_gn_'+time,
            'dir': cmip6_path/'FGOALS-g3',
            'name': 'FGOALS-g3_'+time,
            'file_pattern': 'pr_'+time+'*'+last_digit+'.'+grid_str+'nc',
            'legend_name': 'FGOALS-g3_'+time,
        }
    elif key == 'GFDL-CM':
        asop_dict={
            'desc': 'GFDL-CM4_historical_r1i1p1f1_gr1_'+time,
            'dir': cmip6_path/'GFDL-CM4',
            'name': 'GFDL-CM4_'+time,
            'file_pattern': 'pr_'+time+'*_gr1_*'+last_digit+'.'+grid_str+'nc',
            'legend_name': 'GFDL-CM4_'+time,
        }
    elif key == 'GFDL-ESM':
        asop_dict={
            'desc': 'GFDL-ESM4_historical_r1i1p1f1_gr1_'+time,
            'dir': cmip6_path/'GFDL-ESM4',
            'name': 'GFDL-ESM4_'+time,
            'file_pattern': 'pr_'+time+'*_gr1_*'+last_digit+'.'+grid_str+'nc',
            'legend_name': 'GFDL-ESM4_'+time,
        }
    elif key == 'GISS':
        asop_dict={
            'dir': cmip6_path/'GISS-E2-1-G',
            'name': 'GISS-E2-1-G_'+time,
            'desc': 'GISS-E2-1-G_historical_r1i1p1f1_gn_'+time,
            'file_pattern': 'pr_'+time+'*30.'+grid_str+'nc',
            'legend_name': 'GISS-E2-1-G_'+time,
        }
    elif key == 'HadGEM':
        asop_dict={
            'desc': 'HadGEM3-GC31-MM_historical_r1i1p1f3_gn_'+time,
            'dir': cmip6_path/'HadGEM3-GC31-MM',
            'name': 'HadGEM3-GC31-MM_'+time,
            'file_pattern': 'pr_'+time+'*30.'+grid_str+'nc',
            'legend_name': 'HadGEM3-GC31-MM_'+time,
        }
    elif key == 'IITM-ESM':
        asop_dict={
            'desc': 'IITM-ESM_historical_r1i1p1f1_gn_'+time,
            'dir': cmip6_path/'IITM-ESM',
            'name': 'IITM-ESM_'+time,
            'file_pattern': 'pr_'+time+'*'+last_digit+'.'+grid_str+'nc',
            'legend_name': 'IITM-ESM_'+time
        }
    elif key == 'INM':
        asop_dict={
            'desc': 'INM-CM5-0_historical_r1i1p1f1_gr1_'+time,
            'dir': cmip6_path/'INM-CM5-0',
            'name': 'INM-CM-5-0_'+time,
            'file_pattern': 'pr_'+time+'*'+last_digit+'.'+grid_str+'nc',
            'legend_name': 'INM-CM5_'+time,
        }
    elif key == 'IPSL':
        asop_dict={
            'desc': 'IPSL-CM6A-LR_historical_r1i1p1f1_gr1_'+time,
            'dir': cmip6_path/'IPSL-CM6A-LR',
            'name': 'IPSL-CM6A-LR_'+time,
            'file_pattern': 'pr_'+time+'*'+last_digit+'.'+grid_str+'nc',
            'legend_name': 'IPSL-CM6A-LR_'+time,
        }
    elif key == 'KACE':
        asop_dict={
            'desc': 'KACE-1-0-G_historical_r1i1p1f1_gr_'+time,
            'dir': cmip6_path/'KACE-1-0-G',
            'name': 'KACE-1-0-G_'+time,
            'file_pattern': 'pr_'+time+'*30.'+grid_str+'nc',
            'legend_name': 'KACE-1-0-G_'+time,
        }
    elif key == 'MIROC':
        asop_dict={
            'dir': cmip6_path/'MIROC6',
            'name': 'MIROC6_'+time,
            'desc': 'MIROC6_historical_r1i1p1f1_gn_'+time,
            'file_pattern': 'pr_'+time+'*'+last_digit+'.'+grid_str+'nc',
            'legend_name': 'MIROC6_'+time,
        }
    elif key == 'MPI-HAM':
        asop_dict={
            'dir': cmip6_path/'MPI-ESM1-2-HAM',
            'name': 'MPI-ESM1-2-HAM_'+time,
            'desc': 'MPI-ESM1-2-HAM_historical_r1i1p1f1_gn_'+time,
            'file_pattern': 'pr_'+time+'*'+last_digit+'.'+grid_str+'nc',
            'legend_name': 'MPI-ESM1-2-HAM_'+time,
        }
    elif key == 'MPI-LR':
        asop_dict={
            'dir': cmip6_path/'MPI-ESM1-2-LR',
            'name': 'MPI-ESM1-2-LR_'+time,
            'desc': 'MPI-ESM1-2-LR_historical_r1i1p1f1_gn_'+time,
            'file_pattern': 'pr_'+time+'*'+last_digit+'.'+grid_str+'nc',
            'legend_name': 'MPI-ESM1-2-LR_'+time,
        }
    elif key == 'MPI-HR':
        asop_dict={
            'dir': cmip6_path/'MPI-ESM1-2-HR',
            'name': 'MPI-ESM1-2-HR_'+time,
            'desc': 'MPI-ESM1-2-HR_historical_r1i1p1f1_gn_'+time,
            'file_pattern': 'pr_'+time+'*'+last_digit+'.'+grid_str+'nc',
            'legend_name': 'MPI-ESM1-2-HR_'+time,
        }
    elif key == 'MRI':
        asop_dict={
            'dir': cmip6_path/'MRI-ESM2-0',
            'name': 'MRI-ESM2-0_'+time,
            'desc': 'MPI-ESM2-0_historical_r1i1p1f1_gn_'+time,
            'file_pattern': 'pr_'+time+'*'+last_digit+'.'+grid_str+'nc',
            'legend_name': 'MRI-ESM2-0_'+time,
        }
    elif key == 'NESM':
        asop_dict={
            'dir': cmip6_path/'NESM3',
            'name': 'NESM3_'+time,
            'desc': 'NESM3_historical_r1i1p1f1_gn_'+time,
            'file_pattern': 'pr_'+time+'*'+last_digit+'.'+grid_str+'nc',
            'legend_name': 'NESM3_'+time,
        }
    elif key == 'NorESM':
        asop_dict={
            'dir': cmip6_path/'NorESM2-MM',
            'name': 'NorESM2-MM_'+time,
            'desc': 'NorESM2-MM_historical_r1i1p1f1_gn_'+time,
            'file_pattern': 'pr_'+time+'*'+last_digit+'.'+grid_str+'nc',
            'legend_name': 'NorESM2-MM_'+time,
        }
    elif key == 'SAM':
        asop_dict={
            'dir': cmip6_path/'SAM0-UNICON',
            'name': 'SAM_'+time,
            'desc': 'SAM0-UNICON_historical_r1i1p1f1_gn_'+time,
            'file_pattern': 'pr_'+time+'*'+last_digit+'.'+grid_str+'nc',
            'legend_name': 'SAM0-UNICON_'+time,
        }
    elif key == 'TaiESM':
        asop_dict={
            'dir': cmip6_path/'TaiESM1',
            'name': 'TaiESM1_'+time,
            'desc': 'TaiESM1_historical_r1i1p1f1_gn_'+time,
            'file_pattern': 'pr_'+time+'*'+last_digit+'.'+grid_str+'nc',
            'legend_name': 'TaiESM1_'+time,
        }
    elif key == 'UKESM':
        asop_dict={
            'dir': cmip6_path/'UKESM1-0-LL',
            'name': 'UKESM1-0-LL_'+time,
            'desc': 'UKESM1-0-LL_historical_r1i1p1f2_gn_'+time,
            'file_pattern': 'pr_'+time+'*30.'+grid_str+'nc',
            'legend_name': 'UKESM1-0-LL_'+time,
        }
    else:
        raise Exception('No dictionary for '+key)
    asop_dict['region'] = [-60,60,0,360]
    if 'start_year' not in asop_dict.keys():
        asop_dict['start_year'] = 1980
    if 'stop_year' not in asop_dict.keys():
        asop_dict['stop_year'] = 2014
    if 'months' not in asop_dict.keys():
        asop_dict['months'] = months
    asop_dict['year_range'] = str(asop_dict['start_year'])+'-'+str(asop_dict['stop_year'])
    if grid is not '':
        asop_dict['desc'] = asop_dict['desc']+'_'+grid
        asop_dict['name'] = asop_dict['name']+'_'+grid
        asop_dict['grid_str'] = grid_str
    asop_dict['time_type'] = time
    return(asop_dict)

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
import asop_coherence_global as asop_global
import argparse



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-m',dest='model',help='Model to process (eg AWI)')
    parser.add_argument('-t',dest='time',help='Frequency to process (eg 3hr, day)')
    parser.add_argument('-n',dest='grid',help='Grid spacing to process (in degrees, eg 2x2)')
    parser.add_argument('-o',dest='overwrite',action='store_true',default=False,required=False)
    args = parser.parse_args()
    model = args.model
    timetype = args.time
    grid = args.grid
    masked_overwrite = args.overwrite

    client = Client()
    wet_season_threshold = 1.0/24.0
    wet_season_threshold_str='1d24'
    min_precip_threshold = 1.0 # mm/day
    min_precip_threshold_str='1mmpd'

    print('--> '+model)
    asop_dict = asop_global.get_asop_dict(model,timetype,grid=grid)
    year_range = str(asop_dict['start_year'])+'-'+str(asop_dict['stop_year'])

    masked_precip_file=str(asop_dict['dir'])+'/'+asop_dict['desc']+'_asop_'+year_range+'_masked_precip_wetseason'+wet_season_threshold_str+'.nc'
    masked_min_precip_file=str(asop_dict['dir'])+'/'+asop_dict['desc']+'_asop_'+year_range+'_masked_precip_wetseason'+wet_season_threshold_str+'_minprecip'+min_precip_threshold_str+'.nc'
    spatial_summary_file=str(asop_dict['dir'])+'/'+str(asop_dict['desc'])+'_asop_'+year_range+'_spatial_summary_wetseason'+wet_season_threshold_str+'.nc'
    spatial_corr_file=str(asop_dict['dir'])+'/'+str(asop_dict['desc'])+'_asop_'+year_range+'_spatial_corr_wetseason'+wet_season_threshold_str+'.nc'

    if os.path.exists(masked_min_precip_file) and not masked_overwrite:
        print('-->--> Reading masked precipitation (by wet season and by min precip) from file')
        masked_min_precip = iris.load_cube(masked_min_precip_file)
    elif os.path.exists(masked_precip_file) and not masked_overwrite:
        print('-->--> Reading masked precipitation (by wet season) from file')
        masked_precip = iris.load_cube(masked_precip_file)
        print('-->--> Masking precip for minimum rain rate')
        masked_min_precip = asop_global.mask_min_precip(masked_precip,min_precip_threshold=min_precip_threshold)
        masked_min_precip.var_name='precipitation_flux_masked'
        masked_min_precip.long_name='Masked precipitation for wet season (threshold '+wet_season_threshold_str+' of annual total) and min mean precip (threshold '+min_precip_threshold_str+' mm/day)'
        with dask.config.set(scheduler='synchronous'):
            iris.save(masked_min_precip,masked_min_precip_file)
    else:
        print('-->--> Reading precipitation ')
        precip = asop_global.load_cmip6(asop_dict)
        print('-->--> Masking precipitation for wet season')
        masked_precip = asop_global.mask_wet_season(precip)
        masked_precip.var_name='precipitation_flux_masked'
        masked_precip.long_name='Masked precipitation for wet season (threshold '+wet_season_threshold_str+' of annual total)'
        with dask.config.set(scheduler='synchronous'):
            iris.save(masked_precip,masked_precip_file)
        print('-->--> Masking precipitation for minimum rain rate')
        masked_min_precip = asop_global.mask_min_precip(masked_precip,min_precip_threshold=min_precip_threshold)
        masked_min_precip.var_name='precipitation_flux_masked'
        masked_min_precip.long_name='Masked precipitation for wet season (threshold '+wet_season_threshold_str+' of annual total) and min mean precip (threshold '+min_precip_threshold_str+' mm/day)'
        with dask.config.set(scheduler='synchronous'):
            iris.save(masked_min_precip,masked_min_precip_file)
    print('-->--> Computing spatial correlation metrics')
    spatial_corr = asop_global.compute_equalgrid_corr_global(masked_min_precip,[0,400,600,800,1000,1200])
    print('-->--> Computing spatial summary metrics')
    spatial_summary = asop_global.compute_spatial_summary(masked_min_precip,4)
    with dask.config.set(scheduler='synchronous'):
        iris.save(spatial_summary,spatial_summary_file)
        iris.save(spatial_corr,spatial_corr_file)
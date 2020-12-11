import iris
from pathlib import Path
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import iris.coord_categorisation
from iris.experimental.equalise_cubes import equalise_attributes
from iris.util import unify_time_units
import dask
from dask.distributed import Client,progress,LocalCluster
import os
import asop_dict as adict
import asop_coherence_global as asop_global
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-m',dest='model',help='Model to process (eg AWI)',required=False,default=None)
    parser.add_argument('-s',dest='model_set',help='Model set to process (see cmip6_dict.py)',required=False,default=None)
    parser.add_argument('-t',dest='time',help='Frequency to process (eg 3hr, day)')
    parser.add_argument('-n',dest='grid',help='Grid spacing to process (in degrees, eg 2x2)')
    parser.add_argument('-o',dest='overwrite',action='store_true',default=False,required=False)
    parser.add_argument('-d',dest='diag_overwrite',action='store_true',default=False,required=False)
    parser.add_argument('-w',dest='workers',default=None,required=False,help='Number of workers (processes)')
    args = parser.parse_args()
    model = args.model
    model_set = args.model_set
    if model_set is None and model is None:
        raise Exception("You must specify at least a model (-m) or model set (-s)")
    if model_set is not None:
        key_list = adict.get_key_list(model_set)
    else:
        key_list = [model,]
    timetype = args.time
    grid = args.grid
   # workers = int(args.workers)
    masked_overwrite = args.overwrite
    diag_overwrite = args.diag_overwrite
    
    if args.workers is not None:
        cluster = LocalCluster(scheduler_port=0,dashboard_address=0,n_workers=int(args.workers),connection_limit=4096)
        client = Client(cluster) #,processes=False)
    else:
        cluster = LocalCluster(scheduler_port=0,dashboard_address=0,connection_limit=4096)
        client = Client(cluster) #,processes=False)
    print(client)
#        client = Client(connection_limit=1024) #,processes=False)
    wet_season_threshold = 1.0/24.0
    wet_season_threshold_str='1d24'
    min_precip_threshold = 1.0 # mm/day
    min_precip_threshold_str='1mmpd'

    for model in key_list:
        print('--> '+model)
        asop_dict = adict.get_asop_dict(model,timetype,grid=grid)
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
        
        haversine_file = str(asop_dict['dir'])+'/'+asop_dict['desc']+'_asop_haversine_map.nc'
        if os.path.exists(haversine_file):
            haversine_map = iris.load_cube(haversine_file)
        else:
            print('-->--> Computing Haversine distances')
            haversine_map = asop_global.compute_haversine_map(masked_min_precip)
            with dask.config.set(scheduler='synchronous'):
                iris.save(haversine_map,haversine_file)

        if os.path.exists(spatial_summary_file) and not diag_overwrite:
            print('-->--> Will not overwrite existing spatial summary output file')
        else:
            print('-->--> Computing spatial summary metrics')
            spatial_summary = asop_global.compute_spatial_summary(masked_min_precip,4)
            with dask.config.set(scheduler='synchronous'):
                iris.save(spatial_summary,spatial_summary_file)
        if os.path.exists(spatial_corr_file) and not diag_overwrite:
            print('-->--> Will not overwrite existing spatial correlations output file')
        else:
            print('-->--> Computing spatial correlation metrics')
            spatial_corr = asop_global.compute_equalgrid_corr_global(masked_min_precip,haversine_map,[0,250,450,650,850,1050])
            with dask.config.set(scheduler='synchronous'):
                iris.save(spatial_corr,spatial_corr_file)
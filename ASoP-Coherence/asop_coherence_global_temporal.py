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
import asop_dict as adict
import argparse


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-m',dest='model',help='Model to process (eg AWI)')
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
    masked_overwrite = args.overwrite
    diag_overwrite = args.diag_overwrite
    if args.workers is not None:
        client = Client(n_workers=int(args.workers),connection_limit=4096,processes=False)
    else:
        client = Client(connection_limit=1024,processes=False)
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
        temporal_summary_file=str(asop_dict['dir'])+'/'+str(asop_dict['desc'])+'_asop_'+year_range+'_temporal_summary_wetseason'+wet_season_threshold_str+'.nc'
        temporal_autocorr_file=str(asop_dict['dir'])+'/'+str(asop_dict['desc'])+'_asop_'+year_range+'_temporal_autocorr_wetseason'+wet_season_threshold_str+'.nc'

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
        if os.path.exists(temporal_summary_file) and not diag_overwrite:
            print('-->--> Will not overwrite existing temporal summary output file')
        else:
            print('-->--> Computing temporal summary metrics')
            temporal_summary = asop_global.compute_temporal_summary(masked_min_precip,4)
            with dask.config.set(scheduler='synchronous'):
                iris.save(temporal_summary,temporal_summary_file)
        if os.path.exists(temporal_autocorr_file) and not diag_overwrite:
            print('-->--> Will not overwrite existing temporal correlation output file')
        else:
            print('-->--> Computing temporal autocorrelation metrics')
            temporal_autocorr = asop_global.compute_temporal_autocorr(masked_min_precip,range(17))
            with dask.config.set(scheduler='synchronous'):
                iris.save(temporal_autocorr,temporal_autocorr_file)


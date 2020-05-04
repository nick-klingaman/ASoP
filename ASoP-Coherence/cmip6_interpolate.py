import iris
from pathlib import Path
import glob
import os
import numpy as np

def get_asop_dict(key):
    datapath=Path('/media/nick/lacie_tb3/data_from_gill/CMIP6')
    if key == 'AWI':
        asop_dict={
            'desc': 'AWI-CM-1-1-MR_historical_r1i1p1f1_gn',
            'dir': datapath/'AWI-CM-1-1-MR',
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
            'dir': datapath/'BCC-CSM2-MR',
            'name': 'BCC',
            'desc': 'BCC-CSM2-MR_historical_r1i1p1f1_gn',
            'dt': 10800,
            'legend_name': 'BCC',
            'region': [-90,90,0,360],
            'color': 'blue',
            'symbol': '8'
        }
    elif key == 'ACCESS':
        asop_dict={
            'dir': datapath/'ACCESS-CM2',
            'name': 'ACCESS',
            'desc': 'ACCESS-CM2_historical_r1i1p1f1_gn',
            'dt': 10800,
            'legend_name': 'ACCESS',
            'region': [-90,90,0,360],
            'color': 'purple',
            'symbol': '>'
        }
    elif key == 'FGOALS':
        asop_dict={
            'dir': datapath/'FGOALS-g3',
            'name': 'FGOALS',
            'desc': 'FGOALS-g3_historical_r1i1p1f1_gn',
            'dt': 10800,
            'legend_name': 'FGOALS',
            'region': [-90,90,0,360],
            'color': 'brown',
            'symbol': '<'
        }
    elif key == 'GISS':
        asop_dict={
            'dir': datapath/'GISS-E2-1-G',
            'name': 'GISS',
            'desc': 'GISS-E2-1-G_historical_r1i1p1f1_gn',
            'dt': 10800,
            'legend_name': 'GISS',
            'region': [-90,90,0,360],
            'color': 'brown',
            'symbol': '<'
        }
    elif key == 'MIROC':
        asop_dict={
            'dir': datapath/'MIROC6',
            'name': 'MIROC',
            'desc': 'MIROC6_historical_r1i1p1f1_gn',
            'dt': 10800,
            'legend_name': 'MIROC6',
            'region': [-90,90,0,360],
            'color': 'brown',
            'symbol': '<'
        }
    elif key == 'MPI-ESM1':
        asop_dict={
            'dir': datapath/'MPI-ESM1-2-HR',
            'name': 'MPI-ESM',
            'desc': 'MPI-ESM1-2-HR_historical_r1i1p1f1_gn',
            'dt': 10800,
            'legend_name': 'MPI-ESM',
            'region': [-90,90,0,360],
            'color': 'brown',
            'symbol': '<'
        }
    elif key == 'SAM0-UNICON':
        asop_dict={
            'dir': datapath/'SAM0-UNICON',
            'name': 'SAM',
            'desc': 'SAM0-UNICON_historical_r1i1p1f1_gn',
            'dt': 10800,
            'legend_name': 'SAM',
            'region': [-90,90,0,360],
            'color': 'brown',
            'symbol': '<'
        }
    else:
        raise Exception('No dictionary for '+key)
    return(asop_dict)

def interp_3x3(cube):
    interp_lon = iris.coords.DimCoord(np.arange(1.5,360,3),standard_name='longitude',units='degrees_east',circular=True)
    nlon = len(interp_lon.points)
    interp_lat = iris.coords.DimCoord(np.arange(-88.5,90,3),standard_name='latitude',units='degrees')
    nlat = len(interp_lat.points)
    interp_cube = iris.cube.Cube(data=np.empty((nlat,nlon)),dim_coords_and_dims=[(interp_lat,0),(interp_lon,1)])
    interp_cube.coord('longitude').guess_bounds()
    interp_cube.coord('latitude').guess_bounds()
    out_cube = cube.regrid(interp_cube,iris.analysis.AreaWeighted(mdtol=0))
    return(out_cube)

def avg_daily(cube):
    from iris.coord_categorisation import add_day_of_year,add_year
    add_day_of_year(cube,'time')
    add_year(cube,'time')
    print(cube)
    daily_cube = cube.aggregated_by(['day_of_year','year'],iris.analysis.MEAN)
    print(daily_cube)
    return(daily_cube)

models=['BCC','GISS']#,'MPI-ESM1','ACCESS','FGOALS']
do_daily = True
do_3x3 = False
for model in models:
    print(model)
    asop_dict = get_asop_dict(model)
    infiles = glob.glob(str(asop_dict['dir'])+'/pr_3hr_'+asop_dict['desc']+'_*0.nc')
    print(asop_dict['dir'])
    print(infiles)
    for infile in infiles:
        print(infile)
        cube = iris.load_cube(infile)
        if do_daily:
            daily_cube = avg_daily(cube)
            outfile = os.path.splitext(infile)[0]+'.daily.nc'
            iris.save(daily_cube,outfile)
        if do_3x3:
            interp_cube = interp_3x3(cube)
            outfile = os.path.splitext(infile)[0]+'.3x3.nc'
            iris.save(interp_cube,outfile)
#iris.save(out_interp_cube,basedir+'/3B-HHR.MS.MRG.3IMERG.'+date+'.3hr_means_3x3.V06B.nc',unlimited_dimensions=['time'],zlib=True)
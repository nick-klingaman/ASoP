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
            'dt': 10800,
            'legend_name': 'BCC',
            'region': [-90,90,0,360],
            'color': 'blue',
            'symbol': '8'
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

models=['AWI']
for model in models:
    asop_dict = get_asop_dict(model)
    for year in range(asop_dict['start_year'],asop_dict['stop_year']+1):
        infile = glob.glob(str(asop_dict['dir'])+'/pr_3hr_'+asop_dict['desc']+'*'+str(year)+'*.nc3')
        for infile in infile:
            print(infile)
            cube = iris.load_cube(infile)
            interp_cube = interp_3x3(cube)
            outfile = os.path.splitext(infile)[0]+'.3x3.nc'
            iris.save(interp_cube,outfile)
#iris.save(out_interp_cube,basedir+'/3B-HHR.MS.MRG.3IMERG.'+date+'.3hr_means_3x3.V06B.nc',unlimited_dimensions=['time'],zlib=True)
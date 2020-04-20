import iris
from iris.util import unify_time_units
from iris.experimental.equalise_cubes import equalise_attributes
import iris.coord_categorisation
import argparse
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import datetime as dt
import os

parser = argparse.ArgumentParser()
parser.add_argument('-y',dest='process_year',help='Year to process (YYYY)')
args = parser.parse_args()
year = int(args.process_year)
basedir = '/media/nick/lacie_tb3/datasets/GPM_IMERG/'+str(year)
#year = date[0:4]

overwrite_3hr = False
overwrite_3x3 = True
dt_date = dt.datetime(year=year,month=1,day=1) 
while dt_date < dt.datetime(year=year+1,month=1,day=1):
    date = dt_date.strftime('%Y%m%d')
    outfile_3hr = basedir+'/3B-HHR.MS.MRG.3IMERG.'+date+'.3hr_means.V06B.nc'
    outfile_3hr_3x3 = basedir+'/3B-HHR.MS.MRG.3IMERG.'+date+'.3hr_means_3x3.V06B.nc'
    if not os.path.exists(outfile_3hr) or overwrite_3hr:
        print('Creating 3hr means for '+date)
        cubes = iris.load(basedir+'/3B*.'+date+'-S*-E*.*.V06B.nc')
        equalise_attributes(cubes)
        unify_time_units(cubes)
        cube = cubes.concatenate_cube()
        iris.coord_categorisation.add_hour(cube,'time')
        out_cubelist = iris.cube.CubeList()
        for start_hr in range(0,24,3):
            this_3hr = cube.extract(iris.Constraint(hour=lambda cell: start_hr <= cell < start_hr+3))
            this_3hr_mean = this_3hr.collapsed('time',iris.analysis.MEAN)
            out_cubelist.append(this_3hr_mean)
        equalise_attributes(out_cubelist)
        unify_time_units(out_cubelist)
        out_cube = out_cubelist.merge_cube()
        out_cube = out_cube.intersection(longitude=(0,360))
        out_cube.coord('longitude').guess_bounds()
        out_cube.coord('latitude').guess_bounds()
        out_cube.coord('longitude').circular = True
        iris.save(out_cube,outfile_3hr,unlimited_dimensions='time',zlib=True)
    else:
        out_cube = iris.load_cube(outfile_3hr)
        out_cube.coord('longitude').circular = True
    if not os.path.exists(outfile_3hr_3x3) or overwrite_3x3:
        print('Interpolating 3hr means to 3x3 for '+date)
        interp_lon = iris.coords.DimCoord(np.arange(1.5,360,3),standard_name='longitude',units='degrees_east',circular=True)
        nlon = len(interp_lon.points)
        interp_lat = iris.coords.DimCoord(np.arange(-88.5,90,3),standard_name='latitude',units='degrees')
        nlat = len(interp_lat.points)
        interp_cube = iris.cube.Cube(data=np.empty((nlat,nlon)),dim_coords_and_dims=[(interp_lat,0),(interp_lon,1)])
        interp_cube.coord('longitude').guess_bounds()
        interp_cube.coord('latitude').guess_bounds()
        out_interp_cube = out_cube.regrid(interp_cube,iris.analysis.AreaWeighted(mdtol=1))
        iris.save(out_interp_cube,outfile_3hr_3x3,unlimited_dimensions=['time'],zlib=True)
    dt_date = dt_date+dt.timedelta(days=1)
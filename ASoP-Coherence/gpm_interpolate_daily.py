import iris
from iris.util import unify_time_units
from iris.experimental.equalise_cubes import equalise_attributes
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import datetime as dt
import os
import glob
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-y',dest='process_year',help='Year to process (YYYY)')
args = parser.parse_args()
year = int(args.process_year)
basedir = '/media/nick/lacie_tb3/datasets/GPM_IMERG/daily/'+str(year)
filelist = glob.glob(basedir+'/3B-DAY.MS.MRG.3IMERG.'+str(year)+'*-S000000-E235959.V06.nc4.nc4')

#for myfile in filelist:
cubelist = iris.load(filelist,'Daily accumulated precipitation (combined microwave-IR) estimate')
unify_time_units(cubelist)
equalise_attributes(cubelist)
cube = cubelist.concatenate_cube()

print(cube)
interp_lon = iris.coords.DimCoord(np.arange(1.5,360,3),standard_name='longitude',units='degrees_east',circular=True)
nlon = len(interp_lon.points)
interp_lat = iris.coords.DimCoord(np.arange(-88.5,90,3),standard_name='latitude',units='degrees')
nlat = len(interp_lat.points)
interp_cube = iris.cube.Cube(data=np.empty((nlat,nlon)),dim_coords_and_dims=[(interp_lat,0),(interp_lon,1)])
interp_cube.coord('longitude').guess_bounds()
interp_cube.coord('latitude').guess_bounds()
cube.coord('longitude').guess_bounds()
cube.coord('latitude').guess_bounds()
cube.coord('longitude').circular = True
cube = cube.intersection(longitude=(0,360))
out_cube = cube.regrid(interp_cube,iris.analysis.AreaWeighted())
out_cube.transpose(new_order=[0,2,1])
basefile = basedir+'/3B-DAY.MS.MRG.3IMERG.'+str(year)+'.V06.3x3.nc'
print(basefile)
iris.save(out_cube,basefile)


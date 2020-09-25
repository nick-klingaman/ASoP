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

def interp_nxn(cube,n):
    import iris
    import numpy as np
    
    n = np.int(n)
    print(n)
    if cube.coord('longitude').has_bounds() == False:
        cube.coord('longitude').guess_bounds()
    if cube.coord('latitude').has_bounds() == False:
        cube.coord('latitude').guess_bounds()
    latitude = iris.coords.DimCoord(np.linspace(-90+n/2,90-n/2,180//n), standard_name='latitude', units='degrees_north')
    longitude = iris.coords.DimCoord(np.linspace(n/2,360-n/2,360//n),standard_name='longitude',units='degrees_east',circular=True)
    target_cube = iris.cube.Cube(np.zeros((180//n,360//n),np.float32),dim_coords_and_dims=[(latitude,0),(longitude,1)])
    target_cube.coord('longitude').circular=True
    target_cube.coord('longitude').guess_bounds()
    target_cube.coord('latitude').guess_bounds()
    target_cube.coord('longitude').coord_system=cube.coord('longitude').coord_system
    target_cube.coord('latitude').coord_system=cube.coord('latitude').coord_system
    cube_interp = cube.regrid(target_cube,iris.analysis.AreaWeighted(mdtol=1))
    return(cube_interp)

overwrite=False
parser = argparse.ArgumentParser()
parser.add_argument('-y',dest='process_year',help='Year to process (YYYY)',type=int)
parser.add_argument('-n',dest='interp_n',help='Gridsize to which to interpolate (in degrees n x n)',type=int)
parser.add_argument('--overwrite',dest='overwrite',help='Overwrite existing interpolated files',action='store_true')
args = parser.parse_args()
year = args.process_year
interp_n = args.interp_n
overwrite = args.overwrite

basedir = '/media/nick/lacie_tb3/datasets/GPM_IMERG/daily/'+str(year)
filelist = glob.glob(basedir+'/3B-DAY.MS.MRG.3IMERG.'+str(year)+'*-S000000-E235959.V06.nc4.nc4')

print('-> Process year '+str(year))
print('---> Reading data ...')
cubelist = iris.load(filelist,'Daily accumulated precipitation (combined microwave-IR) estimate')
unify_time_units(cubelist)
equalise_attributes(cubelist)
cube = cubelist.concatenate_cube()
cube.coord('longitude').circular = True
cube = cube.intersection(longitude=(0,360))

basefile = os.path.splitext(basedir+'/3B-DAY.MS.MRG.3IMERG.'+str(year)+'.V06.'+str(interp_n)+'x'+str(interp_n)+'.nc')[0]
if not os.path.exists(basefile) or overwrite:
    print('---> Interpolating ...')
    out_cube = interp_nxn(cube,interp_n)
    out_cube.transpose(new_order=[0,2,1])
    print('---> Output to '+basefile)
    iris.save(out_cube,basefile+'.'+str(interp_n)+'x'+str(interp_n)+'.nc')


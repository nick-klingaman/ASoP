import iris
from iris.time import PartialDateTime
from pathlib import Path
import glob
import os
import numpy as np
import asop_dict as adict
import argparse

def interp_nxn(cube,n):
    import iris
    import numpy as np
    
    n = np.int(n)
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

def avg_daily(cube):
    from iris.coord_categorisation import add_day_of_year,add_year
    add_day_of_year(cube,'time')
    add_year(cube,'time')
    daily_cube = cube.aggregated_by(['day_of_year','year'],iris.analysis.MEAN)
    return(daily_cube)

parser = argparse.ArgumentParser()
parser.add_argument('-m',dest='model',help='Model to process (eg AWI)',required=False,default=None)
parser.add_argument('-s',dest='model_set',help='Set of models to process (see cmip6_dict.py)',required=False,default=None)
parser.add_argument('-t',dest='time',help='Frequency to process (eg 3hr, day)')
parser.add_argument('-n',dest='grid',help='Grid spacing to which to interpolate (in degrees, eg 2 for 2x2')
parser.add_argument('-o',dest='overwrite',action='store_true',default=False,required=False)
args = parser.parse_args()
model = args.model
model_set = args.model_set
if model_set is None and model is None:
    raise Exception("You must specify at least a model (-m) or model set (-s)")
timetype = args.time
overwrite = args.overwrite
gridn = args.grid

if model_set is not None:
    key_list = adict.get_key_list(model_set)
else:
    key_list = [model,]

for model in key_list:
    print('--> '+model)
    asop_dict = adict.get_asop_dict(model,timetype)
    print(asop_dict['dir'],asop_dict['file_pattern'])
    infiles = glob.glob(str(asop_dict['dir'])+'/'+asop_dict['file_pattern'])
    for infile in infiles:
        print('-->--> '+infile)
        outfile = os.path.splitext(infile)[0]+'.'+str(gridn)+'x'+str(gridn)+'.nc'
        if not os.path.exists(outfile) or overwrite:
            print('-->-->--> Interpolating ...')
            cube = iris.load_cube(infile)
            cube = cube.extract(iris.Constraint(time = lambda cell: PartialDateTime(year=1980) <= cell.point <= PartialDateTime(year=2020)))
            if cube is not None:
                interp_cube = interp_nxn(cube,gridn)
                iris.save(interp_cube,outfile)

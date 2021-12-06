# Copyright 2021 Lawrence Livermore National Security, LLC
"""
This script is used by cmec-driver to run the ASoP-Coherence metrics.
It is based on the workflow in asop_coherence_example.py and
can be called with the aruments listed below. If no configuration file
with module settings is provided, the settings will be estimated.

Arguments:
    * model_dir:
        directory containing model data
    * obs_dir:
        directory containing obs data
    * wk_dir:
        output directory
    * config:
        JSON config file (optional)

Author: Ana Ordonez
"""
import argparse
import csv
import glob
import numpy as np
import os
import shutil
import json
import iris
import matplotlib
from matplotlib import cm

from cf_units import Unit
import asop_coherence as asop

# setting output directory names
figure_dir_name = "asop_figures"
metrics_dir_name = "asop_metrics"

def get_dictionary(filename):

    """
    The Coherence package relies on a dataset dictionary. This
    version generates a data dictionary based on metadata, coordinates,
    and attributes in the input datasets.

    Default values are provided for a standard observation dataset.

    Arguments:
        * filename:  The (base) name of the input file.

    Returns:
        * asop_dict: A dictionary containing required and optional parameters for the ASoP Coherence package.
    """
    asop_dict = {}
    # Defaults for standard observational data
    if 'CMORPH_V1.0.mjodiab_period_3hrmeans.precip.nc' in filename or \
       'TRMM_3B42V7A.mjodiab_period_3hrmeans.precip.nc' in filename:
        asop_dict['infile']       = filename
        asop_dict['name']         = ''
        asop_dict['dt']           = 10800
        asop_dict['dx']           = 27
        asop_dict['dy']           = 27
        asop_dict['constraint']   = 'precipitation'
        asop_dict['scale_factor'] = 8.0
        asop_dict['legend_name']  = ''
        asop_dict['region']       = [-10,10,60,90]
        asop_dict['box_size']     = 1680
        asop_dict['color']        = 'red'
        asop_dict['region_size']  = 7
        asop_dict['lag_length']   = 6
        asop_dict['grid_type']    = 'native'
        asop_dict['time_type']    = '3hr'
        asop_dict['grid_desc']    = 'native'
        asop_dict['time_desc']    = '3-hourly'
        asop_dict['autocorr_length'] = 60*60*24
    else:
        asop_dict=build_asop_dict(filename)
    return(asop_dict)

def build_asop_dict(filename):
    """Auto-populate the asop_dict using the input data file. These are
    default settings only.
    Arguments:
    * filename: File name of the dataset to describe.
    """
    cube = iris.load_cube(filename)

    # Look for name keys. Default first 15 characters of filename
    name = filename.split('/')[-1][0:15]
    for key in ['source_id','source_label','short_name','name','long_name']:
        if key in cube.attributes:
            name = cube.attributes[key]
            break
    if 'variant_label' in cube.attributes:
        name += ('_' + cube.attributes['variant_label'])

    constraint = cube.standard_name

    # Get coordinate deltas
    t1 = cube.coords('time')[0][0].cell(0).point
    t2 = cube.coords('time')[0][1].cell(0).point
    # Try assuming datetime object, otherwise int
    try:
        dt = int((t2 - t1).total_seconds())
    except AttributeError: # assume units of days
        dt = int((t2-t1)*60*60*24)
        print('Warning: Time units not found. Units assumed to be "days"')

    # Estimate average grid spacing in km
    dims=[cube.dim_coords[n].standard_name for n in range(0,len(cube.dim_coords))]
    dim_name_lat = None
    dim_name_lon = None
    for dim in dims:
        if 'lat' in dim.lower():
            dim_name_lat = dim
        elif 'lon' in dim.lower():
            dim_name_lon = dim
    if (dim_name_lat is None) or (dim_name_lon is None):
        raise RuntimeError(filename+': latitude or longitude dimension not found.\n'
            + 'Valid latitude names contain "lat" and '
            + 'valid longitude names contain "lon"')
    deltas = {}
    for dvar,coord in zip(['dy','dx'],[dim_name_lat,dim_name_lon]):
        if cube.coords(coord)[0].is_monotonic():
            coord_list = cube.coords(coord)[0].points
            # Estimating at equator for now
            deltas[dvar] = np.absolute(np.mean(np.diff(coord_list))*110)

    # get time descriptions
    if (dt > 86399) and (dt < 86401):
        # This is a day, with some room for error
        time_type = 'day'
        time_desc = 'daily'
    elif dt >= 86401:
        days = round(dt/(60*60*24))
        time_type = str(days) + 'day'
        time_desc = str(days) + '-day'
    elif (dt > 10799) and (dt < 10801):
        time_type = '3hr'
        time_desc = '3-hourly'
    elif (dt > 3599) and (dt <  3601):
        time_type = '1hr'
        time_desc = 'hourly'
    elif dt <= 3599:
        minutes = round(dt/60)
        time_type = str(minutes) + 'min'
        time_desc = str(minutes) + '-min'
    elif dt <= 86399:
        # catch other hour lengths
        hours = round(dt/60*60)
        time_type = str(hours) + 'hour'
        time_desc = str(hours) + '-hourly'

    # Set scale factor, starting with some common units
    pr_units = cube.units
    scale_factor = 1
    if pr_units ==  Unit('mm'):
        scale_factor = round(86400 / dt)
    elif pr_units == Unit('kg m-2 s-1'):
        scale_factor = 86400
    else:
        try:
            # Find conversion factor between 1 in dataset's units
            # and a "benchmark" unit
            bm = Unit('kg m-2 day-1')
            scale_factor = pr_units.convert(1,bm)
        except ValueError:
            try:
                bm = Unit('mm day-1')
                scale_factor = pr_units.convert(1,bm)
            except ValueError:
                print("Warning: Could not determine scale factor. Using default of "+scale_factor)

    asop_dict = {}
    asop_dict['infile']       = filename
    asop_dict['name']         = name
    asop_dict['dt']           = dt
    asop_dict['dx']           = deltas['dx']
    asop_dict['dy']           = deltas['dy']
    asop_dict['constraint']   = constraint
    asop_dict['scale_factor'] = scale_factor
    asop_dict['legend_name']  = ''
    asop_dict['region']       = ''
    asop_dict['box_size']     = 7*deltas['dx']
    asop_dict['color']        = ''
    asop_dict['region_size']  = 7
    asop_dict['lag_length']   = 6
    asop_dict['grid_type']    = ''
    asop_dict['time_type']    = time_type
    asop_dict['grid_desc']    = ''
    asop_dict['time_desc']    = time_desc
    asop_dict['autocorr_length'] = 8*dt

    return asop_dict

def update_asop_dict(asop_dict,region,coords,color,all_settings):
    """Adjust data dictionary quantities based on region and unique settings.
    Args:
    * asop_dict: Dictionary of settings for dataset
    * region: Name of region
    * coords: List of region boundary coordinates
    * color: Code or string designating a color
    * all_settings: Dictionary of data from CMEC settings file
    """
    # Set unique color
    asop_dict['color'] = color

    # Apply any general user settings
    asop_dict['grid_desc'] = all_settings.get('grid','native')
    asop_dict['grid_type'] = all_settings.get('grid','native')
    asop_dict['region_name'] = region
    asop_dict['region_desc'] = region.replace('_',' ')
    asop_dict['region'] = coords

    # Edit dx for region
    mean_lat = np.mean(coords[0:2])
    asop_dict['dx'] = asop_dict['dx'] * np.cos(np.radians(mean_lat))
    all_settings.pop('infile','') # key not allowed
    for key in asop_dict:
        if key in all_settings:
            asop_dict[key] = all_settings[key]

    # Apply any specific file settings
    infile = os.path.basename(asop_dict['infile'])
    file_settings = settings.get(infile,{})
    file_settings.pop('infile','') # key not allowed
    file_settings.pop('region','')
    if file_settings:
        for key in file_settings:
            asop_dict[key] = file_settings[key]
    if 'legend_name' not in file_settings:
        asop_dict['legend_name'] = asop_dict['name'].replace('_',' ')

    print('---> Final data dictionary:')
    print(json.dumps(asop_dict, sort_keys=True, indent=2))

    return asop_dict

def initialize_descriptive_json(json_filename,wk_dir,model_dir,obs_dir):
    """Create the output.json metadata file for the CMEC metrics package.
    Arguments:
    * json_filename: Name of the metadata file (recommended: output.json)
    * wk_dir: Path to output directory (parent of descriptive json)
    * model_dir: Path to model input directory to record in descriptive json
    * obs_dir: Path to obs input directory to record in descriptive json
    """
    output = {'provenance':{},'data':{},'metrics':{},'plots':{},'index': 'index.html','html':'index.html'}
    log_path = wk_dir + '/asop_coherence.log.txt'
    output['provenance'] = {'environment': get_env(),
            'modeldata': model_dir,
            'obsdata': obs_dir,
            'log': log_path}
    with open(json_filename,'w') as output_json:
        json.dump(output,output_json, indent=2)

    return

def initialize_metrics_json(metrics_filename):
    """
    Create the metrics json file with the CMEC results structure.
    Arguments:
    * metrics_filename:
        Name of the metrics file to initialize.
    """
    json_structure = ['dataset','region','metric','statistic']
    metrics = {
        'SCHEMA': {
            'name': 'CMEC',
            'version': 'v1',
            'package': 'ASoP'},
        'DIMENSIONS':{
            'json_structure': json_structure,
            'dimensions': {
                'dataset': {},
                'region': {},
                'metric': {
                    'Temporal intermittency': {},
                    'Spatial intermittency': {}},
                'statistic': {
                    'p(upper|upper)': "Probability of upper quartile precipitation followed by upper quartile precipitation",
                    'p(lower|lower)': "Probability of lower quartile precipitation followed by lower quartile precipitation",
                    'p(upper|lower)': "Probability of upper quartile precipitation followed by lower quartile precipitation",
                    'p(lower|upper)': "Probability of lower quartile precipitation followed by upper quartile precipitation",
                    'combined': 'Metric of coherence (combined probabilities)'
                    }}},
        'RESULTS': {},
        'REFERENCE': 'Klingaman et al. (2017, GMD, doi:10.5194/gmd-10-57-2017)'}
    with open(metrics_filename,'w') as fname:
        json.dump(metrics,fname,indent=2)

    return

def load_and_transpose_csv(metrics_csv):
    """Load csv into a dictionary and transpose rows to columns.
    Args:
    * metrics_csv:
        Path to csv file to load and transpose
    """
    models = []
    with open(metrics_csv,'r') as f:
        reader = csv.reader(f)
        fields = next(reader)
        fields = fields[1:] # remove "Dataset"
        out_dict = dict.fromkeys(fields,{})
        for row in reader:
            model_name = row[0]
            models.append(model_name)
            for i,item in enumerate(row[1:]):
                tmp = out_dict[fields[i]].copy()
                tmp[model_name] = item
                out_dict[fields[i]] = tmp.copy()
    return out_dict

def merge_csvs(metrics_csv_dims,metrics_csv_int,region,wk_dir):
    """Combine the data from the dimensions and metrics csv files
    into one file.
    Args:
    * metrics_dsv_dims:
        Name of csv file with dimension data
    * metrics_csv_int:
        Name of csv file with intermittency metrics
    * region:
        Region name
    * wk_dir:
        Path to output directory
    """
    dict_dims = load_and_transpose_csv(metrics_csv_dims)
    dict_int = load_and_transpose_csv(metrics_csv_int)
    dict_dims.update(dict_int.copy())
    fname = os.path.join(wk_dir,region.replace(' ','_') + '_summary.csv')
    fields = [*dict_dims]
    models = ['Metric']+[*dict_dims[fields[0]]]

    with open(fname,'w') as f:
        writer = csv.DictWriter(f,fieldnames=models)
        writer.writeheader()
        for field in fields:
            row = {'Metric': field}
            row.update(dict_dims.get(field,{}))
            writer.writerow(row)

    data_desc = {
        os.path.basename(fname): {
            'longname': os.path.basename(fname).split('.')[1].replace('_',' '),
            'description': 'Parameters and metrics for ' + region + ' region'}}

    # add metadata to output.json
    asop.update_output_json('metrics', data_desc, wk_dir)

def metrics_to_json(metrics_csv_int, region, coords, metrics_filename):
    """Move intermittency metrics from CSV files to CMEC formatted JSON.
    Args:
    * metrics_csv_int:
        Name of intermittency metrics csv
    * region:
        Name of region
    * coords:
        List of region boundary coordinates
    * metrics_filename:
        Name of metrics JSON to output
    """
    data = {}
    with open(metrics_csv_int,'r') as f:
        reader = csv.reader(f)
        fields = next(reader)
        for row in reader:
            data[row[0]] = {"Temporal intermittency": {},
                    "Spatial intermittency": {}}
            # skip the first key in fields, clean up field name
            for i,field in enumerate(fields[1:6]):
               data[row[0]]["Temporal intermittency"].update({field[5:]:float(row[i+1])})
            for i,field in enumerate(fields[6:]):
               data[row[0]]["Spatial intermittency"].update({field[3:]:float(row[i+6])})
    with open(metrics_filename, 'r') as fname:
        metrics = json.load(fname)

    # Add region to dimensions information
    metrics['DIMENSIONS']['dimensions']['region'].update({region: coords})

    # Update model statistics
    for model in data:
        if not (model in metrics['RESULTS']):
            metrics['RESULTS'][model] = {}
            metrics['DIMENSIONS']['dimensions']['dataset'].update({model: {}})
        metrics['RESULTS'][model][region] = data[model]

    # Write new metrics to same file
    with open(metrics_filename, 'w') as fname:
        json.dump(metrics,fname,indent = 2)

def delete_intermediate_csvs(wk_dir):
    """Move files to a figure or metrics subdirectory as appropriate.
    Delete intermediate csv files.
    Args:
    * wk_dir:
        CMEC output directory path
    """
    # Remove intermediate csv tables
    out_files = os.listdir(wk_dir)
    delete_keys = ["int_metrics","region_dims"]
    delete_list = [f for f in out_files if any(x in f for x in delete_keys)]
    for f in delete_list:
        os.remove(f)

def write_index_html(wk_dir,region_dict,metrics_filename,ext="png"):
    """Create an html page that links users to the metrics json and
    plots created by ASoP-Coherence. Results must be located in the
    output directory "wk_dir".
    Arguments:
        * wk_dir:
            Output directory
        * region_dict:
            Dictionary of region names
        * metrics_filename:
            Path to metrics JSON file
    """
    # Make lists of the metrics and figure files to display
    metrics_dir = os.path.join(wk_dir,metrics_dir_name)
    metric_list = sorted([
        f for f in os.listdir(metrics_dir) if f.endswith('_summary.csv')])
    plot_list=[]
    fig_list=sorted([f for f in os.listdir(wk_dir+'/'+figure_dir_name)])
    for keyword in ['lag','correlations','twodpdf']:
        plot_list.append([f for f in fig_list if (keyword in f)]) # sort datasets
    subtitle_list=['Autocorrelation','2D Histograms','Correlation maps']

    # Start working on html text. Each line is appened to a list that
    # is then written to file.
    html_file=['<html>\n',
        '<body>','<head><title>ASoP-Coherence</title></head>\n',
        '<br><h1>ASoP-Coherence results</h1>\n','<h2>Contents</h2>\n',
        '<dl>\n','<dt><a href="#Metrics">Metrics</a></dt>\n',
        '<dt><a href="#Figures">Figures</a></dt>\n',
        '<dd><a href="#Autocorrelation">Autocorrelation</a></dd>\n',
        '<dd><a href="#2D-Histograms">2D Histograms</a></dd>\n',
        '<dd><a href="#Correlation-maps">Correlation Maps</a></dd>\n',
        '</dl>\n''<section id="Metrics">\n','<br><h2>Metrics</h2>\n']
    html_file.append('<h3>Intermittency Metrics</h3>\n')

    # Display metrics JSON in dashboard option
    metrics_json = os.path.basename(metrics_filename)
    metrics_relocated = os.path.join(metrics_dir_name,metrics_json)
    tmp='<p><a href="'+metrics_relocated+'" target="_blank">'+metrics_json+'</a></p>\n'
    html_file.append(tmp)

    # Link CSV tables for download
    html_file.append('<h3>Tables</h3>\n')
    for metric_file in metric_list:
        metric_path = os.path.join(metrics_dir_name,metric_file)
        html_file.append('<p><a href="{0}">{1}</a></p>\n'.format(metric_path,metric_file))
    html_file.append('<br>\n')
    html_file.append('</section>\n')

    # Add figures
    html_file.append('<section id="Figures">\n')
    html_file.append('<h2>Figures</h2>\n')
    for title,category in zip(subtitle_list,plot_list):
        html_file.append('<section id='+title.replace(' ','-')+'>\n')
        html_file.append('<h3>{0}</h3>\n'.format(title))
        # Adjust figure width for autocorrelation
        fwidth = "647"
        if title=="Autocorrelation":
            fwidth="450"
        for region in region_dict:
            html_file.append('<h4>{0}</h4>\n'.format(region.replace('_',' ')))
            region_fig = [f for f in category if (region.replace(" ","_") in f)]
            for fig in region_fig:
                tmp = '<p><a href="{0}" target="_blank" alt={0}>' + \
                '<img src="{0}" width={1} alt="{0}"></a></p>\n'
                html_file.append(
                    tmp.format(os.path.join(figure_dir_name,fig),fwidth))
        html_file.append('</section>\n')
    html_file.append('</section>\n')

    html_file.append('</body>\n</html>\n')
    filename=wk_dir+'/index.html'
    with open(filename,'w') as html_page:
        html_page.writelines(html_file)

def get_env():
    """
    Return versions of dependencies.
    """
    from platform import python_version
    versions = {}
    versions['iris'] = iris.__version__
    versions['matplotlib'] = matplotlib.__version__
    versions['numpy'] = np.__version__
    versions['python'] = python_version()
    return versions

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='ASoP Coherence parser')
    parser.add_argument('model_dir', help='model directory')
    parser.add_argument('obs_dir', help='observations directory')
    parser.add_argument('wk_dir', help='output directory')
    parser.add_argument('--config', help='configuration file', default=None)
    args = parser.parse_args()
    model_dir=args.model_dir
    obs_dir=args.obs_dir
    wk_dir=args.wk_dir
    config=args.config

    # Load user settings from cmec config
    with open(config) as config_file:
        settings = json.load(config_file)['ASoP/Coherence']
    ext = '.'+settings.get('figure_type','png').replace(".","")
    if 'regions' in settings:
        if isinstance(settings['regions'],dict):
            region_dict = settings['regions']
        else:
            print('Error: Please provide region dictionary')
    else:
        region_dict = {'default': [-10,10,60,90]}

    # Use all files in the obs and model directories
    dataset_list = []
    if obs_dir !='None':
        obs_file_list = glob.glob(os.path.join(obs_dir,'*'))
        dataset_list.extend(obs_file_list)
    mod_file_list = glob.glob(os.path.join(model_dir,'*'))
    dataset_list.extend(mod_file_list)
    n_datasets = len(dataset_list)

    # Create metrics and figure folders
    fig_dir = os.path.join(wk_dir,figure_dir_name)
    met_dir = os.path.join(wk_dir,metrics_dir_name)
    os.mkdir(fig_dir)
    os.mkdir(met_dir)
    json_filename=os.path.join(wk_dir,'output.json')
    initialize_descriptive_json(json_filename,wk_dir,model_dir,obs_dir)
    metrics_filename = os.path.join(wk_dir,met_dir,'intermittency_metrics.json')
    initialize_metrics_json(metrics_filename)

    # Allocate memory for multi-model fields
    max_box_distance,max_timesteps,max_boxes = asop.parameters()
    all_distance_correlations = np.zeros((n_datasets,max_box_distance))
    all_distance_ranges = np.zeros((n_datasets,3,max_box_distance))
    all_distance_max = np.zeros((n_datasets),dtype=np.int64)
    all_time_correlations = np.zeros((n_datasets,max_timesteps))
    all_time_max = np.zeros((n_datasets),dtype=np.int64)
    all_dt = np.zeros((n_datasets),dtype=np.int64)
    all_legend_names = []

    # Initialize colors
    cmap = matplotlib.cm.get_cmap('Paired')
    all_colors = [cmap(n) for n in list(np.linspace(0,1,num=n_datasets))]
    name_cache = []

    for region in region_dict:
        metrics_csv_int=met_dir+'/int_metrics_'+region.replace(' ','_')+'.csv'
        metrics_csv_dims=met_dir+'/region_dims_'+region.replace(' ','_')+'.csv'

        for i,dataset in enumerate(dataset_list):
            print('--> '+region)
            print('--> '+str(dataset))
            asop_dict = get_dictionary(dataset)
            print('----> dt: ', asop_dict['dt'])

            # Update region, colors, other specific settings
            asop_dict = update_asop_dict(asop_dict,region,region_dict[region],all_colors[i],settings)

            # Read precipitation data.
            precip = asop.read_precip(asop_dict)

            # Define edges of bins for 1D and 2D histograms
            # Note that on plots, the first and last edges will be
            # replaced by < and > signs, respectively.
            bins=[0,1,2,4,6,9,12,16,20,25,30,40,60,90,130,180,2e20]
            bins=np.asarray(bins)

            # Compute 1D and 2D histograms
            oned_hist, twod_hist = asop.compute_histogram(precip,bins)

            # Plot 1D and 2D histograms (e.g., Fig. 2a in Klingaman et al. 2017).
            asop.plot_histogram(oned_hist,twod_hist,asop_dict,bins,wk_dir=fig_dir,ext=ext)

            # Compute correlations as a function of native gridpoints, by dividing
            # analysis region into sub-regions (boxes of length region_size).  Also
            # computes lag correlations to a maximum lag of lag_length.
            corr_map,lag_vs_distance,autocorr,npts_map,npts = asop.compute_equalgrid_corr(precip,asop_dict,metrics_csv=metrics_csv_dims,wk_dir=wk_dir)

            # Plot correlations as a function of native gridpoints and time lag
            # (e.g., Figs. 2c and 2e in Klingaman et al. 2017).
            asop.plot_equalgrid_corr(corr_map,lag_vs_distance,autocorr,npts,asop_dict,wk_dir=fig_dir,ext=ext)

            # Compute correlations as a function of physical distance, by dividing
            # analysis region into sub-regions (boxes of length box_size).
            all_distance_correlations[i,:],all_distance_ranges[i,:,:],all_distance_max[i] = asop.compute_equalarea_corr(precip,asop_dict)

            # Compute lagged autocorrelations over all points
            all_time_correlations[i,:],all_time_max[i] = asop.compute_autocorr(precip,asop_dict)

            # Compute spatial and temporal coherence metrics, based on quartiles (4 divisions)
            space_inter, time_inter = asop.compute_spacetime_summary(precip,4,metrics_csv=metrics_csv_int,short_name=asop_dict['name'],wk_dir=wk_dir)

            # Save dataset timestep information
            all_dt[i] = asop_dict['dt']

            # Save color and legend information
            #all_colors.append(asop_dict['color'])
            all_legend_names.append(asop_dict['legend_name'])

        # Add the intermittency metrics for this region to JSON
        metrics_to_json(metrics_csv_int, region, region_dict[region], metrics_filename)

        # Plot correlations as a function of physical distance for all datasets
        asop.plot_equalarea_corr(all_distance_correlations,all_distance_ranges,all_distance_max,colors=all_colors,legend_names=all_legend_names,set_desc=region,wk_dir=fig_dir,ext=ext)

        # Plot correlations as a function of physical time for all datasets
        asop.plot_autocorr(all_time_correlations,all_time_max,dt=all_dt,colors=all_colors,legend_names=all_legend_names,set_desc=region,wk_dir=fig_dir,ext=ext)

        # Load and process csv files for this region to make main csv
        merge_csvs(metrics_csv_dims,metrics_csv_int,region,met_dir)

    # move figures and metrics to new folders and generate html pages
    delete_intermediate_csvs(wk_dir)
    write_index_html(wk_dir,region_dict,metrics_filename,ext=ext)

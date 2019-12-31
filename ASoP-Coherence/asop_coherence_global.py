import asop_coherence as asop
import numpy as np
import iris

def get_dict(key):
	asop_dict={}
	bam='/gws/nopw/j04/klingaman/bam'
	gc2='/gws/nopw/j04/klingaman/metum/gc2'
	trmm='/gws/nopw/j04/klingaman/datasets/TRMM_3B42'
	gpcp='/gws/nopw/j04/klingaman/datasets/GPCP'
	if key == 'bam_t63_2p5':
		asop_dict['dir']=bam
		asop_dict['infile']=bam+'/GPOS1000PREC19810101122010012012C.fct.TQ0062L042_2p5x2p5.nc.1'
		asop_dict['name']='BAM_T63_2p5'
		asop_dict['dt']=86400
		asop_dict['dx']=275
		asop_dict['dy']=275
		asop_dict['constraint']='TIME MEAN TOTAL PRECIPITATION     (mm/day           )'
		asop_dict['scale_factor']=1.0
		asop_dict['legend_name']=r'BAM T63_2.5'
		asop_dict['region']=[-90,90,0,360]
		asop_dict['color']='red'
		asop_dict['grid_type']='2p5'
		asop_dict['time_type']='daily'
		asop_dict['grid_desc']='2.5$\degree$'
		asop_dict['time_desc']='daily'
		asop_dict['mask_file']='/gws/nopw/j04/klingaman/datasets/HADGEM3-KPP_ANCIL/landfrac_2p5x2p5_hadgem3-8.5.nc'
		asop_dict['mask_constraint']='land_area_fraction'
		asop_dict['load_from_file']=True
		asop_dict['symbol']='<'
	if key == 'gc2_n96_2p5':
		asop_dict['dir']=gc2+'/mi-ah288'
		asop_dict['infile']=gc2+'/mi-ah288/ah288.jan-dec_dmeans_ts.1989-2008.precip.2p5x2p5.nc'
		asop_dict['name']='GC2_n96_2p5'
		asop_dict['dt']=86400
		asop_dict['dx']=275
		asop_dict['dy']=275
		asop_dict['constraint']='precipitation_flux' #TOTAL PRECIPITATION RATE     KG/M2/S'
		asop_dict['scale_factor']=86400.0
		asop_dict['legend_name']=r'GC2 N96_2.5'
		asop_dict['region']=[-90,90,0,360]
		asop_dict['color']='blue'
		asop_dict['grid_type']='native'
		asop_dict['time_type']='daily'
		asop_dict['grid_desc']='native'
		asop_dict['time_desc']='daily'
		asop_dict['mask_file']='/gws/nopw/j04/klingaman/datasets/HADGEM3-KPP_ANCIL/landfrac_2p5x2p5_hadgem3-8.5.nc'
		asop_dict['mask_constraint']='land_area_fraction'
		asop_dict['load_from_file']=True
		asop_dict['symbol']='8'
	if key == 'trmm_2p5':
		asop_dict['dir']=trmm+'/two_point_five'
		asop_dict['infile']=asop_dict['dir']+'/test.nc'
		asop_dict['name']='TRMM_2p5'
		asop_dict['dt']=86400
		asop_dict['dx']=275
		asop_dict['dy']=275
		asop_dict['constraint']='daily (0Z-21Z) rainfall total' #TOTAL PRECIPITATION RATE     KG/M2/S'
		asop_dict['scale_factor']=86400.0
		asop_dict['legend_name']='TRMM_2.5'
		asop_dict['region']=[-90,90,0,360]
		asop_dict['color']='black'
		asop_dict['grid_type']='2p5'
		asop_dict['time_type']='daily'
		asop_dict['grid_desc']='2.5$\degree$'
		asop_dict['time_desc']='daily'
		asop_dict['mask_file']='/gws/nopw/j04/klingaman/datasets/HADGEM3-KPP_ANCIL/landfrac_2p5x2p5_hadgem3-8.5.nc'
		asop_dict['mask_constraint']='land_area_fraction'
		asop_dict['load_from_file']=True
		asop_dict['symbol']='x'
	if key == 'gpcp_2p5':
		asop_dict['dir']=gpcp+'/one_degree/two_point_five'
		asop_dict['infile']=asop_dict['dir']+'/gpcp_1dd_v1.3.jan-dec_dmeans.1997-2016.2p5x2p5.nc'
		asop_dict['name']='GPCP_2p5'
		asop_dict['dt']=86400
		asop_dict['dx']=275
		asop_dict['dy']=275
		asop_dict['constraint']='lwe_precipitation_rate' #TOTAL PRECIPITATION RATE     KG/M2/S'
		asop_dict['scale_factor']=86400.0
		asop_dict['legend_name']='GPCP_2.5'
		asop_dict['region']=[-90,90,0,360]
		asop_dict['color']='black'
		asop_dict['grid_type']='2p5'
		asop_dict['time_type']='daily'
		asop_dict['grid_desc']='2.5$\degree$'
		asop_dict['time_desc']='daily'
		asop_dict['mask_file']='/gws/nopw/j04/klingaman/datasets/HADGEM3-KPP_ANCIL/landfrac_2p5x2p5_hadgem3-8.5.nc'
		asop_dict['mask_constraint']='land_area_fraction'
		asop_dict['load_from_file']=True
		asop_dict['symbol']='x'
	if key == 'gc2_n216_2p5':
		asop_dict['dir']=gc2+'/antib'
		asop_dict['infile']=gc2+'/antib/hadgem3_ga6_n216.jan-dec_dmeans_ts.years1-27.precip.2p5x2p5.nc'
		asop_dict['name']='GC2_n216_2p5'
		asop_dict['dt']=86400
		asop_dict['dx']=275
		asop_dict['dy']=275
		asop_dict['constraint']='TOTAL PRECIPITATION RATE     KG/M2/S'
		asop_dict['scale_factor']=86400.0
		asop_dict['legend_name']='GC2 N216_2.5'
		asop_dict['region']=[-90,90,0,360]
		asop_dict['color']='cyan'
		asop_dict['grid_type']='2p5'
		asop_dict['time_type']='daily'
		asop_dict['grid_desc']='2.5'
		asop_dict['time_desc']='daily'
		asop_dict['mask_file']='/gws/nopw/j04/klingaman/datasets/HADGEM3-KPP_ANCIL/landfrac_n216e_hadgem3-10.3.nc'
		asop_dict['mask_constraint']='land_area_fraction'
		asop_dict['load_from_file']=True
		asop_dict['symbol']='o'

	return asop_dict


if __name__ == '__main__':
	datasets=['gpcp_2p5','bam_t63_2p5','gc2_n96_2p5','gc2_n216_2p5']
	n_datasets=len(datasets)

	regions = [([-30,30,0,360],'land','trop_land'),
		   ([-30,30,0,360],'ocean','trop_ocean'),
		   ([-90,-30,0,360],'land','sh_land'),
		   ([-90,-30,0,360],'ocean','sh_ocean'),
		   ([30,90,0,360],'land','nh_land'),
		   ([30,90,0,360],'ocean','nh_ocean'),
		   ([-90,90,0,360],'land','glob_land'),
		   ([-90,90,0,360],'ocean','glob_ocean')]
	n_regions = len(regions)
	space_metrics_plot = np.empty((n_datasets,n_regions))
	time_metrics_plot = np.empty((n_datasets,n_regions))
	all_datasets = []
	all_colors = []
	all_symbols = []
	all_regions = []
	for box,mask_type,region_name in regions:
		all_regions.append(region_name)
	print np.shape(space_metrics_plot)

	for i in xrange(n_datasets):
		print '--> '+datasets[i]
		asop_dict=get_dict(datasets[i])
		all_datasets.append(asop_dict['name'])
		all_colors.append(asop_dict['color'])
		all_symbols.append(asop_dict['symbol'])

		precip = asop.read_precip(asop_dict)
		mask = asop.read_mask(asop_dict)
		land_mask = mask.copy()
		land_mask.data = np.where(mask.data > 0,1,0)
		ocean_mask = mask.copy()
		ocean_mask.data = np.where(mask.data == 0,1,0)

		if asop_dict['load_from_file']:
			space_inter = iris.load_cube(asop_dict['dir']+'/'+asop_dict['name']+'_asop_spatial_metric.nc')
			time_inter = iris.load_cube(asop_dict['dir']+'/'+asop_dict['name']+'_asop_temporal_metric.nc')
			try:
				space_inter.coord('longitude').guess_bounds()
			except:
				pass
			try:
				space_inter.coord('latitude').guess_bounds()
			except:
				pass
			try:
				time_inter.coord('longitude').guess_bounds()
			except:
				pass
			try:
				time_inter.coord('latitude').guess_bounds()
			except:
				pass
		else:
			space_inter, time_inter = asop.compute_spacetime_summary(precip,4,twod=True,cyclic_lon=True)
			iris.save(space_inter,asop_dict['dir']+'/'+asop_dict['name']+'_asop_spatial_metric.nc')
			iris.save(time_inter,asop_dict['dir']+'/'+asop_dict['name']+'_asop_temporal_metric.nc')
		j=0
		for box,mask_type,region_name in regions:
			if mask_type == 'land':
				space_inter_avg = asop.average_spacetime_summary(space_inter,box,mask=land_mask)
				time_inter_avg = asop.average_spacetime_summary(time_inter,box,mask=land_mask)
			elif mask_type == 'ocean':
				space_inter_avg = asop.average_spacetime_summary(space_inter,box,mask=ocean_mask)
				time_inter_avg = asop.average_spacetime_summary(time_inter,box,mask=ocean_mask)
			elif mask_type == 'none':
				space_inter_avg = asop.average_spacetime_summary(space_inter,box,mask=None)
				time_inter_avg = asop.average_spacetime_summary(time_inter,box,mask=None)
			print 'For region '+region_name+', average spatial metric: ',space_inter_avg.data
			print 'For region '+region_name+', average temporal metric: ',time_inter_avg.data
			space_metrics_plot[i,j]=space_inter_avg.data
			time_metrics_plot[i,j]=time_inter_avg.data
			j=j+1

		asop.plot_spacetime_summary_maps(time_inter,precip.coord('latitude').points,precip.coord('longitude').points,'time',asop_dict)
		asop.plot_spacetime_summary_maps(space_inter,precip.coord('latitude').points,precip.coord('longitude').points,'space',asop_dict)
	asop.plot_spacetime_summary_regions(time_metrics_plot,all_regions,all_datasets,all_colors,all_symbols,'time','bam_gc2')
	asop.plot_spacetime_summary_regions(space_metrics_plot,all_regions,all_datasets,all_colors,all_symbols,'space','bam_gc2')

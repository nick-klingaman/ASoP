"""
Function to calculate and plot spectral metric based on Perkins et al (2007) which
calculates the area under the fractional histogram that is covered by overlap between
two individual histograms (e.g. model and obs, but could be two models).

Also calculates spatial rms over specified regions (hardwired here) and outputs to
screen and/or returns as variables.

Gill Martin
November 2020

(c) British Crown Copyright

"""

import sys
import numpy as np

import iris
import iris.coord_categorisation
import matplotlib

import iris.plot as iplt
import iris.quickplot as qplt
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from matplotlib.colors import from_levels_and_colors
from matplotlib.colors import NoNorm

from make_hist_maps import calc_rain_contr, read_data_cube

def plot_metric(filename1,filename2,dataname1,dataname2,season,timescale,dates,maskfile,plotname):
    """
    Args:
    * filename1,filename2:
        files containing histogram counts for two datasets, which must be on the same grid.
        filename2 data are the baseline (e.g. obs) against which data in filename1
        will be compared.
    * dataname1,dataname2:
        names of datasets in filename1, filename2 (e.g. a model name and obs name,
        or two model names)
    * season:
        string descriptor of season for which data in filenames apply (e.g. 'JJA')
    * timescale:
        string descriptor of time frequency of original data (e.g. '3-hourly')
    * dates:
        string descriptor of dates to which histograms apply
    * maskfile:
        filename for land/sea fraction dataset (on same grid as both filenames)
        with values ranging from 0.0 = no land, to 1.0 = all land.
    * plotname:
        filename for plot

    Example usage:
         index, indexl, indexs, indextrop, indexnhml, indexshml=plot_metric(filename1,filename2,dataname,season,'3-hourly','1990-2014',maskfile,plotname)

    """

    seasonc = season.upper()

    ppn_hist_cube1=read_data_cube(filename1)
    ppn_hist_cube2=read_data_cube(filename2)
    tot_rain_bins_seas_a,tot_rain_bins_frac_a=calc_rain_contr(ppn_hist_cube1)
    tot_rain_bins_seas_b,tot_rain_bins_frac_b=calc_rain_contr(ppn_hist_cube2)

    # Limit region to 60S-60N to fit with use of GPM-IMERG observations
    # Could be omitted or made optional according to which two datasets are being compared.

    ce = iris.coords.CoordExtent('longitude', 0.0, 360.0)
    ce2 = iris.coords.CoordExtent('latitude', -59, 59)
    tot_rain_bins_frac_a.coord('longitude').circular=True
    if tot_rain_bins_frac_a.coord('longitude').bounds is None:
        tot_rain_bins_frac_a.coord('longitude').guess_bounds()
        tot_rain_bins_frac_a.coord('latitude').guess_bounds()
    tot_rain_bins_frac_a1 = tot_rain_bins_frac_a.intersection(ce, ce2)
    tot_rain_bins_frac_b.coord('longitude').circular=True
    if tot_rain_bins_frac_b.coord('longitude').bounds is None:
        tot_rain_bins_frac_b.coord('longitude').guess_bounds()
        tot_rain_bins_frac_b.coord('latitude').guess_bounds()
    tot_rain_bins_frac_b1 = tot_rain_bins_frac_b.intersection(ce, ce2)

    minf=tot_rain_bins_frac_a1.copy()
    minf.data=np.minimum(tot_rain_bins_frac_a1.data,tot_rain_bins_frac_b1.data)
    skill=minf.collapsed('precipitation_flux', iris.analysis.SUM,mdtol=1)

    # Get mask (currently total rain <1 mm/day)

    bin_mid2=np.exp(np.log(0.005)+np.sqrt(np.linspace(0.5,98.5,99)*((np.square(np.log(120.)-np.log(0.005)))/59.)))
    bin_mid=np.zeros(100)
    bin_mid[1:]=bin_mid2[0:99]
    bin_mid3=bin_mid.reshape(100,1,1)

    lshista=iris.load_cube(filename1)
    tot_events = lshista.collapsed('precipitation_flux', iris.analysis.SUM)
    tot_events.data = np.ma.masked_where(tot_events.data == 0.0, tot_events.data)
    tot_rain_bins = lshista * bin_mid3
    tot_rain_boxa = tot_rain_bins.collapsed('precipitation_flux', iris.analysis.SUM) /tot_events

    lshistb=iris.load_cube(filename2)
    lshistb.coord('latitude').coord_system = lshista.coord('latitude').coord_system
    lshistb.coord('longitude').coord_system = lshista.coord('longitude').coord_system
    tot_events = lshistb.collapsed('precipitation_flux', iris.analysis.SUM)
    tot_events.data = np.ma.masked_where(tot_events.data == 0.0, tot_events.data)
    tot_rain_bins = lshistb * bin_mid3
    tot_rain_boxb = tot_rain_bins.collapsed('precipitation_flux', iris.analysis.SUM) /tot_events

    tot_rain_boxa.data = np.ma.masked_where(tot_rain_boxa.data <1.0, tot_rain_boxa.data)
    tot_rain_boxb.data = np.ma.masked_where(tot_rain_boxb.data <1.0, tot_rain_boxb.data)
    tot_events = tot_rain_boxa*tot_rain_boxb

    # Extract same area as skill is calculated for

    tot_events.coord('longitude').circular=True
    if tot_events.coord('longitude').bounds is None:
        tot_events.coord('longitude').guess_bounds()
        tot_events.coord('latitude').guess_bounds()
    tot_events1 = tot_events.intersection(ce, ce2)

    # Apply land/sea mask to get sea or land only

    lsmask=iris.load_cube(maskfile)
    lsmask.coord('latitude').coord_system = lshista.coord('latitude').coord_system
    lsmask.coord('longitude').coord_system = lshista.coord('longitude').coord_system
    lsmask.coord('longitude').circular=True
    if lsmask.coord('longitude').bounds is None:
        lsmask.coord('longitude').guess_bounds()
        lsmask.coord('latitude').guess_bounds()
    lsmasks = lsmask.intersection(ce, ce2)
    lsmaskl = lsmask.intersection(ce, ce2)
    lsmasks.data = np.ma.masked_where(lsmasks.data >=0.5, lsmasks.data)
    lsmaskl.data = np.ma.masked_where(lsmaskl.data <0.5, lsmaskl.data)

    lsmasksea=lsmasks*tot_events1
    lsmaskland=lsmaskl*tot_events1

    # Mask the skill cube and calculate index as rms over region

    skill.data.mask = tot_events1.data.mask
    grid_areas = iris.analysis.cartography.area_weights(skill)
    index = skill.collapsed(['latitude','longitude'], iris.analysis.RMS,weights=grid_areas)
    skilltrop=skill.extract(iris.Constraint(latitude = lambda cell: -15 <= cell <= 15))
    grid_areas = iris.analysis.cartography.area_weights(skilltrop)
    indextrop = skilltrop.collapsed(['latitude','longitude'], iris.analysis.RMS,weights=grid_areas)

    skillnhml=skill.extract(iris.Constraint(latitude = lambda cell: 30 <= cell <= 60))
    grid_areas = iris.analysis.cartography.area_weights(skillnhml)
    indexnhml= skillnhml.collapsed(['latitude','longitude'], iris.analysis.RMS,weights=grid_areas)

    skillshml=skill.extract(iris.Constraint(latitude = lambda cell: -60 <= cell <= -30))
    grid_areas = iris.analysis.cartography.area_weights(skillshml)
    indexshml = skillshml.collapsed(['latitude','longitude'], iris.analysis.RMS,weights=grid_areas)

    skill=minf.collapsed('precipitation_flux', iris.analysis.SUM,mdtol=1)
    skill.data.mask = lsmasksea.data.mask
    skilltst=skill.extract(iris.Constraint(latitude = lambda cell: -30 <= cell <= 30))
    grid_areas = iris.analysis.cartography.area_weights(skilltst)
    indexs = skilltst.collapsed(['latitude','longitude'], iris.analysis.RMS,weights=grid_areas)

    skill=minf.collapsed('precipitation_flux', iris.analysis.SUM,mdtol=1)
    skill.data.mask = lsmaskland.data.mask
    skilltst=skill.extract(iris.Constraint(latitude = lambda cell: -30 <= cell <= 30))
    grid_areas = iris.analysis.cartography.area_weights(skilltst)
    indexl = skilltst.collapsed(['latitude','longitude'], iris.analysis.RMS,weights=grid_areas)

    print(dataname1, dataname2, 'Index = %0.2f, Index (land) = %0.2f, Index (sea) = %0.2f, Index (tropics) = %0.2f, Index (NH mid-lat) = %0.2f, Index (SH mid-lat) = %0.2f' % (index.data, indexl.data, indexs.data, indextrop.data, indexnhml.data, indexshml.data))

    skill=minf.collapsed('precipitation_flux', iris.analysis.SUM,mdtol=1)
    skill.data.mask = tot_events1.data.mask

    cf = qplt.pcolormesh(skill,vmin=0.5, vmax=0.95)
    ax = plt.gca()
    gl=ax.gridlines(draw_labels=True)
    gl.xlabels_top = False
    ax.coastlines()
    plt.title(dataname1+' vs '+dataname2+' '+timescale+' '+seasonc+' '+dates+'\nIndex = %0.2f' % (index.data))
    plt.savefig(plotname)
    plt.close()

    return index, indexl, indexs, indextrop, indexnhml, indexshml

def set_descriptive_text():
    """Set explanatory text by Gill Martin. This is a large chunk of text
    so it has been moved to its own file for organization."""

    # Introductory text at top of page
    intro_text = """Following the method of \
<a href="https://www.geosci-model-dev.net/10/57/2017/">Klingaman et al. (2017)</a>, \
the actual and fractional contribution to the total mean rainfall from different \
intensities is calculated, sorted into 100 bins of varying width ranging from \
0.005 - 2,360 mm day-1. This reveals the relative importance of precipitation \
events in a given intensity bin to the total precipitation. The calculation is \
performed at each grid box.</p>\n<p>\
Examples of the output from ASoP-Spectral are given in \
<a href="https://www.geosci-model-dev.net/10/105/2017/">Martin et al. (2017)</a>. \
This includes maps of actual and fractional contributions from different \
intensity ranges to the total mean rainfall for the period analysed, and \
full histograms, regionally averaged over small selected regions in order \
to facilitate direct comparison between datasets.<p>\
ASoP-Spectral can be run on datasets of any grid resolution and any time \
resolution. Datasets from different models, or from the same model, at a \
range of temporal and spatial resolutions, can be compared. However, in \
order to provide a fair comparison we recommend that datasets are \
regridded/averaged beforehand to a horizontal resolution and/or timescale \
that is sufficiently coarse for at least some spatial/temporal averaging to \
have been carried out for all of the models."""

    # Introductory text for the metrics section and metrics map.
    metrics_text = """We use a similarity index \
(<a href="https://journals.ametsoc.org/view/journals/clim/20/17/jcli4253.1.xml">\
Perkins et al., 2007)</a> to compare the fractional histograms from each model \
with observations at each grid point between 60S and 60N. This measures the \
overlap between the model and observed histograms, with values closer to 1.0 \
indicating that the histograms match better and 0.0 if they are entirely \
separated. Gridboxes where the total mean rainfall <1 mm/day are masked. \
A map of the index at each gridpoint is produced and metrics are the spatial \
root mean square of these indices over selected regions. The metric for the \
map as a whole is shown in the plot title, while the metrics for the \
sub-regions are in the histogram metrics JSON linked below."""

    # Explanatory text for the 2-panel histogram maps.
    hist_maps_text = """The contributions from rainfall in each of the 100 \
bins are collected into 4 categories: 0.005-10.0 mm/day, 10-50 mm/day, \
50-100 mm/day, >100 mm/day. Actual contributions (in mm/day) and \
fractional contributions (normalised by the total mean rainfall at each \
gridbox) are plotted. Each plot includes the 4 category maps for each \
pair of datasets that can be made from the files supplied. Maps are \
produced for each of the regions chosen by the user."""

    # Explanatory text for the binned precip vs contribution histograms
    all_hist_text = """Regional averages of the spectra are produced \
for the regions selected by the user. These are created by averaging \
together the histograms calculated at each point in the region, \
NOT by creating a new histogram using all the points within the region \
(doing the latter would remove some of the effects of grid-scale \
temporal variability (as it may be raining at one point when it is \
not at another), which is what this analysis method was designed to \
investigate).</p>\n<p>Regional averages are included for easier comparison \
between timescales and datasets, but since they do introduce spatial \
averaging, they are best done for relatively small \
(or spatially-consistent) regions only."""

    # Explanatory text for the histogram difference plots
    hist_diff_text="""Each plot shows the difference between \
regionally-averaged histograms from each pair of datasets that can be \
made from the files supplied, for each of the regions selected by the user."""

    return intro_text, metrics_text, hist_maps_text, all_hist_text, hist_diff_text

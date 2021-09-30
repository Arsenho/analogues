import rasterio
from rasterio.transform import Affine
import os
import georasters as gr

from analogue import cmip5_table

VALID_RESOLUTIONS = [0.5, 2.5, 5, 10]
VARS = ('tmin', 'tmax', 'tmean', 'prec', 'bio')
RCPS = (2.6, 4.5, 6.0, 8.5)
YEARS = (2030, 2050, 2070, 2080)
MODELS = (i for i in range(1, 36))


def getCMIP5(var, rcp, model, year, res, lon=None, lat=None, path=''):
    """
    Download data from remote repository
    'http://datacgiar.s3.amazonaws.com/ccafs/ccafs-climate/data/ipcc_5ar_ciat_downscaled/rcp4_5/2030s/csiro_mk3_6_0/10min/csiro_mk3_6_0_rcp4_5_2030s_prec_10min_r1i1p1_no_tile_asc.zip'
    :return:
    """

    if res not in VALID_RESOLUTIONS:
        exit('resolution should be one of: 2.5, 5, 10')

    if res == 2.5:
        res = '2_5min'
    elif res == 0.5:
        res = '30s'
    else:
        res = res + 'min'

    # find tile (only for 30s data)
    if res == '30s':
        lon = min(180, max(-180, lon))
        lat = min(90, max(-60, lat))
        transform = Affine(a=3, b=6, c=-180, d=180, e=-60, f=90)
        rs = rasterio.open(
            'rs.tif',
            'w',
            driver='GTiff',
            height=None,
            width=None,
            count=None,
            dtype=None,
            crs='+proj=latlong',
            transform=transform
        )

        # row < - c("a", "b", "c")[rowFromY(rs, lat)]
        # col < - colFromX(rs, lon)
        # ttile < - paste(row, col, sep="")

    ####
    # need to do the tile thing, only for 30s data
    ####

    var = str(var[1]).lower()
    if var not in VARS:
        exit(0)

    if rcp not in RCPS:
        exit(0)

    rcp = "rcp" + "_".join(str(round(rcp, 1)).split('.'))

    if year not in YEARS:
        exit(0)
    year = str(year) + "s" + ""

    if model not in MODELS:
        exit(0)

    # check if combination of rcp and model is available
    i = cmip5_table[model, rcp]
    if not i:
        print('this combination of rcp and model is not available')
        return None  # (invisible(NULL))

    path = path + '/cmip5/' + res + '/' + ''
    os.mkdir(path)  # dir.create(path, recursive=TRUE, showWarnings=FALSE)

    if res == "30s":
        baseurl = "http://datacgiar.s3.amazonaws.com/ccafs/ccafs-climate/data/ipcc_5ar_ciat_tiled/"
        zipp = str(cmip5_table['model'][
                       model] + "_" + rcp + "_" + year + "_" + var + "_" + res + "_r1i1p1_" + ttile + "_asc.zip" + '').lower()
    else:
        baseurl = "http://datacgiar.s3.amazonaws.com/ccafs/ccafs-climate/data/ipcc_5ar_ciat_downscaled/"
        zipp = str(cmip5_table['model'][
                       model] + "_" + rcp + "_" + year + "_" + var + "_" + res + "_r1i1p1_no_tile_asc.zip" + '').lower()

    zipfile = path + zipp + ''

    print('all is ok !')


# getCMIP5(12, 3, 3, 1998, 10)

precipitation = gr.from_file("precipitation.tif")

precipitation = precipitation.to_pandas()

precipitation.head()



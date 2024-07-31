import requests
import json
import configparser

import numpy as np
import csv
import math
import pandas as pd
import os, sys


from datetime import datetime
import rasterio
import warnings
warnings.filterwarnings('ignore')
import requests
import netCDF4 as nc


res='2016-07-01'
doy = 1

#download cloud data
#url = 'https://wvs.earthdata.nasa.gov/api/v1/snapshot?REQUEST=GetSnapshot&TIME='+res+'T00:00:00Z&BBOX=-2334051.0214676396,-414387.78951688844,-1127689.8419350237,757861.8364224486&CRS=EPSG:3413&LAYERS=MODIS_Terra_CorrectedReflectance_TrueColor,MODIS_Terra_Cloud_Fraction_Day&WRAP=day,day&FORMAT=image/tiff&WIDTH=2414&HEIGHT=4184&ts=1678819190554'

#download TC data:
url = 'https://wvs.earthdata.nasa.gov/api/v1/snapshot?REQUEST=GetSnapshot&TIME='+res+'T00:00:00Z&BBOX=-2334051.0214676396,-414387.78951688844,-1127689.8419350237,757861.8364224486&CRS=EPSG:3413&LAYERS=MODIS_Terra_CorrectedReflectance_TrueColor&WRAP=day&FORMAT=image/tiff&WIDTH=4712&HEIGHT=4579&ts=1683675557694'

r = requests.get(url, allow_redirects=True)
f_direc='./data/'
fname='tci_'+res+'_'+str(doy)+'_terra.tiff'
open(f_direc+fname, 'wb').write(r.content)



res='2016-07-01'
url = 'https://wvs.earthdata.nasa.gov/api/v1/snapshot?REQUEST=GetSnapshot&TIME='+res+'T00:00:00Z&BBOX=-2334051.0214676396,-414387.78951688844,-1127689.8419350237,757861.8364224486&CRS=EPSG:3413&LAYERS=MODIS_Terra_CorrectedReflectance_TrueColor,MODIS_Terra_Cloud_Fraction_Day&WRAP=day,day&FORMAT=image/tiff&&WIDTH=4712&HEIGHT=4579&ts=1683675557694'

r = requests.get(url, allow_redirects=True)
f_direc='./data/'
fname='cloud_'+res+'_'+str(doy)+'_terra.tiff'
open(f_direc+fname, 'wb').write(r.content)
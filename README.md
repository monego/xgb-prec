# XGBoost Seasonal Precipitation Forecast

This repository contains the code used in [South America Seasonal Precipitation Prediction by Gradient-Boosting Machine-Learning Approach](https://www.mdpi.com/2073-4433/13/2/243). The original code was run in Google Colab with different package versions than the provided environment, but it "should" work. The code also needs some cleanup.

Download the dataset [here](https://my.owndrive.com/index.php/s/2RQYDixfysQSQc2) and place it in the `data` directory.

Use the `environment.yml` file to create the software environment to run the code (the Conda package manager is required):

``` sh
conda env create -f environment.yml
```

Create two directories called "data" and "results" in the local repository.

# Training

First run the data processing code with `python preparations.py` to generate the seasonal files. Then run `python train.py`.

# Testing

Run `python inference.py` to generate the error tables. The maps are missing and will be added soon. 

# Dataset

The authors use the [GPCP](https://climatedataguide.ucar.edu/climate-data/gpcp-monthly-global-precipitation-climatology-project) precipitation dataset and [NCEP/NCAR](https://psl.noaa.gov/data/reanalysis/reanalysis.shtml) for the atmospheric variables, from 1980 to 2020.

Training set was divided from 1980 to 2016, and testing set from 2017 to 2020.

# References

Anochi, J.A.; de Almeida, V.A.; de Campos Velho, H.F. Machine Learning for Climate Precipitation Prediction Modeling over South America. Remote Sensing 2021, 13, 2468. DOI: 10.3390/rs13132468

Adler, R.F.; Huffman, G.J.; Chang, A.; Ferraro, R.; Xie, P.P.; Janowiak, J.; Rudolf, B.; Schneider, U.; Curtis, S.; Bolvin, D.; others. The Version-2 Global Precipitation Climatology Project (GPCP) Monthly Precipitation Analysis (1979–present). Journal of hydrometeorology 2003, 4, 1147–1167.

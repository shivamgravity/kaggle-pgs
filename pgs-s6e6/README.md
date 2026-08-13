# Predicting Stellar Classes

In June 2026 kaggle playground series, we have a great dataset for astronomy enthusiasts and I am one of them.

Access the competition from here: [Competition Page](https://www.kaggle.com/competitions/playground-series-s6e6/overview).

In this episode we have the following 10 features:

| S.No. | Feature | Description |
|-------|---------|-------------|
| 1 | alpha | Right Ascension, RA. Equivalent to longitude on Earth. Measured in degrees (0&deg;-360&deg;). Specifies east-west poisiton on celestial sphere. |
| 2 | delta | Declination, Dec. Equivalent to latitude on Earth. Measured in degrees (-90&deg; to +90&deg;). Specifies north-south position. |
| 3 | u | Ultravoilet |
| 4 | g | Green |
| 5 | r | Red |
| 6 | i | Near Infrared |
| 7 | z | Infrared |
| 8 | redshift | Tells us how far a celestial object is |
| 9 | spectral_type | It is the classification based on temperature from hottest to coolest |
| 10 | galaxy_population | It tells what kind of object do we have - a blue cloud or red sequence |

**alpha** and **delta** are sky coordinates. They tell where the object is located in the sky.

**u, g, r, i, z** are SDSS photometric bands. Each measures brightness through a different filter. Every celestial object emit light differently.

Due to expansion of the universe, nearby objects have small **redshift** while distant galaxies have larger.

Redshift, *z =* $\frac{\lambda_{obs} - \lambda_{emit}}{\lambda_{emit}}$

$\lambda_{obs}$: observed wavelength

$\lambda_{emit}$: emitted wavelength

**spectral_type** is the stellar spectral class, that classifies objects based on temperature. It is written as **O, B, A, F, G, K, M** from hottest to coolest, repectively.

In our dataset, **galaxy_population** has only two unique values - *Blue_Cloud* and *Red_Sequence*. It generally tells us if the object is newly born or old.

The dataset have 577k rows, so we have a quite a big dataset. Using these feature, we need to predict what kind of celestial object it is from the given target features.

**Target:** Galaxy | QSO | Star

## Kaggle Notebooks

| Notebook | Description |
|----------|-------------|
| [Catboost](https://www.kaggle.com/code/shivamgravity/pgs-s6e6-catboost-benchmark) | Benchmark and newer versions of CatBoost |
| [LightGBM](https://www.kaggle.com/code/shivamgravity/pgs-s6e6-lgbm-benchmark) | Benchmark and newer versions of LightGBM |
| [XG Boost](https://www.kaggle.com/code/shivamgravity/pgs-s6e6-xgb-benchmark) | Benchmark and newer versions of xg-boost |
| [MLP](https://www.kaggle.com/code/shivamgravity/pgs-s6e6-mlp-benchmark) | Benchmark and newer versions of MLP |
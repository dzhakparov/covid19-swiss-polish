# project information

This project takes the data from positive **COVID19-patients** of a hospital in Poland and tries to **predict** if
**illness of each patient** will be severe (death) or 'mild' (not death). 

To start calculating 'main.py' have to be executed.

    $ python3 main.py

All parameters are stored in file config.py.

Source data are in folder 'data/input'. At the moment there are raw 3 files needed:
* CoV-2_motherdatabase_lab_17_08.csv
* CoV_2_motherdatabase_clinics_17_08.csv
* CoV_2_motherdatabase_epidem_symptoms_17_08.csv

To rename columns 3 files with same structure as files up will be needed:
* renaming_lab_17_08.csv
* renaming_clinics_17_08.csv
* renaming_epidem_symptoms_17_08.csv

If running the code an output folder (data/output/) with all plots, stored_outputs and 
logging files will be generated automatically.

In step 'data_preprocessing' a working file (named as defined in config.py) will be generated and stored in 
folder 'data/output'. This file is input for further processes like 'ml_pipeline'. 

## update conda environment to environment.yaml
If new external packages are installed in the local conda environment 'cov19_publication.yaml'-file have to be updated with:

    $  conda env export --name cov19_publication > environment.yml

## build new conda environment if cloned project
If this project is cloned to a new system, the environment can be built with:

*remark:* The 2 own written packages (misc and sklearn_custom) have to be commented out in environment.yml 
before running.

    $ conda env create --file environment.yml

or if the conda environemnt already exists, update this with:

    $ conda env update --file environemnt.yml --prune
    
    
## manual installation of 2 own written packages
2 packages have to be installed manually. They are located in folder pkg/ of this project.
* misc-0.0.3-py3-none-any.whl
* sklearn_custom-0.0.6-py3-none-any.whl
   
```
    $ cd pkg/
    $ pip install misc-0.0.3-py3-none-any.whl
    $ pip install sklearn_custom-0.0.6-py3-none-any.whl1
```

## manual installation of saxpy and pandas-profiling
```
    $ pip install saxpy
    $ pip install pandas-profiling
```
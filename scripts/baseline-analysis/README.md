# project information

This project takes the data from positive and negative **COVID19-patients** of a hospital in Poland and tries to **predict** 
if they are positive or negative depending on several parameters when entering the hospital.

To start calculating 'main.py' have to be executed.

    $ python3 main.py

All parameters are stored in file config.py.

Source data are in folder 'data/input'. At the moment there are raw 3 files needed:
* CoV-2_final_lab_10_29_appendix.csv
* CoV-2_negative_final_lab_17_08.csv
* CoV-2_positive_final_lab_17_08.csv

If running the code an output folder (data/output/) with all plots, stored_outputs and 
logging files will be generated automatically.

In step 'data_preprocessing' a working file (named as defined in config.py) will be generated and stored in 
folder 'data/output'. This file is input for further processes like 'final_train_model.py'. 

## update conda environment to environment.yaml
If new external packages are installed in the local conda environment 'cov19_pub' have to be updated with:

    $  conda env export --name cov19_pub > environment.yml

## build new conda environment if cloned project
If this project is cloned to a new system, the environment can be built with:

*remark:* The 2 own written packages (misc and sklearn_custom) have to be commented out in environment.yml 
before running.

    $ conda env create --file environment.yml

or if the conda environment already exists, update this with:

    $ conda env update --file environment.yml --prune
    
    
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

## info for PyCharm user

If the paths to the files in folder 'src' and to config.py are not shown properly (red -> can't reference to them)

* **delete content root**: File -> settings -> Project: cov19_pub -> Project structure
* **add content root**: File -> settings -> Project: cov19_pub -> Project structure -> 
**+** '/home/schmidmarco/Documents/CODE/PROJECTS/cov19_pub/scripts/baseline-survival-analysis' and 
**+** '/home/schmidmarco/Documents/CODE/PROJECTS/cov19_pub/scripts/sax-analysis' 
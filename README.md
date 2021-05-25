# Quantification of heart cell tracks in 3D #

## Input format ##
The analysis script expects as input two sets of Microsoft Excel files (output from Imaris software), stored in separate folders:
1) **A first set of Excel files that record the positions of the tracks throughout the timelapse.** 
There should be one file for each biological replicate. It should have a sheet named "Position" with columns "Time" (index of time step in the time-lapse), "Position X" (x-coordinate of cell track with respect to imaging volume), "Position Y" (y-coordinate of cell track with respect to imaging volume), "Position Z" (z-coordinate of cell track with respect to imaging volume), and "TrackID" (cell track ID using the same numbering scheme as in the other Excel file).

2) **A second set of Excel files that record the start and end positions of each cell track as well as the heart segment category.** 
There should be one file for each biological replicate. It should contain two sheets: "Calculations" and "Time Since Track Start". The "Calculations" sheet contains time index, cell track ID, and heart segment category. The heart segment categories accepted as entries are: "VV" (ventricle ventral), "VD" (ventricle dorsal), "AV" (atrium ventral), "AD" (atrium dorsal), "AVC" (atrio-ventricular canal). The time index should have the header "Time", cell track ID the header "ID", and the category the header "Cat". The same Excel file should also have a sheet named "Time Since Track Start" with the column named "Time Since Track Start" with the time in seconds. All items in quotations are case-sensitive.

To be able to map Excel files corresponding to the same biological replicate, the filenames should include identical identifiers. For wildtype samples, the identifier is of the format "wt" plus at least one digit. For mutant samples, the identifier is of the format "oug" plus at least one digit.

## Usage example ##

### Input file examples ###

* File named "_040221-SP8MP-wt4-spots.xlsx_" in the folder "_./excel_set_1/_"
* File named "_040221-SP8MP-wt4.xlsx_" in the folder "_./excel_set_2/_"

The filenames both contain the identifier "_wt4_". The code will map identical cell track IDs in both files to each other, and assign the resulting dataset to the wildtype condition.

### Command line input example ###
`python3 heartbending.py ./excel_set_1/ ./excel_set_2/ "*wt*" "*oug*"`

#### Arguments ####
1) Relative folder path for the first set of Excel files.
2) Relative folder path for the second set of Excel files.
3) Identifier for wildtype condition filenames.
4) Identifier for mutant condition filenames.

### Printing movie frames ###
To print the frames of the supplementary movies, change the option `printmovies = False` to `printmovies = True` in [line 927](https://github.com/rmerks/heartbending/blob/6b03a506c7e4078f6680756b93f72e145ec91986/heartbending.py#L927).

## Output format ##
The code outputs the following:
* Plots used in figure panels of the main manuscript and supplement in png and pdf format.
* csv files containing the source data as provided with the manuscript.
* csv files containing the result of all statistical analyses as reported in the manuscript.
* If `printmovies` option changed (see above): Individual frames used to create supplementary movies in the main manuscript in png format.

## Required Python installation and packages ##
The code runs in python 3 (tested with python 3.6 - 3.8) and requires the following additional modules:
* Numpy
* Pandas
* Scipy
* Matplotlib
* Seaborn
* statsmodels

IDF creation for existing buildings
read_idf.py by Chris van Dronkelaar
uses bits from EPGP2 by authors below.

Operating EP2
Original by Jon Taylor
Additional edits from Phillip Biddulph and Phil Symonds
Dec 2014

EPG2 allows for the generation of EnergyPlus .idf files for individual or batch simulation.

1) EPG2 Datafiles
The original EPGenerator was an Excel spreadsheet that detailed the parameters for use in EnergyPlus simulations. Because this led to compatibility issues with Macs, the new EPG2 relies on .csv files rather that a spreadsheet. These .csv files are the same as the tabs in the original spreadsheet, and allows the values for the different parameters used in the .idf files to be defined. The required outputs can also be defined on the ‘Reports’ csv.  These .csv files are then referenced by the Python script when the .idfs are to be built.

2)	 A second folder in the EPG2 Root directory contains the produced .idfs, simulation results, and a .csv with the list of the .idf files that you would like to perform. 
The .csv layout is very much the same as in the old EPGenerator ‘EP_Control’ tab. 
Define different combinations of built form, occupation, environment, efficiency package, simulation period, etc that you want built.

3) Launch Git Bash. Allow Git Bash to use Python command prompt:

	ipython --pylab

4) Launch EPG2 from Git Bash using the Python command
	
	import epg2

5) You are now ready to build some .idfs for simulation. To produce all files listed in the .csv described in (2), type:

 	epg2.unpick_csv(indir=’<folder name>’, csvfile=’<filename.csv’)

Where <foldername> is the folder within the EP2 folder where the .csv file is saved, and <filename> is the name of the .csv.
This will produce the .idfs corresponding to the buildings defined in the .csv file and a batch file that will run all of the .idfs created if launched.

6) If you want to produce a number of .idfs that contain combinations of different parameters, you could use the .csv file and copy and paste a lot. Alternatively, you can produce a script that details the combinations. A script can be launched in Git Bash by typing:

	epg2.unpick_script(indir=’<folder name>’, script=’<scriptname.txt’)

Where scriptname.txt is a script you have written that details the combinations. A script for EP2 may look something like this:

Where the different options are separated by commas. A list of the different versions can be found in the epg2_dump.csv file.

Or steps 3-6 can be done by running run_epg.py from the command line:

	python run_epg.py

Various options (e.g running a latin hypercube design) can be used with this which are displayed with:
	
	python run_epg.py --help (or -h) 

7) If you just want to build one idf instead of all of the runs listed on the .csv file, that make sure that run has a ‘No’ under the Built column, and all the other runs have ‘Yes’.


*********************** The stuff bellow hasn't been added in yet ***********************

8) Drawing building is done through a Python script.  This is done by typing:

	EPG.draw_building(buildname=’Buildingname’,xcut=2.0,ycut=3.0,zcut=1.0,shadingname='Urban_Detatched') 

Buildingname is the name of the building you want to draw.
shadingname is the shading option you want. The default is ‘blank’.
The x,y, and z values for the cross section of the building you want drawn.

9) The old EPGenerator had a macro that could update the spreadsheet to reflect changes to the u-value for different wall types described in the Fabric tab. To update the fab_props.csv with u-values to reflect any changes you’ve made, type:

	epg2.update_fabs()

This will update the u-value column of the .csv.  If you quickly want to know the u-value of a particular fabric type, you can type:

	epg2.calc_fab_u(fab_name=’<fabname>’)

Where <fabname> is the name of the fabric you are interested in.

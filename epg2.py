"""code for EP_Generator2"""

import os
import sys

import copy
import datetime
import matplotlib.pyplot as plt 

import tools.runsfolder as rf
import tools.unpickdata as unp
import tools.idfbuilder as idf
import tools.outputs    as out

##For latin hypercube experiment design
from pyDOE import doe_lhs
import scipy.stats as stats
import random

#Read the root directory and the directory EnergyPlus is installed. Store as a global variable
if sys.platform=='win32':
    print("Operating System: Windows")
    rootdir="/".join(__file__.split('\\')[:-1])
    epdir="C:/EnergyPlusV8-1-0"
elif sys.platform=='darwin':
    print("Operating System: MAC OS")
    rootdir="/Users/PhilSymonds/Documents/workspace/epg2"
    epdir="/Applications/EnergyPlus-8-1-0"
elif sys.platform=='linux2' or sys.platform=='linux':
    print("Operating System: linux")
    if "GCC 4.6.3" in sys.version  or "GCC 4.9.2" in sys.version:  ##For legion
        rootdir="/home/ucqbpsy/Scratch/Simulations/EPG_HPRU/epg2"
        epdir="/home/ucqbpsy/Software/EnergyPlus_v8.1/bin"
    else:
        rootdir="/home/phil/Documents/ucl/epg2"
        epdir="/usr/local/EnergyPlus-8-1-0"
else:
    print("rootdir not found")


def run_hypercube(indir='Debug_runs',build=True,batfile=True,updatecsv=True, build_type = 'Detached'):

    ###Replace script file here:
    #For latin hypercube how many experiments do you want:
    n_exps=[3]#,100,200,400,600,800,1000
    ##You need to specify how many testing runs you want from the above
    n_test = 3
    n_train = n_exps
    
    #time steps per hour
    timesteps=4
    start_day=1
    start_month=1
    end_day=31
    end_month=1
    built='No'
    simulated='No'
    output_all='No'
    use_AirflowNetwork='No'
    #Simulation level (1 for default,3 for HAMT)
    sim_level=1
    
    ##Categorical values:    
    # package properties found in data_files/pack_props.csv
    packages=['Cavity','Solid']
    ##HPRU: 'Cavity','Solid','Cavity-Shutters','Solid-Shutters'
    
    # occupation properties found in data_files/occupation_props.csv
    occupations=['Pensioners']
    #WHO: WHO_PM25Ext
    ##HPRU: Family, Pensioners
        
    # building properties found in data_files/build_props.csv
    buildings=['Semi','Detached','End-Terrace','Mid-Terrace', 'Bungalow', 'ConvertedFlat', 'Lowrise-Pre1990', 'Highrise']
    ##HPRU: 'Semi','Detached','End-Terrace','Mid-Terrace', 'Bungalow', 'ConvertedFlat', 'Lowrise-Pre1990', 'Highrise'
            
    #weather
    weather='cntr_Birmingham_TRY'
    #WHO: BRA_Manaus.823310_SWERA

    ##Continuous values
    # set the permeability (m3/(hm2)
    permeability=20
    perm_stdev=10
    
    # Window open thresh temperature
    window_openthresh=24
    window_openthresh_stdev=5
    
    # Heating on threshold
    heater_thresh=22
    heater_thresh_stdev=3
    
    #building thermal resistance factor if 1 do uniform values if 0 use the default
    wall_u=1    #walls u-value now
    roof_u=1    #roof
    window_u=1  #window
    floor_u=1   #floor
    
    for n_exp in n_exps:
        for building in buildings:
            for package in packages:
                for occ in occupations:
                    
                    if('Shutters' in package):
                        occ+='-Shutters'
                
                    occ_name = occ.split("-")[0]
                    loc_name = weather.split("_")[1]
                    runs_name = building+'_'+package+'_'+occ_name+'_'+loc_name+'_'+ str(n_exp)
                    
                    # set the folder path
                    folder=rf.Folder(rootdir, indir, runs_name = runs_name, n_test = n_test, n_train = n_train) 
                    
                    ##Produce latin hypercube
                    n_experiments= n_exp   ## Get number of experiments from script
                    lhd = doe_lhs.lhs(12, samples=n_exp)
                
                    rinfo={}
                
                    for i in range(n_experiments):
                        
                        # environment (weather etc...) properties found in data_files/env_props.csv
                        
                        ##HPRU: Urban_Semi,Urban_Detached,Urban_EndTerrace,Urban_MidTerrace,Urban_HighRise
                        ##define the terrain
                        terrain = 'Urban'
                        rand = random.randint(1, 3)
                        
                        if rand==1:
                            terrain = 'City'
                        elif rand==2:
                            terrain = 'Urban'
                        elif rand==3:
                            terrain = 'Country'
                        
                        environment=terrain+'-Semi'
                        if(building=='End-Terrace'):
                            environment=terrain+'-EndTerrace'
                        elif(building=='Mid-Terrace' or 'Lowrise-Pre1990' in building):
                            environment=terrain+'-MidTerrace'
                        elif(building=='Detached' or building=='Bungalow'):
                            environment=terrain+'-Detached'
                        elif('HighRise' in building):
                            environment=terrain+'-HighRise'                
                        
                        ### Stuff that is not randomised
                        rinfo['rn']         =i+1
                        rinfo['built']      =built              ##been built?
                        rinfo['simulated']  =simulated          ##bat created?
                        rinfo['building']   =building           ##building type
                        rinfo['occupation'] =occ                ##Occupancy e.g windows etc...
                        rinfo['environment']=environment        ##Environment settings
                        rinfo['package']    =package            ##heating package
                        rinfo['start_day']  =start_day        
                        rinfo['start_month']=start_month
                        rinfo['end_day']    =end_day
                        rinfo['end_month']  =end_month
                        rinfo['sim_level']  =sim_level          ##1 for default 3 for HAMT (heat balance algo)
                        rinfo['output_all'] =output_all        
                        rinfo['use_AirflowNetwork']=use_AirflowNetwork  
                        rinfo['timesteps']  =timesteps
                        rinfo['weather']    =weather
                        rinfo['wall_u']     =wall_u
                        rinfo['roof_u']     =roof_u
                        rinfo['window_u']   =window_u
                        rinfo['floor_u']    =floor_u
                        
                        ##Permeability with normal dist
                        lower, upper = 0.0, 99999999.0
                        mu, sigma = permeability, perm_stdev
                        rinfo['permeability'] = stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma).ppf(lhd[i, 0])
                        
                        ##Orientation with uniform dist
                        rinfo['orientation']  = lhd[i, 1]*360
                        
                        ##Heating with normal dist (may need increasing)
                        lower, upper = 15, 26.0    ##Set upper and lower bounds for normal distribution
                        mu, sigma = heater_thresh, heater_thresh_stdev
                        rinfo['heater_thresh'] =  stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma).ppf(lhd[i, 3])
                        
                        ##Window opening with normal dist (may need increasing)
                        lower, upper = 10.0, 60.0 
                        mu, sigma = window_openthresh, window_openthresh_stdev
                        rinfo['window_openthresh'] = stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma).ppf(lhd[i, 2])
                
                        #rinfo['cook_pm25fact'] = lhd[i, 4]*3  ##Not used so set to 1
                        rinfo['cook_pm25fact'] = 1
                
                        max_f = 2.4
                        if 'Cavity' in rinfo['package']:
                            max_f = 1.6
                
                        ## U-values
                        if(rinfo['wall_u'] != 0):
                            rinfo['wall_u'] = max_f*lhd[i, 4]+0.15
                        else:
                            rinfo['wall_u'] = 0
                        
                        if(rinfo['roof_u'] != 0):
                            rinfo['roof_u'] = 2.15*lhd[i, 5]+0.1
                        else:
                            rinfo['roof_u'] = 0  
                        
                        if(rinfo['window_u'] != 0):
                            rinfo['window_u'] = 3.95*lhd[i, 6]+.85
                        else:
                            rinfo['window_u'] = 0
                            
                        if(rinfo['floor_u'] != 0):
                            rinfo['floor_u'] = 1.05*lhd[i, 7]+0.15
                        else:
                            rinfo['floor_u'] = 0
                        
                        #TODO: Need to check that these ranges are suitable. Maybe use norm?
                        rinfo['glaz_fract'] = 0.5*lhd[i, 8]+0.1
                        rinfo['floor_height'] = lhd[i, 9]+2.
                        rinfo['floor_area_scale'] = 1.45*lhd[i, 10]+0.65
                        
                        rinfo['gain_factor'] = 1.5*lhd[i, 11]+.5

                        rinfo['run_type'] = 'lhd'
                        
                        # get all the information on the various properties from the files

                        ## uses unpickdata.py as unp in tools folder,
                        ## unpick_database is a function containing all functions in the python file.
                        ## it reads all the csv files located in folder data_files, i.e. the inputs
                        ## this it stores in memory?
                        dbdats=unp.unpick_database(rootdir)

                        ## uses runfolder.py as rf in tools folder,
                        ## uses data from unpick_database or unp to create a run object
                        ##
                        run=rf.create_run(dbdats,rinfo=rinfo)

                        ## saves the run in the runs dictionary in the previously defined folder 'indir' (Debug_runs)
                        folder.runs[run.rn]=run

                    ## builds the idf using the idfbuilder.py as idf
                    ## the builds_idf function uses the load_idd function in iddinfo.py as idd
                    if build: # builds the idf uses tools.idfbuilder as idf
                        idf.build_idfs(rootdir, epdir, folder=folder)
                    
                    ##If platform is linux2 create a bash script rather than a bat file
                    if build and batfile and ((sys.platform==('linux2')) or (sys.platform==('linux'))): #Creates bash file for linux to legion
                        out.create_legion_scripts(rootdir, epdir, folder=folder)     
                    elif build and batfile: #Creates batch file for windows
                        out.create_bat(rootdir, epdir, folder=folder)
                    
                    if updatecsv:
                        out.create_csv(rootdir, folder,runs_name)
                    else:
                        print(("script Not found ", runs_name))

    
#Used for producing idfs from information input into the .csv file
def run_csv(n_runs = 1, indir='Debug_runs',csvfile='script.csv',batfile=False,updatecsv=False):
    
    for i in range(1, n_runs+1):
        unpick_csv(i ,indir=indir,csvfile=csvfile,batfile=False,updatecsv=False)


def unpick_csv(number, indir='Debug_runs',csvfile='script.csv',batfile=False,updatecsv=False):

    folder=read_csv(number, indir=indir,csvfile=csvfile)
    
    if folder != None:
        idf.build_idfs(rootdir,epdir, folder=folder)
    
    if batfile:
        out.create_legion_scripts(rootdir, epdir,folder=folder)
    if updatecsv:
        out.create_csv(rootdir, folder,csvfile.split('.')[0])
            


def unpick_script(runstart=1,indir='Debug_runs',script='script.txt',build=True,batfile=False):       
    
    ##for timer
    start = datetime.datetime.now()
    
    # get all the information on the various properties from the files
    dbdats=unp.unpick_database(rootdir)
    
    # set the path to the script
    scriptfile="{0}/{1}/{2}".format(rootdir,indir,script)

    # set the folder path
    folder=rf.Folder(rootdir, indir)
    rn=runstart-1
    rinfos={}
    
    # open the script file and read the parameters 
    if os.path.exists(scriptfile):
        #open script.txt
        with open(scriptfile,mode='r') as inf:
            
            rinfos["run_type"]='script'
        #loop over the lines in script.txt
            for lline in inf:

                linfo=lline.split('#')[0]   ###allows to put comments in
                linfo=linfo.rstrip()           ###removes white space characters at the end of line
                 
                #reads the parameters from the script... will do this stuff before the stuff above   
                dds=linfo.split('=')
        
                if len(dds)>1:
                    param,val=dds            ###assigns param with the parameter name and value with the value from the script.txt file
                    rinfos[param]=val 
                
                if 'CreateRun' in linfo:
                    
                    rn=run_combix(dbdats,rn=rn,rinfos=rinfos,folder=folder)
                
                if 'Stop' in linfo:
                    print("==>stopping")
                    break
                    
        if build: 
            idf.build_idfs(rootdir, epdir, folder=folder)
        
        ##If platform is linux2 create a bash script rather than a bat file
        if build and batfile and (sys.platform==('linux2')):
            out.create_legion_scripts(rootdir, epdir, folder=folder)     
        elif build and batfile:
            out.create_bat(rootdir, epdir, folder=folder)
        
        

        out.create_csv(rootdir, folder,script.split('.')[0])
    else:
        print(("script Not found ",scriptfile))
    
    
    end = datetime.datetime.now()
    runtime = end-start
    print(("Build time: ", runtime.seconds, "s, ", runtime.microseconds, "micro secs"))


def read_csv(number, indir='Debug_runs',csvfile='script.csv'):

    #would prefer to read stuff from csv as you go but will keep this for now... too much spaghetti   
    dbdats = unp.unpick_database(rootdir)
    
    rfile="{0}/{1}/{2}".format(rootdir,indir,csvfile)
        
    folder=rf.Folder(rootdir, indir)
    
    with open(rfile,mode='r') as inf:
        inf.readline()
        inf.readline()
        for lline in inf:
            lline=lline.replace('\n','')
            lline=lline.replace('\r','')
            rinfos=lline.split(',')
            
            if int(rinfos[0]) != number:
                continue
            
            rinfo={}
            rinfo['rn']=int(rinfos[0])                  ##run number
            rinfo['built']=rinfos[1]            ##been built?
            rinfo['simulated']=rinfos[2]        ##bat created?
            rinfo['sim_level']=int(rinfos[4])         ##1 for default 3 for HAMT (heat balance algo) 
            rinfo['weather']=rinfos[5]
            
            rinfo['building']=rinfos[6]            ##building type
            rinfo['occupation']=rinfos[7]        ##Occupancy e.g windows etc...
            rinfo['environment']=rinfos[8]        ##Environment settings
            rinfo['package']=rinfos[9]            ##heating package
            rinfo['start_day']=rinfos[11]        
            rinfo['start_month']=rinfos[12]
            rinfo['end_day']=rinfos[13]
            rinfo['end_month']=rinfos[14]
            rinfo['timesteps']=rinfos[15]
            
            rinfo['permeability']=float(rinfos[17])    ##m/h(@50Pa)
            rinfo['orientation']=rinfos[18]            ##this gets overriden later when reading from csv
            rinfo['window_openthresh']=rinfos[19]
            rinfo['heater_thresh']=rinfos[20]
            rinfo['cook_pm25fact']=rinfos[21]
            rinfo['wall_u']=rinfos[22]
            rinfo['roof_u']=rinfos[23]
            rinfo['window_u']=rinfos[24]
            rinfo['floor_u']=rinfos[25]
            
            rinfo['glaz_fract']=rinfos[26]
            rinfo['floor_height']=rinfos[27]
            rinfo['floor_area_scale']=rinfos[28]
            rinfo['gain_factor']=rinfos[29]
            
            #rinfo['output_all']=rinfos[19]        
            #rinfo['use_AirflowNetwork']=rinfos[21]
            rinfo['run_type']='csv'   
                        
            run=rf.create_run(dbdats,rinfo)
            folder.runs[run.rn]=run
            
    return folder

#find the combinations from script.txt and run             
def run_combix(dbdats,rn=0,rinfos=None,folder=None):
    
    ##make copies of rinfos
    sinfos=copy.deepcopy(rinfos)
    rinfo=copy.deepcopy(rinfos)
    
    ##loop over parameters/values
    for param,val in list(rinfo.items()):   
        
        rinfo[param]=val.split(',')[0]    ## get the 1st value
        
        if len(val.split(','))>1:      ## if there is more than one value for a param    
            sinfos[param]=','.join(val.split(',')[1:])  ##removes the first val so that can find the next
        
            rn=run_combix(dbdats,rn=rn,rinfos=sinfos,folder=folder)
            sinfos[param]=rinfo[param]
                
    rn+=1
    print(("Creating run: ", rn))
    rinfo['rn']=rn
    

    run=rf.create_run(dbdats,rinfo=rinfo)    
    folder.runs[run.rn]=run
    return rn

##For drawing the building
def draw_building(buildname='Semi',xcut=5,ycut=3,zcut=4,shadingname='Blank'):

    fcols={
        'internal_wall':'pink',
        'external_wall':'black',
        'ground':'black',
        'roof':'black',
        'adiabatic':'green',
        'internal_floor':'gray',
        'internal_ceiling':'gray',
        'insulated_floor':'chocolate',
        'insulated_ceiling':'chocolate',
        'window':'blue',
        'door':'red',
        'perm':'skyblue',
        'hang':'lime',
        'shade':'red',
        'trickle':'darkorange'
        }
    
    folder=read_csv(1, indir='Debug_runs',csvfile='script.csv')
    
    mats    =   unp.unpick_materials(rootdir)
    fabs    =   unp.unpick_fabrics(rootdir, mats)
    packs   =   unp.unpick_packages(rootdir, fabs)
    package =   folder.runs[1].props['package']
    
    shades  =   unp.unpick_shades(rootdir)
    builds  =   unp.unpick_buildings(rootdir, pack=packs)  
    
    if shadingname not in shades.keys():
        print(shadingname," Not found")
        shadingname='Blank'
        
        
    if buildname in builds.keys():
        building=builds[buildname]

        building.create(run=folder.runs[1],pack=package,shading=shades[shadingname])
        
        fig=plt.figure(100,figsize=(14,10))
        fig.clf()
        
        ax1=fig.add_subplot(2,2,1)
        ax2=fig.add_subplot(2,2,3)
        ax3=fig.add_subplot(2,2,2)
        
        ax1.set_title("Building -- {0}\nEnvironment -- {1}".format(buildname,shadingname), fontsize=24)
        
        
        for axs,dirm,cut in zip([ax1,ax2,ax3],[2,1,0],[zcut,ycut,xcut]):
            for surface in building.shading_surfaces:
                plot_xyz(axs,surface,dirm,cut,fcols)
        
            for _,zone in building.zones.items():
                if zone.inside:
                    vcentre=zone.centre
                                    
                    name_zone=False
                    for surface in zone.surfaces+zone.hang_surfaces:  
                        zone_there=plot_xyz(axs,surface,dirm,cut,fcols)
                        if zone_there:
                            name_zone=True
                           
                    if name_zone:
                        vcs=[]
                        for ccc in [0,1,2]:
                            if dirm!=ccc:
                                vcs.append(vcentre[ccc])
                        axs.text(vcs[0],vcs[1],"{0}".format(zone.name),ha='center')
                       
        
        ax1.plot([xcut,xcut],[building.buildmin[1]-1.,building.buildmax[1]+1.],'--',color='black')
        ax1.plot([building.buildmin[0]-1.,building.buildmax[0]+1.],[ycut,ycut],'--',color='black')
        
        ax2.plot([xcut,xcut],[building.buildmin[2]-1.,building.buildmax[2]+1.],'--',color='black')            
        ax2.plot([building.buildmin[0]-1.,building.buildmax[0]+1.],[zcut,zcut],'--',color='black')

        ax3.plot([ycut,ycut],[building.buildmin[2]-1.,building.buildmax[2]+1.],'--',color='black')
        ax3.plot([building.buildmin[1]-1.,building.buildmax[1]+1.],[zcut,zcut],'--',color='black')
        
        
        ax1.set_xlabel('X [metres]')
        ax1.set_ylabel('Y [metres]')
        
        ax2.set_xlabel('X [metres]')
        ax2.set_ylabel('Z [metres]')
        
        ax3.set_xlabel('Y [metres]')
        ax3.set_ylabel('Z [metres]')
        
        fig.savefig('Debug_runs/'+buildname+'.png')

def plot_xyz(ax1,surface,axis,cut,fcols): 
    zone_there=False

    ees=surface.draw_xyz(axis=axis,cut=cut)
  
    if len(ees)>0:
        dwall=surface.wall_type
        if surface.outside_BC=='Adiabatic':
            dwall='adiabatic'
        if surface.perm:
            dwall='perm'
        if surface.hang:
            dwall='hang'
        if surface.trick:
            dwall='trickle'
            
        for e in range(0, int(len(ees)/2)):
            i = int(2*e)
            ax1.plot(ees[i],ees[i+1],'-',color=fcols[dwall],linewidth=2)                        
        
        zone_there=True
        
    if surface.window!=None:
        ees=surface.window.draw_xyz(axis=axis,cut=cut)
        if len(ees)>0:
            dwall='window'
            ax1.plot(ees[0],ees[1],'-',color=fcols[dwall],linewidth=4)
            
    if surface.door!=None:
        ees=surface.door.draw_xyz(axis=axis,cut=cut)
        if len(ees)>0:
            dwall='door'
            ax1.plot(ees[0],ees[1],'-',color=fcols[dwall],linewidth=4)  
            
    return zone_there

def calc_fab_u(fab_name=None):
    mats=unp.unpick_materials(rootdir)
    fabs=unp.unpick_fabrics(rootdir, mats,repack=False)
    if fab_name in fabs:
        fab=fabs[fab_name]
        fab.calc_u()
        print(fab.name," U-Value is ",fab.U)
    else:
        print(fab_name,' Not Found')
        

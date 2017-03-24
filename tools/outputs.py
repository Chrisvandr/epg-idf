"""Code to produce bat for multiple runs and csv for post processing"""

import sys
from subprocess import Popen
import os.path

def create_bat(rootdir, epdir, sim=False,folder=None):

    if folder!=None and sys.platform=='win32':

        batfile="{0}\\{1}\\EPG2.bat".format(rootdir,folder.proj_name)
        with open(batfile,mode='w') as inf:
           
            op="chdir {0}".format(epdir)
            inf.write(op+"\n")
            
            for run in list(folder.runs.values()):
                if run.built=='Yes' and run.simulated=='No' and run.simulate_request:
                    print(("Adding to BAT {0}".format(run.rn)))

                    wetfile=run.weather
                    
                    if sys.platform=='win32':
                        idfpath="{0}\\{1}\\EPG2_{2}".format(rootdir,folder.proj_name,run.rn)
                        wetpath="{0}".format(wetfile)
                    elif sys.platform=='darwin' or sys.platform == 'linux2':
                        idfpath="{0}/{1}/EPG2_{2}".format(rootdir,folder.proj_name,run.rn)
                        wetpath="{0}".format(wetfile)
                    
                    op="call Epl-run \"{0}\" \"{1}\" idf \"{2}\" EP N nolimit N N 0 N".format(idfpath,idfpath,wetpath)
                    inf.write(op+"\n")
                    if not run.OutputAll:
                        for exten in ['eso','audit','bnd','svg','eio','mtd','rvaudit','shd']:
                            op="IF EXIST \"{0}.{1}\" DEL \"{0}.{1}\" ".format(idfpath,exten)
                            inf.write(op+"\n")
                        
                        
                    run.simulated='Yes'
                    run.simulate_request=False
            op="chdir {0}\\{1}".format(rootdir,folder.proj_name)
            inf.write(op+"\n")
        if sim:
            Popen(batfile)    

##Creates script to run multiple jobs on legion
def create_legion_scripts(rootdir, epdir, sim=False,folder=None):
    legion_eplus_script(rootdir, epdir, sim=sim,folder=folder)
    legion_write_script(rootdir, epdir, sim=sim,folder=folder)
    legion_train_script(rootdir, epdir, sim=sim,folder=folder)
    legion_test_script(rootdir, epdir, sim=sim,folder=folder)
    
def legion_eplus_script(rootdir, epdir, sim=False,folder=None):
    
    if folder!=None and (sys.platform=='darwin' or sys.platform == 'linux2' or sys.platform == 'linux'):
        sims_per_run = 25    ##stride
        
        ##Move this somewhere more general if more people start using
        wdir = "/home/ucqbpsy/Scratch/Simulations/EPG_HPRU/epg2/"+folder.proj_name+"/"+folder.runs_name
        matter = "#!/bin/bash -l\n"
        matter += "#$ -S /bin/bash\n" 
        matter += "#$ -l h_rt=18:0:0\n"
        matter += "#$ -l mem=1G\n"
        matter += "#$ -l tmpfs=15G\n"
        matter += "#$ -t 1-"+str(len(folder.runs.values()))+":"+str(sims_per_run)+"\n"
        matter += "#$ -N ep_"+folder.runs_name+"\n"+"\n"
        
        matter += "#$ -wd "+wdir+"\n"+"\n"
        matter += "cd $TMPDIR"+"\n"
        matter += "WORK_DIR="+wdir+"\n"
        matter += "mkdir $WORK_DIR/Output\n"+"\n"
        
        matter += "for (( i=$SGE_TASK_ID; i<$SGE_TASK_ID+"+str(sims_per_run)+"; i++ ))"+"\n"
        matter += "do\n"
        matter += "\t if [ ! -d $WORK_DIR/Temp_$SGE_TASK_ID ]; then\n"
        matter += "\t \t mkdir $WORK_DIR/Temp_$SGE_TASK_ID\n"
        matter += "\t fi\n"
        
        matter += "\t if [ ! -f $WORK_DIR/Output/EPG2_$i.csv ]; then\n"
        matter += "\t\t cp $WORK_DIR/EPG2_$i.idf $WORK_DIR/Temp_$SGE_TASK_ID\n"
        matter += "\t\t ( cmdpid=$BASHPID; (sleep 3600; kill $cmdpid) & exec runenergyplus $WORK_DIR/Temp_$SGE_TASK_ID/EPG2_$i.idf "+list(folder.runs.values())[1].weather+" )\n"
        matter += "\t\t cp $WORK_DIR/Temp_$SGE_TASK_ID/Output/EPG2_$i.err $WORK_DIR/Output\n"
        matter += "\t\t cp $WORK_DIR/Temp_$SGE_TASK_ID/Output/EPG2_$i.csv $WORK_DIR/Output\n"
        matter += "\t fi\n"
        matter += "\t rm -fr $WORK_DIR/Temp_$SGE_TASK_ID\n"
        matter += "done"
        
        if not os.path.exists(rootdir+"/"+folder.proj_name+"/scripts/legion_eplus/"):
            os.makedirs(rootdir+"/"+folder.proj_name+"/scripts/legion_eplus/")
        
        shfile="{0}/{1}/scripts/legion_eplus/eplus_{2}.sh".format(rootdir,folder.proj_name, folder.runs_name)
        with open(shfile,mode='w') as inf:
            inf.write(matter)
                     
        ##Write the batch submission script
        sub_file="{0}/{1}/scripts/eplus_{1}.sh".format(rootdir,folder.proj_name)
        op_run="qsub legion_eplus/eplus_{0}.sh\n".format(folder.runs_name) 
        if (os.path.isfile(sub_file)): 
            with open(sub_file,mode='a') as inf:
                inf.write(op_run)
        else:  
            with open(sub_file,mode='w') as inf:
                inf.write("#!/bin/bash\n"+op_run)
                
def legion_write_script(rootdir, epdir, sim=False,folder=None):
    
    stride=100
    
    if len(folder.runs.values())<stride:
        stride = len(folder.runs.values())
    
    in_dir = "/home/ucqbpsy/Scratch/Simulations/EPG_HPRU/epg2/"+folder.proj_name
    w_dir = "/home/ucqbpsy/Scratch/Simulations/EPR/"
    
    matter =  "#!/bin/bash -l \n"
    matter += "#$ -S /bin/bash"+"\n"
    matter += "#$ -l h_rt=8:0:0"+"\n"

    matter += "#$ -l mem=10G"+"\n"
    matter += "#$ -l tmpfs=15G"+"\n"
    matter += "#$ -N write_"+folder.runs_name+"\n"
    matter += "#$ -wd "+w_dir+"\n"+"\n"
 
    matter += "WORK_DIR="+w_dir+"\n"
    matter += "#$ -t 1-"+str(len(folder.runs.values()))+":"+str(stride)+" \n\n"

    matter += "cd $TMPDIR \n"

    matter += "python3 "+w_dir+"run_epr.py -i "+in_dir+" -p "+w_dir+folder.proj_name+" -c "+folder.runs_name+".csv --write -s $SGE_TASK_ID -e $(($SGE_TASK_ID+"+str(stride-1)+"))"
    
    if not os.path.exists(rootdir+"/"+folder.proj_name+"/scripts/legion_write/"):
            os.makedirs(rootdir+"/"+folder.proj_name+"/scripts/legion_write/")
        
    shfile="{0}/{1}/scripts/legion_write/write_{2}.sh".format(rootdir,folder.proj_name, folder.runs_name)
    with open(shfile,mode='w') as inf:
        inf.write(matter)
        
    ##Write the batch submission script
    sub_file="{0}/{1}/scripts/write_{1}.sh".format(rootdir,folder.proj_name)
    op_run="qsub legion_write/write_{0}.sh\n".format(folder.runs_name) 
    if (os.path.isfile(sub_file)): 
        with open(sub_file,mode='a') as inf:
            inf.write(op_run)
    else:  
        with open(sub_file,mode='w') as inf:
            inf.write("#!/bin/bash\n"+op_run)
    
def legion_train_script(rootdir, epdir, sim=False,folder=None):
    
    if folder.n_test != None:
        if folder.n_test != len(folder.runs.values()):
        
            in_dir = "/home/ucqbpsy/Scratch/Simulations/EPG_HPRU/epg2/"+folder.proj_name
            w_dir = "/home/ucqbpsy/Scratch/Simulations/EPR/"
            
            matter =  "#!/bin/bash -l \n"
            matter += "#$ -S /bin/bash"+"\n"
            matter += "#$ -l h_rt=8:0:0"+"\n"
        
            matter += "#$ -l mem=10G"+"\n"
            matter += "#$ -l tmpfs=15G"+"\n"
            matter += "#$ -N train_"+folder.runs_name+"\n"
            matter += "#$ -wd "+w_dir+"\n"+"\n"
         
            matter += "WORK_DIR="+w_dir+"\n"

            matter += "cd $TMPDIR \n"
        
            matter += "python3 "+w_dir+"run_epr.py -i "+in_dir+" -p "+w_dir+folder.proj_name+" -c "+folder.runs_name+".csv --train -e "+str(len(folder.runs.values()))+"\n"
            
            if not os.path.exists(rootdir+"/"+folder.proj_name+"/scripts/legion_train/"):
                    os.makedirs(rootdir+"/"+folder.proj_name+"/scripts/legion_train/")
                
            shfile="{0}/{1}/scripts/legion_train/train_{2}.sh".format(rootdir,folder.proj_name, folder.runs_name)
            with open(shfile,mode='w') as inf:
                inf.write(matter)
                
            ##Write the batch submission script
            sub_file="{0}/{1}/scripts/train_{1}.sh".format(rootdir,folder.proj_name)
            op_run="qsub legion_train/train_{0}.sh\n".format(folder.runs_name) 
            if (os.path.isfile(sub_file)): 
                with open(sub_file,mode='a') as inf:
                    inf.write(op_run)
            else:  
                with open(sub_file,mode='w') as inf:
                    inf.write("#!/bin/bash\n"+op_run)

def legion_test_script(rootdir, epdir, sim=False,folder=None):
    
    if folder.n_test != None:
        if folder.n_test == len(folder.runs.values()):
            
            for n_train in folder.n_train:
                if n_train != folder.n_test:
                    in_dir = "/home/ucqbpsy/Scratch/Simulations/EPG_HPRU/epg2/"+folder.proj_name
                    w_dir = "/home/ucqbpsy/Scratch/Simulations/EPR/"
                    
                    matter =  "#!/bin/bash -l \n"
                    matter += "#$ -S /bin/bash"+"\n"
                    matter += "#$ -l h_rt=8:0:0"+"\n"
                
                    matter += "#$ -l mem=10G"+"\n"
                    matter += "#$ -l tmpfs=15G"+"\n"
                    matter += "#$ -N test_"+folder.runs_name+"\n"
                    matter += "#$ -wd "+w_dir+"\n"+"\n"
                 
                    matter += "WORK_DIR="+w_dir+"\n"
        
                    matter += "cd $TMPDIR \n"
                    
                    name = ""
                    name_split =folder.runs_name.split('_')[0:4]
                    for n in name_split:
                        name+=n+"_"
                    
                    matter += "python3 "+w_dir+"run_epr.py -i "+in_dir+" -p "+w_dir+folder.proj_name+" -t "+w_dir+folder.proj_name+"/"+name+str(n_train)+" -c "+folder.runs_name+".csv --test -e "+str(len(folder.runs.values()))+"\n"
                    
                    if not os.path.exists(rootdir+"/"+folder.proj_name+"/scripts/legion_test/"):
                            os.makedirs(rootdir+"/"+folder.proj_name+"/scripts/legion_test/")
                        
                    shfile="{0}/{1}/scripts/legion_test/test_{2}.sh".format(rootdir,folder.proj_name, name+str(n_train))
                    with open(shfile,mode='w') as inf:
                        inf.write(matter)
                        
                    ##Write the batch submission script
                    sub_file="{0}/{1}/scripts/test_{1}.sh".format(rootdir,folder.proj_name)
                    op_run="qsub legion_test/test_{0}.sh\n".format(name+str(n_train)) 
                    if (os.path.isfile(sub_file)): 
                        with open(sub_file,mode='a') as inf:
                            inf.write(op_run)
                    else:  
                        with open(sub_file,mode='w') as inf:
                            inf.write("#!/bin/bash\n"+op_run)
                     
def create_csv(rootdir,folder,fname):
    indir=folder.proj_name
    csvfile="{0}/{1}/{2}.csv".format(rootdir,indir,fname)
    print(csvfile)
    with open(csvfile,mode='w') as outf:
        outf.write("Directory,{0}\n".format(indir))
        
        run_a = list(folder.runs.values())[0]

        outf.write("Run Number,Built,BAT file Created,Data Good,Simulation Level (1 Default/3 HAMT),Weather,Building,Occupation,Environment,Package,,Start Day,Start Month,End Day,End Month,Time Steps per hour,,Permeability,Orientation,Window Open Thresh,Heater Thresh,Cook Fact,Wall U,Roof U,Window U,Floor U,Glaz Fract,Floor Height,Area Fact(Area),Gains Fact\n")
        
        for run in list(folder.runs.values()):
            print(("Writing Run {0} to CSV ".format(run.rn)))
            csvline=run.pack_csvline()
            outf.write(csvline+"\n")

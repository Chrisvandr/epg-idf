"""Run and folder properties containing runs"""
import os
import numpy as np

class Folder():
    def __init__(self,rootdir, proj_name, runs_name=None, n_test = None, n_train = None):
        
        if not os.path.exists(rootdir+'/'+proj_name):
            os.makedirs(rootdir+'/'+proj_name)
            
        if runs_name!=None:
            if not os.path.exists(proj_name+'/'+runs_name): 
                os.makedirs(rootdir+'/'+proj_name+'/'+runs_name)
    
        self.proj_name=proj_name
        self.runs_name=runs_name
        self.runs={}
        self.n_test = n_test
        self.n_train = n_train
        

def create_run(dbdats,rinfo=None):
    run=None   ## initiate the run to no value
    if rinfo!=None:
        ##set up the run info
        packs,envs,builds,occups,equips,reports=dbdats

        ##create run object
        run = Run(rinfo['rn'])
        run.built=rinfo['built']
        run.simulated=rinfo['simulated']
        run.sim_level=int(rinfo['sim_level'])
                
        ##assign values from script perhaps this should be done inside the objection function
        run.start_day=rinfo['start_day']
        run.start_month=rinfo['start_month']
        run.end_day=rinfo['end_day']
        run.end_month=rinfo['end_month']
        run.timesteps=int(rinfo['timesteps'])
                
        run.permeability=float(rinfo['permeability'])
           
        run.orient=float(rinfo['orientation'])  ##angle wrt north        
        
        ##run occupancy props
        run.window_openthresh=float(rinfo['window_openthresh'])
        run.heater_thresh=float(rinfo['heater_thresh'])
        run.cook_pm25fact=float(rinfo['cook_pm25fact'])
        
        #make sure wall u not too high
        max_f = 2.55
        if 'Cavity' in rinfo['package']:
            max_f = 1.75

        ## U-values check
        if(rinfo['wall_u'] != 0. and float(rinfo['wall_u'])>max_f):
            run.wall_u = max_f
        elif(rinfo['wall_u'] != 0. and float(rinfo['wall_u'])<0.15):
            run.wall_u = 0.15
        else:
            run.wall_u=float(rinfo['wall_u'])
            
        if run.wall_u <= 0.5 and 'Cavity' in rinfo['package']:
            rinfo['package'] = 'FilledCavity'        
        
        if (rinfo['roof_u'] != 0. and float(rinfo['roof_u'])>=2.25):
            run.roof_u=2.24
        elif (rinfo['roof_u'] != 0. and float(rinfo['roof_u'])<=0.1):
            run.roof_u=0.1
        else:    
            run.roof_u=float(rinfo['roof_u'])
        
        if (rinfo['window_u'] != 0. and float(rinfo['window_u'])>=4.8):
            run.window_u=4.8
        elif (rinfo['window_u'] != 0. and float(rinfo['window_u'])<=0.85):
            run.window_u=0.85
        else:
            run.window_u=float(rinfo['window_u'])
        
        if (rinfo['floor_u'] != 0. and float(rinfo['floor_u'])>=1.2):
            run.floor_u=1.2
        elif (rinfo['floor_u'] != 0. and float(rinfo['floor_u'])<=0.15):
            run.floor_u=0.15
        else:
            run.floor_u=float(rinfo['floor_u'])
        
        #choose what type of run to do. e.g. hypercube or norm
        run.type=rinfo['run_type']

        run.glaz_fract = float(rinfo['glaz_fract'])
        if float(rinfo['glaz_fract']) > 0.6 or float(rinfo['glaz_fract']) < 0.:
            run.glaz_fract = 0.25
        
        run.floor_height = float(rinfo['floor_height'])
        if float(rinfo['floor_height']) == 88.8:
            run.floor_height = 2.4
        
        run.floor_area = float(rinfo['floor_area_scale'])
        if(float(rinfo['floor_area_scale']) < 0.65 and float(rinfo['floor_area_scale']) != 0.0):
            run.floor_area = 0.65        
        
        run.gain_fact = 1
        if (float(rinfo['gain_factor'] != 0.)):
            run.gain_fact = float(rinfo['gain_factor'])
        
        run.weather=rinfo['weather']
                                        
        for param,params in zip(['occupation','building','environment','package'],[occups,builds,envs,packs]):             
            if param in rinfo:
                if rinfo[param] in list(params.keys()):
                    run.set_prop(param,params[rinfo[param]])
                else:
                    print((param," ",rinfo[param]," Not Found. Run ",run.rn))
                    run.set_prop(param,"Fail ==> "+rinfo[param])
                    run.data_good=False
            else:
                print(("No ",param," Found in Run ",run.rn))
                run.data_good=False
                      
        run.set_prop('reports',reports)
        run.set_prop('equipment',equips)
    
    return run
    
class Run():
    def __init__(self,rn):
        
        self.rn=rn
        self.data_good=True
        self.props={}
        self.use_AirflowNetwork=True
        self.weather=None
        
        self.built='No'
        self.simulated='No'
        self.build_request=True
        self.simulate_request=True
        self.EPdatas=None
        self.OutputAll=False
        self.type = None
        
    def pack_csvline(self):
        
        csvline="{0}".format(self.rn)
        csvline+=",{0}".format(self.built)
        csvline+=",{0}".format(self.simulated)
        csvline+=",{0}".format(self.data_good)
        csvline+=",{0}".format(self.sim_level)
        csvline+=",{0}".format(self.weather)
        
        for param in ['building','occupation','environment','package']:
            if param in self.props:
                if isinstance(self.props[param],str):
                    csvline+=','+self.props[param]
                else:
                    csvline+=','+self.props[param].name
            else:
                csvline+=','+param+' not found'
            
        csvline+=","  
        csvline+=",{0}".format(self.start_day)
        csvline+=",{0}".format(self.start_month)
        csvline+=",{0}".format(self.end_day)
        csvline+=",{0}".format(self.end_month)
        csvline+=",{0}".format(self.timesteps)
        
#        if self.OutputAll:
#            csvline+=",{0}".format('Yes')
#        else:
#            csvline+=",{0}".format('No')
#        csvline+=","
#        if self.use_AirflowNetwork:
#            csvline+=",{0}".format('Yes')
#        else:
#            csvline+=",{0}".format('No')
        csvline+=","

        csvline+=",{0}".format(self.permeability)
        csvline+=",{0}".format(self.orient)
        csvline+=",{0}".format(self.window_openthresh)
        csvline+=",{0}".format(self.heater_thresh)
        csvline+=",{0}".format(self.cook_pm25fact)
        csvline+=",{0}".format(self.wall_u)
        csvline+=",{0}".format(self.roof_u)
        csvline+=",{0}".format(self.window_u)
        csvline+=",{0}".format(self.floor_u)

        csvline+=",{0}".format(self.glaz_fract)
        csvline+=",{0}".format(self.floor_height)
        csvline+=",{0}".format(self.floor_area)
        csvline+=",{0}".format(self.gain_fact)
            
        return csvline
    
    def set_prop(self,propname=None,prop=None):
        self.props[propname]=prop
        
        
        

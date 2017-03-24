"""Class for a Zone in a building"""

import numpy as np
import properties.surface as surf

class Zone():
    def __init__(self,zname):
        self.name=zname
        self.origin=np.array([0.0,0.0,0.0])
        self.centre=np.array([0.0,0.0,0.0])
        self.joinname=''
        self.joinlength=0.0
        self.doordatas=[]
        self.glazing={}
        self.hang={}
        self.L1=0.0
        self.L2=0.0
        self.h=0    
        self.party=[]
        #for a pitched roof
        self.pitch_roof=[]
        self.pitch_fraction = 0.0

        self.re_init()
        
    def re_init(self):
        
        self.hang_surfaces=[]
        self.windows=[]
        self.floor_area=0.0
        self.int_area=0.0
        self.volume=0.0
        
        ##For CIBSE calc
        self.ground_floor_area=0.0
        self.wall_area=0.0
        self.window_area=0.0
        self.roof_area=0.0
        
        self.join=None
        
        self.heater=None
        self.equiplist=[]
        self.appliances=[]
        self.people={}
        
        self.doors=[]
        self.source_sink=None
        self.contaminant=None
        self.dep_rate=None
        self.dep_vel=None
        self.exhaust_sched=None
        self.hasfan=False
        self.inside=True
        self.surfaces=[]
        self.hscheds=[]      
        
    def set_prop(self,item1,item2,bunit,bvalue):
        if item1=='L1' and item2=='Length':
            self.L1=np.float(bvalue)
        if item1=='L2' and item2=='Length':
            self.L2=np.float(bvalue)  
        if item1=='h' and item2=='Height':
            self.h=np.float(bvalue)
        if item1=='x' and item2=='Coord':
            self.origin[0]=np.float(bvalue)
        if item1=='y' and item2=='Coord':
            self.origin[1]=np.float(bvalue)
        if item1=='z' and item2=='Coord':
            self.origin[2]=np.float(bvalue)
        if item2=='Party':
            self.party.append(item1)
        if item1=='J' and item2=='Join' and bunit=='Zone':
            self.joinname=bvalue    
        if item1=='J' and item2=='Join' and bunit=='length':
            self.joinlength=np.float(bvalue)
        if item2=='Glazing':
            self.glazing[item1]=np.float(bvalue)
        if item2=='Door':
            doordata={}
            doordata['dside']=item1
            doordata['zonen']=bvalue            
            self.doordatas.append(doordata)
        if item2=='Hang':
            self.hang[item1]=np.float(bvalue)
        if item2=='Pitch' and item1=='L1':
            self.pitch_roof.append('L1')
            self.pitch_roof.append('L3')
            if bunit == 'fraction':
                self.pitch_fraction = float(bvalue)
        if item2=='Pitch' and item1=='L2':
            self.pitch_roof.append('L2')
            self.pitch_roof.append('L4')
            if bunit == 'fraction':
                self.pitch_fraction = float(bvalue)
            
    ##Function to set up surface parameters to create surface
    def set_surface(self, s_EPG_type, pack):
        #initialize stuff
        s_surface_type=''
        s_fab=''
        s_name="{0}_{1}".format(self.name, s_EPG_type)
        s_wall_type=''
        s_outside_BC_obj=''
        s_outside_BC='Outdoors'
        s_sunexp='SunExposed'
        s_windexp='WindExposed'
        s_shape='Rectangle'
        r1_x = 0.0
        r1_y = 0.0
        r2_x = 0.0
        r2_y = 0.0
        
        if pack!=None:
            ext_wall=pack.props['external_wall']         
            roof=pack.props['roof']
        
        #If standard floor               
        if s_EPG_type == 'floor':
            s_name="{0}_Floor".format(self.name)
            s_surface_type='Floor'         
            s_outside_BC='Ground'
            s_outside_BC_obj=''
            s_sunexp='NoSun'
            s_windexp='NoWind'
              
        #If standard wall    
        if (s_EPG_type == 'L1' or s_EPG_type == 'L2' or s_EPG_type == 'L3' or s_EPG_type == 'L4') and self.origin[2] >= 0.:        
            s_outside_BC_obj=''
            s_name="{0}_{1}".format(self.name, s_EPG_type)
            s_fab=ext_wall
            s_wall_type='external_wall'
            s_surface_type='Wall'
        elif(s_EPG_type == 'L1' or s_EPG_type == 'L2' or s_EPG_type == 'L3' or s_EPG_type == 'L4') and self.origin[2] < 0.:
            s_outside_BC='Ground'
            s_sunexp='NoSun'
            s_windexp='NoWind'
            s_outside_BC_obj=''
            s_name="{0}_{1}".format(self.name, s_EPG_type)
            s_fab=ext_wall
            s_wall_type='external_wall'
            s_surface_type='Wall'
        
        if s_EPG_type == 'L1' or s_EPG_type == 'L3':
            r2_x = self.L1
                
        if s_EPG_type == 'L2' or s_EPG_type == 'L4':
            r2_y = -self.L2
        
        ##Set the name of walls if pitched ceiling
        if len(self.pitch_roof)>0 and s_EPG_type not in self.pitch_roof and s_EPG_type != 'floor':
            s_name="{0}_Side_{1}".format(self.name, s_EPG_type)
            s_shape='Triangle'
            
            if self.pitch_fraction != 0:
                s_fab=roof
                s_wall_type='roof'
            
            ##Set the r1 vecs for the pitched roof            
            if 'L1' in self.pitch_roof:
                r1_y = self.L2/2.0
                r1_x = self.pitch_fraction*self.L1/2.0
                r2_y = self.L2
            elif 'L2' in self.pitch_roof:
                r1_x = self.L1/2.0
                r1_y = self.pitch_fraction*self.L2/2.0
                r2_x = -self.L1
            
        #If standard ceiling
        if s_EPG_type == 'roof':           
            s_outside_BC_obj=''
            s_name="{0}_Ceiling".format(self.name)
            
        ##If the surface is a pitched roof
        if s_EPG_type in self.pitch_roof:
            s_name = "{0}_PitchCeiling_{1}".format(self.name, s_EPG_type)
            s_surface_type='Ceiling'
            s_fab=roof
            s_wall_type='roof'
            
            if self.pitch_fraction != 0.0:
                s_shape='Trapezium'            
            
            if 'L1' in self.pitch_roof:                
                r1_x = self.pitch_fraction*self.L1/2.0
                r1_y = self.L2/2.0

            elif 'L2' in self.pitch_roof:
                r1_x = self.L1/2.0
                r1_y = self.pitch_fraction*self.L2/2.0
       
        ##Add if there is an l-shaped roof L1 is a normal sidewall
        if self.joinlength != 0 and (s_EPG_type == 'L2' or s_EPG_type == 'L4'):
            r1_y = -self.joinlength
            s_shape='Join'             
        
        ##L3 is triangular shape
        if (self.joinlength != 0 and s_EPG_type == 'L3'):
            r1_y = -self.joinlength
            #s_wall_type='internal_wall'
            s_outside_BC='Surface'
            s_outside_BC_obj=s_name
            s_surface_type='Ceiling'
            s_fab=roof
            s_wall_type='roof'
            s_sunexp='NoSun'
            s_windexp='NoWind'
            s_shape='Reverse'
            
        ##If the surface is partitioned with another surface e.g no wind/sun exposure and adiabatic               
        if s_EPG_type in self.party:
            s_outside_BC='Adiabatic'
            s_outside_BC_obj=s_name
            s_sunexp='NoSun'
            s_windexp='NoWind'
        
        #print s_name, s_surface_type, s_fab, s_wall_type, s_outside_BC, s_outside_BC_obj, s_sunexp, s_windexp
        return s_name, s_surface_type, s_fab, s_wall_type, s_outside_BC, s_outside_BC_obj, s_sunexp, s_windexp, r1_x, r1_y, r2_x, r2_y, s_shape
    
    
    ###If you know what is good for you... Don't mess with the stuff below!
    ##r1(down) and r2 (across) are used to go around the shape and select the vertices. 
    ##Here it is done for the main surface. Permeabilities are done in surface.py: create_perms function
    def create_surfaces(self,pack):
        self.re_init()       
        self.centre=self.origin+np.array([self.L1/2.0,0,0])-np.array([0.0,self.L2/2.0,0.0])+np.array([0.0,0.0,self.h/2.0])
        ground=None
        roof=None
        self.floor_area=np.abs(self.L1*self.L2)
        self.volume=np.abs(self.L1*self.L2*self.h)
        
        if pack!=None:     
            ground=pack.props['ground']
            roof=pack.props['roof']
                           
        ##initialize everything to the floor
        s_name, _, _, _, s_outside_BC, s_outside_BC_obj, _, _, _, _, _, _,_ = self.set_surface('floor',pack)

        ##Create the floor of the zone  (all zones need a floor)         
        surface=surf.Surface(self,
                        name=s_name,
                        EPG2_type='floor',
                        surface_type='Floor',
                        fab=ground,
                        wall_type='ground',
                        outside_BC=s_outside_BC,
                        outside_BC_obj=s_outside_BC_obj,
                        sunexp='NoSun',
                        windexp='NoWind',
                        origin=self.origin,
                        r1=np.array([0.0,-1.0*self.L2,0.0]),
                        r2=np.array([self.L1,0.0,0.0])
                        )
        self.surfaces.append(surface)
        
        ##For top wall: opposite L3: r1_y introduced for pitched roof (default = 0)
        s_name, s_surface_type, s_fab, s_wall_type, s_outside_BC, s_outside_BC_obj, s_sunexp, s_windexp, r1_x, r1_y, r2_x, r2_y,  s_shape = self.set_surface('L1',pack)  
#        print("L1")
#        print("o: ", self.origin)
#        print("origin: ",self.origin+np.array([r1_x,-r1_y,self.h]))
#        print("r1: ", np.array([r1_x, r1_y, -self.h]))    
#        print("r2: ", np.array([r2_x, r2_y, 0.0]))
        surface=surf.Surface(self,
                        name=s_name,
                        EPG2_type='L1',
                        surface_type=s_surface_type,
                        fab=s_fab,
                        wall_type=s_wall_type,
                        outside_BC=s_outside_BC,
                        outside_BC_obj=s_outside_BC_obj,
                        sunexp=s_sunexp,
                        windexp=s_windexp,
                        origin=self.origin+np.array([r1_x,-r1_y,self.h]),###If you know what is good for you... Don't mess with the stuff below!  
                        r1=np.array([r1_x, r1_y, -self.h]),
                        r2=np.array([r2_x, r2_y, 0.0]),
                        shape=s_shape                       
                        )
        self.surfaces.append(surface)
        
        ##For right hand wall: opposite L4
        s_name, s_surface_type, s_fab, s_wall_type, s_outside_BC, s_outside_BC_obj, s_sunexp, s_windexp, r1_x, r1_y,r2_x, r2_y, s_shape = self.set_surface('L2',pack) 

        orig=self.origin+np.array([-r1_x,-r1_y,self.h])+np.array([self.L1,0.0,0.0])
        ##Fix for origin
        if self.joinlength !=0:
            orig=self.origin+np.array([-r1_x,0,self.h])+np.array([self.L1,0.0,0.0])
        #print("L2")
#        print("origin: ", self.origin+np.array([-r1_x,-r1_y,self.h])+np.array([self.L1,0.0,0.0]))
#        print("r1: ", np.array([r1_x,-r1_y, -self.h]))    
#        print("r2: ", np.array([r2_x, r2_y,0.0]))
        surface=surf.Surface(self,
                        name=s_name,
                        EPG2_type='L2',
                        surface_type=s_surface_type,
                        fab=s_fab,
                        wall_type=s_wall_type,
                        outside_BC=s_outside_BC,
                        outside_BC_obj=s_outside_BC_obj,
                        sunexp=s_sunexp,
                        windexp=s_windexp,
                        origin=orig,                        
                        r1=np.array([r1_x,-r1_y, -self.h]),
                        r2=np.array([r2_x, r2_y,0.0]),
                        shape=s_shape                       
                        )
        self.surfaces.append(surface)
        
        ##For bottom wall: opposite L1
        s_name, s_surface_type, s_fab, s_wall_type, s_outside_BC, s_outside_BC_obj, s_sunexp, s_windexp, r1_x, r1_y,r2_x, r2_y, s_shape = self.set_surface('L3',pack)
#        print("L3")
#        print("origin: ",self.origin+np.array([-r1_x, r1_y,self.h])+np.array([self.L1,0.0,0.0])-np.array([0.0,self.L2,0.0]))
#        print("r1: ", np.array([-r1_x, -r1_y, -self.h]))    
#        print("r2: ", np.array([-r2_x,-r2_y,0.0]))
        surface=surf.Surface(self,
                        name=s_name,
                        EPG2_type='L3',
                        surface_type=s_surface_type,
                        fab=s_fab,
                        wall_type=s_wall_type,
                        outside_BC=s_outside_BC,
                        outside_BC_obj=s_outside_BC_obj,
                        sunexp=s_sunexp,
                        windexp=s_windexp,
                        origin=self.origin+np.array([-r1_x, r1_y,self.h])+np.array([self.L1,0.0,0.0])-np.array([0.0,self.L2,0.0]),
                        r1=np.array([-r1_x, -r1_y, -self.h]),
                        r2=np.array([-r2_x,-r2_y,0.0]),
                        shape=s_shape               
                        )
        self.surfaces.append(surface)
        
        ##For left hand wall: opposite L2
        s_name, s_surface_type, s_fab, s_wall_type, s_outside_BC, s_outside_BC_obj, s_sunexp, s_windexp, r1_x, r1_y,r2_x, r2_y, s_shape = self.set_surface('L4',pack) 
#        print("L4")
#        print("origin: ",self.origin+np.array([r1_x,r1_y,self.h])-np.array([0.0,self.L2,0.0]))
#        print("r1: ", np.array([-r1_x, r1_y, -self.h]))    
#        print("r2: ", np.array([r2_x, -r2_y, 0.0]))
        surface=surf.Surface(self,
                        name=s_name,
                        EPG2_type='L4',
                        surface_type=s_surface_type,
                        fab=s_fab,
                        wall_type=s_wall_type,
                        outside_BC=s_outside_BC,
                        outside_BC_obj=s_outside_BC_obj,
                        sunexp=s_sunexp,
                        windexp=s_windexp,
                        origin=self.origin+np.array([r1_x,r1_y,self.h])-np.array([0.0,self.L2,0.0]),
                        r1=np.array([-r1_x, r1_y, -self.h]),
                        r2=np.array([r2_x, -r2_y, 0.0]),
                        shape=s_shape                       
                        )
        self.surfaces.append(surface)
        
        
        ##For the roof/ceiling of a zone. Wont be created if pitched roof already made
        if len(self.pitch_roof) == 0:
            s_name, s_surface_type, s_fab, s_wall_type, s_outside_BC, s_outside_BC_obj, s_sunexp, s_windexp, r1_x, r1_y,r2_x, r2_y, s_shape = self.set_surface('roof',pack)
#            print("L2")
#            print("origin: ", self.origin+np.array([0.0,0.0,self.h]))
#            print("r1: ", np.array([self.L1,0.0,0.0]))     
#            print("r2: ", np.array([0.0,-1.0*self.L2,0.0]))           
            surface=surf.Surface(zone=self,
                            name=s_name,
                            EPG2_type='roof',
                            surface_type='Ceiling',
                            fab=roof,
                            wall_type='roof',
                            outside_BC=s_outside_BC,
                            outside_BC_obj=s_outside_BC_obj,
                            sunexp=s_sunexp,
                            windexp=s_windexp,
                            origin=self.origin+np.array([0.0,0.0,self.h]),
                            r1=np.array([self.L1,0.0,0.0]),
                            r2=np.array([0.0,-1.0*self.L2,0.0])
                            )
                
            self.surfaces.append(surface)        


        
        
            
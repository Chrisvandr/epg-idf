"""File containing the building Class which uses wind and shading Classes""" 

import numpy as np
import copy  as cp
import math

import inputvars as inp
import properties.zone as zone
import properties.schedule as schedule
import properties.surface as surf

class Building():
    def __init__(self,dline):
        self.dline=dline
        self.name=dline[0]        
        self.type=dline[1]
        self.bedrooms=dline[2]
        self.levels=dline[3]
        self.inzones={}
        self.inzones['Outside']=zone.Zone('Outside')
        self.inzones['Outside'].inside=False
          
        self.re_init()
        
    def re_init(self):
        self.buildmax=np.array([-1000.,-1000.,-1000.,])
        self.buildmin=np.array([1000.,1000.,1000.,])
        self.flowC=0.0
        self.perm_area=0.0
        self.ntrickles=0
        self.bedrooms=0
        self.tELA=0.0
        self.floor_area=0.0
        
        self.schedules=[]
        self.shading_surfaces=[]
        self.zones=cp.copy(self.inzones)
    
    ##method to resize the glazing fraction
    def set_glazing_fract(self, run = None):
        
         for zn in self.zones:
            for key in self.zones[zn].glazing:
                self.zones[zn].glazing[key]=run.glaz_fract
        
    
    def create(self,run=None,pack=None,occu=None,equips=None,shading=None):
    
        self.re_init()
        
        ##Get floor and ceiling height scale factors for LHD 
        length_sf = 1
        height_sf = 1
        
        if (run.glaz_fract != 0.):
            self.set_glazing_fract(run=run)
        
        if run.floor_area != 0.:    
            length_sf = math.sqrt(run.floor_area)
     
        ##need to find the living room height to get a scale factor for other zones
        if (self.type == "Flat"):
            height_sf = run.floor_height /self.zones['Living_GF'].h
        else:
            height_sf = run.floor_height /self.zones['Living'].h  
            
        if pack!=None:
            pack.create(run)
 
        breakout=1000000
                
        surfaces=[]
        for zone in sorted(list(self.zones.values()), key=lambda ff:ff.name):
        #for zone in self.zones.values():
            if zone.inside:
                zone.create_surfaces(pack=pack)
                surfaces.extend(zone.surfaces)
                if zone.joinname!='':
                    zone.join=self.zones[zone.joinname]                 
                    
        for _ in [1,2]:        
            kk=0
            for surface1 in surfaces:
                for surface2 in surfaces:
                     
                    bvec=np.cross(surface1.bvec,surface2.bvec)
                    surfdiff=np.dot((surface1.origin-surface2.origin),surface1.bvec)

                    if surface1!=surface2 and np.abs(np.dot(bvec,bvec))<inp.diffthresh and np.abs(surfdiff)<inp.diffthresh:
                    #if surface1!=surface2 and np.abs(np.dot(bvec,bvec))==0 and np.abs(surfdiff)==0:    
                        newsurfaces=[]
                        newsurfaces=surface1.inhere(surface2,pack=pack)
                        if len(newsurfaces)>0:
             
                            surfaces.extend(newsurfaces)       
                     
                    kk+=1
                    if kk>breakout:
                        break
                kk+=1
                if kk>breakout:
                    break   
        
               
        for zonen in list(self.zones.copy().keys()):
            zone=self.zones[zonen]
            if zone.join!=None:
                zone.volume+=zone.join.volume
              
                zone.join.surfaces.extend(cp.copy(zone.surfaces))
                jsurfs=zone.join.surfaces
                
                for surf in jsurfs[:]:

                    if surf.othersidezone==zone:
                        zone.join.surfaces.remove(surf)
                    if surf.othersidezone==zone.join:
                        zone.join.surfaces.remove(surf)
                
                    surf.zone=zone.join                    
                    if surf.othersidesurface!=None:
                        osurf=surf.othersidesurface
                        osurf.othersidezone=zone.join
                     
                del self.zones[zonen]
        
        build_min = self.get_build_origin()
        
        ##Recalc the vertices according to area and height scale factor
        if (run.floor_area != 0. or run.floor_height != 0.):
            for zonen,zone in list(self.zones.items()):
                ##recals                
                zone.L1 = zone.L1*length_sf
                zone.L2 = zone.L2*length_sf
                zone.h = zone.h*height_sf
                zone.centre = np.array([zone.centre[0]*length_sf, zone.centre[1]*length_sf, zone.centre[2]*height_sf])
                for surface in zone.surfaces:                   
                    surface.recalc_vertices(build_min, surface.vertexs, xy_shift=length_sf, z_shift=height_sf)
                    
                      
        self.find_dimentions(run, length_sf=length_sf, height_sf=height_sf)
        
        ##set floor area for LHD
        if run.type == "lhd":
            run.floor_area = self.floor_area
        
        #install various parts of the building
        self.install_perms(pack)      
        self.install_hangs()
        self.install_fensters(pack, run)                   
        self.install_doors(pack)
 
        self.tELA=inp.calc_trickle_ELA(self.floor_area,self.bedrooms,self.ntrickles)
        
        #for whole build
        #house_heat_m2= calc_CIBSE_sizing(self, run)
             
        print(("Total Floor area   ",self.floor_area))
        print(("Ground Floor area   ",self.ground_floor_area))
        print(("Volume   ",self.volume))
        print(("Number of trickles ",self.ntrickles))
        print(("Number of bedrooms ",self.bedrooms))
        print(("Tricle ELA         ",self.tELA))
        
        if occu!=None and equips!=None:
            if len(occu.hgroups)>0:
                for hgroup in occu.hgroups:

                    if len(hgroup.zones_scheds)>0:
                        for zone_sched in hgroup.zones_scheds:
                            
                            if zone_sched.zone_name in self.zones:
                                zone=self.zones[zone_sched.zone_name]
                                zone.hscheds.append(zone_sched)
            
            for zonen in list(self.zones.copy().keys()):
                zone=self.zones[zonen]
                if len(zone.hscheds)>0:
                    for hsched in zone.hscheds:
                        
                        n_bedrooms = self.bedrooms
                        if self.type=='Flat':
                            n_bedrooms = n_bedrooms/3  ##For top, middle and ground floor
                    
                        if hsched.feature=='Metabolism' or hsched.feature=='Presence':
                            ###Assign people to rooms, don't need to do adults or pensioners as always in Bedroom1                            
                            if 'Bedroom1' in hsched.zone_name and 'Child1' in hsched.gname  and n_bedrooms >= 2:
                                hsched.re_init('2')
                            elif 'Bedroom1' in hsched.zone_name and 'Child1' in hsched.gname  and n_bedrooms < 2:
                                hsched.re_init('1')
                                
                            if 'Bedroom1' in hsched.zone_name and 'Child2' in hsched.gname  and n_bedrooms >= 3:
                                hsched.re_init('3')
                            elif 'Bedroom1' in hsched.zone_name and 'Child2' in hsched.gname  and n_bedrooms == 2:
                                hsched.re_init('2')
                            elif 'Bedroom1' in hsched.zone_name and 'Child2' in hsched.gname  and n_bedrooms < 2:
                                hsched.re_init('1')
                                
                            if 'Bedroom1' in hsched.zone_name and 'Child3' in hsched.gname  and n_bedrooms >= 4:
                                hsched.re_init('4')
                            elif 'Bedroom1' in hsched.zone_name and 'Child3' in hsched.gname  and n_bedrooms == 3:
                                hsched.re_init('3')
                            elif 'Bedroom1' in hsched.zone_name and 'Child3' in hsched.gname  and n_bedrooms == 2:
                                hsched.re_init('2')
                            elif 'Bedroom1' in hsched.zone_name and 'Child3' in hsched.gname  and n_bedrooms < 2:
                                hsched.re_init('1')    
                            
                            if hsched.gname not in zone.people:
                                person  = schedule.Person(hsched.gname)
                                zone.people[hsched.gname]=person
                                
                            person=zone.people[hsched.gname]
                            
                            if hsched.feature=='Metabolism':
                                person.metabolic_schedule=hsched
                            else:
                                person.presence_schedule=hsched
                            
                            if hsched not in self.schedules:
                                    self.schedules.append(hsched)
                            
                        if hsched.feature=='Heating':
                            if zone.heater==None:
                                zone.heater=schedule.ElecHeater(zone.name+'_heater')
                                if hsched.scaler=='Floor_Area':
                                    zone.heater.power=hsched.base_value * zone.floor_area                                   
                                elif hsched.scaler=='CIBSE':
                                    zone_heat_m2 = calc_CIBSE_sizing(zone, run)                           
                                    zone.heater.power = zone_heat_m2
                                    #print(zone.name, zone_heat_m2)
                                else:
                                    zone.heater.power=hsched.base_value
                            zone.heater.heating=hsched
                            if hsched not in self.schedules:
                                self.schedules.append(hsched)
                            
                        if hsched.feature=='Comfort':
                            if zone.heater==None:
                                zone.heater=schedule.ElecHeater(zone.name+'_heater')
                                if hsched.scaler=='Floor_Area':
                                    zone.heater.power=hsched.base_value*zone.floor_area
                                else:
                                    zone.heater.power=hsched.base_value

                            zone.heater.comfort=hsched
                            if hsched not in self.schedules:
                                if(hsched.hours[3] == 'X'):
                                    hsched.base_value = run.heater_thresh
                                    hsched.hours[3] = run.heater_thresh
                                self.schedules.append(hsched)
                      
                                
                        if hsched.feature=='Appliance':
                            

                            gain_fact = float(run.gain_fact)
                            if hsched.info in equips:
                                equip=equips[hsched.info]

                                appliance=schedule.Appliance(zone.name+'_'+hsched.name+'_'+equip.name)
                                appliance.power=float(equip.power)*float(gain_fact)
                                appliance.useful=equip.useful
                                appliance.latent=equip.latent
                                appliance.purpose=equip.purpose
                                appliance.schedule=hsched
                                appliance.calculation_method=equip.calculation_method
                                zone.appliances.append(appliance)
                                if hsched not in self.schedules:
                                    self.schedules.append(hsched)
                                
                        if hsched.feature=='Door':
                            for door in zone.doors:
                                if hsched.info==door.surface.othersidezone.name:
                                     
                                    door.control=hsched
                                    if hsched not in self.schedules:
                                        self.schedules.append(hsched)

                        if hsched.feature=='Window':
                            for window in zone.windows:
                                window.control=hsched
                                if hsched not in self.schedules:
                                    self.schedules.append(hsched)

                                    
                        if hsched.feature=='Window_Temperature':
                            for window in zone.windows:
                                window.temperature=hsched
                                if hsched not in self.schedules:
                                    if(hsched.hours[3] == 'X'):
                                        hsched.hours[3] = run.window_openthresh
                                    self.schedules.append(hsched)
                                    
                                    
                        if hsched.feature=='Window_Shading':
                            for window in zone.windows:
                                window.shading=hsched
                                if hsched not in self.schedules:
                                    self.schedules.append(hsched)  
                                    
                        if hsched.feature=='Contaminant':
                            zone.contaminant=hsched
                            if hsched not in self.schedules:
                                self.schedules.append(hsched)
                                    
                        if hsched.feature=='SourceSink':
                            zone.source_sink=hsched
                            if hsched not in self.schedules:
                                if "COOKING" in hsched.name:
                                    hsched.base_value=run.cook_pm25fact*hsched.base_value      ##mult factor for cooking
                                self.schedules.append(hsched)
                                
                        if hsched.feature=='DepRate':
                            zone.dep_rate=hsched
                            if hsched not in self.schedules:
                                self.schedules.append(hsched)
                                
                        if hsched.feature=='DepVel':
                            zone.dep_vel=hsched
                            if hsched not in self.schedules:
                                self.schedules.append(hsched)
                                
                        if hsched.feature=='p2Contaminant':
                            #zone.contaminant=hsched
                            if hsched not in self.schedules:
                                self.schedules.append(hsched)
                                    
                        if hsched.feature=='p2SourceSink':
                            #zone.source_sink=hsched
                            if hsched not in self.schedules:
                                self.schedules.append(hsched)
                                
                        if hsched.feature=='p2DepRate':
                            #zone.dep_rate=hsched
                            if hsched not in self.schedules:
                                self.schedules.append(hsched)
                                
                        if hsched.feature=='p2DepVel':
                            #zone.dep_vel=hsched
                            if hsched not in self.schedules:
                                self.schedules.append(hsched)
                            
                                
                        if hsched.feature=='ExhaustFan':
                            for surface in zone.surfaces:

                                if surface.upper:
                                    surface.exhaust=True
                                    zone.exhaust_sched=hsched
                                    fan=schedule.ZoneEquip(surface.name+'_ExhaustFan')
                                    fan.eptype='Fan:ZoneExhaust'
                                    zone.hasfan=True
                                    zone.equiplist.append(fan)
                                    if hsched not in self.schedules:
                                        self.schedules.append(hsched)
                                    break
                      
                            
                            
            for zonen in list(self.zones.copy().keys()):
                zone=self.zones[zonen]                 
                if zone.heater!=None and zone.heater.all_there():
                    zone.equiplist.append(zone.heater)

        if shading!=None:
            self.install_shading(shading)
                    
    def install_shading(self,shading):
        
        if shading!=None:

            if len(shading.shade_objects)>0:
                for sobjn,sobj in list(shading.shade_objects.items()):
                    
                    b_xx=(self.buildmax-self.buildmin)[0]
                    b_yy=(self.buildmax-self.buildmin)[1]
                    hh=self.buildmax[2]

                    if sobj.hh>0.0:
                        hh=sobj.hh
                    
                    if len(self.roof_footprint) == 0:
                        self.roof_footprint = self.footprint

                    if sobj.command=='Copy-Abs':
                        
                        BOrigin=np.array([sobj.xx,sobj.yy,0.0])
                        msline=self.roof_footprint
                      
                    elif sobj.command=='Copy-Rel':
                        
                        BOrigin=np.array([sobj.xx*b_xx,sobj.yy*b_yy,0.0])
                        msline=cp.copy(self.roof_footprint)
                    
                       
                    elif sobj.command=='Mirror-Rel':
                        msline=[]
                        for sline in self.roof_footprint: 
                            xxs,yys=sline                        
                            if sobj.rotation_axis==0:
                                nsline=-1.0*np.array(xxs),yys
                            else:
                                nsline=xxs,-1.0*np.array(yys)
                            msline.append(nsline)
                        BOrigin=np.array([self.buildmax[0]+sobj.xx*b_xx,sobj.yy*b_yy,0.0])
                    elif sobj.command=='Mirror-Abs':
                        msline=[]
                        for sline in self.roof_footprint: 
                            xxs,yys=sline                        
                            if sobj.rotation_axis==0:
                                nsline=-1.0*np.array(xxs),yys
                            else:
                                nsline=xxs,-1.0*np.array(yys)
                            msline.append(nsline)
                        BOrigin=np.array([sobj.xx,sobj.yy,0.0])
                    elif sobj.command=='Create-Rel':
                        msline=[]
                        
                        msline.append([[0.0,sobj.L1],[0.0,0.0]])
                        msline.append([[sobj.L1,sobj.L1],[0.0,-1.0*sobj.L2]])
                        msline.append([[sobj.L1,0.0],[-1.0*sobj.L2,-1.0*sobj.L2]])
                        msline.append([[0.0,0.0],[-1.0*sobj.L2,0.0]])
                        
                        BOrigin=np.array([sobj.xx*b_xx,sobj.yy*b_yy,sobj.zz])
                        
                    elif sobj.command=='Create-Abs':
                        msline=[]
                        
                        msline.append([[0.0,sobj.L1],[0.0,0.0]])
                        msline.append([[sobj.L1,sobj.L1],[0.0,-1.0*sobj.L2]])
                        msline.append([[sobj.L1,0.0],[-1.0*sobj.L2,-1.0*sobj.L2]])
                        msline.append([[0.0,0.0],[-1.0*sobj.L2,0.0]])

                        BOrigin=np.array([sobj.xx,sobj.yy,sobj.zz])
                
                    for nn,sline in enumerate(msline):
                        xxs,yys=sline

                        surface=surf.SubSurf(sobjn+"_{0}".format(nn))
                        surface.wall_type='shade'
                        origin=BOrigin+np.array([xxs[0],yys[0],0.0])
                        r1=np.array([xxs[1]-xxs[0],yys[1]-yys[0],0.0])
                        r2=np.array([0.0,0.0,hh])
                        surface.calc_vertexes(origin,r1,r2)
                        self.shading_surfaces.append(surface)
                      
                    
                    ##Used if the building has a roof
                    if 'Create' not in sobj.command:
                        for nn, surf_verts in enumerate(self.roof_vertices):
                            
                            surface=surf.SubSurf(sobjn+"_RoofShade_{0}".format(nn))
                            surface.wall_type='shade'
                            
                            xcent = None
                            
                            if sobj.command=='Mirror-Rel':
                                if sobj.rotation_axis==0:
                                    xcent = BOrigin[0]/2  ##have already added bmax
                            
                            surface.shift_vertices(BOrigin, surf_verts, xcent=xcent)                        
                            
                            self.shading_surfaces.append(surface)
    
    def get_build_origin(self):
        buildmin = [1000,1000,1000]
        for _,zone in list(self.zones.items()):

            buildmin=np.minimum(zone.origin,buildmin)

        return buildmin   
                    
    def find_dimentions(self,run, length_sf =1, height_sf=1):
        
        ##Areas required for CIBSE heating scaling
        self.wall_area=0.0
        self.window_area=0.0
        self.roof_area=0.0
        self.ground_floor_area=0.0
        
        self.volume=0.0
        
        self.floor_area=0.0
        
        self.bedrooms=0
        self.flowC=0.0
        self.footprint=[]
        self.roof_footprint=[]
        self.roof_vertices =[]
        
        for zonen,zone in list(self.zones.items()):
            zone.int_area=0.0
            zone.floor_area=0.0
            if "bedroom" in zonen.lower():
                self.bedrooms+=1
            for surface in zone.surfaces:
                
                ###Get surface areas for CIBSE boiler sizing
                if surface.othersidezone != None and ('Attic' in surface.othersidezone.name or 'Loft' in surface.othersidezone.name):
                    zone.roof_area += surface.area
                #print(surface.name, surface.wall_type, surface.outside_BC, surface.sunexp, surface.windexp)
                    
                if surface.EPG2_type=='floor' and surface.origin[2]==0 and 'shaft' not in surface.name.lower():
                    zone.ground_floor_area+=surface.area
                    self.ground_floor_area+=surface.area
                    
                if surface.wall_type=='external_wall' and surface.outside_BC=='Outdoors' and 'shaft' not in surface.name.lower():
                    self.wall_area+=surface.area
                    zone.wall_area+=surface.area
                    if(surface.EPG2_type in zone.glazing):
                        self.window_area+= zone.glazing[surface.EPG2_type]*surface.area
                        zone.window_area+= zone.glazing[surface.EPG2_type]*surface.area
                
                if surface.wall_type=='roof' and surface.outside_BC=='Outdoors':
                    self.roof_area+=surface.area
                elif "Attic+" in surface.name and surface.wall_type=='internal_wall':
                    self.roof_area-=surface.area   
                
                ###Get the building height excluding roof
                if 'Attic' not in surface.name:
                    self.buildmax=np.maximum(surface.urc,self.buildmax)

                self.buildmin=np.minimum(surface.llc,self.buildmin)
                zone.int_area+=surface.area

                if ('internal_floor' in surface.wall_type or 'ground' in surface.wall_type) and 'Cellar' not in surface.name and 'unzoned' not in surface.name.lower() and 'shaft' not in surface.name.lower():
                    #print(("Surface ",surface.name,surface.area))
                    
                    zone.floor_area+=surface.area
                    if self.type == "Flat":
                        if surface.origin[2]==0:
                            self.floor_area+=surface.area
                    else:   
                        self.floor_area+=surface.area
                elif ('internal_floor' in surface.wall_type or 'ground' in surface.wall_type) and 'Cellar' not in surface.name and ('unzoned' in surface.name.lower() or 'shaft' in surface.name.lower()):
                    zone.floor_area+=surface.area
                
                if 'Attic' in surface.name and surface.outside_BC=='Outdoors':
                    surf_verts = []
                    for vert in surface.vertexs:
                        surf_verts.append(vert)
                    
                    self.roof_vertices.append(surf_verts)
                
                
                ##Obtain the roof footprint for drawing roofs
                if 'Attic' in surface.name and surface.EPG2_type=='floor':
                    for nv in range(0, len(surface.vertexs)):                        
                        if nv+1 == len(surface.vertexs):
                            xxs = ([surface.vertexs[nv][0], surface.vertexs[0][0]])
                            yys = ([surface.vertexs[nv][1], surface.vertexs[0][1]])
                            self.roof_footprint.append([xxs,yys])
                        else:    
                            xxs = ([surface.vertexs[nv][0], surface.vertexs[nv+1][0]]) 
                            yys = ([surface.vertexs[nv][1], surface.vertexs[nv+1][1]])
                            self.roof_footprint.append([xxs,yys])
                   
                    
                
                if (surface.outside_BC=='Outdoors' or surface.outside_BC=='Adiabatic') and surface.EPG2_type!='roof':
                    ees=surface.cut_xyz(axis=2,cut=1.0)
                    if len(ees)>0:
                        self.footprint.append([ees[0],ees[1]])
            
            ##Volume
            if 'shaft' not in zone.name.lower():
                if self.type == "Flat":
                    if zone.origin[2]==0:
                        self.volume+=zone.volume
                else:   
                     self.volume+=zone.volume
        
        ##Scale values if LHD will change
        self.floor_area*=length_sf*length_sf
        self.wall_area*=height_sf
        self.window_area*=height_sf*length_sf
        self.roof_area*=length_sf*length_sf
        self.ground_floor_area*=length_sf*length_sf
        
        if height_sf != 0:
            self.volume*=length_sf*length_sf*height_sf

        if run!=None:
            self.flowC = inp.calc_flowC(run.permeability)
                
    def install_perms(self,pack):

        for zone in sorted(list(self.zones.values()), key=lambda ff:ff.name):
            zonen=zone.name
            
            surfaces=sorted(zone.surfaces,key=lambda ff:ff.name)
            for surface in surfaces[:]:
                if surface.outside_BC=='Outdoors' and surface.EPG2_type!='roof' and zonen!='Cellar':
                    ##Roof does have permeability if it is pitched and if zone is attic
                    ##pu is permeability up and pd is down permeability. The wall is split into 3 segments. Airflow only allowed at top and bottom
                    pus,pds=surface.create_perms(pack)

                    pus.flowC=self.flowC*surface.area*(pus.area/(pus.area+pds.area))
                    pds.flowC=self.flowC*surface.area*(pds.area/(pus.area+pds.area)) ##weighted to the crack surface area (up and down)
                    zone.surfaces.extend([pus,pds])
                    
                    if ('Loft' in zonen) or ('Attic' in zonen) and "Pitch" in  surface.name and pack.L_vent:    
                            surface.ELA=True
                            surface.ELA_height=np.dot(surface.centre,np.array([0,0,1.]))
                            ##Get thelength
                            length = 0
                            if "L1" in surface.name or "L3" in surface.name:
                                length = zone.L1
                            elif "L2" in surface.name or "L4" in surface.name:
                                length = zone.L2
                                                           
                            surface.ELA_area=0.01*length

                else:
                    
                    if surface.othersidesurface!=None:
                        
                        if surface.othersidesurface.flowC<inp.diffthresh:
                            surface.flowC=self.flowC*surface.area
                        
                    else:
                        
                        if surface.outside_BC=='Outdoors' and surface.EPG2_type!='roof' and surface.EPG2_type!='floor' and pack!=None:
                            ##Cellar outside_BC should be ground
                            if 'Cellar' in zonen and pack.C_vent:
                                
                                length = 0
                                if "L1" in surface.name or "L3" in surface.name:
                                    length = zone.L1
                                elif "L2" in surface.name or "L4" in surface.name:
                                    length = zone.L2
                                area_by_length = 0.0015 * length
                                                                
                                surface.ELA=True
                                surface.ELA_area=area_by_length
                                surface.ELA_height=inp.ELA_height

                                
    def install_hangs(self):
        for _,zone in list(self.zones.items()):
            if len(zone.hang)>0:
                surfaces=zone.surfaces
                for surface in surfaces[:]:
                    if surface.EPG2_type in list(zone.hang.keys()) and surface.perm==True and surface.name[-2:]=='pu':
                        hsurf=surface.create_hang(zone.hang[surface.EPG2_type])
                        zone.hang_surfaces.append(hsurf)

                    
    def install_fensters(self,pack, run):
        ###Trickle vents for windows with a U-value of <2.0 (this is an override!!)
        if (run.window_u < 2.0):
            pack.trickle = True
        elif(run.window_u >= 2.0):
            pack.trickle = False
        
        self.ntrickles=0
        for _,zone in list(self.zones.items()):
            if len(zone.glazing)>0:
                wallfs=list(zone.glazing.keys())
                surfaces=zone.surfaces
                for surface in surfaces:
                    if not surface.perm and not surface.trick:
                        for wall in wallfs:
                            if wall==surface.EPG2_type and surface.outside_BC=='Outdoors':
                                
                                window=surf.Window(surface,zone.glazing[wall],pack)
                                surface.window=window
                                zone.windows.append(window)
                                
                                if pack==None or (pack.trickle and not surface.trick):
                                    trsurface=surface.create_tricks(pack)
                                    zone.surfaces.append(trsurface)
                                    self.ntrickles+=1
                                
                                    
    def install_doors(self,pack):
        for _,zone in list(self.zones.items()):   #_ is the zone name but can just do zone.name instead
            if len(zone.doordatas)>0:
                for doordata in zone.doordatas:
                #    print zonen,doordata
                    if doordata['zonen'] in self.zones:
                        #print "zone found"
                        surfaces=zone.surfaces
                        for surface in surfaces:
                            #if surface.othersidezone!=None:
                            #print surface.name,surface.othersidezone.name
                            if surface.othersidezone!=None and surface.othersidezone.name==doordata['zonen'] and surface.EPG2_type==doordata['dside']:
                                #   print zone.name,surface.name,' --> ',surface.othersidezone.name,surface.othersidesurface.name
                                #   for vs1,vs2 in zip(surface.vertexs,surface.othersidesurface.vertexs):
                                #       print vs1,vs2
                                door1=surf.Door(surface,pack)
                                surface.door=door1
                                door2=surf.Door(surface.othersidesurface,pack)
                                surface.othersidesurface.door=door2
                                zone.doors.append(door1)
                                zone.doors.append(door2)
                                door1.otherdoor=door2
                                door2.otherdoor=door1
                                #  print door1.name,door2.name
                                #  for vs1,vs2 in zip(door1.vertexs,door2.vertexs):
                                #      print vs1,vs2
                                break
    
    
def calc_CIBSE_sizing(zone, run):
    
    
    ##need to add floor losses for some building types
    fab_heat_loss = inp.delta_T * (run.wall_u*zone.wall_area + run.window_u*zone.window_area + run.roof_u*zone.roof_area + run.floor_u * zone.ground_floor_area)
    
    #if zone.buildmin[2] <= 0:
    #    fab_heat_loss += run.floor_u * zone.ground_floor_area 
     
    ##Perm needs to be converted to air change rate
    ##It's in SAP 2012: http://www.bre.co.uk/filelibrary/SAP/2012/SAP-2012_9-92.pdf
    ##
    ##Section 2.3 on page 11 says you need to divide by 20 to get the permeability at standard pressure (not exactly sure where this figure comes from).
    ##Then you would have to multiply by area and divide by volume to get the ACR per hour.
    ##There is more info on page 179 of that document.
    air_changes =  ((run.permeability/inp.perm_to_ach)*((zone.wall_area+zone.roof_area)/zone.volume))/(60*60)  ##per second     
    vent_heat_loss = air_changes*zone.volume* (inp.air_density  *  inp.air_heat_capacity) * inp.delta_T  ##check this!
           
    return (fab_heat_loss+vent_heat_loss)
        
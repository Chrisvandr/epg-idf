"""File containing things needed for schedules e.g people, appliances, equipment etc..."""

import copy
import numpy as np

class Appliance():
    def __init__(self,name):
        self.name=name
       
        self.power=0.0
        self.useful=0.0
        self.latent=0.0
        self.purpose=None
        self.shedule=None
        self.calculation_method=None
 
        
class Person():
    def __init__(self,name):
        self.name=name
        self.presence_schedule=None
        self.metabolic_schedule=None
 
class ZoneEquip():
    def __init__(self,name):
        self.name=name
        self.eptype=None
        
class ElecHeater(ZoneEquip):
    def __init__(self,name):
        ZoneEquip.__init__(self,name)
        
      
        self.power=0.0
        self.heating=None
        self.comfort=None
        self.eptype='ZoneHVAC:Baseboard:Convective:Electric'
        
        
    def all_there(self):
        flag=False
        if self.heating!=None and self.comfort!=None:
            flag=True
        return flag

class Equipment():
    def __init__(self,dline):
        self.dline=dline
        self.name=dline[0]
        
        self.purpose=dline[1]
        self.useful=(dline[2])
        self.latent=(dline[3])
        self.power=dline[4]
        self.calculation_method=dline[19]
        
class HGroup():
    def __init__(self,dline):
        self.dline=dline
        self.name=dline[0]
        self.zones_scheds=[]
        
class ZoneSched():
    def __init__(self,dline, zone_name):
        
        self.dline = dline
        self.info=''
        if dline[4]!='':
            self.info=dline[4]
    
        self.name=dline[0]+'_'+zone_name+'_'+dline[2]+'_'+self.info
        self.gname=dline[0]+'_'+zone_name
        self.zone_name=zone_name
        self.feature=dline[2]
        self.type=dline[3]
        
        self.scaler=dline[5]
        self.base_value=0.0
        if dline[7]=='X':
            self.base_value='X'
        elif dline[7]!='':
            self.base_value=float(dline[7])
        hours=copy.copy(dline[8:])

        if '' in hours:
            linx=hours.index('')
            self.hours=np.array(copy.copy(hours[0:linx]))
        else:
            self.hours=np.array(copy.copy(hours))
    def re_init(self, room_number):
        
        self.zone_name=self.zone_name.split('1')[0]+room_number+self.zone_name.split('1')[1]
        self.name=self.dline[0]+'_'+self.zone_name+'_'+self.dline[2]+'_'+self.info
        self.gname=self.dline[0]+'_'+self.zone_name
        
      
     
class Occupation():
    def __init__(self,dline):
        self.dline=dline
        self.name=dline[0] 
        
        self.hgroups=[]
       
        
  
                   

"""File containing things needed for building surfaces"""

import copy
import numpy as np
import math

import inputvars as inp

class SubSurf():
    def __init__(self,name):
        self.name=name
        self.wall_type=''
        self.outside_BC=''
        self.perm=False
        self.hang=False
        self.trick=False
        self.window=None
        self.door=None
        self.upper=False
        self.exhaust=False
               
    def calc_vertexes(self,origin,r1,r2, shape='Rectangle'):
        ##need to try and make this cleaner at some stage
        
        if shape == 'Join':
            if ('L2' in self.name):
                self.origin=copy.copy(origin)  
                self.bvec=np.cross(r1,r2)
                self.r1=copy.copy(r1)
                self.r2=copy.copy(r2)
                self.vertexs=[]
                
                vertex=copy.copy(origin)
                self.llc=copy.copy(vertex)
                self.urc=copy.copy(vertex)
                self.vertexs.append(vertex)
                
                vertex=np.around(vertex+r2-np.array([0,r1[1],0]),4)
                self.urc=np.maximum(self.urc,vertex)
                self.llc=np.minimum(self.llc,vertex)
                self.vertexs.append(vertex)
                
                vertex=np.around(vertex+r1,4)
                self.urc=np.maximum(self.urc,vertex)
                self.llc=np.minimum(self.llc,vertex)
                self.vertexs.append(vertex)
                
                vertex=np.around(vertex-r2,4)
                self.urc=np.maximum(self.urc,vertex)
                self.llc=np.minimum(self.llc,vertex)
                self.vertexs.append(vertex)
                
                
            elif ('L4' in self.name):
                self.origin=copy.copy(origin)  
                self.bvec=np.cross(r1,r2)
                self.r1=copy.copy(r1)
                self.r2=copy.copy(r2)
                self.vertexs=[]
                
                vertex=copy.copy(self.origin)
                self.llc=copy.copy(vertex)
                self.urc=copy.copy(vertex)
                self.vertexs.append(vertex)
                
                vertex=np.around(vertex+r2-np.array([0,r1[1],0]),4)
                self.urc=np.maximum(self.urc,vertex)
                self.llc=np.minimum(self.llc,vertex)
                self.vertexs.append(vertex)
                
                vertex=np.around(vertex+np.array([r1[0],0,r1[2]]),4)
                self.urc=np.maximum(self.urc,vertex)
                self.llc=np.minimum(self.llc,vertex)
                self.vertexs.append(vertex)
                
                vertex=np.around(vertex-r2,4)
                self.urc=np.maximum(self.urc,vertex)
                self.llc=np.minimum(self.llc,vertex)
                self.vertexs.append(vertex)
                
            self.centre=self.origin+0.5*(self.r1+self.r2)
            self.area=(abs(r2[1])+abs(0.5*r1[1]))*math.sqrt(self.r1[0]**2+self.r1[2]**2)
            
        elif shape=='Trapezium':
            
            if(r2[0] != 0):
                r3=(r2-np.array([2*r1[0], 0, 0]))
            elif(r2[1] != 0):
                r3=(r2-np.array([0, 2*r1[1], 0]))
            
            self.origin=copy.copy(origin)
            self.bvec=np.cross(r1,r2)
            self.r1=copy.copy(r1)
            self.r2=copy.copy(r2)
            self.r3=copy.copy(r3)
            self.vertexs=[]
    
            vertex=copy.copy(origin)
            self.llc=copy.copy(vertex)
            self.urc=copy.copy(vertex)
            self.vertexs.append(vertex)
            
            vertex=np.around(vertex+r3,4)
            self.urc=np.maximum(self.urc,vertex)
            self.llc=np.minimum(self.llc,vertex)
            self.vertexs.append(vertex)

            vertex=np.around(vertex+r1,4)
            self.urc=np.maximum(self.urc,vertex)
            self.llc=np.minimum(self.llc,vertex)
            self.vertexs.append(vertex)
            
            vertex=np.around(vertex-r2,4)
            self.urc=np.maximum(self.urc,vertex)
            self.llc=np.minimum(self.llc,vertex)
            self.vertexs.append(vertex)

            h = math.sqrt(vmag(r1)**2 - (vmag(r3-r2)/2)**2)
            z_center = -(h/3)*(vmag(r2) + 2*vmag(r3))/(vmag(r2) + vmag(r3))
            x_center = 0          
            y_center = 0
            if(r2[0] != 0):
                self.area=h*(vmag(r3)+abs(r1[0]))
                y_center = (r1[1]/3)*(vmag(r2) + 2*vmag(r3))/(vmag(r2) + vmag(r3))
            if(r2[1] != 0):
                self.area=h*(vmag(r3)+abs(r1[1]))
                x_center = (r1[0]/3)*(vmag(r2) + 2*vmag(r3))/(vmag(r2) + vmag(r3))
                        
            self.centre=(self.origin+0.5*r3+np.array([x_center,y_center,z_center]))

        
        elif shape=='Triangle':
            self.origin=copy.copy(origin)
            self.bvec=np.cross(r1,r2)
            self.r1=copy.copy(r1)
            self.r2=copy.copy(r2)
            self.vertexs=[]
    
            vertex=copy.copy(origin)
            self.llc=copy.copy(vertex)
            self.urc=copy.copy(vertex)
            self.vertexs.append(vertex)
            
            vertex=np.around(vertex+r1,4)
            self.urc=np.maximum(self.urc,vertex)
            self.llc=np.minimum(self.llc,vertex)
            self.vertexs.append(vertex)

            vertex=np.around(vertex+r2,4)
            self.urc=np.maximum(self.urc,vertex)
            self.llc=np.minimum(self.llc,vertex)
            self.vertexs.append(vertex)

            self.centre=self.origin+(self.r1+self.r2)/2.0
            self.area=0.5*vmag(np.cross(self.r1,self.r2))
        
        elif shape=='Reverse':
            self.origin=copy.copy(origin)
            self.bvec=np.cross(r1,r2)
            self.r1=copy.copy(r1)
            self.r2=copy.copy(r2)
            self.vertexs=[]
            temp_verts = []
            
            vertex=copy.copy(origin)
            self.llc=copy.copy(vertex)
            self.urc=copy.copy(vertex)
            temp_verts.append(vertex)
            
            vertex=np.around(vertex+r1,4)
            self.urc=np.maximum(self.urc,vertex)
            self.llc=np.minimum(self.llc,vertex)
            temp_verts.append(vertex)

            vertex=np.around(vertex+r2,4)
            self.urc=np.maximum(self.urc,vertex)
            self.llc=np.minimum(self.llc,vertex)
            temp_verts.append(vertex)
            
            self.vertexs.append(temp_verts[2])
            self.vertexs.append(temp_verts[1])
            self.vertexs.append(temp_verts[0])
            
            self.centre=self.origin+(self.r1+self.r2)/2.0
            self.area=0.5*vmag(np.cross(self.r1,self.r2))
            
        else:   # "PitchCeiling" in self.name:   ###need this (diff order of vertexes) otherwise roof goes the wrong way up
            self.origin=copy.copy(origin)
            self.bvec=np.cross(r1,r2)
            self.r1=copy.copy(r1)
            self.r2=copy.copy(r2)
            self.vertexs=[]
    
            vertex=copy.copy(origin)
            self.llc=copy.copy(vertex)
            self.urc=copy.copy(vertex)
            self.vertexs.append(vertex)
            
            vertex=np.around(vertex+r2,4)   
            self.urc=np.maximum(self.urc,vertex)
            self.llc=np.minimum(self.llc,vertex)
            self.vertexs.append(vertex)

            vertex=np.around(vertex+r1,4)
            self.urc=np.maximum(self.urc,vertex)
            self.llc=np.minimum(self.llc,vertex)
            self.vertexs.append(vertex)

            vertex=np.around(vertex-r2,4)
            self.urc=np.maximum(self.urc,vertex)
            self.llc=np.minimum(self.llc,vertex)
            self.vertexs.append(vertex)
            self.centre=self.origin+(self.r1+self.r2)/2.0
            self.area=vmag(np.cross(self.r1,self.r2))
    
    def recalc_vertices(self, borigin, vertices, xy_shift=1, z_shift=1):
        self.vertexs = []
        
        #print("before: ", self.name, self.r1, self.r2, self.origin, self.centre, self.area, self.urc, self.llc)
        area_fact = vmag(np.cross(np.array([self.r1[0]*xy_shift, self.r1[1]*xy_shift, self.r1[2]*z_shift]),np.array([self.r2[0]*xy_shift, self.r2[1]*xy_shift, self.r2[2]*z_shift])))/vmag(np.cross(self.r1,self.r2))
        
        self.r1 = np.array([self.r1[0]*xy_shift, self.r1[1]*xy_shift, self.r1[2]*z_shift])
        self.r2 = np.array([self.r2[0]*xy_shift, self.r2[1]*xy_shift, self.r2[2]*z_shift])
        
        if self.origin[0] != borigin[0]:
            new_origin_x = borigin[0] + self.origin[0]*xy_shift
        else:
            new_origin_x = borigin[0] + self.origin[0]
        if self.origin[1] != borigin[1]:
            new_origin_y = borigin[1] + self.origin[1]*xy_shift
        else:
            new_origin_y = borigin[1] + self.origin[1]
        if self.origin[2] != 0.0:
            new_origin_z = 0.0 + self.origin[2]*z_shift
        else:
            new_origin_z = 0.0 + self.origin[2]
        
        self.origin = np.array([new_origin_x, new_origin_y, new_origin_z])
        self.centre = np.array([self.centre[0]*xy_shift,self.centre[1]*xy_shift,self.centre[2]*z_shift])
        self.area   =  self.area*area_fact   
        
        for i, vert in enumerate(vertices):
            
            if vert[0] != borigin[0]:
                new_x = borigin[0] + vert[0]*xy_shift
            else:
                new_x = borigin[0] + vert[0]
            if vert[1] != borigin[1]:
                new_y = borigin[1] + vert[1]*xy_shift
            else:
                new_y = borigin[1] + vert[1]
            if vert[2] != 0.0:
                new_z = 0.0 + vert[2]*z_shift
            else:
                new_z = 0.0 + vert[2]
            
            new_vert = np.array([new_x, new_y, new_z])
            
            if i == 0:
                self.llc=copy.copy(new_vert)
                self.urc=copy.copy(new_vert)
            
            self.llc=np.minimum(self.llc,new_vert)
            self.urc=np.maximum(self.urc,new_vert)   
            self.vertexs.append(new_vert)       
            
        #print("after: ", self.name, self.r1, self.r2, self.origin, self.centre, self.area, self.urc, self.llc)
    
    def shift_vertices(self, borigin, vertices, xcent=None):
        
        self.vertexs = []
        for vert in vertices:
            if xcent != None:
                xcorr = borigin[0] + vert[0] - xcent
                new_vert = borigin + np.array([(borigin[0] + vert[0])-2*xcorr, vert[1], vert[2]])
            else:
                new_vert = borigin + vert
            
            self.vertexs.append(new_vert)
        
    def cut_xyz(self,cut=1.0,axis=2): 
       
        ees=[]
        vvs=self.vertexs
        
        axis1=vvs[0][axis]
        axis2=vvs[2][axis]
        
        if (cut>=axis1 and cut<=axis2) or (cut<=axis1 and cut>=axis2):
            for aa in [0,1,2]:
                if aa!=axis:
                    ees.append([vvs[0][aa],vvs[2][aa]])
        return ees

    def draw_xyz(self,cut=1.0,axis=2): 
       
        ees=[]
        vvs=self.vertexs
        
        #print(self.name, vvs)
        
        axis1=vvs[0][axis]
        axis2=vvs[2][axis]
        
        #print("axis1: ", axis1)
        #print("axis2: ", axis2)
        
        if (cut>=axis1 and cut<=axis2) or (cut<=axis1 and cut>=axis2):
            
            for v in range(0,len(vvs)):
                if axis==2:
                    for aa in [0,1]:
                        if v+1 == len(vvs):
                            ees.append([vvs[v][aa],vvs[0][aa]])
                        else:
                            ees.append([vvs[v][aa],vvs[v+1][aa]])
                
                elif axis==1:
                    for aa in [0,2]:
                        if v+1 == len(vvs):
                            ees.append([vvs[v][aa],vvs[0][aa]])
                        else:
                            ees.append([vvs[v][aa],vvs[v+1][aa]])
                
                else:
                    for aa in [1,2]:
                        if v+1 == len(vvs):
                            ees.append([vvs[v][aa],vvs[0][aa]])
                        else:
                            ees.append([vvs[v][aa],vvs[v+1][aa]])
                            
        return ees


class Surface(SubSurf):
    def __init__(self,zone=None,name=None,EPG2_type=None,wall_type=None,surface_type=None,fab=None,outside_BC='Outdoors',outside_BC_obj='',
                 sunexp='NoSun',windexp='NoWind',origin=None,r1=None,r2=None,shape='Rectangle'):
        SubSurf.__init__(self,name)
        
        self.origin = origin
        self.r1 = r1
        self.r2 = r2
        
        self.EPG2_type=EPG2_type
      
        self.surface_type=surface_type
        self.wall_type=wall_type
        self.fab=fab
        self.zone=zone
        self.base_surface=None
        self.ELA=False
        self.ELA_area=0.0
        self.ELA_height=0.0
        
        self.outside_BC=outside_BC
        self.outside_BC_obj=outside_BC_obj
        self.othersidesurface=None
        self.othersidezone=None
     
        self.sunexp=sunexp
        self.windexp=windexp
        self.flowC=0.0
        self.area=0.0
        
        self.shape=shape
        self.calc_vertexes(origin,r1,r2,shape=self.shape)
        
    #Called from inhere if a certain threshold is met
    def spawn_surface(self,qname,Nll,Nur):   
        
        r1hat=np.fabs(self.r1/vmag(self.r1))
        r2hat=np.fabs(self.r2/vmag(self.r2))
    
        rr1=r1hat*np.dot(r1hat,Nur-Nll)
        rr2=r2hat*np.dot(r2hat,Nur-Nll)
        aa=copy.copy(Nll)
        bb=copy.copy(Nll)+rr1
        cc=copy.copy(Nur)
        dd=copy.copy(Nll)+rr2
        
        NOrigin=copy.copy(aa)
        Oaamin=vmag(aa-self.origin)
  
        for vv in [bb,cc,dd]:
            Oaa=vmag(vv-self.origin)
 
            if Oaa<Oaamin:
                Oaamin=Oaa
                NOrigin=copy.copy(vv)
      
        R1=rr1*self.r1/vmag(self.r1)  
        R2=rr2*self.r2/vmag(self.r2)
        
        nname=self.name+qname
        for surface in self.zone.surfaces:
            if surface.name==nname:
                
                nname+='d'
        
        surface=Surface(self.zone,
            name=nname,
            EPG2_type=copy.copy(self.EPG2_type),
            surface_type=copy.copy(self.surface_type),
            fab=self.fab,
            wall_type=copy.copy(self.wall_type),
            outside_BC=copy.copy(self.outside_BC),
            sunexp=copy.copy(self.sunexp),
            windexp=copy.copy(self.windexp),
            origin=copy.copy(NOrigin),
            r1=copy.copy(R1),
            r2=copy.copy(R2)                       
            )
        return surface

    def inhere(self,surface2,pack=None):
        
        r1hat=np.fabs(self.r1/vmag(self.r1))
        r2hat=np.fabs(self.r2/vmag(self.r2))
        
        pldir=np.fabs(np.cross(r1hat,r2hat))
       
        Ull=np.maximum(self.llc,surface2.llc)
        Uur=np.minimum(self.urc,surface2.urc)
       
        DU=Uur-Ull+pldir
        
        newsurfaces=[]
        ## Had to make sure weird surface aren't spawned from the Attic join
        if np.all(DU>inp.tolldiff):
            
            leftll=Ull-self.llc
            leftur=self.urc-Uur
            if (vmag(leftll)>inp.tolldiff or vmag(leftur)>inp.tolldiff):
                Nll=copy.copy(self.llc)
                Nur=copy.copy(Ull)
                NR=Nur-Nll+pldir
                #Narea=np.prod(NR)
                if np.all(NR>inp.tolldiff):
                    
                    surface=self.spawn_surface('_q1',Nll,Nur)
                    newsurfaces.append(surface)
                    self.zone.surfaces.append(surface)
                
                Nll=copy.copy(Uur)
                Nur=copy.copy(self.urc)
                NR=Nur-Nll+pldir
                #Narea=np.prod(NR)
                if np.all(NR>inp.tolldiff):
                    surface=self.spawn_surface('_q2',Nll,Nur)
                    newsurfaces.append(surface)
                    self.zone.surfaces.append(surface)
                
                for hat in [r1hat,r2hat]:
                    
                    Nll=self.llc+hat*np.dot(hat,(Ull-self.llc))
                    Nur=Ull+hat*np.dot(hat,(Uur-Ull))
                    NR=Nur-Nll+pldir
                    
                    if np.all(NR>inp.tolldiff):
                       
                        surface=self.spawn_surface("_qa{0}{1}{2}".format(int(hat[0]),int(hat[1]),int(hat[2])),Nll,Nur)
                        newsurfaces.append(surface)
                        self.zone.surfaces.append(surface)
                    Nll=self.llc+hat*np.dot(hat,(Ull-self.llc))+hat*np.dot(hat,(Uur-Ull))
                    Nur=Ull+hat*np.dot(hat,(Uur-Ull))+hat*np.dot(hat,(self.urc-Uur))
                    NR=Nur-Nll+pldir
                    
                    if np.all(NR>inp.tolldiff):
                        surface=self.spawn_surface("_qb{0}{1}{2}".format(int(hat[0]),int(hat[1]),int(hat[2])),Nll,Nur)                    
                        newsurfaces.append(surface)
                        self.zone.surfaces.append(surface)
                        
                    Nll=Ull+hat*np.dot(hat,(Uur-Ull))
                    Nur=Uur+hat*np.dot(hat,(self.urc-Uur))
                    NR=Nur-Nll+pldir
                    
                    if np.all(NR>inp.tolldiff):                        
                        surface=self.spawn_surface("_qc{0}{1}{2}".format(int(hat[0]),int(hat[1]),int(hat[2])),Nll,Nur)
                        
                        newsurfaces.append(surface)
                        self.zone.surfaces.append(surface)

                        
                self.set_matched(pack,surface2)
                
                rr1=r1hat*np.dot(r1hat,Uur-Ull)
                rr2=r2hat*np.dot(r2hat,Uur-Ull)
                aa=copy.copy(Ull)
                bb=copy.copy(Ull)+rr1
                cc=copy.copy(Uur)
                dd=copy.copy(Ull)+rr2
                              
                NOrigin=aa
                Oaamin=vmag(aa-self.origin)
                for vv in [bb,cc,dd]:
                    Oaa=vmag(vv-self.origin)
              
                    if Oaa<Oaamin:
                        Oaamin=Oaa
                        NOrigin=vv
              
                R1=rr1*self.r1/vmag(self.r1)  
                R2=rr2*self.r2/vmag(self.r2)                    

                self.calc_vertexes(NOrigin,R1,R2)
                
                    
            else:
                self.set_matched(pack,surface2)
                    
        return newsurfaces
        
        
    def set_matched(self,pack,surface2):
        int_wall=None
        int_ceiling=None
        int_floor=None
        insu_floor=None
        insu_ceiling=None

        if pack!=None:
            int_ceiling=pack.props['internal_ceiling']
            int_floor=pack.props['internal_floor']
            int_wall=pack.props['internal_wall']
            insu_floor=pack.props['loft_floor']
            insu_ceiling=pack.props['loft_ceiling']

        self.othersidezone=surface2.zone
        if surface2.othersidezone==self.zone:       
            self.outside_BC_obj=surface2.name
            surface2.outside_BC_obj=self.name
            self.othersidesurface=surface2
            surface2.othersidesurface=self
        self.windexp='NoWind'
        self.sunexp='NoSun'
        if self.wall_type=='external_wall':
            self.wall_type='internal_wall'
            self.outside_BC='Surface'
            self.fab=int_wall
        elif self.wall_type=='ground':
            if "Loft" in self.zone.name or "Attic" in self.name:
                self.wall_type='insulated_floor'
                self.fab=insu_floor
            else:
                self.wall_type='internal_floor'
                self.fab=int_floor
            self.outside_BC='Surface'

        elif self.wall_type=='roof':
            self.outside_BC='Surface'
                     
            if ("Attic" in self.othersidezone.name and "Attic" in self.name):
                self.wall_type='roof'
                self.outside_BC='Outdoors'
                self.windexp='WindExposed'
                self.sunexp='SunExposed'
            elif ("Loft" in self.othersidezone.name or "Attic" in self.othersidezone.name):
                self.wall_type='insulated_ceiling'
                self.fab=insu_ceiling   
            else:
                self.wall_type='internal_ceiling'
                self.fab=int_ceiling
            
    def create_perms(self,pack):

        pheight=np.array([0.0,0.0,inp.perm_height]) ##height of the permeability crack
        vdown=np.array([0.,0.,1])      
                         
        if np.abs(np.dot(vdown,self.r1))>inp.diffthresh:    ##to make sure the surface has a z component (strips only good if variable height)
            
            ##Stuff general to all surfaces
            puorigin=copy.copy(self.origin)
            pur1=-pheight[2]*self.r1/self.r1[2]
            
            ##main mid section                
            nr1=(self.r1-2.0*pur1)
            
            ##set up origin and vectors for down perm
            pdr1=copy.copy(pur1)

            ##defaults for normal wall changes for others below
            norigin = self.origin+pur1
            nr2=copy.copy(self.r2)
            pur2=copy.copy(self.r2)
            pdorigin=self.origin+self.r1-pur1
            pdr2=copy.copy(self.r2)
            
            ###Do by original shape
            if self.shape == "Join":
                pur2 =  self.r2 - np.array([0, self.r1[1] - pur1[1], 0]) 
                
                if 'L2' in self.name:
                    norigin = self.origin+ np.array([pur1[0], 0, pur1[2]])
                    pdorigin=self.origin + np.array([pur1[0]+nr1[0], 0, pur1[2]+nr1[2]])
                elif 'L4' in self.name:
                    norigin = self.origin+ np.array([pur1[0], -pur1[1], pur1[2]])
                    pdorigin=self.origin + np.array([pur1[0]+nr1[0], -(pur1[1]+nr1[1]), pur1[2]+nr1[2]])
                nr2 = self.r2 - np.array([0, pur1[1], 0])
            elif self.shape == "Triangle":
                pur2 =  -pheight[2]*self.r2/self.r1[2]
                self.shape='Trapezium'
                norigin = self.origin+ pur1 + pur2
                nr2 = -(self.r2 - pur2)
                pdorigin=self.origin + pur1 + nr1 - nr2
                pdr2=-copy.copy(self.r2)
            elif self.shape == "Trapezium":
                if self.r2[0]!=0 and nr1[0] != 0:   
                    pur2= self.r2 - np.array([2*pur1[0]+2*nr1[0], 0., 0.])
                    norigin = self.origin+np.array([-pur1[0], pur1[1], pur1[2]])
                    nr2 = self.r2 - np.array([2*pur1[0], 0., 0.])
                    pdorigin=norigin+np.array([-nr1[0],nr1[1],nr1[2]])
                elif self.r2[1]!=0 and nr1[1]!=0:
                    pur2 = self.r2 - np.array([0., 2*pur1[1]+2*nr1[1], 0.])
                    norigin = self.origin+np.array([pur1[0], -pur1[1], pur1[2]])
                    nr2 = self.r2 - np.array([0., 2*pur1[1], 0.])
                    pdorigin=norigin+np.array([nr1[0],-nr1[1],nr1[2]])
               

        else:
            puorigin=copy.copy(self.origin)
            pur2=pheight*self.r2/vmag(self.r2)
            pur1=copy.copy(self.r1)
            
            norigin=self.origin-pheight
            nr2=(self.r2+2.0*pheight)
            nr1=copy.copy(self.r1)
        
            pdorigin=self.origin+self.r2+pheight
            pdr2=pheight*self.r2/vmag(self.r2)
            pdr1=copy.copy(self.r1)
            
        ##Calcs vertices for the central main section
        self.calc_vertexes(norigin,nr1,nr2,self.shape)
        
        ##Make sure get the right shape for the pu
        pushape=self.shape
        if (self.shape == 'Reverse'):
            pushape='Reverse'
        elif "Side" in self.name:
            pushape='Triangle'
        
        
#        print("surf", self.name)
#        print("PU", self.EPG2_type)
#        print("origin: ", puorigin)
#        print("r1: ", pur1)    
#        print("r2: ", pur2)
#        print("shape: ", self.shape)
#        
#        print("N", self.EPG2_type)
#        print("origin: ", norigin)
#        print("r1: ", nr1)    
#        print("r2: ", nr2)
#        print("shape: ", self.shape)
#        
#        print("PD", self.EPG2_type)
#        print("origin: ", pdorigin)
#        print("r1: ", pdr1)    
#        print("r2: ", pdr2)
#        print("shape: ", self.shape)
        
        pusurface=Surface(self.zone,
            name=self.name+'_pu',
            EPG2_type=copy.copy(self.EPG2_type),
            surface_type=copy.copy(self.surface_type),
            fab=self.fab,
            wall_type=copy.copy(self.wall_type),
            outside_BC=copy.copy(self.outside_BC),
            sunexp=copy.copy(self.sunexp),
            windexp=copy.copy(self.windexp),
            origin=copy.copy(puorigin),
            r1=copy.copy(pur1),
            r2=copy.copy(pur2),
            shape=pushape                       
            )
    

        pdsurface=Surface(self.zone,
            name=self.name+'_pd',
            EPG2_type=copy.copy(self.EPG2_type),
            surface_type=copy.copy(self.surface_type),
            fab=self.fab,
            wall_type=copy.copy(self.wall_type),
            outside_BC=copy.copy(self.outside_BC),
            sunexp=copy.copy(self.sunexp),
            windexp=copy.copy(self.windexp),
            origin=copy.copy(pdorigin),
            r1=copy.copy(pdr1),
            r2=copy.copy(pdr2),
            shape=self.shape                       
            )
            
        
        pusurface.perm=True
        pusurface.upper=True
        
        pdsurface.perm=True
        
        return pusurface,pdsurface
 


    def create_tricks(self,pack):

        pheight=np.array([0.0,0.0,0.1])
        vdown=np.array([0.,0.,1.])
        
        
        if np.abs(np.dot(vdown,self.r1))>inp.diffthresh: 
            puorigin=copy.copy(self.origin)
            pur1=pheight*self.r1/vmag(self.r1)
            pur2=copy.copy(self.r2)
            
            norigin=self.origin-pheight
            nr1=(self.r1+pheight)
            nr2=copy.copy(self.r2)
          

        else:
            puorigin=copy.copy(self.origin)
            pur2=pheight*self.r2/vmag(self.r2)
            pur1=copy.copy(self.r1)
            
            norigin=self.origin-pheight
            nr2=(self.r2+pheight)
            nr1=copy.copy(self.r1)

        
        self.calc_vertexes(norigin,nr1,nr2)
        
        trsurface=Surface(self.zone,
            name=self.name+'_tr',
            EPG2_type=copy.copy(self.EPG2_type),
            surface_type=copy.copy(self.surface_type),
            fab=self.fab,
            wall_type=copy.copy(self.wall_type),
            outside_BC=copy.copy(self.outside_BC),
            sunexp=copy.copy(self.sunexp),
            windexp=copy.copy(self.windexp),
            origin=copy.copy(puorigin),
            r1=copy.copy(pur1),
            r2=copy.copy(pur2)                       
            )

        
        trsurface.trick=True
        
        trsurface.ELA_height=np.dot(trsurface.centre,np.array([0,0,1.]))
        return trsurface
 

    def create_hang(self,hlen):
    
    
        vdown=np.array([0.,0.,1.])
    
        horigin=copy.copy(self.origin)
        idir=np.cross(self.r1,self.r2)
        idir=-1.0*idir/vmag(idir)
        hr1=hlen*idir
        
        if np.dot(vdown,self.r1)>inp.diffthresh:
            hr2=copy.copy(self.r1)
        else:
            hr2=copy.copy(self.r2)
        
       
        
        hsurf=Surface(
            zone=self.zone,
            name=self.name+'_hang',
            EPG2_type=copy.copy(self.EPG2_type),
            surface_type=copy.copy(self.surface_type),
            fab=self.fab,
            wall_type=copy.copy(self.wall_type),
            outside_BC=copy.copy(self.outside_BC),
            sunexp=copy.copy(self.sunexp),
            windexp=copy.copy(self.windexp),
            origin=copy.copy(horigin),
            r1=copy.copy(hr1),
            r2=copy.copy(hr2)                       
            )
        hsurf.hang=True
        hsurf.base_surface=self
        
        return hsurf
 

class Window(SubSurf):
    def __init__(self,surface,glazingfraction,pack):
        self.name=surface.name+'_window'
        self.surface=surface
        self.control=None
        self.temperature=None
        self.shading=None
        self.glazingfraction=glazingfraction
        R1=np.sqrt(self.glazingfraction)*surface.r1
        R2=np.sqrt(self.glazingfraction)*surface.r2
        vr1=(surface.r1-R1)/2.0
        vr2=(surface.r2-R2)/2.0
        Origin=surface.origin+vr1+vr2
        
        self.calc_vertexes(Origin,R1,R2)
        
        self.fab=None
        self.shader_fab=None
        if pack!=None:
            self.fab=pack.props['window']
            self.shader_fab=pack.props['window_shading']
            
  
class Door(SubSurf):
    def __init__(self,surface,pack):
        self.name=surface.zone.name+'_door_'+surface.othersidezone.name
        self.outside_BC_obj=surface.othersidezone.name+'_door_'+surface.zone.name
        self.surface=surface
        self.otherdoor=None
        self.control=None
        self.dwidth=0.8
        self.dheight=2.0
        
        
        vdown=np.array([0.,0.,1.])
        if np.abs(np.dot(surface.r1,vdown))>inp.diffthresh:
            R1=self.dheight*surface.r1/vmag(surface.r1)
            dwidth=self.dwidth
            if dwidth>vmag(surface.r2):
                dwidth=vmag(surface.r2)
            R2=dwidth*surface.r2/vmag(surface.r2)
            Origin=surface.origin+surface.r1-R1+(surface.r2-R2)/2.0
        else:
            R1=self.dheight*surface.r2/vmag(surface.r2)
            dwidth=self.dwidth
            if dwidth>vmag(surface.r1):
                dwidth=vmag(surface.r1)
            R2=dwidth*surface.r1/vmag(surface.r1)
            Origin=surface.origin+surface.r2-R2+(surface.r1-R1)/2.0

        self.calc_vertexes(Origin,R1,R2)
        
        self.fab=None
        if pack!=None:
            self.fab=pack.props['door']


def vmag(vect):
    mag=np.sqrt(np.dot(vect,vect))
    return mag
    
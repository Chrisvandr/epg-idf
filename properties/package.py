"""File containing package, which uses fabric and material classes"""
import numpy as np

"""Class containing the package properties"""
class Package():
    def __init__(self,name):
       
        self.name=name
        self.re_init()
        self.props={}
        self.trickle=False
        self.C_vent=False
        self.L_vent=False
        
    def re_init(self):
        
        self.cmats={}
        self.glaze_mats={}
        self.gas_mats={}
        self.shade_mats={}
        self.blind_mats={}
        
    def set_prop(self,propname=None,prop=None):
        self.props[propname]=prop
        
    def create(self, run=None):
        self.re_init()
        for propname in list(self.props.keys()):
            fab=self.props[propname]
            
            for rmat in fab.rmats:
                if (propname == "external_wall" and ("AirGapExt" in rmat.name or  "CelluloseInsExt" in rmat.name)  and run.wall_u != 0):
                    #Thickness to be determined in here
                    rmat.thickness=fab.get_thickness(run.wall_u)
                elif (propname == "window" and ("GasWin" in rmat.name)  and run.window_u != 0):
                    rmat.thickness=fab.get_thickness(run.window_u)                   
                elif ((propname == "loft_ceiling" or propname == "loft_floor") and ("AirGapRoof" in rmat.name) and run.roof_u != 0):
                    rmat.thickness=fab.get_thickness(run.roof_u)
                elif (propname == "ground" and ("GlassFiberFloor" in rmat.name) and run.floor_u != 0):
                    rmat.thickness=fab.get_thickness(run.floor_u)
                    
                if rmat.mat.ep_type=='Material':
                    if rmat.name not in list(self.cmats.keys()):
                        self.cmats[rmat.name]=rmat
                     
                if rmat.mat.ep_type=='WindowMaterial:Glazing':
                    if rmat.name not in list(self.glaze_mats.keys()):
                        ##Calculate solar transmittance
                        if run.window_u != 0:
                            rmat.g_sol_trans=0.0522*run.window_u+0.59  ##From SAP

                        self.glaze_mats[rmat.name]=rmat
        
                if rmat.mat.ep_type=='WindowMaterial:Gas':
                    if rmat.name not in list(self.gas_mats.keys()):
                        self.gas_mats[rmat.name]=rmat
     
                if rmat.mat.ep_type=='WindowMaterial:Shade':
                    if rmat.name not in list(self.shade_mats.keys()):
                        self.shade_mats[rmat.name]=rmat

                if rmat.mat.ep_type=='WindowMaterial:Blind':
                    if rmat.name not in list(self.blind_mats.keys()):
                        self.blind_mats[rmat.name]=rmat
                
            #print(propname, fab.calc_u())
        
        ##Set the u-values such that CIBSE heating still works 
        for propname in list(self.props.keys()):
            fab=self.props[propname]
            if (propname == "external_wall"  and run.wall_u == 0):
                run.wall_u = fab.calc_u()
            elif (propname == "window" and run.window_u == 0):
                run.window_u = fab.calc_u()
            elif (propname == "loft_floor" and run.roof_u == 0):
                run.roof_u = fab.calc_u()
            elif (propname == "ground" and run.floor_u == 0):
                run.floor_u = fab.calc_u()            
                        
"""Class containing the material properties"""
class Material():
    def __init__(self,dline):  #initialised from line of csv file
        
        self.dline=dline
        self.name=dline[0]
        self.ep_type=dline[1]
        self.roughness=dline[2]
        self.conductivity=dline[3]
        self.density=dline[4]
        self.specific_heat=dline[5]
        self.thermal_abs=dline[6]
        self.solar_abs=dline[7]
        self.visible_abs=dline[8]
        self.thickness_abs=dline[58]
        self.thermal_res=dline[59]
        self.shgc=dline[60]
        self.ufactor=dline[61]
        ##heat and moisture properties
        self.porosity=dline[9]
        self.init_water=dline[10]
        self.n_iso=dline[11]
        self.sorb_iso={}
        self.n_suc=dline[14]
        self.suc={}
        self.n_red=dline[17]
        self.red={}
        self.n_mu=dline[20]
        self.mu={}
        self.n_therm=dline[23]
        self.therm={}
        self.vis_ref = dline[44]
        
        ##solar properties
        if len(dline)>31:
            if dline[31]!='':        
                self.g_sol_trans=float(dline[31])
                self.g_F_sol_ref=float(dline[32])
                self.g_B_sol_ref=float(dline[33])
                self.g_vis_trans=float(dline[34])
                self.g_F_vis_ref=float(dline[35])
                self.g_B_vis_ref=float(dline[36])
                self.g_IR_trans=float(dline[37])
                self.g_F_IR_em=float(dline[38])
                self.g_B_IR_em=float(dline[39])
            
            
        if len(dline)>39:
            self.ep_special=dline[40]
            if dline[41]!='':
                self.sol_trans=float(dline[41])
                self.sol_ref=float(dline[42])
                self.vis_trans=float(dline[43])

                self.therm_hem_em=float(dline[45])
                self.therm_trans=float(dline[46])
                self.shade_to_glass_dist=float(dline[47])
                self.top_opening_mult=float(dline[48])
                self.bottom_opening_mult=float(dline[49])
                self.left_opening_mult=float(dline[50])
                self.right_opening_mult=float(dline[51])
                self.air_perm=float(dline[52])
        
        if len(dline)>53:
            self.ep_special=dline[40]
            if dline[53]!='':
                self.orient=dline[53]
                self.width=float(dline[54])
                self.separation=float(dline[55])
                self.angle=float(dline[56])
                self.blind_to_glass_dist=float(dline[57]) 
                
"""Class containing the fabric properties"""
class Fabric():
    def __init__(self,dline):
        self.dline=dline
        self.name=dline[0]
        self.purpose=dline[1]
        self.permeability=dline[2]
        
        self.htc_in=0.0
        if dline[3]!='':
            
            self.htc_in=np.float(dline[3])

        self.htc_out=0.0
        if dline[4]!='':
            self.htc_out=np.float(dline[4])

        self.vtc_in=0.0
        if dline[5]!='':
            self.vtc_in=np.float(dline[5])

        self.vtc_out=0.0
        if dline[6]!='':
            self.vtc_out=np.float(dline[6])
            
        self.rmats=[]
        
        self.U=0.0
        
    def add_mat(self,mat,thick):
        rmat = Rmat(mat,thick)
        self.rmats.append(rmat)
        
    def calc_u(self):
        R=0.0        
        for rmat in self.rmats:  
            R+=np.float(rmat.init_thickness)/np.float(rmat.mat.conductivity)

        if self.htc_in>0.0:
            R+=1.0/np.float(self.htc_in)
        if self.htc_out>0.0:
            R+=1.0/np.float(self.htc_out)
            
        if R>0.0:
            self.U=1.0/R
        
        return self.U
    
    def get_thickness(self, u_value):
        R=0.0
        
        gap_conductivity = 0
        gap_thickness = 0
        
        if self.htc_in>0.0:
            R+=1.0/np.float(self.htc_in)
        if self.htc_out>0.0:
            R+=1.0/np.float(self.htc_out)
                
        for rmat in self.rmats:
            if(("AirGapExt" in rmat.name) or ("GasWin" in rmat.name) or ("AirGapRoof" in rmat.name) or ("CelluloseInsExt" in rmat.name) or ("GlassFiberFloor" in rmat.name)):  
                gap_conductivity=np.float(rmat.mat.conductivity)
            else:
                R+=np.float(rmat.init_thickness)/np.float(rmat.mat.conductivity)
        
        if R>0.0 and gap_conductivity>0.0:
            gap_thickness = gap_conductivity*((1/u_value)-R)
        
        ##Fail safe
        if gap_thickness < 0:
            gap_thickness = 0.
        
        return gap_thickness

##add a another material to fabric             
class Rmat():
    def __init__(self,mat,thickness):
        self.mat=mat                    
        self.thickness=thickness        ##to be changed for each run with lhd
        self.init_thickness=thickness
        self.name="{0}_{1}".format(mat.name,thickness)


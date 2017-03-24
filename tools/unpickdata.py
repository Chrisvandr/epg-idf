"""Read all of the information from the ./data_files folder containing all of the information input by Jon"""

import properties.package as package
import properties.environment as envir
import properties.building as building
import properties.zone as zn
import properties.schedule as sched

def unpick_database(rootdir):
    mats    =   unpick_materials(rootdir)
    fabs    =   unpick_fabrics(rootdir, mats)
    packs   =   unpick_packages(rootdir, fabs)

    wind    =   unpick_wind(rootdir)
    shades  =   unpick_shades(rootdir)
    envs    =   unpick_environments(rootdir, wind,shades)
         
    equips=unpick_equipments(rootdir)
    hsched_groups=unpick_house_schedules(rootdir)
    occups=unpick_occupations(rootdir, hsched_groups)
    
    builds  =  unpick_buildings(rootdir)
    
    reports=unpick_reports(rootdir)

    return packs,envs,builds,occups,equips,reports
                
def read_sheet(rootdir, fname=None):
    sheet=[]
    
    if fname!=None:
        csvfile="{0}/data_files/{1}".format(rootdir,fname)
        with open(csvfile,'r') as inf:
            
            
            for row in inf:
                row=row.rstrip()
                datas=row.split(',')
                sheet.append(datas)
                
    return sheet

def unpick_materials(rootdir):
    #open material properties file and puts everything into sheet
    sheet=read_sheet(rootdir, fname="mat_props.csv")

    #find the material properties from csv file
    for nn,lline in enumerate(sheet):
        if len(lline)>0 and 'Name' in lline[0]:
            d_start=nn+1
            break
   
    mats={}
    mat=None
    for dline in sheet[d_start:]:
            
        if len(dline)>1:
            ##will have to read some stuff here
            if dline[0]!='':
    
                mat=package.Material(dline) 
                mats[mat.name]=mat
            
            ##Code added to read heat and moisture properties for HAMT
            #sorbtion isotherms
            if mat!=None and dline[12]!=''and dline[13]!='':
                mats[mat.name].sorb_iso[dline[12]] = dline[13]
            #suctions
            if mat!=None and dline[15]!=''and dline[16]!='':
                mats[mat.name].suc[dline[15]] = dline[16]
            #REDISTRIBUTION
            if mat!=None and dline[18]!=''and dline[19]!='':
                mats[mat.name].red[dline[18]] = dline[19]
            #DIFFUSION
            if mat!=None and dline[21]!=''and dline[22]!='':
                mats[mat.name].mu[dline[21]] = dline[22]
            #Thermal Conductivity
            if mat!=None and dline[24]!=''and dline[25]!='':
                mats[mat.name].therm[dline[24]] = dline[25]        

    return mats

def unpick_fabrics(rootdir, mats,repack=False):

    sheet=read_sheet(rootdir, fname="fab_props.csv")
    
    for nn,lline in enumerate(sheet):
        if len(lline)>0 and 'Name' in lline[0]:
            d_start=nn+1
            break
 
    fabs={}
    fab=None
    for dline in sheet[d_start:]:
        
        if len(dline)>1:
            if dline[0]!='':
           
                fabname=dline[0]
                fab = package.Fabric(dline)
                fabs[fabname]=fab    
             
            matname=dline[7]
            
            if fab!=None and matname!='':
                if matname in list(mats.keys()):
                    fab.add_mat(mats[matname],dline[8])
                else: 
                    print("Can't find ", matname, " in mat_props")
    
    if repack:   
        for fab in list(fabs.values()):
            fab.calc_u()
        
        repack_fabrics(rootdir, fabs=fabs)
      
    return fabs

        
def repack_fabrics(rootdir, fabs=None):
    
    csvfile="{0}/data_files/fab_props.csv".format(rootdir)
    with open(csvfile,'w') as outf:
        outf.write("Name,Purpose,Permeability,htc_in,htc_out,vtc_in,vtc_out,Material,Thickness,U-Value\n")
        for fab in sorted(list(fabs.values()), key=lambda ff:ff.purpose):
            
            op="{0}".format(fab.name)
            op+=",{0}".format(fab.purpose)
            op+=",{0}".format(fab.permeability)
            op+=",{0}".format(fab.htc_in)
            op+=",{0}".format(fab.htc_out)
            op+=",{0}".format(fab.vtc_in)
            op+=",{0}".format(fab.vtc_out)
            op+=",{0}".format(fab.rmats[0].mat.name)
            op+=",{0}".format(fab.rmats[0].thickness)
            op+=",{0}".format(fab.U)
            outf.write(op+'\n')
            for rmat in fab.rmats[1:]:
                op=",,,,,,,{0},{1}".format(rmat.mat.name,rmat.thickness)
                outf.write(op+'\n')
            outf.write('\n')
                
def unpick_packages(rootdir, fabs):

    sheet=read_sheet(rootdir, fname="pack_props.csv")
    
    for nn,lline in enumerate(sheet):
        if len(lline)>0 and 'Name' in lline[0]:
            d_start=nn+1
            break

    packs={}
    pack=None
    for dline in sheet[d_start:]:    
        if len(dline)>1:
            if dline[0]!='':

                pack=package.Package(dline[0])
                packs[dline[0]]=pack
                if dline[18]=='Yes':
                    pack.trickle=True
                if dline[24]=='Yes':
                    pack.C_vent=True    
                if dline[25]=='Yes':
                    pack.L_vent=True    
                           
                for propname,lln in zip(['external_wall','internal_wall','ground','roof','internal_floor','internal_ceiling','loft_floor','loft_ceiling','door','window','window_shading'],[8,9,10,11,12,13,14,15,16,17,23]):
                    for fab in list(fabs.values()):
                        if fab.name==dline[lln]:
                            pack.set_prop(propname,fab)                       
                            break
                
                
    return packs

def unpick_wind(rootdir):

    sheet=read_sheet(rootdir, fname="wind_props.csv")
    
    for nn,lline in enumerate(sheet):
        if len(lline)>0 and 'External Node' in lline[0]:
            d_start=nn+1
            break
            
            
    dline1=sheet[d_start-1]
    dline2=sheet[d_start]
    wind=envir.Wind(dline1,dline2)
   
    for dline in sheet[d_start+1:]:    
        if len(dline)>1:
            if dline[0]!='':
    
                wind.angles.append(float(dline[1]))
                for nn,vv in enumerate(dline[2:]):
                    wind.coeffs[wind.ninx[nn]].append(vv)
            
            
            
    return wind

def unpick_shades(rootdir):

    sheet=read_sheet(rootdir, fname="shade_props.csv")
    
    for nn,lline in enumerate(sheet):
        if len(lline)>0 and 'Name' in lline[0]:
            d_start=nn+1
            break

    shades={}
    shade=None
    
    for dline in sheet[d_start:]:    
        if len(dline)>1:
            if dline[0]!='':
                shade=envir.Shade(dline[0])
                shades[dline[0]]=shade
            if dline[1]!='' and shade!=None:
                if dline[1] not in list(shade.shade_objects.keys()):
                    sobj=envir.ShadeProps(dline[1])
                    shade.shade_objects[dline[1]]=sobj
                if dline[2]!='':
                    sobj.command=dline[2]
                    sobj.rotation_axis=int(dline[3])
                    sobj.xx=float(dline[4])
                    sobj.yy=float(dline[5])
                    sobj.zz=float(dline[6])
                    sobj.L1=float(dline[7])
                    sobj.L2=float(dline[8])
                    sobj.hh=float(dline[9])
                        
    return shades
 
def unpick_environments(rootdir, wind,shades):

    sheet=read_sheet(rootdir, fname="env_props.csv")
    
    for nn,lline in enumerate(sheet):
        if len(lline)>0 and 'Name' in lline[0]:
            d_start=nn+1
            break

    envs={}
    env=None
    for dline in sheet[d_start:]:    
        if len(dline)>1:
            if dline[0]!='':

                env=envir.Environment(dline,wind,shades)
                envs[dline[0]]=env
                         
    return envs

def unpick_buildings(rootdir, pack=None):
    sheet=read_sheet(rootdir, fname="build_props.csv")
    
    for nn,lline in enumerate(sheet):
        if len(lline)>0 and 'Name' in lline[0]:
            d_start=nn+1
            break

    builds={}
    build=None
    zone=None
    for dline in sheet[d_start:]:    
        if len(dline)>1:
            if dline[0]!='':
            
                build=building.Building(dline)
                builds[dline[0]]=build
            
            if build!=None and dline[5]!='':
                zonename=dline[5]
                #build.inzones.keys() are the zones for a particular building def
                if zonename not in list(build.inzones.keys()):
                    zone=zn.Zone(zonename)
                    build.inzones[zonename]=zone
                zone=build.inzones[zonename]
                #(item1:x,y,z,L1,L2,h, item2:Coord,Length,Height,   
                zone.set_prop(dline[6],dline[7],dline[8],dline[9])
                
    return builds      

def unpick_equipments(rootdir):

    sheet=read_sheet(rootdir, fname="equipment_props.csv")
    
    for nn,lline in enumerate(sheet):
        if len(lline)>0 and 'Appliance' in lline[0]:
            d_start=nn+1
            break
 
    equips={}
    equip=None
 
    for dline in sheet[d_start:]:    
        if len(dline)>1:
            if dline[0]!='':
                equip=sched.Equipment(dline)
            equips[dline[0]]=equip
                         
    return equips

def unpick_house_schedules(rootdir):
    sheet=read_sheet(rootdir, fname="house_scheds.csv")
    
    for nn,lline in enumerate(sheet):
        if len(lline)>0 and 'Name' in lline[0]:
            d_start=nn+1
            break
   
    sched_groups={}
    hgroup=None
    for dline in sheet[d_start:]:
        if len(dline)>1:
            if dline[0]!='':
                
                sname=dline[0]
                if sname not in list(sched_groups.keys()):               
                    hgroup=sched.HGroup(dline)
                    sched_groups[sname]=hgroup
                hgroup=sched_groups[sname]
                
                #Loop her over zones so that can reduce the size of the house_sched csv
                zones = dline[1].split(":")
                if(len(zones)>1):
                    for zone in zones:
                        zonesched=sched.ZoneSched(dline, zone)
                        hgroup.zones_scheds.append(zonesched)
                else:
                    zonesched=sched.ZoneSched(dline, dline[1])
                    hgroup.zones_scheds.append(zonesched)

    return sched_groups

def unpick_occupations(rootdir, hgroups):
    sheet=read_sheet(rootdir, fname="occupation_props.csv")
    
    for nn,lline in enumerate(sheet):
        if len(lline)>0 and 'Name' in lline[0]:
            d_start=nn+1
            break
   
    occups={}
    occup=None
    for dline in sheet[d_start:]:    
        if len(dline)>1:
            if dline[0]!='':

                oname=dline[0]
                if oname not in list(occups.keys()):
                    occup=sched.Occupation(dline)
                    occups[oname]=occup
                occup=occups[oname]
                
                if dline[1]=='House' and dline[2] in hgroups:
                    occup.hgroups.append(hgroups[dline[2]])
                else:
                    print("Can't find: ",  dline[2], " in hscheds")
       
    return occups
       
def unpick_reports(rootdir):

    sheet=read_sheet(rootdir, fname="Reports.csv")
    
    for nn,lline in enumerate(sheet):
        if len(lline)>0 and 'Zone Name' in lline[0]:
            d_start=nn+1
            break
   
    reports=[]
    report=None
    for dline in sheet[d_start:]:    
        if len(dline)>1:
            if dline[0]!='':
                rooms = dline[0].split(":")
                if(len(rooms)>1):
                    for room in rooms:
                        report=Report(room, dline[1], dline[2])
                        reports.append(report)
                else:
                    report=Report(dline[0], dline[1], dline[2])
                    reports.append(report)
    return reports

###not really sure where to put this, can't really put in a file on its own!    
class Report():
    def __init__(self,zone, output, freq):
        self.zone=zone
        self.output=output
        self.freq=freq   
    
        
        
        


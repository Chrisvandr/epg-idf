"""File containing functions to build idf file"""

import tools.iddinfo as idd
import inputvars as inp
import numpy as np

def build_idfs(rootdir,epdir, folder=None):
    
    #do this first so don't have to do it twice
    #loads in all of the mapping information from the idd file
    version,idds=idd.load_idd(epdir)

    if folder!=None:
        for run in list(folder.runs.values()):
            if run.data_good:
                dirname=folder.proj_name
                if folder.runs_name != None:
                    dirname=folder.proj_name+'/'+folder.runs_name

                build_idf(rootdir,run=run,dirname=dirname,version=version,idds=idds)
        
                run.built='Yes'
                run.build_request=False
               
            else:
                run.built='Fail'
    
def build_idf(rootdir,run=None,dirname=None,version=None,idds=None):

    if run.built=='No' and run.build_request:
        print(("Creating idf {0}".format(run.rn)))
        
        building=run.props['building']
        package=run.props['package']
        occupation=run.props['occupation']
        equips=run.props['equipment']
        environ=run.props['environment']

        building.create(run=run,pack=package,occu=occupation,equips=equips,shading=environ.shading)
        
        print((building.name))
        
        idffile="{0}/{1}/EPG2_{2}.idf".format(rootdir,dirname,run.rn)

        with open(idffile,mode='w') as inf:
          

            iddobj=idds['Version']
            op=iddobj.con_obj({'Version Identifier':version})
            inf.write(op+"\n")
            
            iddobj=idds['SimulationControl']               
            op=iddobj.con_obj({
                'Do Zone Sizing Calculation':'No',
                'Do System Sizing Calculation':'No',
                'Do Plant Sizing Calculation':'No',
                'Run Simulation for Sizing Periods':'Yes',
                'Run Simulation for Weather File Run Periods':'Yes'                    
                })
            inf.write(op+"\n")
            
            iddobj=idds['Building']               
            op=iddobj.con_obj({
                'Name':building.name,
                'North Axis':run.orient,
                'Terrain':environ.terrain,
                'Loads Convergence Tolerance Value':inp.lctv,
                'Temperature Convergence Tolerance Value':inp.tctv,
                'Solar Distribution':'FullExterior',
                'Maximum Number of Warmup Days':inp.max_warm_days,
                'Minimum Number of Warmup Days':inp.min_warm_days
                })
            inf.write(op+"\n")
            
            iddobj=idds['SurfaceConvectionAlgorithm:Inside']               
            op=iddobj.con_obj({
                'Algorithm':'TARP'
                })
            inf.write(op+"\n")
            
            iddobj=idds['SurfaceConvectionAlgorithm:Outside']               
            op=iddobj.con_obj({
                'Algorithm':'TARP'
                })
            inf.write(op+"\n")
            
            iddobj=idds['HeatBalanceAlgorithm']               

            if run.sim_level==3:
                op=iddobj.con_obj({
                    'Algorithm':'CombinedHeatAndMoistureFiniteElement',
                    'Surface Temperature Upper Limit':200
                })
            else:
                op=iddobj.con_obj({
                    'Algorithm':'ConductionTransferFunction',
                    'Surface Temperature Upper Limit':200
                })  

            inf.write(op+"\n")
            
            iddobj=idds['Timestep']
            op=iddobj.con_obj({'Number of Timesteps per Hour':run.timesteps})
            inf.write(op+"\n")
            
            
            iddobj=idds['RunPeriod']
            op=iddobj.con_obj({
                'Begin Month':run.start_month,
                'Begin Day of Month':run.start_day,
                'End Month':run.end_month,
                'End Day of Month':run.end_day,
                'Day of Week for Start Day':'Monday',
                'Use Weather File Holidays and Special Days':'Yes',
                'Use Weather File Daylight Saving Period':'Yes',
                'Apply Weekend Holiday Rule':'No',
                'Use Weather File Rain Indicators':'Yes',
                'Use Weather File Snow Indicators':'Yes',
                'Number of Times Runperiod to be Repeated':1
                })
            inf.write(op+"\n")
            
            iddobj=idds['Site:GroundTemperature:BuildingSurface']
            op=iddobj.con_obj({
                'January Ground Temperature':inp.jan_stemp,
                'February Ground Temperature':inp.feb_stemp,
                'March Ground Temperature':inp.mar_stemp,
                'April Ground Temperature':inp.apr_stemp,
                'May Ground Temperature':inp.may_stemp,
                'June Ground Temperature':inp.jun_stemp,
                'July Ground Temperature':inp.jul_stemp,
                'August Ground Temperature':inp.aug_stemp,
                'September Ground Temperature':inp.sep_stemp,
                'October Ground Temperature':inp.oct_stemp,
                'November Ground Temperature':inp.nov_stemp,
                'December Ground Temperature':inp.dec_stemp
                })
            inf.write(op+"\n")
            
            iddobj=idds['ScheduleTypeLimits']
            op=iddobj.con_obj({
                'Name':'Control',
                'Lower Limit Value':0,
                'Upper Limit Value':4,
                'Numeric Type':'Discrete',
                'Unit Type':'Control'
                })
            inf.write(op+"\n")
            
            op=iddobj.con_obj({
                'Name':'Thermal',
                'Lower Limit Value':-20.0,
                'Upper Limit Value':60.0,
                'Numeric Type':'Continuous',
                'Unit Type':'Temperature'
                })
            inf.write(op+"\n")
            
            op=iddobj.con_obj({
                'Name':'Fraction',
                'Lower Limit Value':0.0,
                'Upper Limit Value':10.0,
                'Numeric Type':'Continuous',
                'Unit Type':'Dimensionless'
                })
            inf.write(op+"\n")

            op=iddobj.con_obj({
                'Name':'Number',
                'Lower Limit Value':0.0,
                'Upper Limit Value':1000000000.0,
                'Numeric Type':'Continuous',
                'Unit Type':'Dimensionless'
                })
            inf.write(op+"\n")
            
            
            cmats=package.cmats
            for rmat in list(cmats.values()):
                iddobj=idds['Material']
                op=iddobj.con_obj({
                    'Name':rmat.name,
                    'Roughness':rmat.mat.roughness,
                    'Thickness':rmat.thickness,
                    'Conductivity':rmat.mat.conductivity,
                    'Density':rmat.mat.density,
                    'Specific Heat':rmat.mat.specific_heat,
                    'Thermal Absorptance':rmat.mat.thermal_abs,
                    'Solar Absorptance':rmat.mat.solar_abs,
                    'Visible Absorptance':rmat.mat.visible_abs
                    })
                inf.write(op+"\n")
            
            ##Put heat and moisture properties here! if run.sim_level==3:
            if run.sim_level == 3:
                
                for rmat in list(cmats.values()):
                    iddobj=idds['MaterialProperty:HeatAndMoistureTransfer:Settings']
                    op=iddobj.con_obj({
                    'Material Name':rmat.name,
                    'Porosity':rmat.mat.porosity,
                    'Initial Water Content Ratio':rmat.mat.init_water
                    })
                    inf.write(op+"\n")
            
                for rmat in list(cmats.values()):
                    fdict={}
                    fdict['Material Name']=rmat.name
                    fdict['Number of Isotherm Coordinates']=rmat.mat.n_iso
                    i = 0
                    for  h, m in sorted(rmat.mat.sorb_iso.items()):
                        i+=1
                        fdict['Relative Humidity Fraction '+str(i)]=h
                        fdict['Moisture Content '+str(i)]=m
                   
                    iddobj=idds['MaterialProperty:HeatAndMoistureTransfer:SorptionIsotherm']
                    op=iddobj.con_obj(fdict)
                    inf.write(op+"\n")
                
                for rmat in list(cmats.values()):
                    fdict={}
                    fdict['Material Name']=rmat.name
                    fdict['Number of Suction points']=rmat.mat.n_suc
                    i = 0
                    for  h, m in sorted(rmat.mat.suc.items()):
                        i+=1
                        fdict['Moisture Content '+str(i)]=h
                        fdict['Liquid Transport Coefficient '+str(i)]=m
                   
                    iddobj=idds['MaterialProperty:HeatAndMoistureTransfer:Suction']
                    op=iddobj.con_obj(fdict)
                    inf.write(op+"\n")
                    
                for rmat in list(cmats.values()):
                    fdict={}
                    fdict['Material Name']=rmat.name
                    fdict['Number of Redistribution points']=rmat.mat.n_red
                    i = 0
                    for  h, m in sorted(rmat.mat.red.items()):
                        i+=1
                        fdict['Moisture Content '+str(i)]=h
                        fdict['Liquid Transport Coefficient '+str(i)]=m
                   
                    iddobj=idds['MaterialProperty:HeatAndMoistureTransfer:Redistribution']
                    op=iddobj.con_obj(fdict)
                    inf.write(op+"\n")  
                    
                for rmat in list(cmats.values()):
                    fdict={}
                    fdict['Material Name']=rmat.name
                    fdict['Number of Data Pairs']=rmat.mat.n_mu
                    i = 0
                    for  h, m in sorted(rmat.mat.mu.items()):
                        i+=1
                        fdict['Relative Humidity Fraction '+str(i)]=h
                        fdict['Water Vapor Diffusion Resistance Factor '+str(i)]=m
                   
                    iddobj=idds['MaterialProperty:HeatAndMoistureTransfer:Diffusion']
                    op=iddobj.con_obj(fdict)
                    inf.write(op+"\n") 
                    
                for rmat in list(cmats.values()):
                    fdict={}
                    fdict['Material Name']=rmat.name
                    fdict['Number of Thermal Coordinates']=rmat.mat.n_therm
                    i = 0
                    for  h, m in sorted(rmat.mat.therm.items()):
                        i+=1
                        fdict['Moisture Content '+str(i)]=h
                        fdict['Thermal Conductivity '+str(i)]=m
                   
                    iddobj=idds['MaterialProperty:HeatAndMoistureTransfer:ThermalConductivity']
                    op=iddobj.con_obj(fdict)
                    inf.write(op+"\n")  
                
            
            
                
            glaze_mats=package.glaze_mats
            if len(glaze_mats)>0:
                for rmat in list(glaze_mats.values()):
                    iddobj=idds['WindowMaterial:Glazing']
                    op=iddobj.con_obj({
                        'Name':rmat.name,
                        'Optical Data Type':rmat.mat.ep_special,
                        'Thickness':rmat.thickness,
                        'Solar Transmittance at Normal Incidence':rmat.mat.g_sol_trans,
                        'Front Side Solar Reflectance at Normal Incidence':rmat.mat.g_F_sol_ref,
                        'Back Side Solar Reflectance at Normal Incidence':rmat.mat.g_B_sol_ref,
                        'Visible Transmittance at Normal Incidence':rmat.mat.g_vis_trans,
                        'Front Side Visible Reflectance at Normal Incidence':rmat.mat.g_F_vis_ref,
                        'Back Side Visible Reflectance at Normal Incidence':rmat.mat.g_B_vis_ref,
                        'Infrared Transmittance at Normal Incidence':rmat.mat.g_IR_trans,
                        'Front Side Infrared Hemispherical Emissivity':rmat.mat.g_F_IR_em,
                        'Back Side Infrared Hemispherical Emissivity':rmat.mat.g_B_IR_em,
                        'Conductivity':rmat.mat.conductivity
                        })
                    inf.write(op+"\n")
                    
            gas_mats=package.gas_mats
            if len(gas_mats)>0:
                for rmat in list(gas_mats.values()):
                    iddobj=idds['WindowMaterial:Gas']
                    op=iddobj.con_obj({
                        'Name':rmat.name,
                        'Gas Type':rmat.mat.ep_special,
                        'Thickness':rmat.thickness
                        })
                    inf.write(op+"\n")
                    
            shade_mats=package.shade_mats
            if len(shade_mats)>0:
                for rmat in list(shade_mats.values()):
                    iddobj=idds['WindowMaterial:Shade']
                    op=iddobj.con_obj({
                        'Name':rmat.name,
                        'Solar Transmittance':rmat.mat.sol_trans,
                        'Solar Reflectance':rmat.mat.sol_ref,
                        'Visible Transmittance':rmat.mat.vis_trans,
                        'Visible Reflectance':rmat.mat.vis_ref,
                        'Infrared Hemispherical Emissivity':rmat.mat.therm_hem_em,
                        'Infrared Transmittance':rmat.mat.therm_trans,
                        'Shade to Glass Distance':rmat.mat.shade_to_glass_dist,
                        'Top Opening Multiplier':rmat.mat.top_opening_mult,
                        'Bottom Opening Multiplier':rmat.mat.bottom_opening_mult,
                        'Left-Side Opening Multiplier':rmat.mat.left_opening_mult,
                        'Right-Side Opening Multiplier':rmat.mat.right_opening_mult,
                        'Airflow Permeability':rmat.mat.air_perm,
                        'Thickness':rmat.thickness,
                        'Conductivity':rmat.mat.conductivity
                        })
                    inf.write(op+"\n")
            
            blind_mats=package.blind_mats
            if len(blind_mats)>0:
                for rmat in list(blind_mats.values()):
                    iddobj=idds['WindowMaterial:Blind']
                    op=iddobj.con_obj({
                        'Name':rmat.name,
                        'Slat Orientation':rmat.mat.orient,
                        'Slat Width':rmat.mat.width,
                        'Slat Separation':rmat.mat.separation,
                        'Slat Thickness':rmat.thickness,
                        'Slat Angle':rmat.mat.angle,
                        'Slat Conductivity':rmat.mat.conductivity,
                        'Slat Beam Solar Transmittance':rmat.mat.sol_trans,
                        'Front Side Slat Beam Solar Reflectance':rmat.mat.sol_ref,
                        'Back Side Slat Beam Solar Reflectance':rmat.mat.sol_ref,
                        'Slat Diffuse Solar Transmittance':rmat.mat.vis_trans,
                        'Front Side Slat Diffuse Solar Reflectance':rmat.mat.sol_ref,
                        'Back Side Slat Diffuse Solar Reflectance':rmat.mat.sol_ref,
                        'Slat Beam Visible Transmittance':rmat.mat.vis_trans,
                        'Front Side Slat Beam Visible Reflectance':rmat.mat.sol_ref,
                        'Back Side Slat Beam Visible Reflectance':rmat.mat.sol_ref,
                        'Slat Diffuse Visible Transmittance':rmat.mat.vis_trans,
                        'Front Side Slat Diffuse Visible Reflectance':rmat.mat.sol_ref,
                        'Back Side Slat Diffuse Visible Reflectance':rmat.mat.sol_ref,
                        'Slat Infrared Hemispherical Transmittance':rmat.mat.vis_trans,
                        'Front Side Slat Infrared Hemispherical Emissivity':rmat.mat.therm_hem_em,
                        'Back Side Slat Infrared Hemispherical Emissivity':rmat.mat.therm_hem_em,
                        'Blind to Glass Distance':rmat.mat.blind_to_glass_dist,
                        'Blind Top Opening Multiplier':rmat.mat.top_opening_mult,
                        'Blind Bottom Opening Multiplier':rmat.mat.bottom_opening_mult,
                        'Minimum Slat Angle': ' ',
                        'Maximum Slat Angle': ' '
                        })
                    inf.write(op+"\n")


            for fabtype in list(package.props.keys()):
                fab=package.props[fabtype]
                wrmats=[]

                if fabtype=='window_shading':                       
                    window_fab=package.props['window']
                    wrmats=window_fab.rmats

                outmats=wrmats+fab.rmats
                blind_mats=package.blind_mats
                if len(blind_mats)>0:
                    outmats=fab.rmats+wrmats
                #print(outmats)

                iddobj=idds['Construction']
                fdict={}
                fdict['Name']=fab.name
                
                fdict['Outside Layer']=outmats[0].name
                for nn,rmat in enumerate(outmats[1:]):
                    fdict["Layer {0}".format(nn+2)]=rmat.name
                op=iddobj.con_obj(fdict)
                inf.write(op+"\n")
            
            if run.use_AirflowNetwork:
                iddobj=idds['AirflowNetwork:SimulationControl']
                op=iddobj.con_obj({
                    'Name':'AF1',
                    'AirflowNetwork Control':'MultiZoneWithoutDistribution',
                    'Wind Pressure Coefficient Type':'Input',
                    'AirflowNetwork Wind Pressure Coefficient Array Name':'WParray',
                    'Height Selection for Local Wind Pressure Calculation':'ExternalNode',
                    'Building Type':'LowRise',
                    'Maximum Number of Iterations':500,
                    'Initialization Type':'ZeroNodePressures',
                    'Relative Airflow Convergence Tolerance':0.0001,
                    'Absolute Airflow Convergence Tolerance':'1.E-6',
                    'Convergence Acceleration Limit':-0.5,
                    'Azimuth Angle of Long Axis of Building':0,
                    'Ratio of Building Width Along Short Axis to Width Along Long Axis':1
                    })
                inf.write(op+"\n")
                    
                
                iddobj=idds['AirflowNetwork:MultiZone:ReferenceCrackConditions']
                op=iddobj.con_obj({
                    'Name':'RCC1',
                    'Reference Temperature':inp.ref_temp,
                    'Reference Barometric Pressure':inp.ref_press,
                    'Reference Humidity Ratio':inp.ref_humratio
                    })
                inf.write(op+"\n")
                    
                    
                wind=environ.wind  
                iddobj=idds['AirflowNetwork:MultiZone:WindPressureCoefficientArray']
                fdict={}
                fdict['Name']='WParray'
                for nn,angle in enumerate(wind.angles):
                    fdict["Wind Direction {0}".format(nn+1)]=angle
                op=iddobj.con_obj(fdict)
                inf.write(op+"\n")
                    
                for node, _ in list(wind.orients.items()):   # _ is "orient" of surface e.g L1, L2 ...
                    
                    iddobj=idds['AirflowNetwork:MultiZone:WindPressureCoefficientValues']
                    fdict={}
                    fdict['Name']=node
                    fdict['AirflowNetwork:MultiZone:WindPressureCoefficientArray Name']='WParray'
                    for nn,coeff in enumerate(wind.coeffs[node]):
                        fdict["Wind Pressure Coefficient Value {0}".format(nn+1)]=coeff
                    op=iddobj.con_obj(fdict)
                    inf.write(op+"\n")
                

            if building.zones['Outside'].contaminant!=None:
                iddobj=idds['ZoneAirContaminantBalance']
                op=iddobj.con_obj({
                    'Carbon Dioxide Concentration':'No',
                    'Outdoor Carbon Dioxide Schedule Name':'',
                    'Generic Contaminant Concentration':'Yes',
                    'Outdoor Generic Contaminant Schedule Name':building.zones['Outside'].contaminant.name
                    })
                inf.write(op+"\n")

                
                
            iddobj=idds['GlobalGeometryRules']
            op=iddobj.con_obj({
                'Starting Vertex Position':'UpperLeftCorner',
                'Vertex Entry Direction':'CounterClockwise',
                'Coordinate System':'Relative'
                })
            inf.write(op+"\n")

            
            for shading_surface in building.shading_surfaces:
                if (len(shading_surface.vertexs) == 3):
                    iddobj=idds['Shading:Building:Detailed']
                    op=iddobj.con_obj({
                        'Name':shading_surface.name,
                        'Number of Vertices':len(shading_surface.vertexs),
                        'Vertex 1 X-coordinate':shading_surface.vertexs[0][0],
                        'Vertex 1 Y-coordinate':shading_surface.vertexs[0][1],
                        'Vertex 1 Z-coordinate':shading_surface.vertexs[0][2],
                        'Vertex 2 X-coordinate':shading_surface.vertexs[1][0],
                        'Vertex 2 Y-coordinate':shading_surface.vertexs[1][1],
                        'Vertex 2 Z-coordinate':shading_surface.vertexs[1][2],
                        'Vertex 3 X-coordinate':shading_surface.vertexs[2][0],
                        'Vertex 3 Y-coordinate':shading_surface.vertexs[2][1],
                        'Vertex 3 Z-coordinate':shading_surface.vertexs[2][2]
                        })
                else:
                    iddobj=idds['Shading:Building:Detailed']
                    op=iddobj.con_obj({
                        'Name':shading_surface.name,
                        'Number of Vertices':len(shading_surface.vertexs),
                        'Vertex 1 X-coordinate':shading_surface.vertexs[0][0],
                        'Vertex 1 Y-coordinate':shading_surface.vertexs[0][1],
                        'Vertex 1 Z-coordinate':shading_surface.vertexs[0][2],
                        'Vertex 2 X-coordinate':shading_surface.vertexs[1][0],
                        'Vertex 2 Y-coordinate':shading_surface.vertexs[1][1],
                        'Vertex 2 Z-coordinate':shading_surface.vertexs[1][2],
                        'Vertex 3 X-coordinate':shading_surface.vertexs[2][0],
                        'Vertex 3 Y-coordinate':shading_surface.vertexs[2][1],
                        'Vertex 3 Z-coordinate':shading_surface.vertexs[2][2],
                        'Vertex 4 X-coordinate':shading_surface.vertexs[3][0],
                        'Vertex 4 Y-coordinate':shading_surface.vertexs[3][1],
                        'Vertex 4 Z-coordinate':shading_surface.vertexs[3][2]
                        })
                inf.write(op+"\n")

            
            
            for _ ,zone in list(run.props['building'].zones.items()): #_ is "zname" name of zone (not used)
                if zone.inside:
                    iddobj=idds['Zone']
                    op=iddobj.con_obj({
                        'Name':zone.name,
                        'Direction of Relative North':0.0,
                        'X Origin':0.0,
                        'Y Origin':0.0,
                        'Z Origin':0.0,
                        })
                    inf.write(op+"\n")
                
                    if run.use_AirflowNetwork:
                        iddobj=idds['AirflowNetwork:MultiZone:Zone']
                        op=iddobj.con_obj({
                            'Zone Name':zone.name,
                            'Ventilation Control Mode':'Constant',
                            'Minimum Venting Open Factor':1.0,
                            'Indoor and Outdoor Temperature Difference Lower Limit For Maximum Venting Open Factor':0.0,
                            'Indoor and Outdoor Temperature Difference Upper Limit for Minimun Venting Open Factor':100.0,
                            'Indoor and Outdoor Enthalpy Difference Lower Limit For Maximum Venting Open Factor':0.0,
                            'Indoor and Outdoor Enthalpy Difference Upper Limit for Minimun Venting Open Factor':300000.0
                            })
                        inf.write(op+"\n")

                    if zone.source_sink!=None:
                        schd=zone.source_sink
                        iddobj=idds['ZoneContaminantSourceAndSink:Generic:Constant']
                        op=iddobj.con_obj({
                            'Name':schd.name,
                            'Zone Name':zone.name,
                            'Design Generation Rate':schd.base_value,
                            'Generation Schedule Name':schd.name,
                            'Design Removal Coefficient':0.0,
                            'Removal Schedule Name':schd.name
                            })
                        inf.write(op+"\n")
                        
                    if zone.dep_rate!=None:
                        schd=zone.dep_rate
                        iddobj=idds['ZoneContaminantSourceAndSink:Generic:DepositionRateSink']
                        op=iddobj.con_obj({
                            'Name':schd.name,
                            'Zone Name':zone.name,
                            'Deposition Rate':schd.base_value,
                            'Schedule Name':schd.name
                            })
                        inf.write(op+"\n")
                    
                    
                for hang_surface in zone.hang_surfaces:
                    iddobj=idds['Shading:Zone:Detailed']
                    op=iddobj.con_obj({
                        'Name':hang_surface.name,
                        'Base Surface Name':hang_surface.base_surface.name,
                        'Number of Vertices':len(hang_surface.vertexs),
                        'Vertex 1 X-coordinate':hang_surface.vertexs[0][0],
                        'Vertex 1 Y-coordinate':hang_surface.vertexs[0][1],
                        'Vertex 1 Z-coordinate':hang_surface.vertexs[0][2],
                        'Vertex 2 X-coordinate':hang_surface.vertexs[1][0],
                        'Vertex 2 Y-coordinate':hang_surface.vertexs[1][1],
                        'Vertex 2 Z-coordinate':hang_surface.vertexs[1][2],
                        'Vertex 3 X-coordinate':hang_surface.vertexs[2][0],
                        'Vertex 3 Y-coordinate':hang_surface.vertexs[2][1],
                        'Vertex 3 Z-coordinate':hang_surface.vertexs[2][2],
                        'Vertex 4 X-coordinate':hang_surface.vertexs[3][0],
                        'Vertex 4 Y-coordinate':hang_surface.vertexs[3][1],
                        'Vertex 4 Z-coordinate':hang_surface.vertexs[3][2]
                        })
                    inf.write(op+"\n")
                
                for surface in zone.surfaces:
                    fdict={'Name':surface.name,
                        'Surface Type':surface.surface_type,
                        'Construction Name':surface.fab.name,
                        'Zone Name':zone.name,
                        'Outside Boundary Condition':surface.outside_BC,
                        'Outside Boundary Condition Object':surface.outside_BC_obj,
                        'Sun Exposure':surface.sunexp,
                        'Wind Exposure':surface.windexp,
                        'View Factor to Ground':'autocalculate',
                        'Number of Vertices':'autocalculate'}
                    
                    i=0
                    for vertex in surface.vertexs:
                        i+=1
                        fdict['Vertex '+str(i)+' X-coordinate']=vertex[0]
                        fdict['Vertex '+str(i)+' Y-coordinate']=vertex[1]
                        fdict['Vertex '+str(i)+' Z-coordinate']=vertex[2]
                        
                    iddobj=idds['BuildingSurface:Detailed']
                    op=iddobj.con_obj(fdict)
                    inf.write(op+"\n")
                    
                    if zone.dep_vel!=None:
                        schd=zone.dep_vel
                        iddobj=idds['SurfaceContaminantSourceAndSink:Generic:DepositionVelocitySink']
                        op=iddobj.con_obj({
                            'Name':"{0}_{1}".format(schd.name,surface.name),
                            'Surface Name':surface.name,
                            'Deposition Velocity':schd.base_value,
                            'Schedule Name':schd.name
                            })
                        inf.write(op+"\n")

                  
                    
                    if surface.fab.htc_in>0.0 or surface.fab.htc_out>0.0:
                        iddobj=idds['SurfaceProperty:ConvectionCoefficients']
                        sdict={}
                        sdict['Surface Name']=surface.name
                        cc=1
                        if surface.fab.htc_in>0.0:
                            sdict["Convection Coefficient {0} Location".format(cc)]='Inside'
                            sdict["Convection Coefficient {0} Type".format(cc)]='Value'
                            sdict["Convection Coefficient {0}".format(cc)]=surface.fab.htc_in
                            cc+=1    
                        if surface.fab.htc_out>0.0:
                            sdict["Convection Coefficient {0} Location".format(cc)]='Outside'
                            sdict["Convection Coefficient {0} Type".format(cc)]='Value'
                            sdict["Convection Coefficient {0}".format(cc)]=surface.fab.htc_out
                        op=iddobj.con_obj(sdict)
                        inf.write(op+"\n")
                    
                    if (surface.fab.vtc_in>0.0 or surface.fab.vtc_out>0.0) and run.sim_level == 3:
                        iddobj=idds['SurfaceProperties:VaporCoefficients']
                        sdict={}
                        sdict['Surface Name']=surface.name
                        if surface.fab.vtc_out>0.0:
                            sdict["Constant External Vapor Transfer Coefficient"]='Yes'
                            sdict["External Vapor Coefficient Value"]=surface.fab.vtc_out
                        else:
                            sdict["Constant External Vapor Transfer Coefficient"]='No'
                        if surface.fab.vtc_out>0.0:
                            sdict["Constant Internal vapor Transfer Coefficient"]='Yes'
                            sdict["Internal Vapor Coefficient Value"]=surface.fab.vtc_in
                        else:
                            sdict["Constant Internal vapor Transfer Coefficient"]='No'                                       
                        
                        op=iddobj.con_obj(sdict)
                        inf.write(op+"\n")
                    
                    if run.use_AirflowNetwork:
                    
                        if (surface.perm or surface.othersidezone!=None) and surface.flowC>inp.diffthresh:
                                                        
                            iddobj=idds['AirflowNetwork:MultiZone:Surface']
                            sdict={}
                            sdict['Surface Name']=surface.name
                            if surface.exhaust:
                                sdict['Leakage Component Name']=surface.name+'_ExhaustFan'
                            else:
                                sdict['Leakage Component Name']=surface.name+'_Crack'
                            if surface.perm:
                                sdict['External Node Name']=surface.name+'_extN'
                            sdict['Window/Door Opening Factor, or Crack Factor']=1.0
                            sdict['Ventilation Control Mode']='Constant'
                            sdict['Minimum Venting Open Factor']=1.0
                            sdict['Indoor and Outdoor Temperature Difference Lower Limit For Maximum Venting Open Factor']=0.0
                            sdict['Indoor and Outdoor Temperature Difference Upper Limit for Minimun Venting Open Factor']=100.0
                            sdict['Indoor and Outdoor Enthalpy Difference Lower Limit For Maximum Venting Open Factor']=0.0
                            sdict['Indoor and Outdoor Enthalpy Difference Upper Limit for Minimun Venting Open Factor']=300000.0


                            op=iddobj.con_obj(sdict)
                            inf.write(op+"\n")
                            
                            max_flow_rate = inp.max_flow_rate
                            if("Kitchen" in zone.name):
                                max_flow_rate = inp.max_flow_rate_kitchen
                            
                            if surface.exhaust:
                                
                                iddobj=idds['AirflowNetwork:MultiZone:Component:ZoneExhaustFan']
                                op=iddobj.con_obj({
                                    'Name':surface.name+'_ExhaustFan',
                                    'Air Mass Flow Coefficient When the Zone Exhaust Fan is Off at Reference Conditions':surface.flowC,
                                    'Air Mass Flow Exponent When the Zone Exhaust Fan is Off':inp.flow_exp,
                                    'Reference Crack Conditions':'RCC1'
                                    })
                                inf.write(op+"\n")

                                fan_eff_name = 'Fan Total Efficiency'

                                iddobj=idds['Fan:ZoneExhaust']
                                op=iddobj.con_obj({
                                    'Name':surface.name+'_ExhaustFan',
                                    'Availability Schedule Name':zone.exhaust_sched.name,
                                    fan_eff_name:inp.fan_eff,
                                    'Pressure Rise':inp.press_rise,
                                    'Maximum Flow Rate':max_flow_rate,
                                    'Air Inlet Node Name':zone.name+'_Exhaust',
                                    'Air Outlet Node Name':zone.name+'_Out',
                                    'End-Use Subcategory':'General'
                                    })
                                inf.write(op+"\n")
                                
                                
                                
                            else:
                                iddobj=idds['AirflowNetwork:MultiZone:Surface:Crack']
                                op=iddobj.con_obj({
                                    'Name':surface.name+'_Crack',
                                    'Air Mass Flow Coefficient at Reference Conditions':surface.flowC,
                                    'Air Mass Flow Exponent':inp.flow_exp,
                                    'Reference Crack Conditions':'RCC1'
                                    })
                                inf.write(op+"\n")
                            
                            if surface.perm:
                                iddobj=idds['AirflowNetwork:MultiZone:ExternalNode']
                                op=iddobj.con_obj({
                                    'Name':surface.name+'_extN',
                                    'External Node Height':np.dot(surface.centre,np.array([0,0,1.])),
                                    'Wind Pressure Coefficient Values Object Name':surface.EPG2_type
                                    })
                                inf.write(op+"\n")
                    
                    
                        
                            
                            
                        if surface.ELA:
                            iddobj=idds['AirflowNetwork:MultiZone:Surface']
                            sdict={}
                            sdict['Surface Name']=surface.name
                            sdict['Leakage Component Name']=surface.name+'_ELA'
                            sdict['External Node Name']=surface.name+'_extN'
                            sdict['Window/Door Opening Factor, or Crack Factor']=1.0
                            sdict['Ventilation Control Mode']='Constant'
                            sdict['Minimum Venting Open Factor']=1.0
                            sdict['Indoor and Outdoor Temperature Difference Lower Limit For Maximum Venting Open Factor']=0.0
                            sdict['Indoor and Outdoor Temperature Difference Upper Limit for Minimun Venting Open Factor']=100.0
                            sdict['Indoor and Outdoor Enthalpy Difference Lower Limit For Maximum Venting Open Factor']=0.0
                            sdict['Indoor and Outdoor Enthalpy Difference Upper Limit for Minimun Venting Open Factor']=300000.0


                            op=iddobj.con_obj(sdict)
                            inf.write(op+"\n")
                            
                            iddobj=idds['AirflowNetwork:MultiZone:Surface:EffectiveLeakageArea']
                            op=iddobj.con_obj({
                                'Name':surface.name+'_ELA',
                                'Effective Leakage Area':surface.ELA_area,
                                'Discharge Coefficient':inp.dis_coef,
                                'Reference Pressure Difference':inp.press_diff,
                                'Air Mass Flow Exponent':inp.flow_exp
                                })
                            inf.write(op+"\n")
                            
                            
                            iddobj=idds['AirflowNetwork:MultiZone:ExternalNode']
                            op=iddobj.con_obj({
                                'Name':surface.name+'_extN',
                                'External Node Height':surface.ELA_height,
                                'Wind Pressure Coefficient Values Object Name':surface.EPG2_type
                                })
                            inf.write(op+"\n")
                    
             
                        if surface.trick:
                            iddobj=idds['AirflowNetwork:MultiZone:Surface']
                            sdict={}
                            sdict['Surface Name']=surface.name
                            sdict['Leakage Component Name']=surface.name+'_tELA'
                            sdict['External Node Name']=surface.name+'_extN'
                            sdict['Window/Door Opening Factor, or Crack Factor']=1.0
                            sdict['Ventilation Control Mode']='Constant'
                            sdict['Minimum Venting Open Factor']=1.0
                            sdict['Indoor and Outdoor Temperature Difference Lower Limit For Maximum Venting Open Factor']=0.0
                            sdict['Indoor and Outdoor Temperature Difference Upper Limit for Minimun Venting Open Factor']=100.0
                            sdict['Indoor and Outdoor Enthalpy Difference Lower Limit For Maximum Venting Open Factor']=0.0
                            sdict['Indoor and Outdoor Enthalpy Difference Upper Limit for Minimun Venting Open Factor']=300000.0


                            op=iddobj.con_obj(sdict)
                            inf.write(op+"\n")
                            
                            iddobj=idds['AirflowNetwork:MultiZone:Surface:EffectiveLeakageArea']
                            op=iddobj.con_obj({
                                'Name':surface.name+'_tELA',
                                'Effective Leakage Area':building.tELA,
                                'Discharge Coefficient':inp.dis_coef,
                                'Reference Pressure Difference':inp.press_diff,
                                'Air Mass Flow Exponent':inp.flow_exp
                                })
                            inf.write(op+"\n")
                            
                            
                            iddobj=idds['AirflowNetwork:MultiZone:ExternalNode']
                            op=iddobj.con_obj({
                                'Name':surface.name+'_extN',
                                'External Node Height':surface.ELA_height,
                                'Wind Pressure Coefficient Values Object Name':surface.EPG2_type
                                })
                            inf.write(op+"\n")
                    
                    
                    
                    if surface.window!=None:
                        iddobj=idds['FenestrationSurface:Detailed']
                        
                        wdict={
                            'Name':surface.window.name,
                            'Surface Type':'Window',
                            'Construction Name':surface.window.fab.name,
                            'Building Surface Name':surface.name,
                            'Outside Boundary Condition Object':surface.outside_BC_obj,
                            'View Factor to Ground':'autocalculate',
                            'Number of Vertices':'autocalculate',
                            'Vertex 1 X-coordinate':surface.window.vertexs[0][0],
                            'Vertex 1 Y-coordinate':surface.window.vertexs[0][1],
                            'Vertex 1 Z-coordinate':surface.window.vertexs[0][2],
                            'Vertex 2 X-coordinate':surface.window.vertexs[1][0],
                            'Vertex 2 Y-coordinate':surface.window.vertexs[1][1],
                            'Vertex 2 Z-coordinate':surface.window.vertexs[1][2],
                            'Vertex 3 X-coordinate':surface.window.vertexs[2][0],
                            'Vertex 3 Y-coordinate':surface.window.vertexs[2][1],
                            'Vertex 3 Z-coordinate':surface.window.vertexs[2][2],
                            'Vertex 4 X-coordinate':surface.window.vertexs[3][0],
                            'Vertex 4 Y-coordinate':surface.window.vertexs[3][1],
                            'Vertex 4 Z-coordinate':surface.window.vertexs[3][2]
                            }
                        if surface.window.shading!=None:
                            wdict['Shading Control Name']=surface.window.name+'_Shader'
                        op=iddobj.con_obj(wdict)
                        inf.write(op+"\n")


                        if surface.window.shading!=None:
                            shade_type='InteriorShade'
                            ##Can do this as should only be one type of shade
                            blind_mats=package.blind_mats
                            if len(blind_mats)>0:
                                shade_type='ExteriorBlind'
                            
                            iddobj=idds['WindowProperty:ShadingControl']
                            sdict={}
                            sdict['Name']=surface.window.name+'_Shader'
                            sdict['Shading Type']=shade_type
                            sdict['Construction with Shading Name']=surface.window.shader_fab.name
                            sdict['Shading Control Type']='OnIfScheduleAllows'
                            sdict['Schedule Name']=surface.window.shading.name
                            sdict['Shading Control Is Scheduled']='Yes'
                            sdict['Glare Control Is Active']='No'
                            sdict['Type of Slat Angle Control for Blinds']='FixedSlatAngle'
                            sdict['Slat Angle Schedule Name']=' '
                            
                            op=iddobj.con_obj(sdict)
                            inf.write(op+"\n")
                        
                        
                        if run.use_AirflowNetwork:                            
                            iddobj=idds['AirflowNetwork:MultiZone:Surface']
                            sdict={}
                            sdict['Surface Name']=surface.window.name
                            sdict['Leakage Component Name']=surface.window.name+'_Opening'
                            sdict['External Node Name']=surface.window.name+'_extN'
                            sdict['Window/Door Opening Factor, or Crack Factor']=1.0
                            if surface.window.temperature!=None:
                                sdict['Ventilation Control Mode']='Temperature'
                                sdict['Ventilation Control Zone Temperature Setpoint Schedule Name']=surface.window.temperature.name
                            else:
                                sdict['Ventilation Control Mode']='Constant'
                            sdict['Minimum Venting Open Factor']=1.0
                            sdict['Indoor and Outdoor Temperature Difference Lower Limit For Maximum Venting Open Factor']=0.0
                            sdict['Indoor and Outdoor Temperature Difference Upper Limit for Minimun Venting Open Factor']=100.0
                            sdict['Indoor and Outdoor Enthalpy Difference Lower Limit For Maximum Venting Open Factor']=0.0
                            sdict['Indoor and Outdoor Enthalpy Difference Upper Limit for Minimun Venting Open Factor']=300000.0
                            if surface.window.control!=None:
                                sdict['Venting Availability Schedule Name']=surface.window.control.name
                            op=iddobj.con_obj(sdict)
                            inf.write(op+"\n")
                         

                         
                         
                         
                         
                            iddobj=idds['AirflowNetwork:MultiZone:ExternalNode']
                            op=iddobj.con_obj({
                                'Name':surface.window.name+'_extN',
                                'External Node Height':np.dot(surface.window.centre,np.array([0,0,1.])),
                                'Wind Pressure Coefficient Values Object Name':surface.EPG2_type
                                })
                            inf.write(op+"\n")
                       
                            iddobj=idds['AirflowNetwork:MultiZone:Component:DetailedOpening']
                            op=iddobj.con_obj({
                                'Name':surface.window.name+'_Opening',
                                'Air Mass Flow Coefficient When Opening is Closed':inp.flowC_wclosed,
                                'Air Mass Flow Exponent When Opening is Closed':inp.flow_exp,
                                'Type of Rectanguler Large Vertical Opening (LVO)':'HorizontallyPivoted',
                                'Extra Crack Length or Height of Pivoting Axis':0.0,
                                'Number of Sets of Opening Factor Data':2,
                                'Opening Factor 1':0.0,
                                'Discharge Coefficient for Opening Factor 1':inp.dis_coef1,
                                'Width Factor for Opening Factor 1':0.0,
                                'Height Factor for Opening Factor 1':0.0,
                                'Start Height Factor for Opening Factor 1':0.0,
                                'Opening Factor 2':1.0,
                                'Discharge Coefficient for Opening Factor 2':inp.dis_coef2,
                                'Width Factor for Opening Factor 2':1.0,
                                'Height Factor for Opening Factor 2':0.33333,
                                'Start Height Factor for Opening Factor 2':0.66666
                                })
                            inf.write(op+"\n")
                        
                        
                        
                    if surface.door!=None:
                        
                        iddobj=idds['FenestrationSurface:Detailed']
                        op=iddobj.con_obj({
                            'Name':surface.door.name,
                            'Surface Type':'Door',
                            'Construction Name':surface.door.fab.name,
                            'Building Surface Name':surface.name,
                            'Outside Boundary Condition Object':surface.door.outside_BC_obj,
                            'Number of Vertices':'autocalculate',
                            'Vertex 1 X-coordinate':surface.door.vertexs[0][0],
                            'Vertex 1 Y-coordinate':surface.door.vertexs[0][1],
                            'Vertex 1 Z-coordinate':surface.door.vertexs[0][2],
                            'Vertex 2 X-coordinate':surface.door.vertexs[1][0],
                            'Vertex 2 Y-coordinate':surface.door.vertexs[1][1],
                            'Vertex 2 Z-coordinate':surface.door.vertexs[1][2],
                            'Vertex 3 X-coordinate':surface.door.vertexs[2][0],
                            'Vertex 3 Y-coordinate':surface.door.vertexs[2][1],
                            'Vertex 3 Z-coordinate':surface.door.vertexs[2][2],
                            'Vertex 4 X-coordinate':surface.door.vertexs[3][0],
                            'Vertex 4 Y-coordinate':surface.door.vertexs[3][1],
                            'Vertex 4 Z-coordinate':surface.door.vertexs[3][2]
                            })
                        inf.write(op+"\n")
                        
                        
                        if run.use_AirflowNetwork:
                            
                            if surface.door.control!=None:
                                
                                iddobj=idds['AirflowNetwork:MultiZone:Surface']
                                sdict={}
                                sdict['Surface Name']=surface.door.name
                                sdict['Leakage Component Name']=surface.door.name+'_Opening'
                                #sdict['External Node Name']=surface.window.name+'_extN'
                                sdict['Window/Door Opening Factor, or Crack Factor']=1.0
                               
                                sdict['Ventilation Control Mode']='Constant'
                                sdict['Minimum Venting Open Factor']=1.0
                                sdict['Indoor and Outdoor Temperature Difference Lower Limit For Maximum Venting Open Factor']=0.0
                                sdict['Indoor and Outdoor Temperature Difference Upper Limit for Minimun Venting Open Factor']=100.0
                                sdict['Indoor and Outdoor Enthalpy Difference Lower Limit For Maximum Venting Open Factor']=0.0
                                sdict['Indoor and Outdoor Enthalpy Difference Upper Limit for Minimun Venting Open Factor']=300000.0
                                
                                sdict['Venting Availability Schedule Name']=surface.door.control.name
                                op=iddobj.con_obj(sdict)
                                inf.write(op+"\n")      
                           
                                iddobj=idds['AirflowNetwork:MultiZone:Component:DetailedOpening']
                                op=iddobj.con_obj({
                                    'Name':surface.door.name+'_Opening',
                                    'Air Mass Flow Coefficient When Opening is Closed':inp.flowC_dclosed,
                                    'Air Mass Flow Exponent When Opening is Closed':inp.flow_exp_door,
                                    'Type of Rectanguler Large Vertical Opening (LVO)':'NonPivoted',
                                    'Extra Crack Length or Height of Pivoting Axis':0.0,
                                    'Number of Sets of Opening Factor Data':2,
                                    'Opening Factor 1':0.0,
                                    'Discharge Coefficient for Opening Factor 1':inp.dif_coefd1,
                                    'Width Factor for Opening Factor 1':0.0,
                                    'Height Factor for Opening Factor 1':0.0,
                                    'Start Height Factor for Opening Factor 1':0.0,
                                    'Opening Factor 2':1.0,
                                    'Discharge Coefficient for Opening Factor 2':inp.dif_coefd2,
                                    'Width Factor for Opening Factor 2':1.0,
                                    'Height Factor for Opening Factor 2':1.0,
                                    'Start Height Factor for Opening Factor 2':0.0
                                    })
                                inf.write(op+"\n")
                        

                if len(zone.people)>0:
                    for person in list(zone.people.values()):
                        if person.presence_schedule!=None and person.metabolic_schedule!=None:
                            iddobj=idds['People']
                            adict={}
                            adict['Name']=person.name
                            adict['Zone or ZoneList Name']=person.presence_schedule.zone_name
                            adict['Number of People Schedule Name']=person.presence_schedule.name
                            adict['Number of People Calculation Method']='People'
                            adict['Number of People']=1
                            adict['Fraction Radiant']=0.3
                            adict['Sensible Heat Fraction']='autocalculate'
                            adict['Activity Level Schedule Name']=person.metabolic_schedule.name
                            adict['Carbon Dioxide Generation Rate']=inp.co2_rate
                            adict['Enable ASHRAE 55 Comfort Warnings']='No'
                            adict['Mean Radiant Temperature Calculation Type']='ZoneAveraged'
                            op=iddobj.con_obj(adict)
                            inf.write(op+"\n")
                    
                        
                if len(zone.appliances)>0:
                    for appliance in zone.appliances:
                        
                        iddobj=idds[appliance.purpose]
                        adict={}
                        adict['Name']=appliance.name
                        adict['Zone or ZoneList Name']=zone.name
                        adict['Schedule Name']=appliance.schedule.name
                        adict['Design Level Calculation Method']=appliance.calculation_method

                        if appliance.purpose=='Lights':
                            if appliance.calculation_method=='LightingLevel':
                                adict['Lighting Level']=appliance.power
                            elif appliance.calculation_method=='Watts/Area':
                                adict['Watts per Zone Floor Area']=appliance.power
                            adict['Return Air Fraction']=0
                            adict['Fraction Radiant']=0
                            adict['Fraction Visible']=0
                            adict['Fraction Replaceable']=1
                           
                        if appliance.purpose=='ElectricEquipment':
                            if appliance.calculation_method=='EquipmentLevel':
                                adict['Design Level']=appliance.power
                            elif appliance.calculation_method=='Watts/Area':
                                adict['Watts per Zone Floor Area']=appliance.power
                            
                            adict['Fraction Latent']=appliance.latent
                            adict['Fraction Radiant']=(appliance.useful/2) 
                            ##for radiant fraction /2 so that 50% goes to convective gains
                            adict['Fraction Lost']=1- appliance.useful-appliance.latent
                        op=iddobj.con_obj(adict)
                        inf.write(op+"\n")
                            
                if zone.heater!=None and zone.heater.all_there():
                    
                    
                    iddobj=idds['ZoneControl:Thermostat']
                    op=iddobj.con_obj({
                        'Name':zone.name+'_Thermostat',
                        'Zone or ZoneList Name':zone.name,
                        'Control Type Schedule Name':zone.heater.heating.name,
                        'Control 1 Object Type':'ThermostatSetpoint:SingleHeating',
                        'Control 1 Name':zone.name+'_stat'
                        })
                    inf.write(op+"\n")
                    
                    iddobj=idds['ThermostatSetpoint:SingleHeating']
                    op=iddobj.con_obj({
                        'Name':zone.name+'_stat',
                        'Setpoint Temperature Schedule Name':zone.heater.comfort.name
                        })
                    inf.write(op+"\n")
       
                    iddobj=idds['ZoneHVAC:Baseboard:Convective:Electric']
                    
                    #if statement here with what Energy+ version
                    if "8.2" in version:
                        heater_power_name = 'Heating Design Capacity Per Floor Area'
                    else:
                        heater_power_name = 'Nominal Capacity'

                    op=iddobj.con_obj({
                        'Name':zone.heater.name,
                        'Availability Schedule Name':zone.heater.heating.name,
                        #'Heating Design Capacity Method':'CapacityPerFloorArea',
                        heater_power_name:zone.heater.power,
                        'Efficiency':1.0
                        })
                    inf.write(op+"\n")
                
                if len(zone.equiplist)>0:
                    iddobj=idds['ZoneHVAC:EquipmentList']
                    edict={}
                    ee=0
                    for equip in zone.equiplist:
                        if equip.name not in edict.values():   ##make sure don't repeat equipment
                            edict["Zone Equipment {0} Object Type".format(ee+1)]=equip.eptype
                            edict["Zone Equipment {0} Name".format(ee+1)]=equip.name
                            edict["Zone Equipment {0} Cooling Sequence".format(ee+1)]=ee+1
                            edict["Zone Equipment {0} Heating or No-Load Sequence".format(ee+1)]=ee+1
                            ee+=1
                    edict['Name']=zone.name+'_equiplist'
                    op=iddobj.con_obj(edict)
                    inf.write(op+"\n")
                    
                    iddobj=idds['ZoneHVAC:EquipmentConnections']
                    sdict={}
                    
                    sdict['Zone Name']=zone.name
                    sdict['Zone Conditioning Equipment List Name']=zone.name+'_equiplist'
                    if zone.hasfan:
                        sdict['Zone Air Exhaust Node or NodeList Name']=zone.name+'_Exhaust'
                    sdict['Zone Air Node Name']=zone.name+'_Air'
                    sdict['Zone Return Air Node Name']=zone.name+'_Outlet'
                    op=iddobj.con_obj(sdict)
                    inf.write(op+"\n")
            
            if len(building.schedules)>0:
                for schedule in building.schedules:
                                                                    
                    iddobj=idds['Schedule:Compact']
                    sdict={}
                    sdict['Name']=schedule.name
                    sdict['Schedule Type Limits Name']=schedule.type
                    for ff,hh in enumerate(schedule.hours):
                        sdict["Field {0}".format(ff+1)]=hh
                    op=iddobj.con_obj(sdict)
                    inf.write(op+"\n")
                    

            reports=run.props['reports']
            for report in reports:
                
                if report.zone in building.zones.keys() or "Site" in report.output  or ("Surface" in report.output and report.zone.split("_L")[0] in building.zones.keys()):

                    iddobj=idds['Output:Variable']
                    op=iddobj.con_obj({
                        'Key Value':report.zone,
                        'Variable Name':report.output,
                        'Reporting Frequency':report.freq
                        })
                    inf.write(op+"\n")
            
            #prevent shading problem
            inf.write("Output:Diagnostics,DoNotMirrorDetachedShading; \n")
                
            if run.OutputAll:
            
                iddobj=idds['Output:VariableDictionary']
                op=iddobj.con_obj({'Key Field':'regular'})
                inf.write(op+"\n")
            
                iddobj=idds['Output:Surfaces:Drawing']
                op=iddobj.con_obj({
                    'Report Type':'DXF',
                    'Report Specifications 1':'RegularPolyline'
                    })
                inf.write(op+"\n")
                
                iddobj=idds['Output:Diagnostics']
                op=iddobj.con_obj({
                    'Key 1':'DisplayExtraWarnings'
                    })
                inf.write(op+"\n")
            

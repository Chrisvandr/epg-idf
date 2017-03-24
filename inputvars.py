"""File containing hard coded input parameter"""

diffthresh  =   0.0000001   ###used in building and zones, creates a new surface if  and permeability...
                            # PDB Used as a very small number of various tests involving floats, 
                            # where identical to 0.0 can not be guaranteed.
                            # 1/ Test to make sure two surfaces are in the same plane
                            # 2/ Assign an airflow to an INTERNAL surface that does not 
                            # already have an airflow.
                            # 3/ Generally test that a float value has been set. Mainly for permeabilities.
                            #
air_density =   1.204       ###for calculating FlowC
air_heat_capacity = 1004
perm_to_ach     = 20
delta_T = 27                ##Change between indoor and outdoor for the UK
tolldiff    =  0.00001      ###

perm_height=0.1             ###Height of crack put in for permeability

###Effective leakage areas
ELA_area    =   0.025       ###as it says
ELA_height  =   0         ###as it says

### Calculate the air flow through a crack still not quite convinced...
flow_exp	=	0.66	##Air Mass Flow Exponent (used a lot in 
def calc_flowC(permeability):
    return air_density*permeability/ (60. * 60. * (50.**flow_exp))  ##50pa - pressure used in blower test

###Calculates the trickle vent Effective Leakage Area. Numbers come from: http://www.planningportal.gov.uk/uploads/br/BR_PDF_ADF_2006.pdf
def calc_trickle_ELA(area,n_beds,n_tricks):
    
    calc_ELA = 0.

    ##TODO: sort for flats (quick fix for now which should be fine since the values are the same for 1,2,3 bedrooms)
    if n_beds > 5:
        n_beds = 2
    
    if n_tricks > 0:
        ELAs=[0.0,45000.0,45000.0,45000.0,45000.0,55000.0]
        
        if area < 50.0:
            ELAs=[0.0,25000.0,35000.0,45000.0,45000.0,55000.0]
        elif area < 60.0:
            ELAs=[0.0,25000.0,30000.0,40000.0,45000.0,55000.0]
        elif area < 70.0:
            ELAs=[0.0,30000.0,30000.0,30000.0,45000.0,55000.0]
        elif area < 80.0:
            ELAs=[0.0,35000.0,35000.0,35000.0,45000.0,55000.0]
        elif area < 90.0:
            ELAs=[0.0,40000.0,40000.0,40000.0,45000.0,55000.0]
        elif area < 100.0:
            ELAs=[0.0,45000.0,45000.0,45000.0,45000.0,55000.0]
            

            
        calc_ELA=ELAs[n_beds]
        calc_ELA = calc_ELA + 5000 * (int(area / 10) - 9)
        
        calc_ELA = calc_ELA / (n_tricks * 1000000.0)
        
    
    #print((area, n_beds, calc_ELA))
    
    return calc_ELA

###Build variables, upper and lower limit stuff still in the code
lctv    =   0.04    ##Loads Convergence Tolerance Value
tctv    =   0.4     ##Temperature Convergence Tolerance Value

max_warm_days   =   25  #Maximum Number of Warmup Days
min_warm_days   =   6  #Minimum Number of Warmup Days

###Soil Temps -- PDB. I got these from Ian Ridley. I am not sure where he got them.
# CIBSE perhaps or SAP.

#fcmethod: 7.60,   6.60,   6.20,   6.60,   7.70,   9.20,  10.70,  11.80,  12.20,  11.80,  10.60,   0.00
 
jan_stemp = 8.3
feb_stemp = 6.4
mar_stemp = 5.8
apr_stemp = 6.3
may_stemp = 8.9
jun_stemp = 11.7
jul_stemp = 14.4
aug_stemp = 16.4
sep_stemp = 17
oct_stemp = 16.1
nov_stemp = 13.8
dec_stemp = 11

#manaus
#jan_stemp = 26.0
#feb_stemp = 26.4
#mar_stemp = 26.9
#apr_stemp = 27.2
#may_stemp = 27.7
#jun_stemp = 27.8
#jul_stemp = 27.6
#aug_stemp = 27.2
#sep_stemp = 26.7
#oct_stemp = 26.3
#nov_stemp = 25.9
#dec_stemp = 25.8

###reference condition values
ref_temp 	= 	20	 ##Reference Temperature
ref_press 	= 	101325   ##Reference Barometric Pressure
ref_humratio 	=	0 	 ##Reference Humidity Ratio

###Exhaust Fan 
fan_eff		=	0.5		##Fan Efficiency
press_rise	=	600.0 	##Pressure Rise
max_flow_rate_kitchen	=	0.06	##Maximum Flow Rate
max_flow_rate    =    0.015    ##Maximum Flow Rate

###Leakage Areas
dis_coef	=	1.0	##Discharge Coefficient
press_diff	=	4.0	##Reference Pressure Difference

###Window opening details
flowC_wclosed	=	1.0E-8	##Air Mass Flow Coefficient When Opening is Closed
dis_coef1	=	0.6	##Discharge Coefficient for Opening Factor 1
dis_coef2	=	0.5	##Discharge Coefficient for Opening Factor 2

###Door opening details
flowC_dclosed	=	0.009	##Air Mass Flow Coefficient When Opening is Closed
flow_exp_door	=	0.5	##Air Mass Flow Exponent When Opening is Closed
dif_coefd1	=	0.001	##Discharge Coefficient for Opening Factor 1
dif_coefd2	=	0.78	##Discharge Coefficient for Opening Factor 1

###CO2 generation rate
co2_rate	=	3.82E-8	##Carbon Dioxide Generation Rate

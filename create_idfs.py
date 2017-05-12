__author__ = 'cvdronke'

import sys
import os
import re
import copy
import csv
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
from pyDOE import doe_lhs
import scipy.stats as stats
from collections import defaultdict
import random
import eppy
from time import gmtime, strftime
from eppy import modeleditor
from eppy.modeleditor import IDF

# pathnameto_eppy = 'c:/eppy'
pathnameto_eppy = '../'
sys.path.append(pathnameto_eppy)

if sys.platform=='win32':
    #print("Operating System: Windows")
    #rootdir="/".join(__file__.split('\\')[:-1])
    rootdir=os.path.dirname(os.path.abspath(__file__))
    harddrive_idfs="C:/EngD_hardrive/UCL_DemandLogic"
    idfdir=os.path.dirname(os.path.abspath(__file__))+"\IDFs"
    epdir="C:/EnergyPlusV8-6-0"
else:
    print("rootdir not found")

class DLine(): #use .dline[x] to access different columns in the csv files
    def __init__(self, dline):
        self.dline = dline
        self.name = dline[0]

class Material():
    def __init__(self, dline):  # initialised from line of csv file

        self.dline = dline
        self.name = dline[0]
        self.ep_type = dline[1]
        self.roughness = dline[2]
        self.conductivity = dline[3]
        self.density = dline[4]
        self.specific_heat = dline[5]
        self.thermal_abs = dline[6]
        self.solar_abs = dline[7]
        self.visible_abs = dline[8]
        self.thickness_abs = dline[58]
        self.thermal_res = dline[59]
        self.shgc = dline[60]
        self.ufactor = dline[61]
        ##heat and moisture properties
        self.porosity = dline[9]
        self.init_water = dline[10]
        self.n_iso = dline[11]
        self.sorb_iso = {}
        self.n_suc = dline[14]
        self.suc = {}
        self.n_red = dline[17]
        self.red = {}
        self.n_mu = dline[20]
        self.mu = {}
        self.n_therm = dline[23]
        self.therm = {}
        self.vis_ref = dline[44]

        ##solar properties
        if len(dline) > 31:
            if dline[31] != '':
                self.g_sol_trans = float(dline[31])
                self.g_F_sol_ref = float(dline[32])
                self.g_B_sol_ref = float(dline[33])
                self.g_vis_trans = float(dline[34])
                self.g_F_vis_ref = float(dline[35])
                self.g_B_vis_ref = float(dline[36])
                self.g_IR_trans = float(dline[37])
                self.g_F_IR_em = float(dline[38])
                self.g_B_IR_em = float(dline[39])

        if len(dline) > 39:
            self.ep_special = dline[40]
            if dline[41] != '':
                self.sol_trans = float(dline[41])
                self.sol_ref = float(dline[42])
                self.vis_trans = float(dline[43])

                self.therm_hem_em = float(dline[45])
                self.therm_trans = float(dline[46])
                self.shade_to_glass_dist = float(dline[47])
                self.top_opening_mult = float(dline[48])
                self.bottom_opening_mult = float(dline[49])
                self.left_opening_mult = float(dline[50])
                self.right_opening_mult = float(dline[51])
                self.air_perm = float(dline[52])

        if len(dline) > 53:
            self.ep_special = dline[40]
            if dline[53] != '':
                self.orient = dline[53]
                self.width = float(dline[54])
                self.separation = float(dline[55])
                self.angle = float(dline[56])
                self.blind_to_glass_dist = float(dline[57])

#epdir="C:/Users/cvdronke/Dropbox/01 - EngD/07 - UCL Study/01_CentralHouse\Model"
#csvfile="{0}/{1}/{2}.csv".format(rootdir,indir,fname)
#iddfile="{0}/Energy+.idd".format(epdir)
def read_sheet(rootdir, building_abr, fname=None):
    sheet = []

    if fname != None:
        if building_abr == 'CH':
            csvfile = "{0}/data_files/CentralHouse/{1}".format(rootdir, fname)
        elif building_abr == 'MPEB':
            csvfile = "{0}/data_files/MalletPlace/{1}".format(rootdir, fname)
        print(csvfile)
        with open(csvfile, 'r') as inf:

            for row in inf:
                row = row.rstrip()
                datas = row.split(',')
                sheet.append(datas)
    return sheet
def unpick_equipments(rootdir, building_abr):
    sheet = read_sheet(rootdir, building_abr, fname="equipment_props.csv")

    for nn, lline in enumerate(sheet):
        if len(lline) > 0 and 'Appliance' in lline[0]:
            d_start = nn + 1
            break

    equips = {}
    equip = None

    for dline in sheet[d_start:]:
        if len(dline) > 1:
            if dline[0] != '' and dline[1] != '':
                equip = DLine(dline)
            equips[dline[0]] = equip

    return equips
def unpick_schedules(rootdir, building_abr):
    sheet = read_sheet(rootdir, building_abr, fname="house_scheds.csv")

    for nn, lline in enumerate(sheet):
        if len(lline) > 0 and 'Name' in lline[0]:
            d_start = nn + 1
            break

    sched_groups = {}
    hgroup = None
    for dline in sheet[d_start:]:
        if len(dline) > 1:
            if dline[0] != '':

                #sname = dline[0]
                sched = DLine(dline)
                sched_groups[sched.name] = sched

                # if sname not in list(sched_groups.keys()):
                #     hgroup = sched.HGroup(dline)
                #     sched_groups[sname] = hgroup
                # hgroup = sched_groups[sname]
                #
                # # Loop her over zones so that can reduce the size of the house_sched csv
                # zones = dline[1].split(":")
                # if (len(zones) > 1):
                #     for zone in zones:
                #         zonesched = sched.ZoneSched(dline, zone)
                #         hgroup.zones_scheds.append(zonesched)
                # else:
                #     zonesched = sched.ZoneSched(dline, dline[1])
                #     hgroup.zones_scheds.append(zonesched)

    return sched_groups
def unpick_materials(rootdir, building_abr):
    # open material properties file and puts everything into sheet
    sheet = read_sheet(rootdir, building_abr, fname="mat_props.csv")

    # find the material properties from csv file
    for nn, lline in enumerate(sheet):
        if len(lline) > 0 and 'Name' in lline[0]:
            d_start = nn + 1
            break

    mats = {}
    mat = None
    for dline in sheet[d_start:]:

        if len(dline) > 1:
            ##will have to read some stuff here
            if dline[0] != '' and dline[1] != '':
                mat = Material(dline)
                mats[mat.name] = mat

    return mats

def replace_schedules(run_file, input_values, input_names, var_num, run_no, building_abr):

    # IN ORDER
    # 1. ScheduleTypeLimits
    # 2. Single
    # 3. NoChange
    # 4. TempScheds
    # 5. WaterHeaters
    # 6. Occupancy / Equip / Lights

    scheds = unpick_schedules(rootdir, building_abr)
    scheddict = defaultdict(list)

    if building_abr == 'CH':
        rand = int(round(5*lhd[run_no, var_num]))
        var_num += 1
        if rand == 0:
            HeatingSched = scheds['h_sched_21']
            CoolingSched = scheds['c_sched_23']
            hp, db = 21, 2
        elif rand == 1:
            HeatingSched = scheds['h_sched_21']
            CoolingSched = scheds['c_sched_24']
            hp, db = 21, 3
        elif rand == 2:
            HeatingSched = scheds['h_sched_21']
            CoolingSched = scheds['c_sched_25']
            hp, db = 21, 4
        elif rand == 3:
            HeatingSched = scheds['h_sched_22']
            CoolingSched = scheds['c_sched_24']
            hp, db = 22, 2
        elif rand == 4:
            HeatingSched = scheds['h_sched_22']
            CoolingSched = scheds['c_sched_25']
            hp, db = 22, 3
        elif rand == 5:
            HeatingSched = scheds['h_sched_23']
            CoolingSched = scheds['c_sched_25']
            hp, db = 23, 2
        print(hp, db)
        input_names.append('OfficeHeatingSetPoint')
        input_values.append(hp)
        input_names.append('OfficeHeatingDeadBand')
        input_values.append(db)

    #as inf, open(run_file[:-4]+"s.idf", 'w') as outf
    with open(run_file[:-4]+".idf", 'a') as inf:
        #data = inf.readlines()  # read the whole idf

        timeline = [("0" + str(int(i / 60)) if i / 60 < 10 else str(int(i / 60))) + ":" + (
        "0" + str(i % 60) if i % 60 < 10 else str(i % 60)) for i in range(30, 1470, 30)]
        timeline =  timeline+timeline # one for weekday and weekendday
        #print timeline

        # go through the schedule.csv and assign a sched to a list
        for key in scheds.keys():
            if scheds[key].dline[2] in {'Multiple'}:
                continue #skip heating cooling profiles to be added, are added below

            elif key == 'ScheduleTypeLimits':
                SchedProperties = ['ScheduleTypeLimits', scheds[key].dline[1], scheds[key].dline[8], scheds[key].dline[9], scheds[key].dline[2], scheds[key].dline[3]]
                scheddict[key].append(SchedProperties)

            elif scheds[key].dline[2] in {'Single'}:
                # import schedules that are based on one value and their sigma
                hours = copy.copy(scheds[key].dline[8:])
                if '' in hours:
                    linx=hours.index('')
                    hours=np.array(copy.copy(hours[0:linx]))
                else:
                    hours=np.array(copy.copy(hours))

                new_hours = []
                for i, v in enumerate(hours):
                    sigma = abs(float(scheds[key].dline[4]))
                    mu = abs(float(hours[i]))
                    lower, upper = mu - (3 * sigma), mu + (3 * sigma)
                    if sigma != 0:
                        var_sched = stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu,
                                                    scale=sigma).ppf(lhd[run_no, var_num])
                    else:
                        var_sched = 0
                    new_hours.append(var_sched)

                input_values.append(var_sched)
                input_names.append(key+"_Value")
                print(var_num, scheds[key].dline[0])
                var_num+=1

                SchedProperties = ['Schedule:Compact', scheds[key].name, scheds[key].dline[3]]
                SchedProperties.append('Through: 12/31')
                SchedProperties.append('For: Weekdays WinterDesignDay SummerDesignDay')
                for i, v in enumerate(new_hours):
                    if i == 48:
                        SchedProperties.append('For: Weekends Holiday')
                    SchedProperties.append('Until: ' + timeline[i])
                    if key == 'DeltaTemp_Sched':
                        SchedProperties.append(-v)
                    else:
                        SchedProperties.append(v)
                scheddict[key].append(SchedProperties)

            elif scheds[key].dline[2] in {'NoChange'}:
                hours = copy.copy(scheds[key].dline[8:])
                #print scheds[key].name
                if '' in hours:
                    linx=hours.index('')
                    hours=np.array(copy.copy(hours[0:linx]))
                else:
                    hours=np.array(copy.copy(hours))

                SchedProperties = ['Schedule:Compact', scheds[key].name, scheds[key].dline[3]]
                SchedProperties.append('Through: 12/31')
                SchedProperties.append('For: Weekdays WinterDesignDay SummerDesignDay')
                for i, v in enumerate(hours):
                    if i == 48:
                        SchedProperties.append('For: Weekends Holiday')
                    SchedProperties.append('Until: ' + timeline[i])
                    SchedProperties.append(v)

                scheddict[key].append(SchedProperties)


        # STOCHASTIC HEATING AND COOLING TEMPERATURE SCHEDULES
        if building_abr == 'CH':
            office_c_scheds = ['LectureTheatre_Cooling', 'Meeting_Cooling', 'Office_Cooling', 'PrintRoom_Cooling',
                               'Circulation_Cooling', 'Library_Cooling', 'Kitchen_Cooling', 'ComputerCluster_Cooling', 'Reception_Cooling']
            temp_sched = [CoolingSched, HeatingSched]
            for sched in temp_sched:
                hours = copy.copy(sched.dline[8:])
                if '' in hours:
                    linx = hours.index('')
                    hours = np.array(copy.copy(hours[0:linx]))
                else:
                    hours = np.array(copy.copy(hours))

                if sched == CoolingSched:
                    for y in office_c_scheds:
                        SchedProperties = []
                        SchedProperties = ['Schedule:Compact', y, CoolingSched.dline[3]]
                        SchedProperties.append('Through: 12/31')
                        SchedProperties.append('For: Weekdays WinterDesignDay SummerDesignDay')
                        for i, v in enumerate(hours):
                            if i == 48:
                                SchedProperties.append('For: Weekends Holiday')
                            SchedProperties.append('Until: ' + timeline[i])
                            SchedProperties.append(v)

                        scheddict[y].append(SchedProperties)

                        #print CoolingSched.dline[40], SchedProperties
                elif sched == HeatingSched:
                    for y in office_c_scheds:
                        SchedProperties = []
                        SchedProperties = ['Schedule:Compact', y[:-8] + "_Heating", HeatingSched.dline[3]]
                        SchedProperties.append('Through: 12/31')
                        SchedProperties.append('For: Weekdays WinterDesignDay SummerDesignDay')
                        for i, v in enumerate(hours):
                            if i == 48:
                                SchedProperties.append('For: Weekends Holiday')
                            SchedProperties.append('Until: ' + timeline[i])
                            SchedProperties.append(v)

                        scheddict[y[:-8] + "_Heating"].append(SchedProperties)
                        #print HeatingSched.dline[40], SchedProperties

        #HOT WATER HEATER SCHEDULES
        if building_abr == 'CH':
            hwheaters = ["HWSchedule_Cleaner", "HWSchedule_Kitchenettes", "HWSchedule_Showers", "HWSchedule_Toilets"]
            for heater in hwheaters:
                heater_profile = []
                hprofile = scheds[heater].dline[8:8 + 48+48]
                for i, v in enumerate(hprofile):
                    mu = float(hprofile[i])
                    sigma = mu*20/100
                    lower, upper = 3 * sigma, 1
                    if sigma != 0:
                        hw_sched = stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma).ppf(lhd[run_no, var_num])
                    else:
                        hw_sched = 0
                    heater_profile.append(hw_sched)

                print(var_num, heater, hw_sched)
                var_num += 1

                l = len(heater_profile)
                rand4 = int(round(2 * lhd[run_no, var_num]))

                offset = [1,2,3]
                print(var_num, heater, offset, rand4)
                var_num += 1

                heater_profile_off = []
                for x, val in enumerate(heater_profile):
                    if x > 0 and x < (l - offset[rand4]):
                        if val < heater_profile[x-offset[rand4]]:
                            heater_profile_off.append(heater_profile[x-offset[rand4]])
                            continue
                        if val < heater_profile[x+offset[rand4]]:
                            heater_profile_off.append(heater_profile[x+offset[rand4]])
                            continue
                        else:
                            heater_profile_off.append(heater_profile[x])
                    else:
                        heater_profile_off.append(heater_profile[x])

                input_names.append(heater+'_Offset')
                input_values.append(offset[rand4])

                week_heater_tot, weekend_heater_tot = sum(heater_profile_off[:48]), sum(heater_profile_off[48:96]) #TODO change to sum weekend and weekday? should be in line with coefficient machine learning function
                input_names.append(heater+'_WeekProfile_TotalHours')
                input_values.append(week_heater_tot)
                input_names.append(heater + '_WeekendProfile_TotalHours')
                input_values.append(weekend_heater_tot)

                week_heater_oh, weekend_heater_oh = week_heater_tot - sum(heater_profile_off[13:37]), weekend_heater_tot - sum(heater_profile_off[61:85]) # 7 to 7?
                input_names.append(heater+'_WeekProfile_OH')
                input_values.append(week_heater_oh)
                input_names.append(heater + '_WeekendProfile_OH')
                input_values.append(weekend_heater_oh)

                SchedProperties = []
                for sched, sname in enumerate(heater_profile_off):
                    SchedProperties = ['Schedule:Compact', heater, scheds[heater].dline[3]]
                    SchedProperties.append('Through: 12/31')
                    SchedProperties.append('For: Weekdays WinterDesignDay SummerDesignDay')
                    for i, v in enumerate(heater_profile_off):
                        if i == 48:
                            SchedProperties.append('For: Weekends Holiday')
                        SchedProperties.append('Until: ' + timeline[i])
                        SchedProperties.append(v)
                scheddict[heater].append(SchedProperties)

            # fig = plt.figure()
            # ax = fig.add_subplot(111)
            # ax.plot(hprofile, label='hprofile')
            # ax.plot(heater_profile, label='heater_dev')
            # ax.plot(heater_profile_off, label='heater_off')
            # plt.legend()
            # #plt.show()


        # FOR OCCUPANCY PROFILES TO INSERT STANDARD DEVIATION FROM THE SCHEDULES
        rand2 = int(round(2 * lhd[run_no, var_num]))
        var_num += 1
        week = ['Weekday', 'Weekend']
        equip_week, occ_week, light_week = [], [], []
        for day in week:
            if day == 'Weekday':
                profile = scheds["WifiSum"].dline[8:8 + 48]
                std = [float(a_i) - float(b_i) for a_i, b_i in
                       zip(scheds["WifiSum"].dline[8:8 + 48], scheds["WifiSumMinusStd"].dline[8:8 + 48])]
            elif day == 'Weekend':
                profile = scheds["WifiSum"].dline[8+48:8+48+48]
                std = [float(a_i) - float(b_i) for a_i, b_i in
                       zip(scheds["WifiSum"].dline[8+48:8 + 48+48], scheds["WifiSumMinusStd"].dline[8+48:8 + 48+48])]

            ## EQUIPMENT PROFILES - BASED ON OCCUPANCY ##
            occ_profile = []
            for k in range(2): #zero for light, one for equip
                equip_profile = []
                equip_profile_offset = []
                if k == 0:
                    overtime_hours_perc = [5, 10, 15] #overtime for light percentage*peakvalue
                else:
                    overtime_hours_perc = [20, 40, 60]

                for i, v in enumerate(profile):
                    sigma = abs(std[i])
                    lower, upper = 2*sigma, 1
                    mu = float(profile[i])
                    #print mu, sigma, lower, upper
                    if sigma != 0:
                        var_sched = stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma).ppf(lhd[run_no, var_num])
                    else:
                        var_sched = 0
                    occ_profile.append(var_sched)
                    #print(occ_profile)
                    overtime = float(max(profile))*overtime_hours_perc[rand2]/100

                    if var_sched < overtime:
                        var_sched = overtime
                    equip_profile.append(var_sched)

                var_num += 1 # TODO check if the profiles make sense, previously this var_num was in the for loop for creating a profile, likely causing a very random profile
                print(var_num, var_sched, 'k 0 = light, 1=equip:', k)
                #print var_sched, overtime, max(profile), overtime_hours_perc[rand2]

                ## Creating an offset from previous profile
                rand3 = int(round(2 * lhd[run_no, var_num]))
                print(var_num, 'offset from previous profile:', rand3)
                var_num += 1

                offset = [0, 1, 2, 3]
                l = len(equip_profile)
                for x, val in enumerate(equip_profile):
                    if x > 0 and x < (l - offset[rand3]):
                        if val < equip_profile[x-offset[rand3]]:
                            equip_profile_offset.append(equip_profile[x-offset[rand3]])
                            continue
                        if val < equip_profile[x+offset[rand3]]:
                            equip_profile_offset.append(equip_profile[x+offset[rand3]])
                            continue
                        else:
                            equip_profile_offset.append(equip_profile[x])
                    else:
                        equip_profile_offset.append(equip_profile[x])

                if k == 0:
                    light_profile_offset = equip_profile_offset
                    light_week.extend(light_profile_offset)
                    overtime_hours_light = overtime_hours_perc[rand2]
                elif k == 1:
                    equip_week.extend(equip_profile_offset)

        #print(len(equip_profile_offset), equip_profile_offset) # offset is the one I want to have i think, but it seems that it is only 48 long now (and low so for the weekend?)
        occ_week.extend(occ_profile[0:47]) #occ profile and occ_week exactly the same????
        #print(len(occ_week)) # 47 long?

        # fig = plt.figure()
        # ax = fig.add_subplot(111)
        # #ax.plot(equip_week, label='equip')
        # ax.plot(equip_profile, label='light')
        # ax.plot(equip_profile_offset, label='light_offset')
        # ax.plot(occ_week, label='occ')
        # ax.plot(occ_profile, label='occ_profile')
        # plt.legend()
        # plt.show()

        #print 'offset [no*30Min] = ', offset[rand3]
        #print 'overtime equip multiplier [%] = ', overtime_hours_perc[rand2]
        #print 'overtime light multiplier [%] = ', overtime_hours_light

        office_scheds = [occ_profile, equip_profile_offset, light_profile_offset]

        #print len(light_profile_offset)
        office_scheds_names = ['Office_OccSched', 'Office_EquipSched', 'Office_LightSched'] # has to align with previous profiles
        #print len(timeline)

        for sched, sname in enumerate(office_scheds_names):
            SchedProperties = []
            SchedProperties = ['Schedule:Compact', sname, 'Fraction']
            SchedProperties.append('Through: 12/31')
            SchedProperties.append('For: Weekdays WinterDesignDay SummerDesignDay')

            # TODO there is no weekendday schedule for equipment and lighting in the CentralHouse idfs???
            for i, v in enumerate(office_scheds[sched]):
                if i == 48:
                    SchedProperties.append('For: Weekends Holiday')

                SchedProperties.append('Until: ' + timeline[i])
                SchedProperties.append(v)
            scheddict[sname].append(SchedProperties)

            week_sched_tot, weekend_sched_tot = sum(office_scheds[sched][:48]), sum(office_scheds[sched][48:96])
            input_names.append(sname+'_WeekProfile_TotalHours')
            input_values.append(week_sched_tot)
            input_names.append(sname + '_WeekendProfile_TotalHours')
            input_values.append(weekend_sched_tot)

            week_sched_oh, weekend_sched_oh = week_sched_tot - sum(office_scheds[sched][13:37]), weekend_sched_tot - sum(office_scheds[sched][61:85]) # 7 to 7?
            input_names.append(sname+'_WeekProfile_OH')
            input_values.append(week_sched_oh)
            input_names.append(sname + '_WeekendProfile_OH')
            input_values.append(weekend_sched_oh)

        #print scheddict['PrintRoom_Cooling'][0]
        #print scheddict.keys()
        #print scheddict['Office_OccSched'][0][1]

        # Write to idf file
        #print len(scheddict[key][0])
        for key in scheddict.keys():
            for i,v in enumerate(scheddict[key][0]):
                #print v


                if i + 1 == len(scheddict[key][0]):  # if last element in list then put semicolon
                    inf.write('\n')
                    inf.write(str(v) + ';')
                    inf.write('\n\n')
                else:
                    inf.write('\n')
                    inf.write(str(v) + ',')

    return input_values, input_names, var_num

def remove_schedules():
    # remove existing schedules
    schedule_types = ["SCHEDULE:DAY:INTERVAL", "SCHEDULE:WEEK:DAILY", "SCHEDULE:YEAR"]
    for y in schedule_types:
        existing_scheds = idf1.idfobjects[y]
        print(len(existing_scheds), "existing schedule objects removed in ", y)
        length_object = len(existing_scheds)
        for i in range(0, length_object):
            idf1.popidfobject(y, 0)

def remove_existing_outputs():
    output_types = ["OUTPUT:VARIABLE", "OUTPUT:METER:METERFILEONLY"]
    for y in output_types:
        existing_outputs = idf1.idfobjects[y]
        print(len(existing_outputs), "existing output objects removed in", y)
        length_object = len(existing_outputs)
        if length_object == 0:
            print("no existing outputs to remove in", y)
        for i in range(0, length_object):
            idf1.popidfobject(y,0)

def replace_materials_eppy(input_values, input_names, var_num, run_no, building_abr):
    mats = unpick_materials(rootdir, building_abr)

    print(mats.keys())
    materials, material_types = [], []
    for key, val in mats.items():
        if key != '' and val != '':
            materials.append(key)
            material_types.append(val.dline[1].upper())
            material_types = list(set(material_types))  # make unique names
    print(materials)
    print(material_types)

    mats_types = ['MATERIAL', 'MATERIAL:NOMASS', 'MATERIAL:AIRGAP', 'WINDOWMATERIAL:SIMPLEGLAZINGSYSTEM','WINDOWMATERIAL:BLIND','WINDOWPROPERTY:SHADINGCONTROL']

    for mats_type in material_types: # For every material type, create an object with its material then run through loop
        mats_idf = idf1.idfobjects[mats_type]

        for material in mats_idf: # For each material in object replace content with that defined in csv files
            if material.Name not in mats.keys():
                continue
            else:
                sigma = float(mats[material.Name].dline[62])
                lower, upper = float(mats[material.Name].dline[63]), float(mats[material.Name].dline[64])
                input_names.append(mats[material.Name].name)
                if mats[material.Name].name == material.Name and mats[material.Name].dline[1] == 'Material:NoMass':
                    material.Roughness = mats[material.Name].roughness
                    material.Thermal_Absorptance = mats[material.Name].thermal_abs
                    material.Solar_Absorptance = mats[material.Name].solar_abs
                    material.Visible_Absorptance = mats[material.Name].visible_abs
                    mu = float(mats[material.Name].thermal_res)
                    if sigma > 0:
                        eq_mats = stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma).ppf(lhd[run_no, var_num])

                    else:
                        eq_mats = 0
                    material.Thermal_Resistance = eq_mats
                    input_values.append(eq_mats)
                elif mats[material.Name].name == material.Name and mats[material.Name].dline[1] == 'Material:AirGap':
                    mu = float(mats[material.Name].thermal_res)
                    if sigma > 0:
                        eq_mats = stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma).ppf(lhd[run_no, var_num])
                    else:
                        eq_mats = 0
                    material.Thermal_Resistance = eq_mats
                    input_values.append(eq_mats)
                elif mats[material.Name].name == material.Name and mats[material.Name].dline[1] == 'Material':
                    material.Roughness = mats[material.Name].roughness
                    material.Thickness = mats[material.Name].thickness_abs
                    material.Density = mats[material.Name].density
                    material.Specific_Heat = mats[material.Name].specific_heat
                    material.Thermal_Absorptance = mats[material.Name].thermal_abs
                    material.Solar_Absorptance = mats[material.Name].solar_abs
                    material.Visible_Absorptance = mats[material.Name].visible_abs
                    mu = float(mats[material.Name].conductivity)
                    if sigma > 0:
                        eq_mats = stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma).ppf(lhd[run_no, var_num])
                    else:
                        eq_mats = 0
                    material.Conductivity = eq_mats
                    input_values.append(eq_mats)

                elif mats[material.Name].name == material.Name and mats[material.Name].dline[1] == 'WindowMaterial:SimpleGlazingSystem':
                    material.UFactor = mats[material.Name].dline[61] #ufactor
                    material.Solar_Heat_Gain_Coefficient = mats[material.Name].shgc
                    material.Visible_Transmittance = mats[material.Name].vis_ref

                    mu = float(mats[material.Name].dline[61]) #ufactor
                    if sigma > 0:
                        eq_mats = stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma).ppf(lhd[run_no, var_num])
                    else:
                        eq_mats = 0
                    material.UFactor = eq_mats
                    input_values.append(eq_mats)

                elif mats[material.Name].name == material.Name and mats[material.Name].dline[1] == 'WindowMaterial:Blind':
                    mu = float(mats[material.Name].conductivity)
                    if sigma > 0:
                        eq_mats = stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma).ppf(
                            lhd[run_no, var_num])
                    else:
                        eq_mats = 0
                    material.Slat_Conductivity = eq_mats
                    input_values.append(eq_mats)

                elif mats[material.Name].name == material.Name and mats[material.Name].dline[1] == 'WindowProperty:ShadingControl':
                    mu = float(mats[material.Name].conductivity) # is actually solar radiation set point control
                    if sigma > 0:
                        eq_mats = stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma).ppf(
                            lhd[run_no, var_num])
                    else:
                        eq_mats = 0
                    material.Setpoint = eq_mats
                    input_values.append(eq_mats)
                print(var_num, mats[material.Name].name, round(eq_mats,3))
                var_num +=1

    return input_values, input_names, var_num


def replace_equipment_eppy(input_values, input_names, var_num, run_no, building_abr):
    equips = unpick_equipments(rootdir, building_abr)
    appliances, app_purposes = [], []
    for key, val in equips.items():
        appliances.append(key)
        app_purposes.append(val.dline[1].upper())
    app_purposes = list(set(app_purposes)) #make unique names
    print(appliances)
    print(app_purposes)

    x = 0
    for equip_type in app_purposes: # For every type, create an object with its material then run through loop
        equip_idf = idf1.idfobjects[equip_type]

        print(equip_type)
        for equip in equip_idf: # For each instance of object replace content with that defined in csv files
            # for all ventilation objects, change to the same value
            #print(equip)
            if equip_type == 'ZONEVENTILATION:DESIGNFLOWRATE':
                len_zonevent = len(idf1.idfobjects[equip_type])
                object_name = "ZoneVentilation"

                equip.Design_Flow_Rate_Calculation_Method = equips[object_name].dline[19]
                sigma = float(equips[object_name].dline[30])
                lower, upper = float(equips[object_name].dline[31]), float(equips[object_name].dline[32])
                mu = float(equips[object_name].dline[27])
                if sigma > 0:
                    eq_vent = stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma).ppf(lhd[run_no, var_num])
                    equip.Flow_Rate_per_Person = eq_vent  # dline[19]
                    x += 1
                else:
                    eq_vent = 0
                if x == len_zonevent:
                    equip.Flow_Rate_per_Person = eq_vent  # dline[19]
                    input_values.append(eq_vent)
                    input_names.append(object_name)
                    print(eq_vent, var_num)
                    var_num += 1

            elif equip_type == 'FAN:ZONEEXHAUST':
                len_objects = len(idf1.idfobjects[equip_type])
                object_name = "ExhaustFans"

                sigma = float(equips[object_name].dline[30])
                lower, upper = float(equips[object_name].dline[31]), float(equips[object_name].dline[32])
                mu = float(equips[object_name].dline[7])
                if sigma > 0:
                    eq_efficiency = stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma).ppf(
                        lhd[run_no, var_num])
                    equip.Fan_Total_Efficiency = eq_efficiency
                    x+=1
                else:
                    eq_efficiency = 0
                if x == len_objects:
                    equip.Fan_Total_Efficiency = eq_efficiency
                    input_values.append(eq_efficiency)
                    input_names.append(object_name)
                    #print(eq_efficiency, var_num)
                    var_num += 1

            else:
                if equip_type == 'AIRCONDITIONER:VARIABLEREFRIGERANTFLOW':
                    equip_name = equip.Heat_Pump_Name
                else:
                    equip_name = equip.Name

                #print('name', equips[equip_name].name, equips[equip_name].purpose)

                if equips[equip_name].name == equip_name and equips[equip_name].dline[1] == 'ElectricEquipment':
                    equip.Design_Level_Calculation_Method = equips[equip_name].dline[19] #dline[19]
                    sigma = float(equips[equip_name].dline[30])
                    lower, upper = float(equips[equip_name].dline[31]), float(equips[equip_name].dline[32])
                    mu = float(equips[equip_name].dline[21])
                    if sigma > 0:
                        eq_equip = stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma).ppf(lhd[run_no, var_num])
                    else:
                        eq_equip = 0
                    equip.Watts_per_Zone_Floor_Area = eq_equip
                    input_values.append(eq_equip)
                    input_names.append(equips[equip_name].name)
                    var_num += 1

                elif equips[equip_name].name == equip_name and equips[equip_name].dline[1] == 'People':
                    equip.Number_of_People_Calculation_Method = equips[equip_name].dline[19] #dline[19]
                    sigma = float(equips[equip_name].dline[30])
                    lower, upper = float(equips[equip_name].dline[31]), float(equips[equip_name].dline[32])
                    mu = float(equips[equip_name].dline[25])
                    if sigma > 0:
                        eq_people = stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma).ppf(lhd[run_no, var_num])
                    else:
                        eq_people = 0
                    equip.People_per_Zone_Floor_Area = eq_people
                    input_values.append(eq_people)
                    input_names.append(equips[equip_name].name)
                    var_num += 1

                elif equips[equip_name].name == equip_name and equips[equip_name].dline[1] == 'Lights':
                    equip.Design_Level_Calculation_Method = equips[equip_name].dline[19]  # dline[19]
                    sigma = float(equips[equip_name].dline[30])
                    lower, upper = float(equips[equip_name].dline[31]), float(equips[equip_name].dline[32])
                    mu = float(equips[equip_name].dline[21])
                    if sigma > 0:
                        eq_light = stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma).ppf(lhd[run_no, var_num])
                    else:
                        eq_light = 0
                    equip.Watts_per_Zone_Floor_Area = eq_light
                    input_values.append(eq_light)
                    input_names.append(equips[equip_name].name)
                    var_num += 1

                elif equips[equip_name].name == equip_name and equips[equip_name].dline[1] == 'ZoneInfiltration:DesignFlowRate':
                    ## TODO Should infiltration values be the same everywhere, each spacetype.
                    equip.Design_Flow_Rate_Calculation_Method = equips[equip_name].dline[19]  # dline[19]
                    sigma = float(equips[equip_name].dline[30])
                    lower, upper = float(equips[equip_name].dline[31]), float(equips[equip_name].dline[32])
                    mu = float(equips[equip_name].dline[26])
                    if sigma > 0:
                        eq_infil = stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma).ppf(lhd[run_no, var_num])
                    else:
                        eq_infil = 0
                    equip.Flow_per_Exterior_Surface_Area = eq_infil
                    input_values.append(eq_infil)
                    input_names.append(equips[equip_name].name)
                    var_num += 1

                elif equips[equip_name].name == equip_name and equips[equip_name].dline[1] == 'DesignSpecification:OutdoorAir':
                    equip.Outdoor_Air_Method = equips[equip_name].dline[19]  # dline[19]
                    sigma = float(equips[equip_name].dline[30])
                    lower, upper = float(equips[equip_name].dline[31]), float(equips[equip_name].dline[32])
                    mu = float(equips[equip_name].dline[28])
                    if sigma > 0:
                        eq_oa = stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma).ppf(lhd[run_no, var_num])
                    else:
                        eq_oa = 0
                    equip.Outdoor_Air_Flow_Air_Changes_per_Hour = eq_oa
                    input_values.append(eq_oa)
                    input_names.append(equips[equip_name].name)
                    var_num += 1

                elif equips[equip_name].name == equip_name and equips[equip_name].dline[1] == 'AirConditioner:VariableRefrigerantFlow':
                    print(equips[equip_name].name) # equips[equip_name].dline

                    #ccop
                    mu = float(equips[equip_name].dline[5])
                    sigma = float(equips[equip_name].dline[33])/100*mu
                    print('sigma in AC units', sigma)
                    lower, upper = mu - (3 * sigma), mu + (3 * sigma)
                    ccop = stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma).ppf(lhd[run_no, var_num])
                    var_num += 1

                    #hcop
                    mu = float(equips[equip_name].dline[6])
                    sigma = float(equips[equip_name].dline[33]) / 100 * mu
                    lower, upper = mu - (3 * sigma), mu + (3 * sigma)
                    hcop = stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma).ppf(
                        lhd[run_no, var_num])
                    var_num += 1

                    equip.Gross_Rated_Cooling_COP = ccop
                    input_values.append(ccop)
                    input_names.append(equips[equip_name].name+"ccop")
                    equip.Gross_Rated_Heating_COP = hcop
                    input_values.append(hcop)
                    input_names.append(equips[equip_name].name+"hcop")


                elif equips[equip_name].name == equip_name and equips[equip_name].dline[1] == 'Boiler:HotWater':
                    sigma = float(equips[equip_name].dline[30])
                    lower, upper = float(equips[equip_name].dline[31]), float(equips[equip_name].dline[32])
                    mu = float(equips[equip_name].dline[7])
                    if sigma > 0:
                        input = stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma).ppf(lhd[run_no, var_num])
                    else:
                        input = 0
                    equip.Nominal_Thermal_Efficiency = input
                    input_values.append(input)
                    input_names.append(equips[equip_name].name)
                    var_num += 1

                elif equips[equip_name].name == equip_name and equips[equip_name].dline[1] == 'WaterUse:Equipment':
                    sigma = float(equips[equip_name].dline[30])
                    lower, upper = float(equips[equip_name].dline[31]), float(equips[equip_name].dline[32])
                    mu = float(equips[equip_name].dline[8])
                    if sigma > 0:
                        input = stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma).ppf(lhd[run_no, var_num])
                    else:
                        input = 0
                    equip.Peak_Flow_Rate = input
                    input_values.append(input)
                    input_names.append(equips[equip_name].name)
                    var_num += 1

                else:
                    continue

                print(var_num, equip_name)
                #input_names.append(equips[equip_name].name)
    return input_values, input_names, var_num

def add_groundtemps():
    out = idf1.newidfobject("Site:GroundTemperature:BuildingSurface".upper())
    out.January_Ground_Temperature = 8.3
    out.February_Ground_Temperature = 6.4
    out.March_Ground_Temperature =5.8
    out.April_Ground_Temperature =6.3
    out.May_Ground_Temperature = 8.9
    out.June_Ground_Temperature =11.7
    out.July_Ground_Temperature =14.4
    out.August_Ground_Temperature =16.4
    out.September_Ground_Temperature =17
    out.October_Ground_Temperature =16.1
    out.November_Ground_Temperature =13.8
    out.December_Ground_Temperature =11

def add_outputs(): # add outputvariables and meters as new idf objects
    output_variables = ["Site Outdoor Air Drybulb Temperature",
                        "Site Direct Solar Radiation Rate per Area",
                        "Site Diffuse Solar Radiation Rate per Area",
                        "Site Outdoor Air Relative Humidity"]
    # "Pump Electric Energy",
    # "Fan Electric Power",
    # "Zone Ventilation Mass Flow Rate",
    # "Zone Infiltration Mass Flow Rate",
    # "VRF Heat Pump Cooling Electric Energy",
    # "VRF Heat Pump Heating Electric Energy",
    # "VRF Heat Pump Cooling Electric Energy",
    # "VRF Heat Pump Heating Electric Energy",
    # "VRF Heat Pump Operating Mode",
    # "Zone VRF Air Terminal Cooling Electric Energy",
    # "Zone VRF Air Terminal Heating Electric Energy",
    # "Zone Cooling Setpoint Not Met Time",
    # "Zone Heating Setpoint Not Met Time",
    # "Zone Thermostat Heating Setpoint Temperature",
    # "Zone Thermostat Cooling Setpoint Temperature",
    # "Zone Air Temperature",

    for name in output_variables:
        outvar = idf1.newidfobject("Output:Variable".upper())
        outvar.Key_Value = ''
        outvar.Variable_Name = name
        outvar.Reporting_Frequency = 'hourly'

    output_meters = ["Fans:Electricity",
                     "InteriorLights:Electricity",
                     "InteriorEquipment:Electricity",
                     "WaterSystems:Electricity",
                     "Cooling:Electricity",
                     "Heating:Electricity",
                     "Gas:Facility",
                     "Pumps:Electricity"]

    for name in output_meters:
        outmeter = idf1.newidfobject("Output:Meter:MeterFileOnly".upper())
        outmeter.Name = name
        outmeter.Reporting_Frequency = 'hourly'

# Remove all comments from idf file so as to make them smaller.
def remove_comments(run_file):
    with open(run_file, 'r+') as f:
        data = f.readlines()
        for num, line in enumerate(data):
            data[num] = re.sub(re.compile("!-.*?\n"), "\n", data[num])  # remove all occurances singleline comments (!-COMMENT\n ) from string
            #print data[num]
        f.seek(0) #move to front of file
        f.writelines(data)
        f.truncate()

def run_lhs(idf1, building_name, building_abr):
    idf1.popidfobject('Output:SQLite'.upper(), 0) # remove sql output, have all the outputs in the .eso and meter data in .mtr
    idf1.popidfobject('Output:VariableDictionary'.upper(), 0)
    idf1.popidfobject('Output:Table:SummaryReports'.upper(), 0)
    idf1.popidfobject('OutputControl:Table:Style'.upper(), 0)
    #TODO don't create html and eso outputs...
    #ifd1.popidfobject('Output:Table:SummaryReports'.upper(), 0)
    #http: // bigladdersoftware.com / epx / docs / 8 - 0 / input - output - reference / page - 088.
    #html  # outputmeter-and-outputmetermeterfileonly
    # change the base idf first by adding ground temps, and removing existing objects before adding new ones.

    add_groundtemps()
    remove_schedules()
    remove_existing_outputs()
    add_outputs()

    # Explanation
    # 1. replace_materials_eppy uses mat_props.csv, replace_equipment_eppy uses equipment_props.csv, replace_schedules uses house_scheds.csv
    # 2. adds ground temperatures and output variables
    # 3. define number of idf files to be created and base model file name
    # 4. the functions sample the inputs using lhs, inputs are in the csv files to be manually copied
    # 5. then create separate idf files

    collect_inputs = []
    for run_no in range(n_samples):
        var_num = 0
        input_names = []
        input_values = []

        input_values, input_names, var_num = replace_equipment_eppy(input_values,
                                                                    input_names,
                                                                    var_num,
                                                                    run_no,
                                                                    building_abr)

        var_equip = var_num
        print("number of variables changed for equipment ", var_equip)
        input_values, input_names, var_num = replace_materials_eppy(input_values,
                                                                    input_names,
                                                                    var_num,
                                                                    run_no,
                                                                    building_abr)

        var_mats = var_num-var_equip
        print("number of variables changed for materials ", var_mats)

        # TODO output all used schedules to a file for easy review
        # format the run_no to zero pad to be four values always
        if building_abr == 'CH':
            save_dir = harddrive_idfs+"/01_CentralHouse_Project/IDFs/"
        elif building_abr == 'MPEB':
            save_dir = harddrive_idfs + "05_MaletPlaceEngineering_Project/IDFs/"

        if not os.path.exists(save_dir): # check if folder exists
            os.makedirs(save_dir) # create new folder
        run_file = save_dir+"/"+building_name+"_" + str(format(run_no, '04')) + ".idf" # use new folder for save location, zero pad to 4 numbers

        idf1.saveas(run_file)
        remove_comments(run_file)

        #replace_schedules append to file instead of using eppy, it will open the written idf file
        var_scheds = var_num-var_mats-var_equip
        input_values, input_names, var_num = replace_schedules(run_file,
                                                               input_values,
                                                               input_names,
                                                               var_num,
                                                               run_no,
                                                               building_abr) #create_schedules = the heating and coolign profiles. Which are not to be created for MPEB (however, the remove_schedules shouldn't delete existing ones. Or create them...
        print("number of variables changed for schedules", var_scheds)

        collect_inputs.append(input_values)

        print("total number of variables ", var_num)
        print(input_names)
        print(input_values)

    #Write inputs to csv file
    collect_inputs.insert(0, input_names) # prepend names to list
    csv_outfile = save_dir+"/inputs_" + building_name + strftime("%d_%m_%H_%M", gmtime()) + ".csv"
    with open(csv_outfile, 'w') as outf:
        writer = csv.writer(outf, lineterminator='\n')
        writer.writerows(collect_inputs)

#uniform dist

# TODO rewrite to SOOBOL/?
iddfile = "C:/EnergyPlusV8-6-0/Energy+.idd"
IDF.setiddname(iddfile)

# 'MalletPlace_139', 'CentralHouse_222' # building_name
# 'CH', 'MPEB' # building_abr
building_abr = 'CH'
building_name = 'CentralHouse_222'
idf1 = IDF("{}".format(rootdir) + "/" + building_name + ".idf")

n_samples = 300
if building_abr == 'MPEB':
    no_variables = 300
elif building_abr == 'CH':
    no_variables = 150
lhd = doe_lhs.lhs(no_variables, samples=n_samples)
#print lhd

run_lhs(idf1, building_name, building_abr)

## TODO When I have set up all the replacements blocks of text and values, I need to create the Latin hypercube sampling before creating every new idf, based on the input values.
## TODO Or sobol:
## TODO https://github.com/naught101/sobol_seq, https://github.com/crankycoder/sobol_seq
## TODO which outputs have to be defined and used for sensitivity analysis/calibration?

# print idf1.idfobjects['PEOPLE'][0].objls #fieldnames
# print idf1.idfobjects['ELECTRICEQUIPMENT'][0].objls #fieldnames
# print idf1.idfobjects['LIGHTS'][0].fieldnames


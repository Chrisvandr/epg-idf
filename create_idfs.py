__author__ = 'cvdronke'

import sys
import os
import re
import copy
import csv
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import pandas as pd
from pyDOE import doe_lhs
import scipy.stats as stats
from collections import defaultdict
import random
import eppy
from time import gmtime, strftime
from eppy import modeleditor
from eppy.modeleditor import IDF
import getpass
# pathnameto_eppy = 'c:/eppy'
pathnameto_eppy = '../'
sys.path.append(pathnameto_eppy)
UserName = getpass.getuser()

#own scripts
sys.path.append('C:/Users/' + UserName + '\Dropbox/01 - EngD/07 - UCL Study/UCLDataScripts')
from PlotSubmetering import lineno

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
                #self.therm_hem_em = float(dline[45])
                #self.therm_trans = float(dline[46])
                # self.shade_to_glass_dist = float(dline[47])
                # self.top_opening_mult = float(dline[48])
                # self.bottom_opening_mult = float(dline[49])
                # self.left_opening_mult = float(dline[50])
                # self.right_opening_mult = float(dline[51])
                # self.air_perm = float(dline[52])

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
        elif building_abr == '71':
            csvfile = "{0}/data_files/BH71/{1}".format(rootdir, fname)

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
    print(lineno(), [v for v in equips.keys()])
    # exit()

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

def replace_schedules(run_file, lhd, input_values, input_names, var_num, run_no, building_abr, base_case, seasonal_occ_factor_week, seasonal_occ_factor_weekend, overtime_multiplier_equip, overtime_multiplier_light, multiplier_variation):

    # IN ORDER
    # 1. ScheduleTypeLimits
    # 2. Single
    # 3. NoChange
    # 4. TempScheds
    # 5. WaterHeaters
    # 6. Occupancy / Equip / Lights

    scheds = unpick_schedules(rootdir, building_abr)
    scheddict = defaultdict(list)
    # todo scheds for MPEB or stick to those from O&M

    timeline = [("0" + str(int(i / 60)) if i / 60 < 10 else str(int(i / 60))) + ":" + (
        "0" + str(i % 60) if i % 60 < 10 else str(i % 60)) for i in range(30, 1470, 30)]
    timeline = timeline + timeline  # one for weekday and weekendday
    # print timeline

    #Randomly pick a schedule for zone set-point temperatures
    if building_abr in {'CH'}: #'MPEB'
        rand = int(round(21*lhd[run_no, var_num])) # the number signifies how many options there are minus 1?
        if base_case == True:
            #rand = 4 # 23,23
            rand = 11 # 23,24
        var_num += 1

        if rand == 0:
            HeatingSched = scheds['h_sched_225']
            CoolingSched = scheds['c_sched_225']
            hp, db = 22, 0

        elif rand == 1:
            HeatingSched = scheds['h_sched_225']
            CoolingSched = scheds['c_sched_225']
            hp, db = 22.5, 0

        elif rand == 2:
            HeatingSched = scheds['h_sched_23']
            CoolingSched = scheds['c_sched_23']
            hp, db = 23, 0

        elif rand == 3:
            HeatingSched = scheds['h_sched_215']
            CoolingSched = scheds['c_sched_22']
            hp, db = 21.5, 0.5

        elif rand == 4:
            HeatingSched = scheds['h_sched_22']
            CoolingSched = scheds['c_sched_225']
            hp, db = 22, 0.5

        elif rand == 5:
            HeatingSched = scheds['h_sched_225']
            CoolingSched = scheds['c_sched_23']
            hp, db = 225, 0.5

        elif rand == 6:
            HeatingSched = scheds['h_sched_23']
            CoolingSched = scheds['c_sched_235']
            hp, db = 23, 0.5

        elif rand == 7:
            HeatingSched = scheds['h_sched_21']
            CoolingSched = scheds['c_sched_22']
            hp, db = 21, 1

        elif rand == 8:
            HeatingSched = scheds['h_sched_215']
            CoolingSched = scheds['c_sched_225']
            hp, db = 21.5, 1

        elif rand == 9:
            HeatingSched = scheds['h_sched_22']
            CoolingSched = scheds['c_sched_23']
            hp, db = 22, 1

        elif rand == 10:
            HeatingSched = scheds['h_sched_225']
            CoolingSched = scheds['c_sched_235']
            hp, db = 22.5, 1

        elif rand == 11:
            HeatingSched = scheds['h_sched_23']
            CoolingSched = scheds['c_sched_24']
            hp, db = 23, 1

        elif rand == 12:
            HeatingSched = scheds['h_sched_21']
            CoolingSched = scheds['c_sched_225']
            hp, db = 21, 1.5

        elif rand == 13:
            HeatingSched = scheds['h_sched_215']
            CoolingSched = scheds['c_sched_23']
            hp, db = 21.5, 1.5

        elif rand == 14:
            HeatingSched = scheds['h_sched_22']
            CoolingSched = scheds['c_sched_235']
            hp, db = 22, 1.5

        elif rand == 15:
            HeatingSched = scheds['h_sched_225']
            CoolingSched = scheds['c_sched_24']
            hp, db = 22.5, 1.5

        elif rand == 16:
            HeatingSched = scheds['h_sched_23']
            CoolingSched = scheds['c_sched_245']
            hp, db = 23, 1.5

        elif rand == 17:
            HeatingSched = scheds['h_sched_21']
            CoolingSched = scheds['c_sched_23']
            hp, db = 21, 2

        elif rand == 18:
            HeatingSched = scheds['h_sched_215']
            CoolingSched = scheds['c_sched_235']
            hp, db = 21.5, 2

        elif rand == 19:
            HeatingSched = scheds['h_sched_22']
            CoolingSched = scheds['c_sched_24']
            hp, db = 22, 2

        elif rand == 20:
            HeatingSched = scheds['h_sched_225']
            CoolingSched = scheds['c_sched_245']
            hp, db = 22.5, 2

        elif rand == 21:
            HeatingSched = scheds['h_sched_23']
            CoolingSched = scheds['c_sched_25']
            hp, db = 23, 2

        print(hp, db)
        input_names.append('OfficeHeatingSetPoint')
        input_values.append(hp)
        input_names.append('OfficeHeatingDeadBand')
        input_values.append(db)

    #as inf, open(run_file[:-4]+"s.idf", 'w') as outf
    print('run_file', run_file[:-4])
    with open(run_file[:-4]+".idf", 'a') as inf:
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
                #print(lineno(), scheds[key].dline[0], scheds[key].dline[4], type(scheds[key].dline[4]))
                sigma = abs(float(scheds[key].dline[4]))
                for i, v in enumerate(hours):
                    if base_case is True:
                        sigma=0
                    mu = abs(float(hours[i]))
                    lower, upper = mu - (3 * sigma), mu + (3 * sigma)
                    if sigma != 0:
                        var_sched = stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu,
                                                    scale=sigma).ppf(lhd[run_no, var_num])
                    else:
                        var_sched = mu
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

        # elif building_abr == 'MPEB':
        #     office_c_scheds = ['Office_Cooling']

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
        # todo it should automatically identify which schedules are the water heaters instead of manually picking up the names everytime.
        #todo should be easy enough to just select 'purpose' on dline 2
        if building_abr == 'CH':
            hwheaters = ["HWSchedule_Cleaner", "HWSchedule_Kitchenettes", "HWSchedule_Showers", "HWSchedule_Toilets"]

        elif building_abr == 'MPEB':
            hwheaters = ["CWS_Sched", "HWS_Labs_Sched"]

        elif building_abr == '71':
            hwheaters = ["EWH_Fraction", "EWH_Kitchen_Fraction", "EWH_Shower_Fraction"]

        for heater in hwheaters:
            heater_profile = []
            hprofile = scheds[heater].dline[8:8 + 48+48]
            for i, v in enumerate(hprofile):
                mu = float(hprofile[i])
                sigma = mu*20/100
                if base_case is True:
                    sigma = 0

                lower, upper = mu - (3 * sigma), 1
                if sigma != 0:
                    hw_sched = stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma).ppf(lhd[run_no, var_num])
                else:
                    hw_sched = mu
                heater_profile.append(hw_sched)

            print(var_num, heater, hw_sched)
            var_num += 1

            l = len(heater_profile)
            rand4 = int(round(2 * lhd[run_no, var_num]))
            offset = [1, 2, 3]

            if base_case is True:
                offset_heater = 2
            else:
                offset_heater = offset[rand4]

            print(var_num, heater, offset_heater, rand4)
            var_num += 1

            heater_profile_off = []
            for x, val in enumerate(heater_profile):
                if x > 0 and x < (l - offset_heater):
                    if val < heater_profile[x-offset_heater]:
                        heater_profile_off.append(heater_profile[x-offset_heater])
                        continue
                    if val < heater_profile[x+offset_heater]:
                        heater_profile_off.append(heater_profile[x+offset_heater])
                        continue
                    else:
                        heater_profile_off.append(heater_profile[x])
                else:
                    heater_profile_off.append(heater_profile[x])

            input_names.append(heater+'_Offset')
            input_values.append(offset_heater)

            week_heater_tot, weekend_heater_tot = sum(heater_profile_off[:48]), sum(heater_profile_off[48:96])
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
            if base_case is True:
                heater_profile_off = heater_profile # use standard profile?

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

        # FOR OCCUPANCY PROFILES TO INSERT STANDARD DEVIATION FROM THE SCHEDULES
        if building_abr in {'CH', 'MPEB'}:
            #overtime percentage
            sigma = overtime_multiplier_equip*(multiplier_variation/100) # * 20% variation

            lower, upper = overtime_multiplier_equip - (3 * sigma), overtime_multiplier_equip + (3 * sigma)
            if base_case is not True:
                overtime_multiplier_equip = stats.truncnorm((lower - overtime_multiplier_equip) / sigma, (upper - overtime_multiplier_equip) / sigma, loc=overtime_multiplier_equip,
                                                            scale=sigma).ppf(lhd[run_no, var_num])

            sigma = overtime_multiplier_light*(multiplier_variation/100)
            lower, upper = overtime_multiplier_light - (3 * sigma), overtime_multiplier_light + (3 * sigma)

            if base_case is not True:
                overtime_multiplier_light = stats.truncnorm((lower - overtime_multiplier_light) / sigma, (upper - overtime_multiplier_light) / sigma, loc=overtime_multiplier_light, scale=sigma).ppf(
                    lhd[run_no, var_num])
            var_num += 1

            # offset variable
            rand3 = int(round(2 * lhd[run_no, var_num]))
            var_num += 1

            occ_week = []
            equip_week = []
            light_week = []
            week = ['Weekday', 'Weekend']

            # fill occ_profile for week and weekend
            for day in week:
                occ_profile = []
                if day == 'Weekday':
                    profile = scheds["WifiSum"].dline[8:8 + 48]
                    std = [float(a_i) - float(b_i) for a_i, b_i in
                           zip(scheds["WifiSum"].dline[8:8 + 48], scheds["WifiSumMinusStd"].dline[8:8 + 48])]
                elif day == 'Weekend':
                    profile = scheds["WifiSum"].dline[8+48:8+48+48]
                    std = [float(a_i) - float(b_i) for a_i, b_i in
                           zip(scheds["WifiSum"].dline[8+48:8 + 48+48], scheds["WifiSumMinusStd"].dline[8+48:8 + 48+48])]

                for i, v in enumerate(profile):
                    sigma = abs(std[i])
                    if base_case is True:
                        sigma = 0
                    lower, upper = 2 * sigma, 1
                    mu = float(profile[i])
                    # print mu, sigma, lower, upper
                    if sigma != 0:
                        var_sched = stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma).ppf(
                            lhd[run_no, var_num])
                    else:
                        var_sched = mu
                    occ_profile.append(var_sched) #append values
                occ_week.extend(occ_profile) #extend list
                print(lineno(), day, len(occ_profile), occ_profile)

                ## EQUIPMENT PROFILES - BASED ON OCCUPANCY ##
                for k in range(2): #zero for light, one for equip
                    LP_profile = []
                    equip_profile_offset = []
                    if k == 0: #Lighting

                        if day == 'Weekday':
                            overtime_light_weekday = float(max(occ_profile)) * overtime_multiplier_light / 100
                            overtime = overtime_light_weekday
                            #overtime_light_weekday.append(overtime) # this makes sure that the weekend overtime is as high as during the week, which is typical...
                        if day == 'Weekend':
                            overtime = overtime_light_weekday # this makes sure that the weekend overtime is as high as during the week, which is typical...

                    if k == 1: #Equipment
                        if day == 'Weekday':
                            overtime_equip_weekday = float(max(occ_profile)) * overtime_multiplier_equip / 100
                            overtime = overtime_equip_weekday # this makes sure that the weekend overtime is as high as during the week, which is typical...
                        if day == 'Weekend':
                            overtime = overtime_equip_weekday # this makes sure that the weekend overtime is as high as during the week instead of a fraction of highest during weekend, which is typical...

                    for i, v in enumerate(occ_profile): #replace values which are lower than the overtime
                        if v < overtime:
                            v = overtime
                        LP_profile.append(v)

                    print('LP 1st profile', len(LP_profile), LP_profile)
                    print(var_num, day, 'actual overtime:', overtime, 'k is', k, '(0 = light, 1 = equip)', 'day=', day)

                    ## Creating an offset from previous profile
                    offset = [1, 2, 3, 4, 5]
                    if base_case is True:
                        offset_LP = 3
                    else:
                        offset_LP = offset[rand3]

                    len_equip = len(LP_profile)
                    for x, val in enumerate(LP_profile):
                        if x > 0 and x < (len_equip - offset_LP):
                            if val < LP_profile[x-offset_LP]:
                                equip_profile_offset.append(LP_profile[x-offset_LP])
                                #continue
                            elif val < LP_profile[x+offset_LP]:
                                equip_profile_offset.append(LP_profile[x+offset_LP])
                                #continue
                            else:
                                equip_profile_offset.append(LP_profile[x])
                        else:
                            equip_profile_offset.append(LP_profile[x])
                    #print('LP profile', len(LP_profile), LP_profile)

                    if k == 0:
                        light_profile_offset = equip_profile_offset
                        light_week.extend(light_profile_offset)
                    elif k == 1:
                        #print('equip profile offset', len(equip_profile_offset))
                        #print('equip_week', len(equip_week))
                        equip_week.extend(equip_profile_offset)

            #print('offset with rand3:', offset_LP)
            print('EquipmentOvertimeMultiplier:', overtime_multiplier_equip)
            print('LightOvertimeMultiplier:', overtime_multiplier_light)

            input_names.append('LandPsched' + '_Offset')
            input_values.append(offset_LP)
            input_names.append('EquipmentOvertimeMultiplier')
            input_values.append(overtime_multiplier_equip)
            input_names.append('LightOvertimeMultiplier')
            input_values.append(overtime_multiplier_light)

            def plot_schedules():
                occ_plot = pd.DataFrame(occ_week, index=timeline, columns=['occupancy'])
                equip_plot = pd.DataFrame(equip_week, index=timeline, columns=['equipment'])
                light_plot = pd.DataFrame(light_week, index=timeline, columns=['lighting'])

                ax = occ_plot.plot(drawstyle='steps')
                equip_plot.plot(ax=ax, drawstyle='steps')
                light_plot.plot(ax=ax, drawstyle='steps')
                ax.set_ylim(0)
                plt.show()
            #plot_schedules()

        if base_case is not True:
            seasons = [seasonal_occ_factor_week, seasonal_occ_factor_weekend]
            for q, x in enumerate(seasons):
                seasonal_occ_factor_varied = []
                sigmas = [i * (10 / 100) for i in x] # set monthly variation to define sigma
                lower, upper = [x[v] - (2 * i) for v, i in enumerate(sigmas)], [x[v] + (2 * i) for v, i in enumerate(sigmas)]
                for i in range(len(sigmas)):
                    replace_seasonal = stats.truncnorm((lower[i]-x[i])/sigmas[i], (upper[i]-x[i])/sigmas[i], loc=x[i], scale=sigmas[i]).ppf(lhd[run_no, var_num])
                    seasonal_occ_factor_varied.append(replace_seasonal)
                    var_num += 1

                # seasonal_occ_factor_varied = [i if i < 1 else 1 for i in seasonal_occ_factor_varied]
                # seasonal_occ_factor_varied = [i if i > 0 else 0.1 for i in seasonal_occ_factor_varied]

                seasonal_occ_factor_varied = [i / np.max(seasonal_occ_factor_varied) for i in seasonal_occ_factor_varied]

                if q == 0:
                    seasonal_occ_factor_week = seasonal_occ_factor_varied
                if q == 1:
                    seasonal_occ_factor_weekend = seasonal_occ_factor_varied

        months_in_year = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        seasonal_occ_factor_week_names = [str('SeasonWeekOccFactor_') + str(month) for month in months_in_year]
        seasonal_occ_factor_weekend_names = [str('SeasonWeekendOccFactor_') + str(month) for month in months_in_year]

        print(seasonal_occ_factor_week)
        print(seasonal_occ_factor_weekend)

        [input_values.append(i) for i in seasonal_occ_factor_week]
        [input_names.append(i) for i in seasonal_occ_factor_week_names]
        [input_values.append(i) for i in seasonal_occ_factor_weekend]
        [input_names.append(i) for i in seasonal_occ_factor_weekend_names]


        if building_abr == 'CH':
            office_scheds_names = ['Office_OccSched', 'Office_EquipSched', 'Office_LightSched'] # has to align with previous profiles
            office_scheds = [occ_week, equip_week, light_week]
        elif building_abr == 'MPEB':
            office_scheds_names = ['Office_OccSched', 'Office_EquipSched', 'Office_LightSched']  # has to align with previous profiles
            office_scheds = [occ_week, equip_week, light_week]
        elif building_abr == '71':
            # For Office 71 only the seasonal factor is applied and nothing is done with regards to the overtime or offset of hours, so schedules are taken directly from those defined in the csv
            office_scheds_names = ['Office_LightSched_B', 'Office_LightSched_GF', 'Office_LightSched_1',\
                            'Office_LightSched_2','Office_LightSched_3', 'Office_EquipSched_B','Office_EquipSched_GF', \
                            'Office_EquipSched_1', 'Office_EquipSched_2', 'Office_EquipSched_3']
            office_scheds = []
            for i, v in enumerate(office_scheds_names):
                add_schedule = [float(c) for c in scheds[v].dline[8:8+48+48]]
                office_scheds.append(add_schedule)

        days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31] # typical 365 year
        print(lineno(), 'days in year', sum(days_in_month))
        seasonal_factor = [] # 12 monthly values, with seasonal max factor of unity
        for sched, sname in enumerate(office_scheds_names):
            # this to prevent the overtime factors to be overridden by the seasonal weekend occupancy changes?
            if building_abr in {'CH', 'MPEB'}:
                if sname in {'Office_EquipSched', 'Office_LightSched'}:
                    seasonal_occ_factor_weekend = seasonal_occ_factor_week # weekend overtime stays same as week (baseload)

            SchedProperties = ['Schedule:Compact', sname, 'Fraction']

            for t_month in range(0, 12): #include seasonal variability
                SchedProperties.append('Through: '+str(format(t_month+1, '02')) + '/'+str(days_in_month[t_month])) # where i is the month and then last day of the month
                SchedProperties.append('For: Weekdays WinterDesignDay SummerDesignDay')
                for i, v in enumerate(office_scheds[sched]): # run through a 96 element long list
                    if i == 48: #  if element 48 is reached, then
                        SchedProperties.append('For: Weekends Holiday')
                    SchedProperties.append('Until: ' + timeline[i])

                    if i < 48:
                        SchedProperties.append(v*seasonal_occ_factor_week[t_month]) # monthly seasonal factor
                    elif i >= 48: #  if element 48 is reached, then start appending the weekends and holidays profile
                        SchedProperties.append(v*seasonal_occ_factor_weekend[t_month])  # monthly seasonal factor
            #print(SchedProperties)
            scheddict[sname].append(SchedProperties)
            print(lineno(), 'Office', len(SchedProperties), SchedProperties)

            # no offset or standard deviation is used for changing the profiles in 71!
            if building_abr in {'CH', 'MPEB'}:
                week_sched_tot, weekend_sched_tot = sum(office_scheds[sched][:48]), sum(office_scheds[sched][48:96])
                input_names.append(sname + '_WeekProfile_TotalHours')
                input_values.append(week_sched_tot)
                input_names.append(sname + '_WeekendProfile_TotalHours')
                input_values.append(weekend_sched_tot)

                week_sched_oh, weekend_sched_oh = week_sched_tot - sum(office_scheds[sched][13:37]), weekend_sched_tot - sum(office_scheds[sched][61:85])  # 7 to 7?
                input_names.append(sname + '_WeekProfile_OH')
                input_values.append(week_sched_oh)
                input_names.append(sname + '_WeekendProfile_OH')
                input_values.append(weekend_sched_oh)

        print(scheddict.keys())
        for key in scheddict.keys(): # Write to idf file

            for i,v in enumerate(scheddict[key][0]):
                if i + 1 == len(scheddict[key][0]):  # if last element in list then put semicolon
                    inf.write('\n')
                    inf.write(str(v) + ';')
                    inf.write('\n\n')
                else:
                    inf.write('\n')
                    inf.write(str(v) + ',')

        #print scheddict['PrintRoom_Cooling'][0]
        #print scheddict.keys()
        #print scheddict['Office_OccSched'][0][1]

    return input_values, input_names, var_num

def remove_schedules(idf1, building_abr): #todo should I have the option where I remove only the schedules that are going to be replaced?
    # remove existing schedules and make compact ones from scheds sheet
    scheds = unpick_schedules(rootdir, building_abr)
    schedule_types = ["SCHEDULE:DAY:INTERVAL", "SCHEDULE:WEEK:DAILY", "SCHEDULE:YEAR"]

    # for sched_type in schedule_types:
    #     scheds_idf = idf1.idfobjects[sched_type]
    #     print(len(scheds_idf))
    #
    #     for i, v in enumerate(scheds_idf):
    #         print(i)
    #         # if v.Name in scheds.keys() and scheds[v.Name].dline[2] == 'OpenStudioSchedule':
    #         #         print('Did not remove schedule:', v.Name) #scheds[v.Name].dline[0]
    #         #
    #         # else:
    #         print('remove', v.Name)
    #         idf1.popidfobject(sched_type, 0)
    # Warning when popping objects with eppy, the number objects decreases so the list will end at half if all are being popped.
    # Instead use a for loop that counts backwards through the objects.
    for y in schedule_types:
        #print(len(idf1.idfobjects[y]), "existing schedule objects removed in ", y)
        no_of_objects = len(idf1.idfobjects[y])
        to_remove = []
        for i in range(0, no_of_objects):
            if idf1.idfobjects[y][i].Name in scheds.keys() and scheds[idf1.idfobjects[y][i].Name].dline[2] == 'OpenStudioSchedule':
                print('Did not remove schedule:', idf1.idfobjects[y][i].Name)  # scheds[v.Name].dline[0]
            else:
                to_remove.append(i)
        for i in to_remove[::-1]:
            print('remove', idf1.idfobjects[y][i].Name)
            idf1.popidfobject(y, i)


    # for y in schedule_types:
    #     print(len(idf1.idfobjects[y]), "existing schedule objects removed in ", y)
    #     for i in range(0, len(idf1.idfobjects[y])):
    #         idf1.popidfobject(y, 0)


def remove_existing_outputs(idf1):
    output_types = ["OUTPUT:VARIABLE", "OUTPUT:METER:METERFILEONLY"]
    for y in output_types:
        existing_outputs = idf1.idfobjects[y]
        print(len(existing_outputs), "existing output objects removed in", y)
        if len(existing_outputs) == 0:
            print("no existing outputs to remove in", y)
        for i in range(0, len(existing_outputs)):
            idf1.popidfobject(y, 0)
def replace_materials_eppy(idf1, lhd, input_values, input_names, var_num, run_no, building_abr, base_case):
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
                if base_case is True:
                    sigma = 0
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
                        eq_mats = mu
                    material.Thermal_Resistance = eq_mats
                    input_values.append(eq_mats)
                elif mats[material.Name].name == material.Name and mats[material.Name].dline[1] == 'Material:AirGap':
                    mu = float(mats[material.Name].thermal_res)
                    if sigma > 0:
                        eq_mats = stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma).ppf(lhd[run_no, var_num])
                    else:
                        eq_mats = mu
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
                        eq_mats = mu
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
                        eq_mats = mu
                    material.UFactor = eq_mats
                    input_values.append(eq_mats)

                elif mats[material.Name].name == material.Name and mats[material.Name].dline[1] == 'WindowMaterial:Blind':
                    mu = float(mats[material.Name].conductivity)
                    if sigma > 0:
                        eq_mats = stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma).ppf(
                            lhd[run_no, var_num])
                    else:
                        eq_mats = mu
                    material.Slat_Conductivity = eq_mats
                    input_values.append(eq_mats)

                elif mats[material.Name].name == material.Name and mats[material.Name].dline[1] == 'WindowMaterial:Shade':
                    mu = float(mats[material.Name].conductivity)
                    if sigma > 0:
                        eq_mats = stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma).ppf(
                            lhd[run_no, var_num])
                    else:
                        eq_mats = mu
                    material.Conductivity = eq_mats
                    input_values.append(eq_mats)

                elif mats[material.Name].name == material.Name and mats[material.Name].dline[1] == 'WindowMaterial:Glazing':
                    mu = float(mats[material.Name].conductivity)
                    if sigma > 0:
                        eq_mats = stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma).ppf(
                            lhd[run_no, var_num])
                    else:
                        eq_mats = mu
                    material.Conductivity = eq_mats
                    input_values.append(eq_mats)

                elif mats[material.Name].name == material.Name and mats[material.Name].dline[1] == 'WindowProperty:ShadingControl':
                    mu = float(mats[material.Name].conductivity) # is actually solar radiation set point control
                    if sigma > 0:
                        eq_mats = stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma).ppf(
                            lhd[run_no, var_num])
                    else:
                        eq_mats = mu
                    material.Setpoint = eq_mats
                    input_values.append(eq_mats)

                print(lineno(), var_num, mats[material.Name].name, round(eq_mats,3))
                var_num +=1

    return input_values, input_names, var_num

def replace_equipment_eppy(idf1, lhd, input_values, input_names, var_num, run_no, building_abr, base_case):
    equips = unpick_equipments(rootdir, building_abr)

    # quick script to delete additional created load definitions (don't know where they come from).
    # If the name is not in the equips.csv then it will be removed from the idf file.
    # Those not removed are adjusted.
    load_types = ["People", "Lights", "ElectricEquipment", "ZoneInfiltration:DesignFlowRate"]
    for y in load_types:
        #print(len(idf1.idfobjects[y.upper()]), "existing schedule objects removed in ", y)
        for i in range(0, len(idf1.idfobjects[y.upper()])):

            to_replace = [v for v in equips.keys()]
            existing_loads = [v for v in idf1.idfobjects[y.upper()] if v.Name not in to_replace]

            for load in existing_loads:
                idf1.removeidfobject(load)

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

        print(lineno(), equip_type)
        counter = 0
        for equip in equip_idf: # For each instance of object replace content with that defined in csv files
            # for all ventilation objects, change to the same value
            #print(equip)
            if equip_type == 'ZONEVENTILATION:DESIGNFLOWRATE':
                len_zonevent = len(idf1.idfobjects[equip_type])
                object_name = "ZoneVentilation"

                print(lineno(), 'name', equips[object_name].name, )

                equip.Design_Flow_Rate_Calculation_Method = equips[object_name].dline[19]
                sigma = float(equips[object_name].dline[30])
                if base_case is True:
                    sigma = 0

                lower, upper = float(equips[object_name].dline[31]), float(equips[object_name].dline[32])
                mu = float(equips[object_name].dline[27])
                if sigma > 0:
                    eq_vent = stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma).ppf(lhd[run_no, var_num])
                    equip.Flow_Rate_per_Person = eq_vent  # dline[19]
                    x += 1
                else:
                    eq_vent = mu

                if counter == 0:
                    equip.Flow_Rate_per_Person = eq_vent  # dline[19]
                    input_values.append(eq_vent)
                    input_names.append(object_name)
                    print(eq_vent, var_num)
                    var_num += 1
                    counter += 1

            elif equip_type == 'FAN:ZONEEXHAUST':
                len_objects = len(idf1.idfobjects[equip_type])
                object_name = "ExhaustFans"

                print(lineno(), 'name', equips[object_name].name, )

                sigma = float(equips[object_name].dline[30])
                if base_case is True:
                    sigma = 0

                lower, upper = float(equips[object_name].dline[31]), float(equips[object_name].dline[32])
                mu = float(equips[object_name].dline[7])
                if sigma > 0:
                    eq_efficiency = stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma).ppf(
                        lhd[run_no, var_num])
                    equip.Fan_Total_Efficiency = eq_efficiency
                    x+=1
                else:
                    eq_efficiency = mu

                if counter == 0:
                    equip.Fan_Total_Efficiency = eq_efficiency
                    input_values.append(eq_efficiency)
                    input_names.append(object_name)
                    print(eq_efficiency, var_num)
                    var_num += 1
                    counter += 1

            else:
                if equip_type == 'AIRCONDITIONER:VARIABLEREFRIGERANTFLOW':
                    equip_name = equip.Heat_Pump_Name
                else:
                    equip_name = equip.Name

                #print(equip_name)
                print(lineno(), 'name', equips[equip_name].name,)
                if equips[equip_name].name == equip_name and equips[equip_name].dline[1] == 'ElectricEquipment':
                    equip.Design_Level_Calculation_Method = equips[equip_name].dline[19] #dline[19]
                    sigma = float(equips[equip_name].dline[30])
                    if base_case is True:
                        sigma = 0

                    lower, upper = float(equips[equip_name].dline[31]), float(equips[equip_name].dline[32])
                    mu = float(equips[equip_name].dline[21])
                    if sigma > 0:
                        eq_equip = stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma).ppf(lhd[run_no, var_num])
                    else:
                        eq_equip = mu
                    equip.Watts_per_Zone_Floor_Area = eq_equip
                    input_values.append(eq_equip)
                    input_names.append(equips[equip_name].name)
                    var_num += 1

                elif equips[equip_name].name == equip_name and equips[equip_name].dline[1] == 'People':
                    equip.Number_of_People_Calculation_Method = equips[equip_name].dline[19] #dline[19]
                    sigma = float(equips[equip_name].dline[30])
                    if base_case is True:
                        sigma = 0

                    lower, upper = float(equips[equip_name].dline[31]), float(equips[equip_name].dline[32])
                    mu = float(equips[equip_name].dline[25])
                    if sigma > 0:
                        eq_people = stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma).ppf(lhd[run_no, var_num])
                    else:
                        eq_people = mu
                    equip.Zone_Floor_Area_per_Person = eq_people
                    #equip.People_per_Zone_Floor_Area = eq_people
                    input_values.append(eq_people)
                    input_names.append(equips[equip_name].name)
                    var_num += 1

                elif equips[equip_name].name == equip_name and equips[equip_name].dline[1] == 'Lights':
                    equip.Design_Level_Calculation_Method = equips[equip_name].dline[19]  # dline[19]
                    sigma = float(equips[equip_name].dline[30])
                    if base_case is True:
                        sigma = 0

                    lower, upper = float(equips[equip_name].dline[31]), float(equips[equip_name].dline[32])
                    mu = float(equips[equip_name].dline[21])
                    if sigma > 0:
                        eq_light = stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma).ppf(lhd[run_no, var_num])
                    else:
                        eq_light = mu
                    equip.Watts_per_Zone_Floor_Area = eq_light
                    input_values.append(eq_light)
                    input_names.append(equips[equip_name].name)
                    var_num += 1


                elif equips[equip_name].name == equip_name and equips[equip_name].dline[1] == 'ZoneInfiltration:DesignFlowRate':
                    equip.Design_Flow_Rate_Calculation_Method = equips[equip_name].dline[19]  # dline[19]
                    sigma = float(equips[equip_name].dline[30])
                    if base_case is True:
                        sigma = 0

                    lower, upper = float(equips[equip_name].dline[31]), float(equips[equip_name].dline[32])
                    mu = float(equips[equip_name].dline[26])
                    if sigma > 0:
                        eq_infil = stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma).ppf(lhd[run_no, var_num])
                    else:
                        eq_infil = mu
                    equip.Flow_per_Exterior_Surface_Area = eq_infil
                    input_values.append(eq_infil)
                    input_names.append(equips[equip_name].name)
                    var_num += 1

                elif equips[equip_name].name == equip_name and equips[equip_name].dline[1] == 'DesignSpecification:OutdoorAir':
                    equip.Outdoor_Air_Method = equips[equip_name].dline[19]  # dline[19]
                    sigma = float(equips[equip_name].dline[30])
                    if base_case is True:
                        sigma = 0

                    lower, upper = float(equips[equip_name].dline[31]), float(equips[equip_name].dline[32])
                    mu = float(equips[equip_name].dline[28])
                    if sigma > 0:
                        eq_oa = stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma).ppf(lhd[run_no, var_num])
                    else:
                        eq_oa = mu
                    #todo is this right???
                    equip.Outdoor_Air_Flow_Air_Changes_per_Hour = eq_oa
                    #equip.Outdoor_Air_Flow_per_Zone = eq_oa
                    input_values.append(eq_oa)
                    input_names.append(equips[equip_name].name)
                    var_num += 1

                elif equips[equip_name].name == equip_name and equips[equip_name].dline[1] == 'AirConditioner:VariableRefrigerantFlow':
                    print(lineno(), equips[equip_name].name) # equips[equip_name].dline

                    #ccop
                    mu = float(equips[equip_name].dline[5])
                    sigma = float(equips[equip_name].dline[33])/100*mu
                    if base_case is True:
                        sigma = 0
                    print('sigma in AC units', sigma)
                    if mu > 0:
                        lower, upper = mu - (3 * sigma), mu + (3 * sigma)
                    else:
                        lower, upper, sigma = 0, 0, 0

                    if sigma > 0:
                        ccop = stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma).ppf(
                            lhd[run_no, var_num])
                    else:
                        ccop = mu
                    var_num += 1

                    #hcop
                    mu = float(equips[equip_name].dline[6])
                    sigma = float(equips[equip_name].dline[33]) / 100 * mu
                    if base_case is True:
                        sigma = 0
                    lower, upper = mu - (3 * sigma), mu + (3 * sigma)
                    if sigma > 0:
                        hcop = stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma).ppf(lhd[run_no, var_num])
                    else:
                        hcop = mu
                    var_num += 1

                    equip.Gross_Rated_Cooling_COP = ccop
                    input_values.append(ccop)
                    input_names.append(equips[equip_name].name+"ccop")
                    equip.Gross_Rated_Heating_COP = hcop
                    input_values.append(hcop)
                    input_names.append(equips[equip_name].name+"hcop")

                elif equips[equip_name].name == equip_name and equips[equip_name].dline[1] == 'Boiler:HotWater':
                    sigma = float(equips[equip_name].dline[30])
                    if base_case is True:
                        sigma = 0
                    lower, upper = float(equips[equip_name].dline[31]), float(equips[equip_name].dline[32])
                    mu = float(equips[equip_name].dline[7])
                    if sigma > 0:
                        input = stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma).ppf(lhd[run_no, var_num])
                    else:
                        input = mu
                    equip.Nominal_Thermal_Efficiency = input
                    input_values.append(input)
                    input_names.append(equips[equip_name].name)
                    var_num += 1

                elif equips[equip_name].name == equip_name and equips[equip_name].dline[1] == 'WaterUse:Equipment':
                    sigma = float(equips[equip_name].dline[30])
                    if base_case is True:
                        sigma = 0
                    lower, upper = float(equips[equip_name].dline[31]), float(equips[equip_name].dline[32])
                    mu = float(equips[equip_name].dline[8])
                    if sigma > 0:
                        input = stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma).ppf(lhd[run_no, var_num])
                    else:
                        input = mu
                    equip.Peak_Flow_Rate = input
                    input_values.append(input)
                    input_names.append(equips[equip_name].name)
                    var_num += 1

                else:
                    continue

                print(lineno(), var_num, equip_name)
                #input_names.append(equips[equip_name].name)
    return input_values, input_names, var_num

def add_groundtemps(idf1):
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

#From City of London weatherfile
#	3	0.5				6.65	6.05	7.12	8.8	13.19	16.58	18.87	19.56	18.38	15.75	12.23	8.96	2	8.88	7.8	8.02	8.91	11.85	14.51	16.63	17.77	17.53	16.04000	13.61000	11.05	4				10.72	9.61	9.37	9.71	11.37	13.16	14.8	15.94	16.21	15.56	14.14	12.4

def add_outputs(idf1, base_case, add_variables, building_abr): # add outputvariables and meters as new idf objects
    # add variables only when base case is simulated

    if base_case is True:
        output_variables = ["Site Outdoor Air Drybulb Temperature",
                            "Site Outdoor Air Relative Humidity",
                            "Site Diffuse Solar Radiation Rate per Area",
                            "Site Direct Solar Radiation Rate per Area"
                            ]

        if add_variables is True:
            extra_variables = [#"Zone Infiltration Mass Flow Rate",
                                "Zone Ventilation Mass Flow Rate"
                                "VRF Heat Pump Cooling Electric Energy",
                                "VRF Heat Pump Heating Electric Energy",
                                "Zone VRF Air Terminal Cooling Electric Energy",
                                "Zone VRF Air Terminal Heating Electric Energy",
                                "Zone Cooling Setpoint Not Met Time",
                                "Zone Heating Setpoint Not Met Time",
                                "Zone Air Temperature",
                                "System Node Temperature",
                                "Zone Ventilation Air Change Rate",
                                "Zone Mechanical Ventilation Air Changes per Hour",
                                "System Node Mass Flow Rate",
                                "Air System Gas Energy",
                                "Air System Heating Coil Gas Energy",
                                ]
            output_variables.extend(extra_variables)

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

        output_diagnostics = []
        # "ReportDuringWarmup"
        # "DisplayAllWarnings",
        # "ReportDuringWarmup",
        # "ReportDuringWarmupConvergence",
        # "ReportDuringHVACSizingSimulation"
        if not output_diagnostics:
            print('No output diagnostics')
        else:
            for name in output_diagnostics:
                outvar = idf1.newidfobject("Output:Diagnostics".upper())
                outvar.Key_1 = name

        for name in output_variables:
            outvar = idf1.newidfobject("Output:Variable".upper())
            outvar.Key_Value = ''
            outvar.Variable_Name = name
            if base_case is True:
                outvar.Reporting_Frequency = 'timestep'  # 'timestep', 'hourly', 'detailed',
            else:
                outvar.Reporting_Frequency = 'hourly' #'timestep', 'hourly', 'detailed',

    # First create any CUSTOM METERS, these also need to be added to the MeterFileOnly object
    if building_abr == 'MPEB':
        custom_meter_names = ['Plant_Fans', 'AHU_Fans', 'Workshops_Power', 'Workshops_Lights', 'Chillers']
        custom_meters = [
             [["REF1", "Fan Electric Energy"],
              ["REF2", "Fan Electric Energy"],
              ["REF4A", "Fan Electric Energy"],
              ["REF4B", "Fan Electric Energy"],
              ["REF5", "Fan Electric Energy"]],

            [["AHU1_EXTRACT", "Fan Electric Energy"],
              ["AHU1_SUPPLY", "Fan Electric Energy"],
              ["AHU2_EXTRACT", "Fan Electric Energy"],
              ["AHU2_SUPPLY", "Fan Electric Energy"],
              ["AHU3_SUPPLY", "Fan Electric Energy"],
              ["AHU4_SUPPLY", "Fan Electric Energy"],
              ["AHU5_EXTRACT", "Fan Electric Energy"],
              ["AHU5_SUPPLY", "Fan Electric Energy"]],

            [['THERMAL ZONE: B13 CIRCULATION CIRCULATION_EQUIPMENT', 'Electric Equipment Electric Energy'],
            ['THERMAL ZONE: B70 CORRIDOR CIRCULATION_EQUIPMENT', 'Electric Equipment Electric Energy'],
            ['THERMAL ZONE: B71 SERVICE CORRIDOR 1 CIRCULATION_EQUIPMENT', 'Electric Equipment Electric Energy'],
            ['THERMAL ZONE: B72 SERVICE CORRIDOR 2 CIRCULATION_EQUIPMENT', 'Electric Equipment Electric Energy'],
            ['THERMAL ZONE: B73 SERVICE CORRIDOR 3 CIRCULATION_EQUIPMENT', 'Electric Equipment Electric Energy'],
            ['THERMAL ZONE: B75 CORRIDOR CIRCULATION_EQUIPMENT', 'Electric Equipment Electric Energy'],
            ['THERMAL ZONE: B77 CORRIDOR CIRCULATION_EQUIPMENT', 'Electric Equipment Electric Energy'],
            ['THERMAL ZONE: B77A_CORRIDOR CIRCULATION_EQUIPMENT', 'Electric Equipment Electric Energy'],
            ['THERMAL ZONE: B82 CIRCULATION CIRCULATION_EQUIPMENT', 'Electric Equipment Electric Energy'],
            ['THERMAL ZONE: B84 CIRCULATION_EQUIPMENT', 'Electric Equipment Electric Energy'],
            ['THERMAL ZONE: B13A TEA POINT KITCHEN_EQUIPMENT', 'Electric Equipment Electric Energy'],
            ['THERMAL ZONE: B01 ENGINE DYNAMOMETER CELL 1 LABORATORY_EQUIPMENT', 'Electric Equipment Electric Energy'],
            ['THERMAL ZONE: B01A CONTROL ROOM CELL 1 LABORATORY_EQUIPMENT', 'Electric Equipment Electric Energy'],
            ['THERMAL ZONE: B02 ENGINE DYNAMOMETER CELL 2 LABORATORY_EQUIPMENT', 'Electric Equipment Electric Energy'],
            ['THERMAL ZONE: B02A CELL 2 CONTROL ROOM LABORATORY_EQUIPMENT', 'Electric Equipment Electric Energy'],
            ['THERMAL ZONE: B03 ENGINE DYNAMOMETER CELL 3 LABORATORY_EQUIPMENT', 'Electric Equipment Electric Energy'],
            ['THERMAL ZONE: B03A CELL 3 CONTROL ROOM LABORATORY_EQUIPMENT', 'Electric Equipment Electric Energy'],
            ['THERMAL ZONE: B04 ENGINE DYNAMOMETER CELL 4 LABORATORY_EQUIPMENT', 'Electric Equipment Electric Energy'],
            ['THERMAL ZONE: B04A CELL 4 CONTROL ROOM LABORATORY_EQUIPMENT', 'Electric Equipment Electric Energy'],
            ['THERMAL ZONE: B05 EXPERIMENT ROOM LABORATORY_EQUIPMENT', 'Electric Equipment Electric Energy'],
            ['THERMAL ZONE: B06 FSTF A LABORATORY_EQUIPMENT', 'Electric Equipment Electric Energy'],
            ['THERMAL ZONE: B07 CONTROL ROOM LABORATORY_EQUIPMENT', 'Electric Equipment Electric Energy'],
            ['THERMAL ZONE: B08 FSTF B LABORATORY_EQUIPMENT', 'Electric Equipment Electric Energy'],
            ['THERMAL ZONE: B09 EXPERIMENT ROOM 2 LABORATORY_EQUIPMENT', 'Electric Equipment Electric Energy'],
            ['THERMAL ZONE: B09A EXPERIMENT ROOM 3 LABORATORY_EQUIPMENT', 'Electric Equipment Electric Energy'],
            ['THERMAL ZONE: B15C MARINETECH LAB LABORATORY_EQUIPMENT', 'Electric Equipment Electric Energy'],
            ['THERMAL ZONE: B17 WC LAVATORY_EQUIPMENT', 'Electric Equipment Electric Energy'],
            ['THERMAL ZONE: B18 WC LAVATORY_EQUIPMENT', 'Electric Equipment Electric Energy'],
            ['THERMAL ZONE: B19 TECHNICIANS REST ROOM LAVATORY_EQUIPMENT', 'Electric Equipment Electric Energy'],
            ['THERMAL ZONE: B20 DIS WC LAVATORY_EQUIPMENT', 'Electric Equipment Electric Energy'],
            ['THERMAL ZONE: B15B OFFICE OFFICE_EQUIPMENT', 'Electric Equipment Electric Energy'],
            ['THERMAL ZONE: B08A PLANT ROOM PLANT_EQUIPMENT', 'Electric Equipment Electric Energy'],
            ['THERMAL ZONE: B81 PLANT_EQUIPMENT', 'Electric Equipment Electric Energy'],
            ['THERMAL ZONE: B83 LIFT MOTOR ROOM PLANT_EQUIPMENT', 'Electric Equipment Electric Energy'],
            ['THERMAL ZONE: B87 SERVICES PLANT_EQUIPMENT', 'Electric Equipment Electric Energy'],
            ['THERMAL ZONE: B95 LIFT MOTOR ROOM PLANT_EQUIPMENT', 'Electric Equipment Electric Energy'],
            ['THERMAL ZONE: B97 HYDRAULIC LIFT/ACCESS HATCH PLANT_EQUIPMENT', 'Electric Equipment Electric Energy'],
            ['THERMAL ZONE: B80 CORE 1 STAIRS_EQUIPMENT', 'Electric Equipment Electric Energy'],
            ['THERMAL ZONE: B91 CORE 3 STAIRS STAIRS_EQUIPMENT', 'Electric Equipment Electric Energy'],
            ['THERMAL ZONE: B14 WELDING WORKSHOP WORKSHOP_EQUIPMENT', 'Electric Equipment Electric Energy'],
            ['THERMAL ZONE: B15 WORKSHOP WORKSHOP_EQUIPMENT', 'Electric Equipment Electric Energy'],
            ['THERMAL ZONE: B15A CNC OPER ROOM WORKSHOP_EQUIPMENT', 'Electric Equipment Electric Energy'],
            ['THERMAL ZONE: B22 LABORATORY WORKSHOP_EQUIPMENT', 'Electric Equipment Electric Energy']],

            [['THERMAL ZONE: B13 CIRCULATION CIRCULATION_LIGHTING', 'Lights Electric Energy'],
             ['THERMAL ZONE: B70 CORRIDOR CIRCULATION_LIGHTING', 'Lights Electric Energy'],
             ['THERMAL ZONE: B71 SERVICE CORRIDOR 1 CIRCULATION_LIGHTING', 'Lights Electric Energy'],
             ['THERMAL ZONE: B72 SERVICE CORRIDOR 2 CIRCULATION_LIGHTING', 'Lights Electric Energy'],
             ['THERMAL ZONE: B73 SERVICE CORRIDOR 3 CIRCULATION_LIGHTING', 'Lights Electric Energy'],
             ['THERMAL ZONE: B75 CORRIDOR CIRCULATION_LIGHTING', 'Lights Electric Energy'],
             ['THERMAL ZONE: B77 CORRIDOR CIRCULATION_LIGHTING', 'Lights Electric Energy'],
             ['THERMAL ZONE: B77A_CORRIDOR CIRCULATION_LIGHTING', 'Lights Electric Energy'],
             ['THERMAL ZONE: B82 CIRCULATION CIRCULATION_LIGHTING', 'Lights Electric Energy'],
             ['THERMAL ZONE: B84 CIRCULATION_LIGHTING', 'Lights Electric Energy'],
             ['THERMAL ZONE: B13A TEA POINT KITCHEN_LIGHTING', 'Lights Electric Energy'],
             ['THERMAL ZONE: B01 ENGINE DYNAMOMETER CELL 1 LABORATORY_LIGHTING', 'Lights Electric Energy'],
             ['THERMAL ZONE: B01A CONTROL ROOM CELL 1 LABORATORY_LIGHTING', 'Lights Electric Energy'],
             ['THERMAL ZONE: B02 ENGINE DYNAMOMETER CELL 2 LABORATORY_LIGHTING', 'Lights Electric Energy'],
             ['THERMAL ZONE: B02A CELL 2 CONTROL ROOM LABORATORY_LIGHTING', 'Lights Electric Energy'],
             ['THERMAL ZONE: B03 ENGINE DYNAMOMETER CELL 3 LABORATORY_LIGHTING', 'Lights Electric Energy'],
             ['THERMAL ZONE: B03A CELL 3 CONTROL ROOM LABORATORY_LIGHTING', 'Lights Electric Energy'],
             ['THERMAL ZONE: B04 ENGINE DYNAMOMETER CELL 4 LABORATORY_LIGHTING', 'Lights Electric Energy'],
             ['THERMAL ZONE: B04A CELL 4 CONTROL ROOM LABORATORY_LIGHTING', 'Lights Electric Energy'],
             ['THERMAL ZONE: B05 EXPERIMENT ROOM LABORATORY_LIGHTING', 'Lights Electric Energy'],
             ['THERMAL ZONE: B06 FSTF A LABORATORY_LIGHTING', 'Lights Electric Energy'],
             ['THERMAL ZONE: B07 CONTROL ROOM LABORATORY_LIGHTING', 'Lights Electric Energy'],
             ['THERMAL ZONE: B08 FSTF B LABORATORY_LIGHTING', 'Lights Electric Energy'],
             ['THERMAL ZONE: B09 EXPERIMENT ROOM 2 LABORATORY_LIGHTING', 'Lights Electric Energy'],
             ['THERMAL ZONE: B09A EXPERIMENT ROOM 3 LABORATORY_LIGHTING', 'Lights Electric Energy'],
             ['THERMAL ZONE: B15C MARINETECH LAB LABORATORY_LIGHTING', 'Lights Electric Energy'],
             ['THERMAL ZONE: B17 WC LAVATORY_LIGHTING', 'Lights Electric Energy'],
             ['THERMAL ZONE: B18 WC LAVATORY_LIGHTING', 'Lights Electric Energy'],
             ['THERMAL ZONE: B19 TECHNICIANS REST ROOM LAVATORY_LIGHTING', 'Lights Electric Energy'],
             ['THERMAL ZONE: B20 DIS WC LAVATORY_LIGHTING', 'Lights Electric Energy'],
             ['THERMAL ZONE: B15B OFFICE OFFICE_LIGHTING', 'Lights Electric Energy'],
             ['THERMAL ZONE: B08A PLANT ROOM PLANT_LIGHTING', 'Lights Electric Energy'],
             ['THERMAL ZONE: B81 PLANT_LIGHTING', 'Lights Electric Energy'],
             ['THERMAL ZONE: B83 LIFT MOTOR ROOM PLANT_LIGHTING', 'Lights Electric Energy'],
             ['THERMAL ZONE: B87 SERVICES PLANT_LIGHTING', 'Lights Electric Energy'],
             ['THERMAL ZONE: B95 LIFT MOTOR ROOM PLANT_LIGHTING', 'Lights Electric Energy'],
             ['THERMAL ZONE: B97 HYDRAULIC LIFT/ACCESS HATCH PLANT_LIGHTING', 'Lights Electric Energy'],
             ['THERMAL ZONE: B80 CORE 1 STAIRS_LIGHTING', 'Lights Electric Energy'],
             ['THERMAL ZONE: B91 CORE 3 STAIRS STAIRS_LIGHTING', 'Lights Electric Energy'],
             ['THERMAL ZONE: B14 WELDING WORKSHOP WORKSHOP_LIGHTING', 'Lights Electric Energy'],
             ['THERMAL ZONE: B15 WORKSHOP WORKSHOP_LIGHTING', 'Lights Electric Energy'],
             ['THERMAL ZONE: B15A CNC OPER ROOM WORKSHOP_LIGHTING', 'Lights Electric Energy'],
             ['THERMAL ZONE: B22 LABORATORY WORKSHOP_LIGHTING', 'Lights Electric Energy'],
             ['THERMAL ZONE: B10 MATERIAL STORE STORAGE_LIGHTING', 'Lights Electric Energy'],
             ['THERMAL ZONE: B11 ENGINE STORE RACK STORAGE_LIGHTING', 'Lights Electric Energy'],
             ['THERMAL ZONE: B12 ENGINE PARTS STORE STORAGE_LIGHTING', 'Lights Electric Energy'],
             ['THERMAL ZONE: B16 EQUIPSTORE STORAGE_LIGHTING', 'Lights Electric Energy'],
             ['THERMAL ZONE: B21 LV SWITCHROOM STORAGE_LIGHTING', 'Lights Electric Energy'],
             ['THERMAL ZONE: B71A GAS BOTTLES STORAGE_LIGHTING', 'Lights Electric Energy'],
             ['THERMAL ZONE: B72A GAS BOTTLES STORAGE_LIGHTING', 'Lights Electric Energy'],
             ['THERMAL ZONE: B73 GAS BOTTLES STORAGE_LIGHTING', 'Lights Electric Energy'],
             ['THERMAL ZONE: B88 GAS BOTTLES STORAGE_LIGHTING', 'Lights Electric Energy'],
             ['THERMAL ZONE: B89 GAS BOTTLES STORAGE_LIGHTING', 'Lights Electric Energy'],
             ['THERMAL ZONE: B90 GAS BOTTLES STORAGE_LIGHTING', 'Lights Electric Energy']],

            [['CHILLER - AIR COOLED 1', 'Chiller Electric Energy'],
            ['CHILLER - AIR COOLED', 'Chiller Electric Energy']]
            ]
    elif building_abr == 'CH':
        custom_meters = []
    elif building_abr == '71':

        custom_meter_names = ['B_Lights', 'GF_Lights', '1st_Lights', '2nd_Lights', '3rd_Lights', 'B_Power', 'GF_Power','1st_Power', '2nd_Power', '3rd_Power', 'Plant_Fans']
        #custom_meters = []
        custom_meters = [
            [['B_ELEVATOR CIRCULATION_B_LIGHTS', 'Lights Electric Energy'],
             ['B_ELEVATORFRONT CIRCULATION_B_LIGHTS', 'Lights Electric Energy'],
             ['B_PLANTROOM PLANT_LIGHTS', 'Lights Electric Energy'],
             ['B_STORE1 STORAGE_LIGHTS', 'Lights Electric Energy'],
             ['B_STORE2 STORAGE_LIGHTS', 'Lights Electric Energy'],
             ['B_STORE3 STORAGE_LIGHTS', 'Lights Electric Energy'],
             ['B_TOILETSHOWERS TOILET_LIGHTS', 'Lights Electric Energy'],
             ['B_TOILETS TOILET_LIGHTS', 'Lights Electric Energy']],

            [['GF_CORRIDOR CIRCULATION_GF_LIGHTS', 'Lights Electric Energy'],
             ['GF_CORRIDOREXITBACK CIRCULATION_GF_LIGHTS', 'Lights Electric Energy'],
             ['GF_ELEVATORBACK CIRCULATION_GF_LIGHTS', 'Lights Electric Energy'],
             ['GF_ELEVATORFRONT CIRCULATION_GF_LIGHTS', 'Lights Electric Energy'],
             ['GF_ENTRANCE CIRCULATION_GF_LIGHTS', 'Lights Electric Energy'],
             ['GF_BACKSPACE STORAGE_LIGHTS', 'Lights Electric Energy'],
             ['GF_BACKSPACESTORE2 STORAGE_LIGHTS', 'Lights Electric Energy'],
             ['GF_STORAGE1 STORAGE_LIGHTS', 'Lights Electric Energy'],
             ['GF_TOILET TOILET_LIGHTS', 'Lights Electric Energy'],
             ['GF_HAPPOLDSUITE MEETING_GF_LIGHTS', 'Lights Electric Energy'],
             ['GF_HAPPOLDSUITE1 MEETING_GF_LIGHTS', 'Lights Electric Energy'],
             ['GF_HAPPOLDSUITE2 MEETING_GF_LIGHTS', 'Lights Electric Energy'],
             ['GF_HAPPOLDSUITE3 MEETING_GF_LIGHTS', 'Lights Electric Energy'],
             ['GF_LOBBYPRESENTATION RECEPTION_LIGHTS', 'Lights Electric Energy']],

            [['1_ELEVATORFRONT CIRCULATION_1_LIGHTS', 'Lights Electric Energy'],  # meter 2
             ['1_ELEVATORBACK CIRCULATION_1_LIGHTS', 'Lights Electric Energy'],
             ['1_MEETING MEETING_1_LIGHTS', 'Lights Electric Energy'],
             ['1_OFFICE OFFICE_1_LIGHTS', 'Lights Electric Energy'],
             ['1_TOILETSTORAGE STORAGE_LIGHTS', 'Lights Electric Energy'],
             ['1_TOILETS TOILET_LIGHTS', 'Lights Electric Energy']],

            [['2_ELEVATORBACK CIRCULATION_2_LIGHTS', 'Lights Electric Energy'],
             ['2_ELEVATORFRONT CIRCULATION_2_LIGHTS', 'Lights Electric Energy'],
             ['2_TOILETSTORAGE STORAGE_LIGHTS', 'Lights Electric Energy'],
             ['2_OFFICE OFFICE_2_LIGHTS', 'Lights Electric Energy'],
             ['2_TOILETS TOILET_LIGHTS', 'Lights Electric Energy'],
             ['2_MEETING MEETING_2_LIGHTS', 'Lights Electric Energy']],

            [['3_ELEVATORBACK CIRCULATION_3_LIGHTS', 'Lights Electric Energy'],
             ['3_ELEVATORFRONT CIRCULATION_3_LIGHTS', 'Lights Electric Energy'],
             ['3_OFFICE OFFICE_3_LIGHTS', 'Lights Electric Energy'],
             ['3_TOILETSTORAGE STORAGE_LIGHTS', 'Lights Electric Energy'],
             ['3_TOILETS TOILET_LIGHTS', 'Lights Electric Energy']],

            [['B_PLANTROOM PLANT_EQUIPMENT', 'Electric Equipment Electric Energy'],
            ['B_ELEVATOR CIRCULATION_B_EQUIPMENT', 'Electric Equipment Electric Energy'],
            ['B_ELEVATORFRONT CIRCULATION_B_EQUIPMENT', 'Electric Equipment Electric Energy'],
            ['B_TOILETSHOWERS TOILET_EQUIPMENT', 'Electric Equipment Electric Energy'],
            ['B_TOILETS TOILET_EQUIPMENT', 'Electric Equipment Electric Energy'],
            ['B_STORE1 STORAGE_EQUIPMENT', 'Electric Equipment Electric Energy'],
            ['B_STORE2 STORAGE_EQUIPMENT', 'Electric Equipment Electric Energy'],
            ['B_STORE3 STORAGE_EQUIPMENT', 'Electric Equipment Electric Energy'],
            ['B_WATERHEATER_E','Water Heater Electric Energy'],
            ['B_WATERHEATER_E','Water Heater Off Cycle Parasitic Electric Energy'],
            ['B_WATERHEATER_E','Water Heater On Cycle Parasitic Electric Energy']],

            [['GF_CORRIDOR CIRCULATION_GF_EQUIPMENT', 'Electric Equipment Electric Energy'],
            ['GF_CORRIDOREXITBACK CIRCULATION_GF_EQUIPMENT', 'Electric Equipment Electric Energy'],
            ['GF_ELEVATORBACK CIRCULATION_GF_EQUIPMENT', 'Electric Equipment Electric Energy'],
            ['GF_ELEVATORFRONT CIRCULATION_GF_EQUIPMENT', 'Electric Equipment Electric Energy'],
            ['GF_ENTRANCE CIRCULATION_GF_EQUIPMENT', 'Electric Equipment Electric Energy'],
            ['GF_HAPPOLDSUITE MEETING_GF_EQUIPMENT', 'Electric Equipment Electric Energy'],
            ['GF_HAPPOLDSUITE1 MEETING_GF_EQUIPMENT', 'Electric Equipment Electric Energy'],
            ['GF_HAPPOLDSUITE2 MEETING_GF_EQUIPMENT', 'Electric Equipment Electric Energy'],
            ['GF_HAPPOLDSUITE3 MEETING_GF_EQUIPMENT', 'Electric Equipment Electric Energy'],
            ['GF_LOBBYPRESENTATION RECEPTION_EQUIPMENT', 'Electric Equipment Electric Energy'],
            ['GF_BACKSPACE STORAGE_EQUIPMENT', 'Electric Equipment Electric Energy'],
            ['GF_BACKSPACESTORE2 STORAGE_EQUIPMENT', 'Electric Equipment Electric Energy'],
            ['GF_STORAGE1 STORAGE_EQUIPMENT', 'Electric Equipment Electric Energy'],
            ['GF_TOILET TOILET_EQUIPMENT', 'Electric Equipment Electric Energy'],
            ['GF_WATERHEATER_C','Water Heater Electric Energy'],
            ['GF_WATERHEATER_C','Water Heater Off Cycle Parasitic Electric Energy'],
            ['GF_WATERHEATER_C','Water Heater On Cycle Parasitic Electric Energy']],

            [['1_ELEVATORBACK CIRCULATION_1_EQUIPMENT', 'Electric Equipment Electric Energy'],
            ['1_ELEVATORFRONT CIRCULATION_1_EQUIPMENT', 'Electric Equipment Electric Energy'],
            ['1_MEETING MEETING_1_EQUIPMENT', 'Electric Equipment Electric Energy'],
            ['1_OFFICE OFFICE_1_EQUIPMENT', 'Electric Equipment Electric Energy'],
            ['1_TOILETS TOILET_EQUIPMENT', 'Electric Equipment Electric Energy'],
            ['1_TOILETSTORAGE STORAGE_EQUIPMENT', 'Electric Equipment Electric Energy'],
            ['1_WATERHEATER_A','Water Heater Electric Energy'],
            ['1_WATERHEATER_A','Water Heater Off Cycle Parasitic Electric Energy'],
            ['1_WATERHEATER_A','Water Heater On Cycle Parasitic Electric Energy']],

            [['2_ELEVATORBACK CIRCULATION_2_EQUIPMENT', 'Electric Equipment Electric Energy'],
            ['2_ELEVATORFRONT CIRCULATION_2_EQUIPMENT', 'Electric Equipment Electric Energy'],
            ['2_MEETING MEETING_2_EQUIPMENT', 'Electric Equipment Electric Energy'],
            ['2_OFFICE OFFICE_2_EQUIPMENT', 'Electric Equipment Electric Energy'],
            ['2_TOILETSTORAGE STORAGE_EQUIPMENT', 'Electric Equipment Electric Energy'],
            ['2_TOILETS TOILET_EQUIPMENT', 'Electric Equipment Electric Energy'],
            ['2_WATERHEATER_A','Water Heater Electric Energy'],
            ['2_WATERHEATER_A','Water Heater Off Cycle Parasitic Electric Energy']],

            [['3_OFFICE OFFICE_3_EQUIPMENT', 'Electric Equipment Electric Energy'],
            ['3_ELEVATORBACK CIRCULATION_3_EQUIPMENT', 'Electric Equipment Electric Energy'],
            ['3_ELEVATORFRONT CIRCULATION_3_EQUIPMENT', 'Electric Equipment Electric Energy'],
            ['3_TOILETSTORAGE STORAGE_EQUIPMENT', 'Electric Equipment Electric Energy'],
            ['3_TOILETS TOILET_EQUIPMENT', 'Electric Equipment Electric Energy'],
             ['3_WATERHEATER_B', 'Water Heater Electric Energy'],
             ['3_WATERHEATER_B', 'Water Heater Off Cycle Parasitic Electric Energy'],
             ['3_WATERHEATER_B', 'Water Heater On Cycle Parasitic Electric Energy']],

            [['AHU_ExtractFan', 'Fan Electric Energy'],
            ['AHU_SupplyFan', 'Fan Electric Energy'],
            ['1_TOILETEXHAUST', 'Fan Electric Energy'],
            ['2_TOILETEXHAUST', 'Fan Electric Energy']]
        ]

    if custom_meters: # check if any custom outputs in list
        # adding custom meter objects, make a list of list with inner lists containing other output meters and each separate list is a custom meter.
        # default max on number of meters  is 22, set by .idd file.

        for meter in range(len(custom_meters)):
            for output in range(len(custom_meters[meter])):
                new_meter = idf1.newidfobject("Output:Meter".upper())
                new_meter.Name = str(custom_meters[meter][output][0])+':'+str(custom_meters[meter][output][1])
                if base_case is True:
                    new_meter.Reporting_Frequency = 'timestep'  # 'timestep', 'hourly', 'detailed',
                else:
                    new_meter.Reporting_Frequency = 'hourly'  # 'timestep', 'hourly', 'detailed',

        for meter in range(len(custom_meters)):
            print(custom_meters[meter])
            print('meter', meter, custom_meter_names[meter])
            outmeter = idf1.newidfobject("Meter:Custom".upper())
            outmeter.Name = custom_meter_names[meter]
            outmeter.Fuel_Type = 'Electricity'
            print(len(custom_meters[meter]))
            for output in range(len(custom_meters[meter])):
                print(output, custom_meters[meter][output][0])
                if output == 0:
                    outmeter.Key_Name_1 = custom_meters[meter][output][0]
                    outmeter.Output_Variable_or_Meter_Name_1 = custom_meters[meter][output][1]
                if output == 1:
                    outmeter.Key_Name_2 = custom_meters[meter][output][0]
                    outmeter.Output_Variable_or_Meter_Name_2 = custom_meters[meter][output][1]
                if output == 2:
                    outmeter.Key_Name_3 = custom_meters[meter][output][0]
                    outmeter.Output_Variable_or_Meter_Name_3 = custom_meters[meter][output][1]
                if output == 3:
                    outmeter.Key_Name_4 = custom_meters[meter][output][0]
                    outmeter.Output_Variable_or_Meter_Name_4 = custom_meters[meter][output][1]
                if output == 4:
                    outmeter.Key_Name_5 = custom_meters[meter][output][0]
                    outmeter.Output_Variable_or_Meter_Name_5 = custom_meters[meter][output][1]
                if output == 5:
                    outmeter.Key_Name_6 = custom_meters[meter][output][0]
                    outmeter.Output_Variable_or_Meter_Name_6 = custom_meters[meter][output][1]
                if output == 6:
                    outmeter.Key_Name_7 = custom_meters[meter][output][0]
                    outmeter.Output_Variable_or_Meter_Name_7 = custom_meters[meter][output][1]
                if output == 7:
                    outmeter.Key_Name_8 = custom_meters[meter][output][0]
                    outmeter.Output_Variable_or_Meter_Name_8 = custom_meters[meter][output][1]
                if output == 8:
                    outmeter.Key_Name_9 = custom_meters[meter][output][0]
                    outmeter.Output_Variable_or_Meter_Name_9 = custom_meters[meter][output][1]
                if output == 9:
                    outmeter.Key_Name_10 = custom_meters[meter][output][0]
                    outmeter.Output_Variable_or_Meter_Name_10 = custom_meters[meter][output][1]
                if output == 10:
                    outmeter.Key_Name_11 = custom_meters[meter][output][0]
                    outmeter.Output_Variable_or_Meter_Name_11 = custom_meters[meter][output][1]
                if output == 11:
                    outmeter.Key_Name_12 = custom_meters[meter][output][0]
                    outmeter.Output_Variable_or_Meter_Name_12 = custom_meters[meter][output][1]
                if output == 12:
                    outmeter.Key_Name_13 = custom_meters[meter][output][0]
                    outmeter.Output_Variable_or_Meter_Name_13 = custom_meters[meter][output][1]
                if output == 13:
                    outmeter.Key_Name_14 = custom_meters[meter][output][0]
                    outmeter.Output_Variable_or_Meter_Name_14 = custom_meters[meter][output][1]
                if output == 14:
                    outmeter.Key_Name_15 = custom_meters[meter][output][0]
                    outmeter.Output_Variable_or_Meter_Name_15 = custom_meters[meter][output][1]
                if output == 15:
                    outmeter.Key_Name_16 = custom_meters[meter][output][0]
                    outmeter.Output_Variable_or_Meter_Name_16 = custom_meters[meter][output][1]
                if output == 16:
                    outmeter.Key_Name_17 = custom_meters[meter][output][0]
                    outmeter.Output_Variable_or_Meter_Name_17 = custom_meters[meter][output][1]
                if output == 17:
                    outmeter.Key_Name_18 = custom_meters[meter][output][0]
                    outmeter.Output_Variable_or_Meter_Name_18 = custom_meters[meter][output][1]
                if output == 18:
                    outmeter.Key_Name_19 = custom_meters[meter][output][0]
                    outmeter.Output_Variable_or_Meter_Name_19 = custom_meters[meter][output][1]
                if output == 19:
                    outmeter.Key_Name_20 = custom_meters[meter][output][0]
                    outmeter.Output_Variable_or_Meter_Name_20 = custom_meters[meter][output][1]
                if output == 20:
                    outmeter.Key_Name_21 = custom_meters[meter][output][0]
                    outmeter.Output_Variable_or_Meter_Name_21 = custom_meters[meter][output][1]
                if output == 21:
                    outmeter.Key_Name_22 = custom_meters[meter][output][0]
                    outmeter.Output_Variable_or_Meter_Name_22 = custom_meters[meter][output][1]
                if output == 22:
                    outmeter.Key_Name_23 = custom_meters[meter][output][0]
                    outmeter.Output_Variable_or_Meter_Name_23 = custom_meters[meter][output][1]
                if output == 23:
                    outmeter.Key_Name_24 = custom_meters[meter][output][0]
                    outmeter.Output_Variable_or_Meter_Name_24 = custom_meters[meter][output][1]
                if output == 24:
                    outmeter.Key_Name_25 = custom_meters[meter][output][0]
                    outmeter.Output_Variable_or_Meter_Name_25 = custom_meters[meter][output][1]
                if output == 25:
                    outmeter.Key_Name_26 = custom_meters[meter][output][0]
                    outmeter.Output_Variable_or_Meter_Name_26 = custom_meters[meter][output][1]
                if output == 26:
                    outmeter.Key_Name_27 = custom_meters[meter][output][0]
                    outmeter.Output_Variable_or_Meter_Name_27 = custom_meters[meter][output][1]
                if output == 27:
                    outmeter.Key_Name_28 = custom_meters[meter][output][0]
                    outmeter.Output_Variable_or_Meter_Name_28 = custom_meters[meter][output][1]
                if output == 28:
                    outmeter.Key_Name_29 = custom_meters[meter][output][0]
                    outmeter.Output_Variable_or_Meter_Name_29 = custom_meters[meter][output][1]
                if output == 29:
                    outmeter.Key_Name_30 = custom_meters[meter][output][0]
                    outmeter.Output_Variable_or_Meter_Name_30 = custom_meters[meter][output][1]
                if output == 30:
                    outmeter.Key_Name_31 = custom_meters[meter][output][0]
                    outmeter.Output_Variable_or_Meter_Name_31 = custom_meters[meter][output][1]
                if output == 31:
                    outmeter.Key_Name_32 = custom_meters[meter][output][0]
                    outmeter.Output_Variable_or_Meter_Name_32 = custom_meters[meter][output][1]
                if output == 32:
                    outmeter.Key_Name_33 = custom_meters[meter][output][0]
                    outmeter.Output_Variable_or_Meter_Name_33 = custom_meters[meter][output][1]
                if output == 33:
                    outmeter.Key_Name_34 = custom_meters[meter][output][0]
                    outmeter.Output_Variable_or_Meter_Name_34 = custom_meters[meter][output][1]
                if output == 34:
                    outmeter.Key_Name_35 = custom_meters[meter][output][0]
                    outmeter.Output_Variable_or_Meter_Name_35 = custom_meters[meter][output][1]
                if output == 35:
                    outmeter.Key_Name_36 = custom_meters[meter][output][0]
                    outmeter.Output_Variable_or_Meter_Name_36 = custom_meters[meter][output][1]
                if output == 36:
                    outmeter.Key_Name_37 = custom_meters[meter][output][0]
                    outmeter.Output_Variable_or_Meter_Name_37 = custom_meters[meter][output][1]
                if output == 37:
                    outmeter.Key_Name_38 = custom_meters[meter][output][0]
                    outmeter.Output_Variable_or_Meter_Name_38 = custom_meters[meter][output][1]
                if output == 38:
                    outmeter.Key_Name_39 = custom_meters[meter][output][0]
                    outmeter.Output_Variable_or_Meter_Name_39 = custom_meters[meter][output][1]
                if output == 39:
                    outmeter.Key_Name_40 = custom_meters[meter][output][0]
                    outmeter.Output_Variable_or_Meter_Name_40 = custom_meters[meter][output][1]
                if output == 40:
                    outmeter.Key_Name_41 = custom_meters[meter][output][0]
                    outmeter.Output_Variable_or_Meter_Name_41 = custom_meters[meter][output][1]
                if output == 41:
                    outmeter.Key_Name_42 = custom_meters[meter][output][0]
                    outmeter.Output_Variable_or_Meter_Name_42 = custom_meters[meter][output][1]
                if output == 42:
                    outmeter.Key_Name_43 = custom_meters[meter][output][0]
                    outmeter.Output_Variable_or_Meter_Name_43 = custom_meters[meter][output][1]
                if output == 43:
                    outmeter.Key_Name_44 = custom_meters[meter][output][0]
                    outmeter.Output_Variable_or_Meter_Name_44 = custom_meters[meter][output][1]
                if output == 44:
                    outmeter.Key_Name_45= custom_meters[meter][output][0]
                    outmeter.Output_Variable_or_Meter_Name_45 = custom_meters[meter][output][1]
                if output == 45:
                    outmeter.Key_Name_46 = custom_meters[meter][output][0]
                    outmeter.Output_Variable_or_Meter_Name_46 = custom_meters[meter][output][1]
                if output == 46:
                    outmeter.Key_Name_47 = custom_meters[meter][output][0]
                    outmeter.Output_Variable_or_Meter_Name_47 = custom_meters[meter][output][1]
                if output == 47:
                    outmeter.Key_Name_48 = custom_meters[meter][output][0]
                    outmeter.Output_Variable_or_Meter_Name_48 = custom_meters[meter][output][1]
                if output == 48:
                    outmeter.Key_Name_49 = custom_meters[meter][output][0]
                    outmeter.Output_Variable_or_Meter_Name_49 = custom_meters[meter][output][1]
                if output == 49:
                    outmeter.Key_Name_50 = custom_meters[meter][output][0]
                    outmeter.Output_Variable_or_Meter_Name_50 = custom_meters[meter][output][1]
                if output == 50:
                    outmeter.Key_Name_51= custom_meters[meter][output][0]
                    outmeter.Output_Variable_or_Meter_Name_51 = custom_meters[meter][output][1]
                if output == 51:
                    outmeter.Key_Name_52 = custom_meters[meter][output][0]
                    outmeter.Output_Variable_or_Meter_Name_52= custom_meters[meter][output][1]
                if output == 52:
                    outmeter.Key_Name_53 = custom_meters[meter][output][0]
                    outmeter.Output_Variable_or_Meter_Name_53= custom_meters[meter][output][1]
                if output == 53:
                    outmeter.Key_Name_54 = custom_meters[meter][output][0]
                    outmeter.Output_Variable_or_Meter_Name_54= custom_meters[meter][output][1]
                if output == 54:
                    outmeter.Key_Name_55 = custom_meters[meter][output][0]
                    outmeter.Output_Variable_or_Meter_Name_55= custom_meters[meter][output][1]
                if output == 55:
                    outmeter.Key_Name_56 = custom_meters[meter][output][0]
                    outmeter.Output_Variable_or_Meter_Name_56= custom_meters[meter][output][1]
                if output == 56:
                    outmeter.Key_Name_57 = custom_meters[meter][output][0]
                    outmeter.Output_Variable_or_Meter_Name_57= custom_meters[meter][output][1]
                if output == 57:
                    outmeter.Key_Name_58 = custom_meters[meter][output][0]
                    outmeter.Output_Variable_or_Meter_Name_58= custom_meters[meter][output][1]
                if output == 58:
                    outmeter.Key_Name_59 = custom_meters[meter][output][0]
                    outmeter.Output_Variable_or_Meter_Name_59= custom_meters[meter][output][1]
                if output == 59:
                    outmeter.Key_Name_60 = custom_meters[meter][output][0]
                    outmeter.Output_Variable_or_Meter_Name_60= custom_meters[meter][output][1]
                if output == 60:
                    outmeter.Key_Name_61= custom_meters[meter][output][0]
                    outmeter.Output_Variable_or_Meter_Name_61= custom_meters[meter][output][1]
                if output == 61:
                    outmeter.Key_Name_62 = custom_meters[meter][output][0]
                    outmeter.Output_Variable_or_Meter_Name_62= custom_meters[meter][output][1]
                if output == 62:
                    outmeter.Key_Name_63 = custom_meters[meter][output][0]
                    outmeter.Output_Variable_or_Meter_Name_63= custom_meters[meter][output][1]
                if output == 63:
                    outmeter.Key_Name_64 = custom_meters[meter][output][0]
                    outmeter.Output_Variable_or_Meter_Name_64= custom_meters[meter][output][1]
                if output == 64:
                    outmeter.Key_Name_65 = custom_meters[meter][output][0]
                    outmeter.Output_Variable_or_Meter_Name_65= custom_meters[meter][output][1]
                if output == 65:
                    outmeter.Key_Name_66 = custom_meters[meter][output][0]
                    outmeter.Output_Variable_or_Meter_Name_66= custom_meters[meter][output][1]
                if output == 66:
                    outmeter.Key_Name_67 = custom_meters[meter][output][0]
                    outmeter.Output_Variable_or_Meter_Name_67= custom_meters[meter][output][1]
                if output == 67:
                    outmeter.Key_Name_68= custom_meters[meter][output][0]
                    outmeter.Output_Variable_or_Meter_Name_68= custom_meters[meter][output][1]
                if output == 68:
                    outmeter.Key_Name_69 = custom_meters[meter][output][0]
                    outmeter.Output_Variable_or_Meter_Name_69= custom_meters[meter][output][1]
                if output == 69:
                    outmeter.Key_Name_70 = custom_meters[meter][output][0]
                    outmeter.Output_Variable_or_Meter_Name_70= custom_meters[meter][output][1]
                if output == 70:
                    outmeter.Key_Name_71 = custom_meters[meter][output][0]
                    outmeter.Output_Variable_or_Meter_Name_71= custom_meters[meter][output][1]
                if output == 71:
                    outmeter.Key_Name_72 = custom_meters[meter][output][0]
                    outmeter.Output_Variable_or_Meter_Name_72= custom_meters[meter][output][1]
                if output == 72:
                    outmeter.Key_Name_73 = custom_meters[meter][output][0]
                    outmeter.Output_Variable_or_Meter_Name_73= custom_meters[meter][output][1]
                if output == 73:
                    outmeter.Key_Name_74 = custom_meters[meter][output][0]
                    outmeter.Output_Variable_or_Meter_Name_74= custom_meters[meter][output][1]
                if output == 74:
                    outmeter.Key_Name_75 = custom_meters[meter][output][0]
                    outmeter.Output_Variable_or_Meter_Name_75= custom_meters[meter][output][1]
                if output == 75:
                    outmeter.Key_Name_76 = custom_meters[meter][output][0]
                    outmeter.Output_Variable_or_Meter_Name_76= custom_meters[meter][output][1]
                if output == 76:
                    outmeter.Key_Name_77 = custom_meters[meter][output][0]
                    outmeter.Output_Variable_or_Meter_Name_77= custom_meters[meter][output][1]
                if output == 77:
                    outmeter.Key_Name_78 = custom_meters[meter][output][0]
                    outmeter.Output_Variable_or_Meter_Name_78= custom_meters[meter][output][1]
                if output == 78:
                    outmeter.Key_Name_79 = custom_meters[meter][output][0]
                    outmeter.Output_Variable_or_Meter_Name_79= custom_meters[meter][output][1]
                if output == 79:
                    outmeter.Key_Name_80 = custom_meters[meter][output][0]
                    outmeter.Output_Variable_or_Meter_Name_80= custom_meters[meter][output][1]
                if output == 80:
                    outmeter.Key_Name_81= custom_meters[meter][output][0]
                    outmeter.Output_Variable_or_Meter_Name_81= custom_meters[meter][output][1]
                if output == 81:
                    outmeter.Key_Name_82 = custom_meters[meter][output][0]
                    outmeter.Output_Variable_or_Meter_Name_82= custom_meters[meter][output][1]
                if output == 82:
                    outmeter.Key_Name_83= custom_meters[meter][output][0]
                    outmeter.Output_Variable_or_Meter_Name_83= custom_meters[meter][output][1]
                if output == 83:
                    outmeter.Key_Name_84 = custom_meters[meter][output][0]
                    outmeter.Output_Variable_or_Meter_Name_84= custom_meters[meter][output][1]
                if output == 84:
                    outmeter.Key_Name_85 = custom_meters[meter][output][0]
                    outmeter.Output_Variable_or_Meter_Name_85= custom_meters[meter][output][1]
                if output == 85:
                    outmeter.Key_Name_86 = custom_meters[meter][output][0]
                    outmeter.Output_Variable_or_Meter_Name_86= custom_meters[meter][output][1]
                if output == 86:
                    outmeter.Key_Name_87 = custom_meters[meter][output][0]
                    outmeter.Output_Variable_or_Meter_Name_87= custom_meters[meter][output][1]
                if output == 87:
                    outmeter.Key_Name_88 = custom_meters[meter][output][0]
                    outmeter.Output_Variable_or_Meter_Name_88= custom_meters[meter][output][1]
                if output == 88:
                    outmeter.Key_Name_89= custom_meters[meter][output][0]
                    outmeter.Output_Variable_or_Meter_Name_89= custom_meters[meter][output][1]
                if output == 89:
                    outmeter.Key_Name_90 = custom_meters[meter][output][0]
                    outmeter.Output_Variable_or_Meter_Name_90= custom_meters[meter][output][1]
                if output == 90:
                    outmeter.Key_Name_91 = custom_meters[meter][output][0]
                    outmeter.Output_Variable_or_Meter_Name_91= custom_meters[meter][output][1]
                if output == 91:
                    outmeter.Key_Name_92 = custom_meters[meter][output][0]
                    outmeter.Output_Variable_or_Meter_Name_92= custom_meters[meter][output][1]
                if output == 92:
                    outmeter.Key_Name_93 = custom_meters[meter][output][0]
                    outmeter.Output_Variable_or_Meter_Name_93= custom_meters[meter][output][1]
                if output == 93:
                    outmeter.Key_Name_94 = custom_meters[meter][output][0]
                    outmeter.Output_Variable_or_Meter_Name_94= custom_meters[meter][output][1]
                if output == 94:
                    outmeter.Key_Name_95 = custom_meters[meter][output][0]
                    outmeter.Output_Variable_or_Meter_Name_95= custom_meters[meter][output][1]
                if output == 95:
                    outmeter.Key_Name_96 = custom_meters[meter][output][0]
                    outmeter.Output_Variable_or_Meter_Name_96= custom_meters[meter][output][1]
                if output == 96:
                    outmeter.Key_Name_97 = custom_meters[meter][output][0]
                    outmeter.Output_Variable_or_Meter_Name_97= custom_meters[meter][output][1]
                if output == 97:
                    outmeter.Key_Name_98 = custom_meters[meter][output][0]
                    outmeter.Output_Variable_or_Meter_Name_98= custom_meters[meter][output][1]
                if output == 98:
                    outmeter.Key_Name_99 = custom_meters[meter][output][0]
                    outmeter.Output_Variable_or_Meter_Name_99= custom_meters[meter][output][1]
                if output == 99:
                    outmeter.Key_Name_100 = custom_meters[meter][output][0]
                    outmeter.Output_Variable_or_Meter_Name_100= custom_meters[meter][output][1]
                if output == 100:
                    outmeter.Key_Name_101 = custom_meters[meter][output][0]
                    outmeter.Output_Variable_or_Meter_Name_101= custom_meters[meter][output][1]
                if output == 101:
                    outmeter.Key_Name_102 = custom_meters[meter][output][0]
                    outmeter.Output_Variable_or_Meter_Name_102= custom_meters[meter][output][1]
                if output == 102:
                    outmeter.Key_Name_103 = custom_meters[meter][output][0]
                    outmeter.Output_Variable_or_Meter_Name_103= custom_meters[meter][output][1]
                if output == 103:
                    outmeter.Key_Name_104 = custom_meters[meter][output][0]
                    outmeter.Output_Variable_or_Meter_Name_104= custom_meters[meter][output][1]
                if output == 104:
                    outmeter.Key_Name_105 = custom_meters[meter][output][0]
                    outmeter.Output_Variable_or_Meter_Name_105= custom_meters[meter][output][1]
                if output == 105:
                    outmeter.Key_Name_106 = custom_meters[meter][output][0]
                    outmeter.Output_Variable_or_Meter_Name_106= custom_meters[meter][output][1]
                if output == 106:
                    outmeter.Key_Name_107 = custom_meters[meter][output][0]
                    outmeter.Output_Variable_or_Meter_Name_107= custom_meters[meter][output][1]
                if output == 107:
                    outmeter.Key_Name_108 = custom_meters[meter][output][0]
                    outmeter.Output_Variable_or_Meter_Name_108= custom_meters[meter][output][1]
                if output == 108:
                    outmeter.Key_Name_109 = custom_meters[meter][output][0]
                    outmeter.Output_Variable_or_Meter_Name_109= custom_meters[meter][output][1]
                if output == 109:
                    outmeter.Key_Name_110 = custom_meters[meter][output][0]
                    outmeter.Output_Variable_or_Meter_Name_110= custom_meters[meter][output][1]

    # add the TYPICAL METERS
    output_meters = ["Fans:Electricity",
                     "InteriorLights:Electricity",
                     "InteriorEquipment:Electricity",
                     "WaterSystems:Electricity",
                     "Cooling:Electricity",
                     "Heating:Electricity",
                     "Gas:Facility",
                     "Pumps:Electricity",
                     "HeatRejection:Electricity",
                     "HeatRecovery:Electricity",
                     "DHW:Electricity",
                     "ExteriorLights:Electricity",
                     "Humidifier:Electricity",
                     "Cogeneration:Electricity",
                     "Refrigeration:Electricity",
                     "DistrictCooling:Facility",
                     "DistrictHeating:Facility"
                     ]

    # extend any building specific meters
    if building_abr == 'MPEB':
        output_meters.extend(
            [
            "InteriorEquipment:Electricity:Zone:THERMAL ZONE: 409 MACHINE ROOM",
            "InteriorEquipment:Electricity:Zone:THERMAL ZONE: G01B MACHINE ROOM",
            'Cooling:EnergyTransfer:Zone:THERMAL ZONE: 409 MACHINE ROOM',
            'Cooling:EnergyTransfer:Zone:THERMAL ZONE: G01B MACHINE ROOM'
             ])

    if building_abr == '71':
        print('no extra meters')

    if custom_meters:
        output_meters.extend(custom_meter_names)  # add the custom meters to be added to the .mtr file!
        for name in output_meters:
            outmeter = idf1.newidfobject("Output:Meter:MeterFileOnly".upper())
            outmeter.Name = name
            if base_case is True:
                outmeter.Reporting_Frequency = 'timestep'  # 'timestep', 'hourly', 'detailed',
            else:
                outmeter.Reporting_Frequency = 'hourly'  # 'timestep', 'hourly', 'detailed',

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

def set_runperiod(idf1, building_abr, run_period):
    runperiods = idf1.idfobjects['RunPeriod'.upper()]
    if building_abr in {'CH', 'MPEB'}: # 2017
        start_of_week = 'Sunday'
    elif building_abr in {'71'}: # 2014
        start_of_week = 'Wednesday'

    for runperiod in runperiods:
        runperiod.Begin_Month = run_period[0]
        runperiod.Begin_Day_of_Month = run_period[1]
        runperiod.End_Month = run_period[2]
        runperiod.End_Day_of_Month = run_period[3]

        runperiod.Day_of_Week_for_Start_Day = start_of_week
        runperiod.Use_Weather_File_Holidays_and_Special_Days = 'No'
        runperiod.Use_Weather_File_Daylight_Saving_Period = 'No'

def set_holidays(idf1, building_abr):
    if building_abr in {'CH'}: #2017
        holidays = [['Summer bank holiday', '8/28'], ['Spring bank holiday', '5/29'], ['Early may bank holiday', '5/1'],
                    ['Easter Monday', '4/17'], ['Good Friday', '4/14'], ['New Years Day', '1/2']]  # 2017
    elif building_abr in {'MPEB'}: #2017
        holidays = [['Summer bank holiday', '8/28'], ['Spring bank holiday', '5/29'], ['Early may bank holiday', '5/1'],
                    ['Easter Monday', '4/17'], ['Easter2', '4/18'],['Easter3', '4/19'],['Good Friday', '4/14'], ['New Years Day', '1/2'], ['New Years Day', '1/2'], ['New Years Day', '1/2'], ['Christmas', '12/25'],
                    ['Christmas2', '12/26'], ['Christmas3', '12/27'], ['Christmas4', '12/28'], ['Christmas5', '12/29']]  # 2017
    elif building_abr == '71': # 2014
        holidays = [['Boxing Day', '12/26'], ['Christmas Day', '12/25'], ['Summer bank holiday', '8/25'],
                    ['Spring bank holiday', '5/25'], ['Early may bank holiday', '5/5'],
                    ['Easter Monday', '4/21'], ['Good Friday', '4/18'], ['New Years Day', '1/1']]  # 2014

    for h in range(len(holidays)): # create new object for every holiday
        holiday = idf1.newidfobject("RunPeriodControl:SpecialDays".upper())
        holiday.Name = holidays[h][0]
        holiday.Start_Date = str(holidays[h][1])
        holiday.Duration = 1
        holiday.Special_Day_Type = 'Holiday'

def run_lhs(idf1, lhd, building_name, building_abr, base_case, remove_sql, add_variables, run_period, seasonal_occ_factor_week, seasonal_occ_factor_weekend, n_samples, overtime_multiplier_equip, overtime_multiplier_light, multiplier_variation):
    if base_case is False | remove_sql is True:
        idf1.popidfobject('Output:SQLite'.upper(), 0) # remove sql output, have all the outputs in the .eso and meter data in .mtr
    idf1.popidfobject('Output:VariableDictionary'.upper(), 0)
    idf1.popidfobject('Output:Table:SummaryReports'.upper(), 0)
    idf1.popidfobject('OutputControl:Table:Style'.upper(), 0)
    #TODO don't create html and eso outputs...

    #http: // bigladdersoftware.com / epx / docs / 8 - 0 / input - output - reference / page - 088.
    #html  # outputmeter-and-outputmetermeterfileonly
    # change the base idf first by adding ground temps, and removing existing objects before adding new ones.

    add_groundtemps(idf1)
    set_runperiod(idf1, building_abr, run_period)
    set_holidays(idf1, building_abr)
    remove_schedules(idf1, building_abr)
    remove_existing_outputs(idf1)
    add_outputs(idf1, base_case,add_variables, building_abr)

    # Explanation
    # 1. replace_materials_eppy uses mat_props.csv, replace_equipment_eppy uses equipment_props.csv, replace_schedules uses house_scheds.csv
    # 2. adds ground temperatures and output variables
    # 3. define number of idf files to be created and base model file name
    # 4. the functions sample the inputs using lhs, inputs are in the csv files to be manually copied
    # 5. then create separate idf files

    # NOTE:
    # var_num +=1 implies a new random number for another variable generated with LHS

    collect_inputs = []
    if base_case == True: # search script for base_case to adjust internal base_case values
        n_samples = 1

    for run_no in range(n_samples):
        var_num = 0
        input_names = []
        input_values = []

        input_values, input_names, var_num = replace_equipment_eppy(idf1, lhd, input_values,input_names,var_num,run_no,building_abr, base_case)

        var_equip = var_num
        print("number of variables changed for equipment ", var_equip)
        input_values, input_names, var_num = replace_materials_eppy(idf1, lhd, input_values,input_names,var_num,run_no,building_abr, base_case)

        var_mats = var_num-var_equip
        print("number of variables changed for materials ", var_mats)

        # TODO output all used schedules to a file for easy review
        # format the run_no to zero pad to be four values always
        if building_abr == 'CH':
            if base_case is True:
                save_dir = harddrive_idfs + "/01_CentralHouse_Project/BaseCase/"
            else:
                save_dir = harddrive_idfs+"/01_CentralHouse_Project/IDFs/"
        elif building_abr == 'MPEB':
            if base_case is True:
                save_dir = harddrive_idfs + "/05_MaletPlaceEngineering_Project/BaseCase/"
            else:
                save_dir = harddrive_idfs + "/05_MaletPlaceEngineering_Project/IDFs/"
        elif building_abr == '71':
            if base_case is True:
                save_dir = harddrive_idfs + "/03_BuroHappold_71/BaseCase/"
            else:
                save_dir = harddrive_idfs + "/03_BuroHappold_71/IDFs/"

        if not os.path.exists(save_dir): # check if folder exists
            os.makedirs(save_dir) # create new folder

        if base_case is True:
            run_file = save_dir+building_name+"_basecase.idf" # use new folder for save location, zero pad to 4 numbers
        else:
            run_file = save_dir+"/"+building_name+"_" + str(format(run_no, '04')) + ".idf" # use new folder for save location, zero pad to 4 numbers

        idf1.saveas(run_file)
        remove_comments(run_file)

        #replace_schedules append to file instead of using eppy, it will open the written idf file
        var_scheds = var_num-var_mats-var_equip
        input_values, input_names, var_num = replace_schedules(run_file, lhd, input_values,input_names,var_num,run_no,building_abr, base_case, seasonal_occ_factor_week, seasonal_occ_factor_weekend, overtime_multiplier_equip, overtime_multiplier_light, multiplier_variation) #create_schedules = the heating and coolign profiles. Which are not to be created for MPEB (however, the remove_schedules shouldn't delete existing ones. Or create them...

        print("number of variables changed for schedules", var_scheds)

        collect_inputs.append(input_values)

        print("total number of variables ", var_num)
        print(input_names)
        print(input_values)
        print('file saved here', run_file)
        print('file used', "{}".format(rootdir) + "/" + building_name + ".idf")

    #Write inputs to csv file
    collect_inputs.insert(0, input_names) # prepend names to list
    if base_case is True:
        csv_outfile = save_dir+"/inputs_basecase_" + building_name + strftime("_%d_%m_%H_%M", gmtime()) + ".csv"
    elif base_case is not True:
        csv_outfile = save_dir + "/inputs_" + building_name + strftime("%d_%m_%H_%M", gmtime()) + ".csv"

    with open(csv_outfile, 'w') as outf:
        writer = csv.writer(outf, lineterminator='\n')
        writer.writerows(collect_inputs)

def main():
    iddfile = "C:/EnergyPlusV8-6-0/Energy+.idd"
    IDF.setiddname(iddfile)

    # VARIABLES
    base_case = False #todo run basecase straight away with eppy http://pythonhosted.org/eppy/runningeplus.html
    add_variables = False # these are additional variables written to .eso (which are only done when basecase is true, but can be turned of here if not necessary.
    remove_sql = True # when doing MPEB, sql is too big (1gb+)
    building_abr = '71' # 'CH', 'MPEB', # '71' # building_abr
    n_samples = 2000 # how many idfs need to be created
    run_period = [1, 1, 12, 31] #first month, first day, last month, last day

    if building_abr == 'CH':
        building_name = 'CentralHouse_222'  # 'MalletPlace', 'CentralHouse_222' # building_name
    elif building_abr == 'MPEB':
        building_name = 'MaletPlace'
    elif building_abr == '71':
        building_name = 'BH71'

    idf1 = IDF("{}".format(rootdir) + "/" + building_name + ".idf")
    print(idf1)

    if building_abr == 'MPEB':
        no_variables = 350
        seasonal_occ_factor_week = [0.85, .9, 0.85, 0.65, 0.55, 0.55, 0.55, 0.4, 0.45, 0.65, 0.65, 0.65]
        seasonal_occ_factor_weekend = [0.85, .9, 0.85, 0.65, 0.55, 0.55, 0.55, 0.4, 0.45, 0.65, 0.65, 0.65]
        overtime_multiplier_equip = 95
        overtime_multiplier_light = 60
        multiplier_variation = 20
        start_of_week = 'Sunday'
    elif building_abr == 'CH':
        no_variables = 300
        seasonal_occ_factor_week = [0.77277540295697877, 1.0, 0.97137313142714898, 0.87928938389740785, 0.62508333660143534, 0.55137456370838078, 0.47487284241325745, 0.50672575395113539, 0.40279654460602732, 0.77115323595577723, 0.95529236440644738, 0.71144989488078814]
        seasonal_occ_factor_weekend = [0.50020120724346073, 0.72635814889336014, 1.0, 0.92331768388106417, 0.84949698189134804, 0.47183098591549294, 0.51358148893360156, 0.61287726358148897, 0.20422535211267606, 0.41493404873686562, 0.76861167002012076, 0.8138832997987927]
        overtime_multiplier_equip = 65
        overtime_multiplier_light = 30
        multiplier_variation = 20
        start_of_week = 'Sunday' #2017
    elif building_abr == '71':
        no_variables = 200 # make sure there are more than necessary
        seasonal_occ_factor_week = [1 for i in range(1, 13)]
        seasonal_occ_factor_weekend = [1 for i in range(1, 13)]
        overtime_multiplier_equip = 20 # not used
        overtime_multiplier_light = 15 # not used
        multiplier_variation = 20 # not used
        start_of_week = 'Wednesday' #2014

    lhd = doe_lhs.lhs(no_variables, samples=n_samples)
    run_lhs(idf1, lhd, building_name, building_abr, base_case, remove_sql, add_variables, run_period, seasonal_occ_factor_week, seasonal_occ_factor_weekend, n_samples, overtime_multiplier_equip, overtime_multiplier_light, multiplier_variation)

    # print idf1.idfobjects['PEOPLE'][0].objls #fieldnames
    # print idf1.idfobjects['ELECTRICEQUIPMENT'][0].objls #fieldnames
    # print idf1.idfobjects['LIGHTS'][0].fieldnames

if __name__ == '__main__':
    main()
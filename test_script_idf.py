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

# pathnameto_eppy = 'c:/eppy'
pathnameto_eppy = '../'
sys.path.append(pathnameto_eppy)



'''Move idfs files that have not been run, so that they can be run again'''
new_dir = "C:/EngD_hardrive/UCL_DemandLogic/01_CentralHouse_Project/Rerun from 1000"
THIS_DIR = "C:/EngD_hardrive/UCL_DemandLogic/01_CentralHouse_Project/IDFs"
def move_outputfiles(): # moves output files form runfolders to one defined path.
    # home = expanduser("~")  # does this work on legion?

    # os.rename("path/to/current/file.foo", "path/to/new/desination/for/file.foo")
    files_number = ['0000', '0003', '0011', '0024', '0035', '0040', '0042', '0082', '0095', '0117', '0119', '0133', '0134',
             '0139', '0151', '0157', '0162', '0166', '0189', '0190', '0195', '0204', '0206', '0212', '0222', '0238',
             '0246', '0268', '0273', '0275', '0277', '0379', '0391', '0420', '0434', '0440', '0480', '0482', '0490',
             '0498', '0508', '0522', '0533', '0536', '0541', '0545', '0551', '0563', '0602', '0613', '0619', '0660',
             '0666', '0682', '0694', '0720', '0728', '0741', '0748', '0783', '0794', '0795', '0805', '0814', '0817',
             '0819', '0824', '0869', '0874', '0885', '0913', '0934', '0936', '0947', '0962', '0966', '0977', '0980',
             '0982', '0993']

    for subdir, dirs, files in os.walk(THIS_DIR):
        for file in files:
            for i in files_number:
                file_name = 'CentralHouse_222_' + i + '.idf'

                if file == file_name:
                    print(file_name)
                    file_in = subdir + '/' + file_name
                    file_out = new_dir + '/' + file_name
                    os.rename(file_in, file_out)
move_outputfiles()





# if sys.platform=='win32':
#     #print("Operating System: Windows")
#     #rootdir="/".join(__file__.split('\\')[:-1])
#     rootdir=os.path.dirname(os.path.abspath(__file__))
#     harddrive_idfs="C:/EngD_hardrive/UCL_DemandLogic"
#     idfdir=os.path.dirname(os.path.abspath(__file__))+"\IDFs"
#     epdir="C:/EnergyPlusV8-6-0"
# else:
#     print("rootdir not found")
#
# class DLine(): #use .dline[x] to access different columns in the csv files
#     def __init__(self, dline):
#         self.dline = dline
#         self.name = dline[0]
#
# def read_sheet(rootdir, building_abr, fname=None):
#     sheet = []
#
#     if fname != None:
#         if building_abr == 'CH':
#             csvfile = "{0}/data_files/CentralHouse/{1}".format(rootdir, fname)
#         elif building_abr == 'MPEB':
#             csvfile = "{0}/data_files/MalletPlace/{1}".format(rootdir, fname)
#         elif building_abr == '71':
#             csvfile = "{0}/data_files/BH71/{1}".format(rootdir, fname)
#
#         print(csvfile)
#         with open(csvfile, 'r') as inf:
#
#             for row in inf:
#                 row = row.rstrip()
#                 datas = row.split(',')
#                 sheet.append(datas)
#     return sheet
#
#
# def unpick_schedules(rootdir, building_abr):
#     sheet = read_sheet(rootdir, building_abr, fname="house_scheds.csv")
#
#     for nn, lline in enumerate(sheet):
#         if len(lline) > 0 and 'Name' in lline[0]:
#             d_start = nn + 1
#             break
#
#     sched_groups = {}
#     hgroup = None
#     for dline in sheet[d_start:]:
#         if len(dline) > 1:
#             if dline[0] != '':
#
#                 #sname = dline[0]
#                 sched = DLine(dline)
#                 sched_groups[sched.name] = sched
#
#                 # if sname not in list(sched_groups.keys()):
#                 #     hgroup = sched.HGroup(dline)
#                 #     sched_groups[sname] = hgroup
#                 # hgroup = sched_groups[sname]
#                 #
#                 # # Loop her over zones so that can reduce the size of the house_sched csv
#                 # zones = dline[1].split(":")
#                 # if (len(zones) > 1):
#                 #     for zone in zones:
#                 #         zonesched = sched.ZoneSched(dline, zone)
#                 #         hgroup.zones_scheds.append(zonesched)
#                 # else:
#                 #     zonesched = sched.ZoneSched(dline, dline[1])
#                 #     hgroup.zones_scheds.append(zonesched)
#
#     return sched_groups
# def unpick_equipments(rootdir, building_abr):
#     sheet = read_sheet(rootdir, building_abr, fname="equipment_props.csv")
#
#     for nn, lline in enumerate(sheet):
#         if len(lline) > 0 and 'Appliance' in lline[0]:
#             d_start = nn + 1
#             break
#
#     equips = {}
#     equip = None
#
#     for dline in sheet[d_start:]:
#         if len(dline) > 1:
#             if dline[0] != '' and dline[1] != '':
#                 equip = DLine(dline)
#             equips[dline[0]] = equip
#
#     return equips
#
# def remove_schedules(idf1, building_abr):  # todo should I have the option where I remove only the schedules that are going to be replaced?
#     # remove existing schedules
#     equips = unpick_equipments(rootdir, building_abr)
#     scheddict = defaultdict(list)
#
#     load_types = ["People", "Lights", "ElectricEquipment", "ZoneInfiltration:DesignFlowRate"]
#     for y in load_types:
#         print(len(idf1.idfobjects[y.upper()]), "existing schedule objects removed in ", y)
#         for i in range(0, len(idf1.idfobjects[y.upper()])):
#
#             to_replace = [v for v in equips.keys()]
#             existing_loads = [v for v in idf1.idfobjects[y.upper()] if v.Name not in to_replace]
#
#             for load in existing_loads:
#                 idf1.removeidfobject(load)
#                 print(load)
#
# iddfile = "C:/EnergyPlusV8-6-0/Energy+.idd"
# IDF.setiddname(iddfile)
# building_abr = '71'
# building_name = 'BH71_copy'
# idf1 = IDF("{}".format(rootdir) + "/" + building_name + ".idf")
# remove_schedules(idf1, building_abr)
#
# idf1.saveas(building_name+'adjusted.idf')
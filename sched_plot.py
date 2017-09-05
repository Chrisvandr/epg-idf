import numpy as np
import matplotlib.pyplot as plt
import math
import array
import operator
import random
import getpass
import os
import time
import pickle
from collections import defaultdict
import matplotlib.pyplot as plt
#from create_idfs import read_sheet
import pandas as pd

import matplotlib.dates as mdates


colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#8c564b', '#d62728', '#9467bd', '#aec7e8', '#ffbb78', '#98df8a', '#c49c94',
          '#ff9896', '#c5b0d5', 'red', 'green', 'blue', 'black', '#1f77b4', '#ff7f0e', '#2ca02c', '#8c564b', '#d62728',
          '#9467bd', '#aec7e8', '#ffbb78', '#98df8a', '#c49c94', '#ff9896', '#c5b0d5', 'red', 'green', 'blue', 'black']

UserName = getpass.getuser()
DataPath = 'C:/Users/' + UserName + '\Dropbox/01 - EngD/07 - UCL Study/Legion and Eplus/EPG2/data_files/CentralHouse/'
rootdir=os.path.dirname(os.path.abspath(__file__))

class DLine(): #use .dline[x] to access different columns in the csv files
    def __init__(self, dline):
        self.dline = dline
        self.name = dline[0]

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
def unpick_schedules(rootdir, building_abr):
    sheet = read_sheet(rootdir, building_abr, fname="plot_scheds.csv")

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


scheds = unpick_schedules(rootdir, 'CH') #'CH', 'MPEB'
#scheddict = defaultdict(list)

## Binary ##
sched = [0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,0,0,0,0,0,0]
ind = np.arange(len(sched))

print(len(sched))

#fig = plt.figure(0)
#ax = plt.subplot(111)
#ax.step(ind, sched)

timeline = [("0" + str(int(i / 60)) if i / 60 < 10 else str(int(i / 60))) + ":" + (
        "0" + str(i % 60) if i % 60 < 10 else str(i % 60)) for i in range(30, 1470, 30)]

## Temp ##
def plot_temp(timeline):
    HeatingSched = scheds['h_sched_21'].dline[8:8+48] #weekday
    CoolingSched = scheds['c_sched_24'].dline[8:8+48]
    x_ax = np.arange(len(HeatingSched))
    print(x_ax)
    fig = plt.figure(1, figsize=(16/2.54, 4/2.54))

    x_ticks = np.arange(0, len(timeline), 2)
    print(len(timeline),timeline)
    print('xticks', len(x_ticks), x_ticks)
    ax2, ax3 = plt.subplot(111), plt.subplot(111)
    ax2.step(x_ax, CoolingSched)
    #ax3.step(x_ax, cooling, label='Cooling')
    ax2.set_xlabel('Hour')
    ax2.set_ylabel('[degC]')
    ax2.grid(linestyle='--', which='major', linewidth=.25, color='black')

    timeline = timeline[1::2] #remove every other element
    ax2.set_xlim([0, len(CoolingSched)-1])
    ax2.set_xticks(x_ticks)
    ax2.set_xticklabels(timeline)
    print(max(CoolingSched))

    for index, label in enumerate(ax2.xaxis.get_ticklabels()):
        if index % 4 != 1:
            label.set_visible(False)

    ax2.set_yticks([int(min(CoolingSched)), int(max(CoolingSched))])
    ax2.set_ylim([int(min(CoolingSched))-2,int(max(CoolingSched))+2])
    ax2.set_title('Cooling setpoint', fontsize=9)

def plot_occ(timeline):
    Occ = scheds['WifiSum'].dline[8:8 + 48]  # weekday
    OccWkPlus = scheds['WifiSumPlusStd'].dline[8:8 + 48]
    OccWkMinus = scheds['WifiSumMinusStd'].dline[8:8 + 48]
    OccWknd = scheds['WifiSum'].dline[8+48:48+48+8]
    OccWkndPlus = scheds['WifiSumPlusStd'].dline[8+48:48+48+8]
    OccWkndMinus = scheds['WifiSumMinusStd'].dline[8 + 48:48 + 48 + 8]
    x_ax = np.arange(len(Occ))
    print(x_ax)

    ##FIGURE##
    fig = plt.figure(1, figsize=(16 / 2.54, 4 / 2.54))
    ax2, ax3, ax4, ax5, ax6, ax7 = plt.subplot(111), plt.subplot(111), plt.subplot(111), plt.subplot(111), plt.subplot(111), plt.subplot(111)
    ax2.step(x_ax, Occ, color=colors[0], label='Week')
    ax3.step(x_ax, OccWknd, color=colors[1], label='Weekend')
    ax4.step(x_ax, OccWkPlus, linestyle='--', color=colors[0], alpha=.5)
    ax5.step(x_ax, OccWkMinus, linestyle='--', color=colors[0], alpha=.5)
    ax6.step(x_ax, OccWkndPlus, linestyle='--', color=colors[1], alpha=.5)
    ax7.step(x_ax, OccWkndMinus, linestyle='--', color=colors[1], alpha=.5)
    # ax3.step(x_ax, cooling, label='Cooling')

    ##XAXIS##
    xticks = np.arange(0, len(timeline), 2) #xticks needs to be the same length as the timeline
    print(len(timeline), timeline)
    print('xticks', len(xticks), xticks)
    #ax2.set_xlabel('Hour')
    #ax2.set_ylabel('[degC]')
    timeline = timeline[1::2]  # remove every other element
    ax2.set_xlim([0, len(Occ) - 1])
    ax2.set_xticks(xticks)
    ax2.set_xticklabels(timeline)
    print(max(Occ))

    # do not hide every 4th xlabel, so only showing label every two hours (timeline = 48 half hours)
    for index, label in enumerate(ax2.xaxis.get_ticklabels()):
        if index % 4 != 1:
            label.set_visible(False)

    ##YLABEL##
    yticks = np.arange(0, 1.1, .2)
    print(yticks)
    ax2.set_yticks(yticks)
    ax2.set_ylim([0, 1])

    ax2.grid(linestyle='--', which='major', linewidth=.25, color='black')
    ax2.set_title('Occupancy', fontsize=9)
#plot_occ(timeline)

def plot_equip(timeline):
    Eq20 = scheds['Equip_MinStd_OH20_OFF0'].dline[8:8 + 48]
    Eq40 = scheds['Equip_Base_OH40_OFF0'].dline[8:8 + 48]
    Eq60 = scheds['Equip_PlusStd_OH60_OFF0'].dline[8:8 + 48]  # weekday
    Eq20off1 = scheds['Equip_Base_OH20_OFF1'].dline[8:8 + 48]
    Eq20off2 = scheds['Equip_Base_OH20_OFF2'].dline[8 :48 + 8]
    #OccWkndPlus = scheds['WifiSumPlusStd'].dline[8 + 48:48 + 48 + 8]
    #OccWkndMinus = scheds['WifiSumMinusStd'].dline[8 + 48:48 + 48 + 8]
    x_ax = np.arange(len(Eq20))
    print(x_ax)

    ##FIGURE##
    fig = plt.figure(1, figsize=(16 / 2.54, 4 / 2.54))
    ax2, ax3, ax4, ax5, ax6, ax7 = plt.subplot(111), plt.subplot(111), plt.subplot(111), plt.subplot(111), plt.subplot(
        111), plt.subplot(111)
    ax2.step(x_ax, Eq20, color=colors[0], label='20% OH & min occupancy')
    ax3.step(x_ax, Eq40, color=colors[1], label='40% OH & base occupancy')
    ax4.step(x_ax, Eq60, color=colors[2], label='60% OH & max occupancy')
    #ax5.step(x_ax, Eq20off1, color=colors[3], label='Offset 1hr')
    #ax6.step(x_ax, Eq20off2, color=colors[4], label='20% out of hours & min occupancy,oOffset 2hrs')
    #ax7.step(x_ax, OccWkndMinus, linestyle='--', color=colors[1], alpha=.5)
    # ax3.step(x_ax, cooling, label='Cooling')

    ##XAXIS##
    xticks = np.arange(0, len(timeline), 2)  # xticks needs to be the same length as the timeline
    print(len(timeline), timeline)
    print('xticks', len(xticks), xticks)
    # ax2.set_xlabel('Hour')
    # ax2.set_ylabel('[degC]')
    timeline = timeline[1::2]  # remove every other element
    ax2.set_xlim([0, len(Eq20) - 1])
    ax2.set_xticks(xticks)
    ax2.set_xticklabels(timeline)
    print(max(Eq20))

    # do not hide every 4th xlabel, so only showing label every two hours (timeline = 48 half hours)
    for index, label in enumerate(ax2.xaxis.get_ticklabels()):
        if index % 4 != 1:
            label.set_visible(False)

    ##YLABEL##
    yticks = np.arange(0, 1.1, .2)
    print(yticks)
    ax2.set_yticks(yticks)
    ax2.set_ylim([0, 1])

    ax2.grid(linestyle='--', which='major', linewidth=.25, color='black')
    ax2.set_title('Equipment', fontsize=9)


plot_equip(timeline)


def plot_equip_off(timeline):
    #Eq20 = scheds['Equip_MinStd_OH20_OFF0'].dline[8:8 + 48]
    #Eq40 = scheds['Equip_Base_OH40_OFF0'].dline[8:8 + 48]
    #Eq60 = scheds['Equip_PlusStd_OH60_OFF0'].dline[8:8 + 48]  # weekday
    Eq20off1 = scheds['Equip_Base_OH20_OFF0'].dline[8:8 + 48]
    Eq20off2 = scheds['Equip_Base_OH20_OFF2'].dline[8 :48 + 8]
    Eq20off0 = scheds['Equip_Base_OH60_OFF2'].dline[8 :48 + 8]
    #OccWkndMinus = scheds['WifiSumMinusStd'].dline[8 + 48:48 + 48 + 8]
    x_ax = np.arange(len(Eq20off1))
    print(x_ax)

    ##FIGURE##
    fig = plt.figure(1, figsize=(16 / 2.54, 4 / 2.54))
    ax2, ax3, ax4, ax5, ax6, ax7 = plt.subplot(111), plt.subplot(111), plt.subplot(111), plt.subplot(111), plt.subplot(
        111), plt.subplot(111)
    #ax2.step(x_ax, Eq20, color=colors[0], label='20% out of hours & min occupancy')
    #ax3.step(x_ax, Eq40, color=colors[1], label='40% out of hours & base occupancy')
    #ax4.step(x_ax, Eq60, color=colors[2], label='60% out of hours & max occupancy')
    ax5.step(x_ax, Eq20off1, color=colors[0], label='20% OH & no offset')
    ax6.step(x_ax, Eq20off2, color=colors[1], label='20% OH & offset 2hrs')
    #ax7.step(x_ax, Eq20off0, color = colors[5], label='20% OH & base occupancy, no offset')
    # ax3.step(x_ax, cooling, label='Cooling')

    ##XAXIS##
    xticks = np.arange(0, len(timeline), 2)  # xticks needs to be the same length as the timeline
    print(len(timeline), timeline)
    print('xticks', len(xticks), xticks)
    # ax2.set_xlabel('Hour')
    # ax2.set_ylabel('[degC]')
    timeline = timeline[1::2]  # remove every other element
    ax2.set_xlim([0, len(Eq20off1) - 1])
    ax2.set_xticks(xticks)
    ax2.set_xticklabels(timeline)
    print(max(Eq20off1))

    # do not hide every 4th xlabel, so only showing label every two hours (timeline = 48 half hours)
    for index, label in enumerate(ax2.xaxis.get_ticklabels()):
        if index % 4 != 1:
            label.set_visible(False)

    ##YLABEL##
    yticks = np.arange(0, 1.1, .2)
    print(yticks)
    ax2.set_yticks(yticks)
    ax2.set_ylim([0, 1])

    ax2.grid(linestyle='--', which='major', linewidth=.25, color='black')
    ax2.set_title('Equipment', fontsize=9)


#plot_equip_off(timeline)

plt.tight_layout()
#plt.savefig(DataPathImages + 'No_swipe.png', bbox_inches='tight', dpi=300)
plt.legend()
plt.show()

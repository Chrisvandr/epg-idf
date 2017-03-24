__author__ = 'cvdronke'

import getpass
import pandas as pd
import numpy as np
import collections

UserName = getpass.getuser()
DataPath = 'C:\Users/' + UserName + '\Dropbox/01 - EngD/05 - BHprojects\BH projects - EPC vs DEC/'

EPC_database = 'EPC_20150521.csv'
DEC_database = 'DEC_20150521.csv'
BH_projects = 'Projects.csv'
df_epc = pd.read_csv(DataPath + EPC_database, header=0)
df_dec = pd.read_csv(DataPath  + DEC_database, header=0)
df_projects = pd.read_csv(DataPath + BH_projects, header=0)
df_projects = df_projects.drop_duplicates(subset='Project Title') # remove duplicate projects
cols = df_epc.columns.tolist()
cols_prj = df_projects.columns.tolist()

print cols
project_names = pd.Series(list(df_projects['Project Title']))
epc_buildings = pd.Series(list(df_epc['compa']))
dec_buildings = pd.Series(list(df_dec['BUILDING_NAME']))
epc_addr = pd.Series(list(df_epc['ADDR2']))
dec_addr = pd.Series(list(df_dec['ADDR2']))
duplicate_names_epc1 = set(epc_buildings[epc_buildings.isin(project_names)].tolist()) #check if epc-building names are in projects, use set for unique names
duplicate_names_epc2 = set(epc_addr[epc_addr.isin(project_names)].tolist()) #check if epc address names are in projects, use set for unique names
duplicate_names_epc = {i for j in (duplicate_names_epc1, duplicate_names_epc2) for i in j} # append two sets
duplicate_names_dec1 = set(dec_buildings[dec_buildings.isin(project_names)].tolist()) #check if epc-building names are in projects, use set for unique names
duplicate_names_dec2 = set(dec_addr[dec_addr.isin(project_names)].tolist()) #check if epc address names are in projects, use set for unique names
duplicate_names_dec = {i for j in (duplicate_names_dec1, duplicate_names_dec2) for i in j} # append two sets
print duplicate_names_epc
print duplicate_names_dec

var_epc, var_dec = 0, 0
epc_rows, dec_rows = [], []
for i, v in enumerate(df_projects['Project Title']): #iterate through BH projects
    if v in duplicate_names_epc: # if BH project in EPC register
        for j, w in enumerate(df_epc['compa']): # iterate through epc projects
            if v == '':
                v = df_epc['ADDR2'][j]

            if w == v:
                nline = {"PROJECT":v,
                         "EPC":df_epc['ENERGY_RATING'][j],
                         "ENERGYRATING_EPC":df_epc["ENERGYRATING_DESC"][j],
                         "LODGEMENT_DATETIME_EPC": df_epc["LODGEMENT_DATETIME"][j],
                         "FLOOR_AREA_EPC":df_epc["FLOOR_AREA"][j],
                         "ADDR1_EPC":df_epc["ADDR1"][j]}

                dict_epc = {}
                dict_epc.update(nline)
                epc_rows.append(dict_epc)
        var_epc += 1

    if v in duplicate_names_dec: # if BH project in DEC register
        for j, w in enumerate(df_dec['BUILDING_NAME']):  # iterate through dec projects
            if w == '':
                w = df_dec['ADDR2'][j]

            if w == v:
                nline = {"PROJECT": v,
                         "DEC": df_dec['OPERATIONAL_RATING'][j],
                         "ENERGYRATING_DEC": df_dec['ENERGYRATING_DESC'][j],
                         "LODGEMENT_DATETIME_DEC": df_epc["LODGEMENT_DATETIME"][j],
                         "FLOOR_AREA_DEC": df_dec['FLOOR_AREA'][j],
                         "ADDR1_DEC": df_dec["ADDR1"][j]}

                dict_dec = {}
                dict_dec.update(nline)
                dec_rows.append(dict_dec)

        var_dec += 1
print 'no. variables EPC: ', var_epc, 'no. variables DEC:', var_dec
new_rows = epc_rows + dec_rows

df_empty = pd.DataFrame()
df_projects = pd.concat([df_empty, pd.DataFrame(new_rows)])
ncols_prj = df_projects.columns.tolist() #new columns

#sort rows based on project no.
df_projects.sort_values(by=['PROJECT'], inplace=True)

cols1 = df_projects.columns.tolist()
print cols1
cols1 = ['PROJECT', 'ADDR1_EPC', 'ADDR1_DEC', 'EPC', 'ENERGYRATING_EPC', 'FLOOR_AREA_EPC', 'LODGEMENT_DATETIME_EPC', 'DEC', 'ENERGYRATING_DEC', 'FLOOR_AREA_DEC', 'LODGEMENT_DATETIME_DEC']
df_projects = df_projects[cols1]
df_projects.to_csv(DataPath + 'Projects_adj.csv', index=False)

# #TODO check where a project has both an epc and dec rating
# df_projects = pd.read_csv(DataPath + 'Projects_adj.csv', header=0)
#
# df_prj_epc = pd.concat([df_empty, pd.DataFrame(epc_rows)])
# df_prj_dec = pd.concat([df_empty, pd.DataFrame(dec_rows)])
# epc_buildings = pd.Series(list(df_prj_epc['PROJECT']))
# dec_buildings = pd.Series(list(df_prj_dec['PROJECT']))
# duplicate_names_epcdec = set(epc_buildings[epc_buildings.isin(dec_buildings)].tolist())
# print duplicate_names_epcdec
#
#
# appended_data = []
# project_names = set(pd.Series(list(df_projects['PROJECT'])))
# for i, v in enumerate(project_names):
#     if v in(duplicate_names_epcdec):
#         print df_projects.iloc[[i]]
#         data = df_projects.iloc[[i]]
#         appended_data.append(data)
# df_epcdec = pd.DataFrame()
# df_epcdec = pd.concat(appended_data, axis=0)
# print df_epcdec.head()
# df_epcdec.to_csv(DataPath + 'Project_epcdec.csv', index=False)

## TODO it currently only does exact duplicate project names and address, option might be to use wildcards.
## TODO there's duplicate names in the projects.csv and there's duplicate names in the EPC database.
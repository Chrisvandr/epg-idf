__author__ = 'cvdronke'

import getpass

## This replaces the radiators in the .os files


UserName = getpass.getuser()  # get computer username
baseidf = 'C:/Users/'+UserName+'/Dropbox/01 - EngD/07 - UCL Study/05_MaletPlaceEngineering/Model/MalletPlace_v138'

#in OSM file
#to change radiant convective baseboard to just convective...
#delete lines
with open(baseidf + '.osm', 'r') as inf, open(baseidf + '_1' + '.osm', 'w') as outf:
    data = inf.readlines() #read the whole idf
    # for every material replace idf contents with new info
    for num, line in enumerate(data, start=0):
        if 'OS:ZoneHVAC:Baseboard:RadiantConvective:Water' in line:
            data[num] = 'OS:ZoneHVAC:Baseboard:Convective:Water' + ',\n'
            for i in xrange(7): #no of fields
                if i == 5: # this will remove two consecutive lines, if one is removed, the line number changes so delete num+i twice.
                    del data[num+i]
                    del data[num+i]
                    print 'delete this line: ', data[num+i]
                else:
                    print 'skip this line', data[num+i]
                    data[num+4] = data[num+4].replace(',',';')

    for num, line in enumerate(data, start=0):
        if 'OS:Coil:Heating:Water:Baseboard:Radiant' in line:
            data[num] = 'OS:Coil:Heating:Water:Baseboard' + ',\n'

            inlet = data[num+3]
            outlet = data[num+4]


            ufactor = '  autosize'+',\n'


            data.insert(num + 11, ufactor)
            data[num+10] = data[num+10].replace('1', '0.8')
            data.insert(num + 14, outlet)
            data.insert(num+14, inlet)

            del data[num+3]
            del data[num+3]
            del data[num + 3]
            del data[num + 3]
            data[num + 4] = data[num + 4].replace(data[num + 4], ufactor)
            data[num+9] = data[num+9].replace(';', ',')
            data[num+11] = data[num+11].replace(',', ';')

    outf.writelines(data)



# OS:ZoneHVAC:Baseboard:Convective:Water,
#   {d450127f-c959-4d7e-9894-e46a44fcf222}, !- Handle
#   HW Baseboard,                           !- Name
#   {c510c28e-5e4b-4819-92bb-56abb908f2e8}, !- Availability Schedule Name
#   {b761b90b-e0bf-4196-ae99-a2191f9249fa}; !- Heating Coil Name
#
# OS:Coil:Heating:Water:Baseboard,
#   {b761b90b-e0bf-4196-ae99-a2191f9249fa}, !- Handle
#   Baseboard HW Htg Coil,                  !- Name
#   HeatingDesignCapacity,                  !- Heating Design Capacity Method
#   Autosize,                               !- Heating Design Capacity {W}
#   0,                                      !- Heating Design Capacity Per Floor Area {W/m2}
#   0.8,                                    !- Fraction of Autosized Heating Design Capacity
#   autosize,                               !- U-Factor Times Area Value {W/K}
#   autosize,                               !- Maximum Water Flow Rate {m3/s}
#   0.001,                                  !- Convergence Tolerance
#   ,                                       !- Water Inlet Node Name
#   ;                                       !- Water Outlet Node Name
#
#
# OS:ZoneHVAC:Baseboard:RadiantConvective:Water,
#   {691b266e-6a92-4cd0-95ca-015a333fc6c2}, !- Handle
#   R103,                                   !- Name
#   {c510c28e-5e4b-4819-92bb-56abb908f2e8}, !- Availability Schedule Name
#   {13b50bdd-be7d-43c7-be67-331ce5060b45}, !- Heating Coil Name
#   0.3,                                    !- Fraction Radiant
#   0.3;                                    !- Fraction of Radiant Energy Incident on People
#
# OS:Coil:Heating:Water:Baseboard:Radiant,
#   {13b50bdd-be7d-43c7-be67-331ce5060b45}, !- Handle
#   Coil Heating Water Baseboard Radiant 55, !- Name
#   {61b1b6b7-4ec7-4500-b608-4271062f8502}, !- Inlet Node Name
#   {3a5e13b4-d8c0-4363-942e-077673edfdd0}, !- Outlet Node Name
#   87.78,                                  !- Rated Average Water Temperature {C}
#   0.063,                                  !- Rated Water Mass Flow Rate {kg/s}
#   HeatingDesignCapacity,                  !- Heating Design Capacity Method
#   605,                                    !- Heating Design Capacity {W}
#   0,                                      !- Heating Design Capacity Per Floor Area {W/m2}
#   1,                                      !- Fraction of Autosized Heating Design Capacity
#   autosize,                               !- Maximum Water Flow Rate {m3/s}
#   0.001;                                  !- Convergence Tolerance
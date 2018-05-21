import os
from os.path import expanduser
home = expanduser("~") #does this work on legion?

new_dir = 'C:/EngD_hardrive/UCL_DemandLogic/01_CentralHouse_Project/LegionRuns/'
THIS_DIR = os.path.abspath(os.path.dirname(__file__))
#os.rename("path/to/current/file.foo", "path/to/new/desination/for/file.foo")

string = 'eplusmtr.csv'
print(string[:-4])
print(string[-4:])

for subdir, dirs, files in os.walk(THIS_DIR):
    for file in files:
        file_in = subdir + '/' + file
        file_out = THIS_DIR + '/' + file[:-4] + subdir[-4:] + '.csv'
        if file == 'eplusmtr.csv':
            print(file_in)
            print(file_out)
            os.rename(file_in, file_out)
        elif file == 'eplusout.csv':
            print(file_in)
            print(file_out)
            os.rename(file_in, file_out)

"""File containing code to read EnergyPlus idd file"""

import re

def load_idd(epdir):

    iddfile="{0}/Energy+.idd".format(epdir)
    #Needed to add encoding to use python2 encoding as python3 changed the default to uf8 which gives errors
    with open(iddfile,mode='r', encoding='iso-8859-1') as inf:
        iddobjs={}
        thisobj=None
        groupname=None
        fieldname=None
        fieldtype=None
        objectname=None
        for lline in inf:
            if '!IDD_Version' in lline: # returns the version
                version=lline.split(' ')[1][:-1] #returns list of lines split by whitespace
                version=version.strip() #removes whitespace characters
                
            rinfo=lline.split('!')[0][:-1] # returns the idd without comments (!)

            if '\\group' in rinfo:
                groupname=rinfo[7:] # returns groupname without "\group "

            elif '\\field' in rinfo:           
                fieldtype,fieldname=rinfo.split('\\field')
                if ';' in fieldtype:
                    objectname=None
                fieldtype=fieldtype.rstrip()
                fieldtype=fieldtype.rstrip(',')
                fieldtype=fieldtype.strip(';, ')
                fieldname=fieldname.strip(' ')
                fieldname=fieldname.rstrip()
                fieldname=fieldname.rstrip(',')
                if thisobj!=None:
                    thisobj.add_field(fieldname,fieldtype)
                
            elif objectname==None and groupname!=None:
                linfo=rinfo.split('\\')[0]
                linfo=linfo.strip(', ')
                linfo=linfo.rstrip()
                linfo=linfo.rstrip(',')
            
                if linfo!="":
                    thisobj=iddobj(groupname,linfo)
                    iddobjs[linfo]=thisobj
    return version,iddobjs
 
class iddobj():
    def __init__(self,groupname,objectname):
        self.group=groupname
        self.name=objectname
        self.fields={}
        self.fieldnames=[]
    def add_field(self,fieldname,fieldtype):
        self.fieldnames.append(fieldname)
        self.fields[fieldname]=fieldtype
        
    def con_obj(self,indatas):
        objstr="{0}".format(self.name)
        for indata in list(indatas.keys()):
            if indata not in self.fieldnames:
                print((indata," NOT found in ",self.name))
        for field in self.fieldnames:
            
            if field in indatas:
                objstr+=",\n{0}".format(indatas[field])
            else:
                objstr+=",\n"
        objstr+=";\n"
        
        objstr=re.sub('(,\\n)+;\\n',';\\n',objstr)
       
        return objstr
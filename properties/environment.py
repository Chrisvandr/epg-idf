"""File containing the environment Class which uses wind and shading Classes""" 

"""Environment Class"""
class Environment():
    def __init__(self,dline,wind,shades):
        self.dline=dline
        self.name=dline[0]
                
        self.terrain=dline[1]
        
        self.shading=None
        if dline[2] in shades:
            self.shading=shades[dline[2]]

        self.wind=wind
                    
"""Wind Class""" 
class Wind():
    def __init__(self,dline1,dline2):
        
        self.angles=[]
        self.coeffs={}
        self.ninx={}
        self.orients={}
        for nn,(node,orient) in enumerate(zip(dline1[2:],dline2[2:])):
            self.coeffs[node]=[]
            self.ninx[nn]=node
            self.orients[node]=orient

"""Shade Classes"""                                
class Shade():
    def __init__(self,name):
        self.name=name
        self.shade_objects={}
        
class ShadeProps():
    def __init__(self,name):
        self.name=name
        self.command=None
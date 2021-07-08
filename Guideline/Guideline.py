import re
import pandas as pd

data = pd.read_excel('./GC_CSCO_2019_FDU_190619.xlsx')

Scenario = sorted(set(data['Scenario']))

IF = sorted(set(data['IF']))
IF_split = [re.split('AND|OR', x) for x in IF]
IF_dict = {}
for lines in IF_split:
    for line in lines:
        key, value = line.split('=')
        key = key.strip(' "')
        value = value.strip(' "')
        if key not in IF_dict:
            IF_dict[key] = [value]
        else:
            if value not in IF_dict[key]:
                IF_dict[key].append(value)
                
                
THEN = sorted(set(data['THEN']))
THEN_split = [re.split('AND|OR', x) for x in THEN]
THEN_dict = {}
for lines in THEN_split:
    for line in lines:
        key, value = line.split('=')
        key = key.strip(' "')
        value = value.strip(' "')
        _value = re.split('\(Evidence .+?\)|\+', value)
        __value = [x.strip(', ') for x in _value] 
        for x in __value:
            if x != '':
                if key not in THEN_dict:
                    THEN_dict[key] = [x]
                else:
                    if x not in THEN_dict[key]:
                        THEN_dict[key].append(x)
    
       
       
#data = pd.read_excel('./GC_NCCN_2019.1_FDU_191112.xlsx').iloc[1:,]
#
#Scenario = sorted(set(data['Scenario']))
#
#IF = sorted(set(data['IF']))
#IF_split = [re.split('AND|OR', x) for x in IF]
#IF_dict = {}
#for lines in IF_split:
#    for line in lines:
#        key, value = line.split('=')
#        key = key.strip(' "')
#        value = value.strip(' "')
#        if key not in IF_dict:
#            IF_dict[key] = [value]
#        else:
#            if value not in IF_dict[key]:
#                IF_dict[key].append(value)
#                
#                
#THEN = sorted(set(data['THEN']))
#THEN_split = [re.split('AND|OR', x) for x in THEN]
#THEN_dict = {}
#for lines in THEN_split:
#    for line in lines:
#        key, value = line.split('=')
#        key = key.strip(' "')
#        value = value.strip(' "')
#        _value = re.split('\(Evidence .+?\)|\+', value)
#        __value = [x.strip(', ') for x in _value] 
#        for x in __value:
#            if x != '':
#                if key not in THEN_dict:
#                    THEN_dict[key] = [x]
#                else:
#                    if x not in THEN_dict[key]:
#                        THEN_dict[key].append(x)            
    
     
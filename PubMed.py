import os
import json
import pickle
import requests
import pandas as pd
from bs4 import BeautifulSoup as BS


class PubMed():
    """Get PubMed article's Impact Factor (IF)."""
    
    def __init__(self):
        self.header = {'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64;' \
                       'x64)AppleWebKit/537.36 (KHTML, like Gecko) Chrome/' \
                       '86.0.4240.198 Safari/537.36'}
            

    def get_ids(self):
        """Get all PubMedIDs from raw dataset."""
        
        p = 'id_count.json'
        if os.path.exists(p):
            with open(p) as file:
                self.id_count = json.load(file)
        else:
            D = ['alzheimer_disease', 'chronic_obstructive_pulmonary',
                 'colon_cancer', 'coronary_heart_disease', 'diabetes',
                 'gallbladder_cancer', 'gastric_cancer', 'heart_failure',
                 'liver_cancer', 'lung_cancer', 'rheumatatoid_arthritis'] 
            self.id_count = {}
            for d in D:
                _p = '../Dataset/' + d + '/raw/raw_data.csv'
                ID = list(pd.read_csv(_p, encoding = 'ISO-8859-1')['PubMedID'])
                for x in ID:
                    if x not in self.id_count:
                        self.id_count[x] = 1
                    else:
                        self.id_count[x] += 1
            with open(p, 'w') as file:
                json.dump(self.id_count, file)
            
        self.n_raw_id = sum(list(self.id_count.values()))
        self.ids = sorted(set(self.id_count.keys()))    
        self.n_id = len(self.ids)
        print('\n>>  Load {} raw ids. ({:^6} id)'.format(self.n_raw_id,
                                                         self.n_id))
        
    
    def get_id2j(self):
        """Get ID to journal name dict."""
        
        p = 'id2j.json'
        if os.path.exists(p):
            with open(p) as file:
                self.id2j = json.load(file)
        else:            
           self.id2j = {}
              
        self.js = sorted(set(self.id2j.values()))
        self.n_j = len(self.js)                  
        print('>>  Load {:6^}  id2j. ({:^5} journal)'. \
              format(len(self.id2j), self.n_j))
        

    def complete_id2j(self):
        """If id2j dict is not completed, complete id2j."""
        
        if len(self.id2j) == self.n_id:
            return None
        
        print('\n ### Start Completing id2j ###')
        for i in range(self.n_id):
            ID = self.ids[i]
            if ID not in self.id2j:
                print('>>  [{} | {}]: '.format(i + 1, self.n_id), end = '')
                url = 'https://pubmed.ncbi.nlm.nih.gov/' + ID
                obj = BS(requests.get(url, headers = self.header).\
                         text, features = 'html5lib') 
                try:
                    j = obj.find('div', {'class': 'journal-actions dropd' \
                        'own-block'}).find('button').get_text().strip(' \n')
                    print('Done!')
                except:
                    j = '-'
                    print('Missing!')
                self.id2j[ID] = j
                        
            if (i + 1) % 100 == 0 or (i + 1) == self.n_id:
                with open('./id2j.json', 'w') as file:
                    json.dump(self.id2j, file)
        
    
    def get_j2if(self):
        """Get journal name to Impact Factor (IF) dict."""
        
        p = 'j2if.json'
        if os.path.exists(p):
            with open(p) as file:
                self.j2if = json.load(file)
        else:
           self.j2if = {}
                                   
        print('>>  Load {:^6}  j2if. ({:^5} unknown)'.format(len(self.j2if),
              len(['-' for x in self.j2if.values() if x == '-'])))


    def complete_j2if(self):
        """If j2if dict is not completed, complete j2if."""
        
        if len(self.j2if) == self.n_j:
            return None
        
        print('\n ### Start Completing j2if ###')
        for i in range(self.n_j):
            j = self.js[i]
            if j not in self.j2if:
                print('>>  [{} | {}]: {} ==> '.format(i + 1, self.n_j, j),
                      end = '')
                url = 'http://www.greensci.net/search?kw=' + j
                obj = BS(requests.get(url, headers = self.header).\
                         text, features = 'html5lib') 
                self.j2if[j] = self._get_if(obj)
        
        with open('./j2if.json', 'w') as file:
            json.dump(self.j2if, file)            
        
    
    def _get_if(self, obj):
        """
        Get Impact Factor (IF) for a journal name.
        
        Args:
            obj: BeautifulSoup obeject.
        """
        
        try:
            lines = obj.findAll('ul', {'class': 'mail-table'})
            for line in lines:
                scores = [x.get_text() for x in line. \
                          findAll('span', {'class': 'table-content'})]
                for score in scores[::-1]:
                    if '.' in score:
                        IF = float(score)
                        print('{} Done!'.format(IF))
                        return IF
            print('Missing!')
            return '-'
        except:
            print('Missing!')
            return '-'
        

    def get_id2if(self, save = False):
        """
        Get ID to Impact Factor (IF) dict.
        Set the unknown socre of journal as 0.
        
        Args:
            save (False): whether to save new id2if dict.
        """
                
        self.id2if = {}
        n_UNK, n_VAL = 0, 0
        for _id, j in self.id2j.items():
            if self.j2if[j] != '-':
                _if = self.j2if[j]
                self.id2if[_id] = _if
                if _if >= 1:
                    n_VAL += self.id_count[_id]
            else:
                self.id2if[_id] = 0
                n_UNK += self.id_count[_id]
        
        print('>>  Load {:^6} id2if.'.format(self.n_id))
        print('    {:^7} ({:>6.2%}) unknown.'.format(n_UNK, n_UNK / self.n_raw_id))
        print('    {:^7} ({:>6.2%})   valid (IF >= 1).'. \
              format(n_VAL, n_VAL / self.n_raw_id))
        print('    {:^7} ({:>6.2%}) unvalid (IF < 1).'. \
              format(self.n_raw_id - n_VAL - n_UNK,
                     (self.n_raw_id - n_VAL - n_UNK) / self.n_raw_id))
        
        if save:
            with open('./ID2IF.json', 'w') as file:
                json.dump(self.id2if, file)          
        
    

obj = PubMed()
obj.get_ids()
obj.get_id2j()
#obj.complete_id2j()
obj.get_j2if() 
#obj.complete_j2if()
obj.get_id2if(save = True)

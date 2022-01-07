import os
import json


p = 'E_dict.json'
if os.path.exists(p):
    with open(p) as file:
        E_dict = json.load(file)
    print('>>  E_dict already exists !')
else:
    E_dict = {}
    for k in range(6):
        with open('E_dict_' + str(k) + '.json') as file:
            tmp = json.load(file)
        E_dict.update(tmp)
    with open(p, 'w') as file:
        json.dump(E_dict, file)
    print('>>  Generate E_dict Done !')


n_E = len(E_dict)
keys = [x.split('-')[0] for x in sorted(E_dict.keys())]
print('\n>>  Totally {} entity in the E_dict.'.format(n_E))
print('    {:^14}  {:^6}  {:^6}'.format(' *Entity_Type ', 'Number', 'Ratio '))
for g in ['gene', 'mirna', 'protein', 'small_molecule', 'drug',
          'phenotype', 'disease']:
    n, n_c, n_d = 0, 0, 0 
    for i in range(keys.index(g), n_E):
        if keys[i] != g:
            break
        else:
            n += 1
    print('    {:<14}  {:>6}  {:>6.2%}'.format(g, n, n / n_E))

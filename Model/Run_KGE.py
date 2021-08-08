import os
import argparse
import tensorflow.compat.v1 as tf
from Models import *
    
parser = argparse.ArgumentParser(description = 'KGE')

parser.add_argument('--model', type = str, default = 'ConvKB',
                    help = 'model name') #'TransE', 'TransH', 'ConvKB'
parser.add_argument('--disease', type = str, default = 'can',
                    help = 'disease abbreviaton') #look at disease_dict
parser.add_argument('--dim', type = int, default = 256,
                    help = 'embedding dimension')
parser.add_argument('--margin', type = float, default = None,
                    help = 'margin value for TransE and TransH')
parser.add_argument('--n_filter', type = int, default = 4,
                    help = 'number of filters for ConvKB')
parser.add_argument('--dropout', type = float, default = 0.0, 
                    help = 'dropout rate for ConvKB')
parser.add_argument('--lanta_c', type = float, default = 0.00,
                    help = 'lanta for category')
parser.add_argument('--lanta_d', type = float, default = 0.00, 
                    help = 'lanta for description')
parser.add_argument('--l_r', type = float, default = 1e-4, 
                    help = 'learning rate')
parser.add_argument('--batch_size', type = int, default = None,
                    help = 'batch size for SGD')
parser.add_argument('--epoches', type = int, default = 200,
                    help = 'training epoches')
parser.add_argument('--do_train', type = bool, default = True,
                    help = 'whether to train')
parser.add_argument('--save_model', type = bool, default = False,
                    help = 'whether to save trained model')
parser.add_argument('--do_predict', type = bool, default = True,
                    help = 'whether to predict')
parser.add_argument('--do_evaluate', type = bool, default = False,
                    help = 'whether to evaluate for drug-gene, gene-disease, '
                    'and disease-drug triplets')
    
args = parser.parse_args()
disease_dict = \
    {'ad'  : ['alzheimer_disease', 512],
     'cop' : ['chronic_obstructive_pulmonary', 128],
     'cc'  : ['colon_cancer', 1024],
     'chd' : ['coronary_heart_disease', 512],
     'dia' : ['diabetes', 2048],
     'gbc' : ['gallbladder_cancer', 128],
     'gsc' : ['gastric_cancer', 512],
     'hf'  : ['heart_failure', 512],
     'lic' : ['liver_cancer', 2048],
     'luc' : ['lung_cancer', 2048],
     'ra'  : ['rheumatoid_arthritis', 512],
     'can' : ['_cancer', 4096]} 
if not args.batch_size:
    #batch_size is fixed as 1/40 of dataset
    args.batch_size = disease_dict[args.disease][1] 
args.disease = disease_dict[args.disease][0]

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #shielding warning
os.environ['CUDA_VISIBLE_DEVICES'] = '2' #GPU number
config = tf.ConfigProto() 
config.gpu_options.allow_growth = True 
model = eval(args.model + '(args)')
model.run(config) 
        
                        
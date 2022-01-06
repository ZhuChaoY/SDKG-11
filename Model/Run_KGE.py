import os
import argparse
import tensorflow as tf
from Models import *
    
parser = argparse.ArgumentParser(description = 'KGE')

parser.add_argument('--model', type = str, default = 'TransE',
                    help = 'model name') #'TransE', 'TransH', 'ConvKB'
parser.add_argument('--disease', type = str, default = 'ald',
                    help = 'disease abbreviaton') #look at disease_dict blow
parser.add_argument('--dim', type = int, default = 200,
                    help = 'embedding dimension')
parser.add_argument('--margin', type = float, default = None,
                    help = 'margin value for TransX')
parser.add_argument('--n_filter', type = int, default = None,
                    help = 'number of filters for ConvKB')
parser.add_argument('--lanta_c', type = float, default = 0.00,
                    help = 'lanta for category')
parser.add_argument('--lanta_d', type = float, default = 0.00, 
                    help = 'lanta for description')
parser.add_argument('--n_neg', type = int, default = 1,
                    help = 'number of negative samples')
parser.add_argument('--l2', type = float, default = 5e-4,
                    help = 'l2 penalty coefficient')
parser.add_argument('--l_r', type = float, default = 5e-3, 
                    help = 'learning rate')
parser.add_argument('--batch_size', type = int, default = None,
                    help = 'batch size for SGD')
parser.add_argument('--epoches', type = int, default = 1000,
                    help = 'training epoches')
parser.add_argument('--earlystop', type = int, default = 2,
                    help = 'earlystop steps')
parser.add_argument('--save_em', type = bool, default = False,
                    help = 'whether to save embedding result')
parser.add_argument('--do_train', type = bool, default = True,
                    help = 'whether to train')
parser.add_argument('--do_predict', type = bool, default = True,
                    help = 'whether to predict')
parser.add_argument('--do_evaluate', type = bool, default = False,
                    help = 'whether to evaluate for drug-gene, gene-disease, '
                    'and disease-drug triplets')
parser.add_argument('--gpu', type = str, default = '0',
                    help = 'gpu number')
    
args = parser.parse_args()
disease_dict = \
    {'ald' : 'alzheimer_disease',
     'coc' : 'colon_cancer',
     'cop' : 'copd',
     'chd' : 'coronary_heart_disease',
     'dia' : 'diabetes',
     'gac' : 'gallbladder_cancer',
     'gsc' : 'gastric_cancer',
     'hef' : 'heart_failure',
     'lic' : 'liver_cancer',
     'luc' : 'lung_cancer',
     'rha' : 'rheumatoid_arthritis',
     'can' : '_cancer5',
     'dis' : '_disease11'} 
args.disease = disease_dict[args.disease]

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu 
             
config = tf.ConfigProto() 
config.gpu_options.allow_growth = True 
model = eval(args.model + '(args)')
model.run(config) 
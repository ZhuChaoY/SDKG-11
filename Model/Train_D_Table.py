import os
import re
import timeit
import random
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf 
from Tokenization import FullTokenizer
from KGE import myjson, mypickle
from Optimization import AdamWeightDecayOptimizer


class Train_D_Table():
    """Train D table by biobert."""
    
    def __init__(self, args):
        """
        (1) Initialized train_D_table as args dict.
        (2) Load train dataset.
        (3) Construct bioBERT model.
        """
        
        for key, value in dict(args._get_kwargs()).items():
            exec('self.{} = {}'.format(key, value))
            
        self.out_dir = 'C&D/'
        self.initializer = tf.truncated_normal_initializer(stddev = 0.02)    
        self.tokenizer = FullTokenizer('Biobert/vocab.txt')
        
        self._dt_data()
        self._dt_structure()
        
        
    def _dt_data(self):
        """
        (1) Load _all triple and delete relation, then drop duplicates.
        (2) Generate entity index dict.
        (3) Generate description dict.
        (4) Generate train dataset.
        """
        
        print('\n' + '# ' * 3 + ' Loading Dataset ' + ' #' * 3)
        ds = ['alzheimer_disease', 'chronic_obstructive_pulmonary',
              'colon_cancer', 'coronary_heart_disease', 'diabetes', 
              'gallbladder_cancer', 'gastric_cancer', 'heart_failure', 
              'liver_cancer', 'lung_cancer', 'rheumatatoid_arthritis']
        
        fds = [pd.read_csv('../Dataset/' + d + '/train.csv') for d in ds]
        raw = pd.concat(fds)
        raw = raw.drop_duplicates()
        self.n_train = raw.shape[0]
        print('    #triple   : {}'.format(self.n_train))

        H, R, Ta = list(raw['H']), list(raw['R']), list(raw['T'])
        
        self.E = sorted(set(H) | set(Ta))
        self.n_E = len(self.E)
        self.E_index = dict(zip(self.E, range(self.n_E)))
        myjson(self.out_dir + '/E_index', self.E_index)
        print('    #entity   : {}'.format(self.n_E))
        
        self.R = sorted(set(R))
        self.n_R = len(self.R)
        self.R_index = dict(zip(self.R, range(self.n_R)))
        print('    #relation : {}'.format(self.n_R))
                  
        A = myjson('../Annotation/_E_dict')
        self.D_dict = []
        for e in self.E:
            tokens = self.tokenizer.tokenize(A[e]['D'])
            if len(tokens) > self.len_d - 2:
                tokens = tokens[0: (self.len_d - 2)]
            tokens = ['[CLS]'] + tokens + ['[SEP]']
            ids = self.tokenizer.convert_tokens_to_ids(tokens)
            mask = [1] * len(ids)
            while len(ids) < self.len_d:
                ids.append(0)
                mask.append(0)
            self.D_dict.append([ids, mask])
        print('    #D_dict   : {}'.format(self.n_E))         
        
        self.train_E = np.array([[self.D_dict[self.E_index[H[i]]],
                                  self.D_dict[self.E_index[Ta[i]]]]
                                  for i in range(self.n_train)])
        self.train_R = np.array([self.R_index[R[i]] 
                                 for i in range(self.n_train)])
        print('    #train    : {}'.format(self.n_train))
        
        
    def _dt_structure(self):
        """Main structure of training D table."""

        print('\n' + '# ' * 3 + ' Constructing Model ' + ' #' * 3)
        print('    *len_d         : {}'.format(self.len_d))
        print('    *Learning_Rate : {}'.format(self.l_r))
        print('    *Batch_Size    : {}'.format(self.batch_size))
        print('    *Epoches       : {}\n'.format(self.epoches))
        
        tf.reset_default_graph()
        with tf.variable_scope('bert'):
            self._bert_layer()
        with tf.variable_scope('loss'):
            self._finetune_layer()
            
        self._dt_init()
            
        
    def _bert_layer(self):
        """BERT layer, initialized with biobert."""
        
        self.keep_prob = tf.placeholder(tf.float32)
        self.batch_D = tf.placeholder(tf.int32, [None, 2, self.len_d])
        ids = tf.reshape(self.batch_D[:, 0, :], [-1]) #(B * L)
        #(1, 1, L, 1) * (B, 1, 1, L) ==> (B, 1, L, L)
        mask = tf.cast(tf.reshape(self.batch_D[:, 1, :], 
                                  [-1, 1, 1, self.len_d]), tf.float32)
        att_mask = -10000.0 * (1.0 - tf.ones([1, 1, self.len_d, 1]) * mask) 
        
        with tf.variable_scope('embeddings'): #(B, L, 768)
            #(28996, 768) ==> (B * L, 768) ==> (B, L, 768)
            word_table = tf.get_variable('word_embeddings', [28996, 768],
                         initializer = self.initializer)
            em_out = tf.reshape(tf.gather(word_table, ids), 
                                [-1, self.len_d, 768]) 
            #(B, L, 768) + [(512, 768) ==> (1, L, 768)] ==> (B, L, 768)
            position_table = tf.get_variable('position_embeddings',
                             [512, 768], initializer = self.initializer)
            em_out += tf.reshape(tf.slice(position_table, [0, 0],
                      [self.len_d, -1]), [1, self.len_d, 768]) 
            em_out = tf.nn.dropout(layer_norm(em_out), self.keep_prob)
        
        with tf.variable_scope('encoder'): #(B * L, 768)
            prev_out = tf.reshape(em_out, [-1, 768]) #(B * L, 768)    
            for i in range(12):
                with tf.variable_scope('layer_{}'.format(i)):
                    with tf.variable_scope('attention'): #(B * L, 768)
                        att_out = self._attention_layer(prev_out, att_mask)
                    with tf.variable_scope('intermediate'): #(B * L, 3072)
                        mid_out = tf.layers.dense(att_out, 3072, 
                                  activation = gelu,
                                  kernel_initializer = self.initializer)
                    with tf.variable_scope('output'): #(B * L, 768)
                        prev_out = tf.layers.dense(mid_out, 768,
                                   kernel_initializer = self.initializer)
                        prev_out = layer_norm(tf.nn.dropout(prev_out,
                                   self.keep_prob) + att_out)
        
        with tf.variable_scope('pooler'): #(B, 768) 
            #(B * L, 768) ==> (B, L, 768) ==> (B, 768) ==> (B, 768)
            prev_out = tf.squeeze(tf.reshape(prev_out, [-1, self.len_d, 768]) \
                                  [:, 0: 1, :], axis = 1)
            self.bert_out = tf.layers.dense(prev_out, 768, activation = \
                            tf.tanh, kernel_initializer = self.initializer)
        

    def _attention_layer(self, prev_out, att_mask):
        """Attention layer for bert layer"""
        
        with tf.variable_scope('self'): #(B * L, 768)
            #(B * L, 768) => (B * L, 768) => (B, L, 12, 64) => (B, 12, L, 64)
            query = tf.transpose(tf.reshape(tf.layers.dense(prev_out, 768,
                    name = 'query', kernel_initializer = self.initializer), 
                    [-1, self.len_d, 12, 64]), [0, 2, 1, 3])
            key = tf.transpose(tf.reshape(tf.layers.dense(prev_out, 768,
                  name = 'key', kernel_initializer = self.initializer),
                  [-1, self.len_d, 12, 64]), [0, 2, 1, 3])
            value = tf.transpose(tf.reshape(tf.layers.dense(prev_out, 768,
                    name = 'value', kernel_initializer = self.initializer),
                    [-1, self.len_d, 12, 64]), [0, 2, 1, 3])
           
            #(B, 12, L, 64) * (B, 12, 64, L) + (B, 1, L, L) ==> (B, 12, L, L)
            probs = tf.nn.dropout(tf.nn.softmax(0.125 * tf.matmul(query, key, 
                    transpose_b = True) + att_mask), self.keep_prob)
            #(B, 12, L, L) * (B, 12, L, 64) ==> (B, 12, L, 64) 
            # ==> (B, L, 12, 64) ==> (B * L, 768)   
            self_out = tf.reshape(tf.transpose(tf.matmul(probs, value), 
                       [0, 2, 1, 3]), [-1, 768])

        with tf.variable_scope('output'): #(B * L, 768)
            att_out = layer_norm( \
                tf.nn.dropout(tf.layers.dense(self_out, 768,
                kernel_initializer = self.initializer),
                self.keep_prob) + prev_out)
        
        return att_out
            

    def _finetune_layer(self):
        """Finetune layer D_table's generating."""
        
        bs = self.batch_size        
        
        self.batch_R = tf.placeholder(tf.int32, [None])
        self.R_table = tf.nn.l2_normalize(tf.get_variable('relation_table',
                       [self.n_R, 256], initializer = self.initializer), 1)
        
        w = tf.get_variable('weight', [768, 256],
                            initializer = self.initializer)
        
        h_pos = tf.nn.dropout(tf.matmul(self.bert_out[: bs], w),
                              self.keep_prob)
        t_pos = tf.nn.dropout(tf.matmul(self.bert_out[bs: ], w),
                              self.keep_prob)
        h_neg = tf.concat([h_pos[1: ], h_pos[: 1]], 0)
        t_neg = tf.concat([t_pos[1: ], t_pos[: 1]], 0)
                         
        r = tf.gather(self.R_table, self.batch_R)
        
        s_pos = tf.reduce_sum((h_pos + r - t_pos) ** 2, 1)
        s_neg1 = tf.reduce_sum((h_pos + r - t_neg) ** 2, 1)
        s_neg2 = tf.reduce_sum((h_neg + r - t_pos) ** 2, 1)
                        
        self.loss = tf.reduce_sum(tf.nn.relu(2.0 + \
                                             2 * s_pos - s_neg1 - s_neg2))
                        
        n_step = (self.n_train // bs + 1) * self.epoches
        self.train_op = AdamWeightDecayOptimizer(self.loss, self.l_r,
                                                 n_step, n_step // 10)
        

    def _dt_init(self):
        """Initialize dt structure trainable variables."""
        
        tvs = [re.match('^(.*):\\d+$', v.name).group(1)
               for v in tf.trainable_variables()]
        
        if not self.do_train:
            p = self.out_dir + 'model.ckpt'           
            ivs = {v[0]: v[0] for v in tf.train.list_variables(p) 
                   if v[0] in tvs}
        else:
            p = 'Biobert/bert_model.ckpt'          
            ivs = {v[0]: v[0] for v in tf.train.list_variables(p) 
                   if v[0] in tvs and 'bert' in v[0]}
        tf.train.init_from_checkpoint(p, ivs)            
        
        print('\n>>  {} of {} trainable variables initialized.'. \
              format(len(ivs), len(tvs))) 
            
        
    def _dt_train(self, sess):
        """
        Training process of D_table's generating.
        
        Args:
            sess: tf.Session.
        """
        
        bs = self.batch_size
        print('\n' + '# ' * 6 + ' Training ({} EPOCHES) '. \
              format(self.epoches) + ' #' * 6)
        print('    EPOCH   LOSS   time   TIME')
        t0 = t1 = timeit.default_timer()         
        for epoch in range(self.epoches):
            print('    {:^5}'.format(epoch + 1), end = '')
            n_batch = self.n_train // bs
            sample = random.sample(range(self.n_train), self.n_train)
            idxes = [sample[i * bs: (i + 1) * bs] for i in range(n_batch)]
            
            Loss = 0.0
            for idx in idxes:
                pos_E = self.train_E[idx]
                batch_D = np.vstack([pos_E[:, 0], pos_E[:, 1]])
                loss, _ = sess.run([self.loss, self.train_op], 
                                   {self.batch_D: batch_D, 
                                    self.batch_R: self.train_R[idx],
                                    self.keep_prob: 0.9})
                Loss += loss
            
            _t = timeit.default_timer()
            print(' {:>6.4f} {:>6.2f} {:>6.2f}'.format(Loss / n_batch / bs,
                  (_t - t1) / 60, (_t - t0) / 60))  
            t1 = _t
            
        tf.train.Saver().save(sess, self.out_dir + 'model.ckpt')
        
    
    def _dt_predict(self, sess):
        """
        Prediction process of D_table's generating.
        
        Args:
            sess: tf.Session.
        """
                
        bs = self.batch_size
        n_batch = self.n_E // bs
        ds = [self.D_dict[i * bs: (i + 1) * bs] for i in range(n_batch)]
        if n_batch * bs != self.n_E:
            ds.append(self.D_dict[n_batch * bs: ])
        
        print('\n' + '# ' * 6 + ' Prediction ' + ' #' * 6)
        t0 = timeit.default_timer()
        D_table = []
        for d in ds:
            batch_D = np.array(d)
            D_table.append(sess.run(self.bert_out, {self.batch_D: batch_D, 
                                                    self.keep_prob: 1.0}))                            
        print('>>  Total Time: {:.2f}min'. \
              format((timeit.default_timer() - t0) / 60))
        mypickle(self.out_dir + 'D_table', np.vstack(D_table))
        
    
    def run(self, config):
        """
        Running Process.
        
        Args:
            config: tf.ConfigProto
        """
        
        with tf.Session(config = config) as sess:
            tf.global_variables_initializer().run()   
            if self.do_train:
                self._dt_train(sess)
            if self.do_predict:
                self._dt_predict(sess)
    
            
def gelu(x):
    return 0.5 * x * (1.0 + tf.tanh((0.797884 * (x + 0.044715 * x * x * x))))


def layer_norm(inputs):
    return tf.contrib.layers.layer_norm(inputs = inputs,
           begin_norm_axis = -1, begin_params_axis = -1) 
        


parser = argparse.ArgumentParser(description = 'Train_D_table')

parser.add_argument('--len_d', type = str, default = 128,
                    help = 'length of the text') 
parser.add_argument('--l_r', type = float, default = 1e-5, 
                    help = 'learning rate')
parser.add_argument('--batch_size', type = int, default = 8,
                    help = 'batch size for SGD')
parser.add_argument('--epoches', type = int, default = 5,
                    help = 'training epoches')
parser.add_argument('--do_train', type = bool, default = True,
                    help = 'whether to train')
parser.add_argument('--do_predict', type = bool, default = True,
                    help = 'whether to predict')

args = parser.parse_args()
    
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '3' 
config = tf.ConfigProto() 
config.gpu_options.allow_growth = True

model = Train_D_Table(args)
model.run(config)
    
    
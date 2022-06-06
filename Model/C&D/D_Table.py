import re
import time
import json
import random
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf 
import Tokenization as tkz
from Optimization import AdamWeightDecayOptimizer


class D_Table():
    """Train D table by BioBERT."""
    
    def __init__(self, args):
        """
        (1) Initialized D_table as args dict.
        (2) Load train and dev dataset.
        (3) Construct BioBERT model.
        """
        
        self.args = dict(args._get_kwargs())
        for key, value in self.args.items():
            if type(value) == str:
                exec('self.{} = "{}"'.format(key, value))
            else:
                exec('self.{} = {}'.format(key, value))
            
        self.model_dir = '../Pretrained BioBERT/'
            
        with open(self.model_dir + 'bert_config.json') as file:
            config = json.load(file)
            self.dropout = config['hidden_dropout_prob']
            self.hidden = config['hidden_size']
            self.init_range = config['initializer_range']
            self.intermediate = config['intermediate_size']
            self.max_position = config['max_position_embeddings']
            self.n_head = config['num_attention_heads']
            self.n_layer = config['num_hidden_layers']
            self.vocab_size = config['vocab_size']
            
        self.initializer = tf.truncated_normal_initializer(self.init_range)   
        self.tokenizer = tkz.FullTokenizer(self.model_dir + '/vocab.txt')
        
        self.load_data()
        self.construct_model()
        
        
    def load_data(self):
        """
        (1) Load _all triple and delete relation, then drop duplicates.
        (2) Generate entity index dict.
        (3) Generate description dict.
        (4) Generate train and dev dataset by 9:1.
        """
        
        print('\n' + '# ' * 3 + ' Loading Dataset ' + ' #' * 3)
        ds = ['alzheimer_disease', 'colon_cancer', 'copd',
              'coronary_heart_disease', 'diabetes', 'gallbladder_cancer',
              'gastric_cancer', 'heart_failure', 'liver_cancer',
              'lung_cancer', 'rheumatoid_arthritis']
        
        fds = [pd.read_csv('../../Dataset/' + d + '/train.csv') for d in ds]
        raw = pd.concat(fds)
        raw = raw.drop_duplicates()
        self.n_raw = raw.shape[0]
        print('    #triple   : {}'.format(self.n_raw))

        H, R, Ta = list(raw['H']), list(raw['R']), list(raw['T'])
        
        self.E = sorted(set(H) | set(Ta))
        self.n_E = len(self.E)
        self.E_index = dict(zip(self.E, range(self.n_E)))
        print('    #entity   : {}'.format(self.n_E))
        
        self.R = sorted(set(R))
        self.n_R = len(self.R)
        self.R_index = dict(zip(self.R, range(self.n_R)))
        print('    #relation : {}'.format(self.n_R))
                  
        with open('../../Annotation/E_dict.json') as file:
            A = json.load(file)
        self.D_dict = []
        for e in self.E:
            tokens = self.tokenizer.tokenize(tkz.convert_to_unicode(A[e]['D']))
            if len(tokens) > self.len_d - 2:
                tokens = tokens[: (self.len_d - 2)]
            tokens = ['[CLS]'] + tokens + ['[SEP]']
            ids = self.tokenizer.convert_tokens_to_ids(tokens)
            mask = [1] * len(ids)
            while len(ids) < self.len_d:
                ids.append(0)
                mask.append(0)
            self.D_dict.append([ids, mask])       
        
        dev_idx = random.sample(range(self.n_raw), self.n_raw // 10)
        train_E, train_R, dev_E, dev_R = [], [], [], []
        for i in range(self.n_raw):
            if i in dev_idx:
                dev_E.append([self.D_dict[self.E_index[H[i]]],
                              self.D_dict[self.E_index[Ta[i]]]])
                dev_R.append(self.R_index[R[i]])
            else:
                train_E.append([self.D_dict[self.E_index[H[i]]],
                              self.D_dict[self.E_index[Ta[i]]]])
                train_R.append(self.R_index[R[i]])
        self.train_E = np.array(train_E)
        self.train_R = np.array(train_R)
        self.n_train = len(self.train_E)
        self.dev_E = np.array(dev_E)
        self.dev_R = np.array(dev_R)
        self.n_dev = len(self.dev_E)
        print('\n    #train    : {}'.format(self.n_train))
        print('    #dev      : {}'.format(self.n_dev))
        
        
    def construct_model(self):
        """Main structure of training D table."""

        print('\n' + '# ' * 3 + ' Constructing Model ' + ' #' * 3)
        print('    *Length of Sequence : {}'.format(self.len_d))
        print('    *Embedding Dim      : {}'.format(self.dim))
        print('    *Learning_Rate      : {}'.format(self.l_r))
        print('    *Batch_Size         : {}'.format(self.batch_size))
        print('    *Max_Epoch          : {}'.format(self.epoches))
        print('    *Earlystop Steps    : {}'.format(self.earlystop))

        tf.reset_default_graph()
        self.keep = tf.placeholder(tf.float32) 
        with tf.variable_scope('bert'):
            self.transformer_layer()
        with tf.variable_scope('loss'):
            self.finetune_layer()
            n_step = (self.n_train // self.batch_size + 1) * self.epoches
            self.train_op = AdamWeightDecayOptimizer(self.loss, self.l_r,
                                                     n_step, n_step // 10)
            
        
    def transformer_layer(self):
        """BERT layer, initialized with biobert."""
        
        self.batch_D = tf.placeholder(tf.int32, [None, 2, self.len_d])
        ids = tf.reshape(self.batch_D[:, 0, :], [-1]) #(B * L)
        #(1, 1, L, 1) * (B, 1, 1, L) ==> (B, 1, L, L)
        mask = tf.cast(tf.reshape(self.batch_D[:, 1, :], 
                                  [-1, 1, 1, self.len_d]), tf.float32)
        att_mask = -10000.0 * (1.0 - tf.ones([1, 1, self.len_d, 1]) * mask) 
        
        with tf.variable_scope('embeddings'): 
            #(vocab_size, H) ==> (B * L, H) ==> (B, L, H)
            word_table = tf.get_variable('word_embeddings', [self.vocab_size,
                         self.hidden], initializer = self.initializer)
            em_out = tf.reshape(tf.gather(word_table, ids), 
                                [-1, self.len_d, self.hidden]) 
            #(B, L, H) + [(max_position, H) ==> (1, L, H)] ==> (B, L, H)
            position_table = tf.get_variable('position_embeddings',
                             [self.max_position, self.hidden],
                             initializer = self.initializer)
            em_out += tf.reshape(tf.slice(position_table, [0, 0],
                      [self.len_d, -1]), [1, self.len_d, self.hidden]) 
            em_out = self.dropout_layer(norm_layer(em_out))

        with tf.variable_scope('encoder'): #(B * L, H)
            prev_out = tf.reshape(em_out, [-1, self.hidden]) #(B * L, H)    
            for i in range(self.n_layer):
                with tf.variable_scope('layer_{}'.format(i)):
                    att_out = self.attention_layer(prev_out, att_mask)
                    prev_out = self.ffn_layer(att_out)
                    
        with tf.variable_scope('pooler'): #(B, H) 
            #(B * L, H) ==> (B, L, H) ==> (B, H) ==> (B, H)
            self.sequence_out = prev_out
            prev_out = tf.squeeze(tf.reshape(prev_out, [-1, self.len_d, 
                       self.hidden])[:, 0: 1, :], axis = 1)
            self.pooled_out = \
                self.dense_layer(prev_out, self.hidden, None, tf.tanh)
        
    
    def attention_layer(self, prev_out, att_mask):
        """Attention layer for bert layer"""
        
        with tf.variable_scope('attention'): #(B * L, H)
            with tf.variable_scope('self'): 
                #(B * L, H)=>(B * L, H)=>(B, L, head, 64)=>(B, head, L, 64)
                Q = self.dense_layer(prev_out, self.hidden, 'query')
                Q = tf.transpose(tf.reshape(Q, [-1, self.len_d,
                                 self.n_head, 64]), [0, 2, 1, 3])
                K = self.dense_layer(prev_out, self.hidden, 'key')
                K = tf.transpose(tf.reshape(K, [-1, self.len_d,
                                 self.n_head, 64]), [0, 2, 1, 3])
                V = self.dense_layer(prev_out, self.hidden, 'value')
                V = tf.transpose(tf.reshape(V, [-1, self.len_d,
                                 self.n_head, 64]), [0, 2, 1, 3])
                #(B, head, L, 64)*(B, head, 64, L)+(B, 1, L, L)==>(B,head,L,L)
                probs = self.dropout_layer(tf.nn.softmax(0.125 * tf.matmul(Q,
                        K, transpose_b = True) + att_mask))
                #(B, head, L, L) * (B, head, L, 64) ==> (B, head, L, 64) 
                # ==> (B, L, head, 64) ==> (B * L, H)   
                self_out = tf.reshape(tf.transpose(tf.matmul(probs, V), 
                           [0, 2, 1, 3]), [-1, self.hidden])
    
            with tf.variable_scope('output'): #(B * L, H)
                att_out = self.dense_layer(self_out, self.hidden)
                att_out = norm_layer(self.dropout_layer(att_out) + \
                                          prev_out)
        
        return att_out  
        

    def ffn_layer(self, att_out):
        """Feed Forward Network layer."""
        
        with tf.variable_scope('intermediate'): #(B * L, intermediate)
            mid_out = self.dense_layer(att_out, self.intermediate, None, gelu)
        with tf.variable_scope('output'): #(B * L, H)
            prev_out = self.dense_layer(mid_out, self.hidden)
            prev_out = norm_layer(self.dropout_layer(prev_out) + att_out)
        
        return prev_out
    

    def finetune_layer(self):
        """Finetune layer D_table's generating."""
        
        bs = self.batch_size        
        
        self.batch_R = tf.placeholder(tf.int32, [None])
        K1 = np.sqrt(6.0 / self.dim)
        self.R_table = tf.nn.l2_normalize(tf.get_variable('relation_table',
                       initializer = tf.random_uniform([self.n_R, self.dim],
                                                       -K1, K1)), 1)
        K2 = np.sqrt(6.0 / (self.hidden + self.dim))
        W = tf.get_variable('weight', initializer = tf.random_uniform( \
                            [self.hidden, self.dim], -K2, K2))
        
        h_pos = self.dropout_layer(tf.matmul(self.pooled_out[: bs], W))
        t_pos = self.dropout_layer(tf.matmul(self.pooled_out[bs: ], W))
        h_neg = tf.concat([h_pos[1: ], h_pos[: 1]], 0)
        t_neg = tf.concat([t_pos[1: ], t_pos[: 1]], 0)
                         
        r = tf.gather(self.R_table, self.batch_R)
        
        s_pos = tf.reduce_sum((h_pos + r - t_pos) ** 2, 1)
        s_neg1 = tf.reduce_sum((h_pos + r - t_neg) ** 2, 1)
        s_neg2 = tf.reduce_sum((h_neg + r - t_pos) ** 2, 1)
                        
        self.loss = tf.reduce_sum(tf.nn.relu(2.0 + \
                                             2 * s_pos - s_neg1 - s_neg2))
        
        
    def dt_train(self, sess):
        """
        (1) Training process of D_table's generating.
        (2) Evaluate for dev dataset each epoch.
        
        Args:
            sess: tf.Session.
        """
                        
        dev_batches = self.get_batches('dev')
        print('    EPOCH  Dev-LOSS   time   TIME')
        result = {'args': self.args}
        temp_kpi, KPI = [], []
        t0 = t1 = time.time()      
        for ep in range(self.epoches):
            print('    {:^5}'.format(ep + 1), end = '')
            train_batches = self.get_batches('train')          
            for batch_E, batch_R in train_batches:
                batch_D = np.vstack([batch_E[:, 0], batch_E[:, 1]])
                _ = sess.run(self.train_op, {self.batch_D: batch_D, 
                             self.batch_R: batch_R,
                             self.keep: 1 - self.dropout})
                
            kpi = 0.0
            for batch_E, batch_R in dev_batches:
                batch_D = np.vstack([batch_E[:, 0], batch_E[:, 1]])
                loss = sess.run(self.loss, {self.batch_D: batch_D, 
                                self.batch_R: batch_R, self.keep: 1.0})
                kpi += loss
            kpi = round(kpi / self.n_dev, 4)
            
            _t = time.time()
            print(' {:>10.4f} {:>6.2f} {:>6.2f}'.format(kpi, (_t - t1) / 60,
                                                 (_t - t0) / 60), end = '')  
            t1 = _t
            
            if ep == 0 or kpi < KPI[-1]:
                print(' *')
                if len(temp_kpi) > 0:
                    KPI.extend(temp_kpi)
                    temp_kpi = []
                KPI.append(kpi)
                tf.train.Saver().save(sess, 'model.ckpt')     
                result['dev-kpi'] = KPI
                result['best-epoch'] = len(KPI)
                with open('result.json', 'w') as file: 
                    json.dump(result, file) 
            else:
                print('')
                if len(temp_kpi) == self.earlystop:
                    break
                else:
                    temp_kpi.append(kpi)
            
        if ep != self.epoches - 1:
            print('\n    Early stop at epoch of {} !'.format(len(KPI)))
        
    
    def get_batches(self, key):    
        """Get input batches."""
        
        bs = self.batch_size
        n = eval('self.n_' + key)
        sample = random.sample(range(n), n)
        idxes = [sample[i * bs: (i + 1) * bs] for i in range(n // bs)]
        E = eval('self.' + key + '_E')
        R = eval('self.' + key + '_R')
        batches = []
        for idx in idxes:
            batches.append((E[idx], R[idx]))
        return batches
    
    
    def dt_predict(self, sess):
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
        
        t0 = time.time()
        D_table = []
        for d in ds:
            batch_D = np.array(d)
            D_table.append(sess.run(self.pooled_out, {self.batch_D: batch_D, 
                                                      self.keep: 1.0}))
        with open('D_table', 'wb') as file:
            pickle.dump(np.vstack(D_table), file) 
        
        print('    Total Time: {:.2f}min'.format((time.time() - t0) / 60))
        
    
    def initialize_variables(self, mode):
        """
        Initialize D_table structure trainable variables.
        
        Args:
            mode: 'train' or 'predict'
        """
        
        tvs = [re.match('^(.*):\\d+$', v.name).group(1)
               for v in tf.trainable_variables()]
        
        if mode == 'train':
            p = self.model_dir + 'bert_model.ckpt'          
            ivs = {v[0]: v[0] for v in tf.train.list_variables(p) 
                   if v[0] in tvs and 'bert' in v[0]}
        elif mode == 'predict':
            p = 'model.ckpt'           
            ivs = {v[0]: v[0] for v in tf.train.list_variables(p) 
                   if v[0] in tvs}
        tf.train.init_from_checkpoint(p, ivs)   
    
    
    def run(self, config):
        """
        Running Process.
        
        Args:
            config: tf.ConfigProto
        """
        
        if self.do_train:
            print('\n>>  Training Process.')
            self.initialize_variables('train')        
            with tf.Session(config = config) as sess:
                tf.global_variables_initializer().run()   
                self.dt_train(sess)
                
        if self.do_predict:
            print('\n>>  Predict Process.')
            self.initialize_variables('predict')   
            with tf.Session(config = config) as sess:
                tf.global_variables_initializer().run()  
                self.dt_predict(sess)
    

    def dropout_layer(self, _input):
        return tf.nn.dropout(_input, self.keep)

    
    def dense_layer(self, _input, out_dim, name = None, activation = None):
        return tf.layers.dense(_input, out_dim, activation, name = name,
                               kernel_initializer = self.initializer)
    
    
def norm_layer(_input):
    return tf.contrib.layers.layer_norm(inputs = _input,
           begin_norm_axis = -1, begin_params_axis = -1) 
    
            
def gelu(x):
    return 0.5 * x * (1.0 + tf.tanh((0.797884 * (x + 0.044715 * x * x * x))))

import os
import re
import timeit
import random
import numpy as np
import pandas as pd
import tensorflow.compat.v1 as tf
from KGE import makedirs, exists, myjson, mypickle


class PathwayCommons():
    """
    A class of training and prediction for PC's 12 relations
    classification, pretrained by cancer KGE result.
    gene-gene         : CAI, EF, ER, MI, SR, SRP
    gene-molecule     : Ep, TRc
    molecule-gene     : cA, sE
    molecule-molecule : sp, rw
    """
    
    def __init__(self, args):
        """
        (1) Initialize PC with args dict.
        (2) Named out dir and init dir.
        (3) Load entity and model structure.
            E : the entity of PC dataset.
            cE: the entity of cancer dataset.
            E1: the entity both in PC and cancer.
            E2: the entity in PC but not in cancer.
        """
        
        self.args = args
        for key, value in args.items():
            if type(value) == str:
                exec('self.{} = "{}"'.format(key, value))
            else:
                exec('self.{} = {}'.format(key, value))
                                        
        self.dir = '../MBI/PathwayCommons/'        
        if self.pre_train:
            self.out_dir = self.dir + 'Pretrain-' + self.pre_train
            self.init_dir = '../Dataset/_cancer/ConvKB/' + self.pre_train
        else:
            self.out_dir = self.dir + 'No Pretrain'           
        if not exists(self.out_dir):
            makedirs(self.out_dir)
            
        self.initializer = tf.truncated_normal_initializer(stddev = 0.02)
                        
        print('\n' + '= ' * 10 + ' < {} >'.format(self.pre_train if
              self.pre_train else 'No pretrain') + ' =' * 10)   
        self.E = list(pd.read_csv(self.dir + 'entity.csv')['E'])
        self.n_E = len(self.E)
        self.E_dict = dict(zip(self.E, range(self.n_E)))
        
        self.cE = list(pd.read_csv('../Dataset/_cancer/entity.csv')['E'])
        self.n_cE = len(self.cE)
        self.cE_dict = dict(zip(self.cE, range(self.n_cE)))
        
        self.E1 = sorted(set(self.E) & set(self.cE))
        self.E2 = sorted(set(self.E) - set(self.cE))
        
        print('    #Entity           : {} ({} E1 + {} E2)'. \
              format(self.n_E, len(self.E1), len(self.E2)))
        self._structure()
            

    def _structure(self):
        """Main structure of predicting PC."""
        
        print('    *Dim              : {}'.format(self.dim))
        print('    *N_filter         : {}'.format(self.n_filter))
        print('    *Batch_Size       : {}'.format(self.batch_size))
        print('    *Learning_Rate_2  : {}'.format(self.l_r_2))
        print('    *Epoches_2        : {}'.format(self.epoches_2))
        print('    *Learning_Rate_12 : {}'.format(self.l_r_12))
        print('    *Epoches_12       : {}'.format(self.epoches_12))
                        
        tf.reset_default_graph()
        self.X = tf.placeholder(tf.int32, [None, 2])
        self.Y_2 = tf.placeholder(tf.int32, [None])
        self.Y_12 = tf.placeholder(tf.int32, [None])
        self.M = tf.placeholder(tf.float32, [None, 12])
        one_hot_Y_2 = tf.one_hot(self.Y_2, 2)
        one_hot_Y_12 = tf.one_hot(self.Y_12, 12)
        self.keep_prob = tf.placeholder(tf.float32)
        D = self.dim
        
        with tf.variable_scope('structure'): #(B, D, 1, 1)
            if self.do_train_2 and self.pre_train:
                E_pre = mypickle(self.init_dir + '/_E_pre')
                init_E = self._get_init_table(E_pre[:, :, 0, 0])
                self.E_table = tf.get_variable('entity_table', [self.n_E, D],
                               initializer = tf.constant_initializer(init_E))
            else:
                self.E_table = tf.get_variable('entity_table', [self.n_E, D],
                               initializer = self.initializer)           
            self.E_table = tf.nn.l2_normalize(tf.reshape(self.E_table,
                                              [-1, D, 1, 1]), 1)
            h_s = tf.gather(self.E_table, self.X[:, 0])
            t_s = tf.gather(self.E_table, self.X[:, 1])
        
            cols = [h_s, t_s]
            if self.pre_train:
                args = myjson(self.init_dir + '/result')
                lanta_c, lanta_d = args['lanta_c'], args['lanta_d']
                if lanta_c:
                    h_c, t_c = self._C_layer()
                    cols = [lanta_c * t_c] + cols + [lanta_c * h_c]
                if lanta_d:
                    h_d, t_d = self._D_layer()
                    cols = [lanta_d * t_d] + cols + [lanta_d * h_d]
        
        with tf.variable_scope('conv'): #(B, D)        
            F, M = self.n_filter, len(cols)
            #(B, D, M, 1) conv (1, 2, 1, F) ==> (B, D, M - 1, F)
            f1 = tf.get_variable('filter1', [1, 2, 1, F],
                  initializer = self.initializer)
            conv1 = tf.nn.dropout(tf.nn.conv2d(tf.concat(cols, 2), f1, 
                    strides = [1, 1, 1, 1], padding = 'VALID'), self.keep_prob)
            if M == 2: 
                conv_out = conv1
            else:
                #(B, D, M - 1, F) conv (1, 3, F, F) ==> (B, D, M - 3, F)
                f2 = tf.get_variable('filter2', [1, 3, F, F],
                      initializer = self.initializer)
                conv2 = tf.nn.dropout(tf.nn.conv2d(conv1, f2, strides = [1, 1,
                        1, 1], padding = 'VALID'), self.keep_prob)
                if M == 4: 
                    conv_out = conv2
                elif M == 6: 
                    #(B, D, 3, F) conv (1, 3, F, F) ==> (B, D, 1, F)
                    f3 = tf.get_variable('filter3', [1, 3, F, F],
                          initializer = self.initializer)
                    conv3 = tf.nn.dropout(tf.nn.conv2d(conv2, f3, strides = \
                            [1, 1, 1, 1], padding = 'VALID'), self.keep_prob)
                    conv_out = conv3
            
            #(B, D, 1, F) ==> (B, D)
            conv_out = tf.reshape(tf.reduce_mean(conv_out, -1), [-1, D])     
            
        with tf.variable_scope('loss_2'): #(1)
            #(B, D) * (D, 2) + (2) ==> (B, 2)
            w_2 = tf.get_variable('weight_2', [D, 2],                                                    
                                  initializer = self.initializer)
            b_2 = tf.get_variable('bias_2', [2],
                                  initializer = tf.zeros_initializer())
            Y_2_ = tf.clip_by_value(tf.nn.softmax(tf.matmul( \
                    conv_out, w_2) + b_2), 1e-6, 1.0 - 1e-6)
            self.pre_2 = tf.argmax(Y_2_, 1)
            self.loss_2 = tf.reduce_mean(-tf.reduce_sum(one_hot_Y_2 * \
                                                        tf.log(Y_2_), 1))
            self.train_op_2 = tf.train.AdamOptimizer(self.l_r_2). \
                              minimize(self.loss_2)
        
        with tf.variable_scope('loss_12'): #(1)
            #(B, D) * (D, 12) + (12) ==> (B, 12)
            w_12 = tf.get_variable('weight_12', [D, 12],                                                    
                                  initializer = self.initializer)
            b_12 = tf.get_variable('bias_12', [12],
                                  initializer = tf.zeros_initializer())
            Y_12_ = tf.clip_by_value(tf.nn.softmax((tf.matmul( \
                    conv_out, w_12) + b_12) * self.M), 1e-6, 1.0 - 1e-6)
            self.pre_12 = tf.argmax(Y_12_, 1)
            self.loss_12 = tf.reduce_mean(-tf.reduce_sum(one_hot_Y_12 * \
                                                          tf.log(Y_12_), 1))
            self.train_op_12 = tf.train.AdamOptimizer(self.l_r_12). \
                                minimize(self.loss_12)
                
        
    def _C_layer(self):
        """Category layer."""
        
        D = self.dim
        if self.do_train_2:
            C_pre = mypickle(self.init_dir + '/_C_pre')
            init_C = self._get_init_table(C_pre)
            C_table = tf.get_variable('category_table', [self.n_E, D],
                      initializer = tf.constant_initializer(init_C))
        else:
            C_table = tf.get_variable('category_table', [self.n_E, D],
                      initializer = self.initializer)   
        C_table = tf.nn.l2_normalize(tf.reshape(C_table, [-1, D, 1, 1]), 1)
        h_c = tf.gather(C_table, self.X[:, 0])
        t_c = tf.gather(C_table, self.X[:, 1]) 
        return h_c, t_c


    def _D_layer(self):
        """Description layer."""
              
        D = self.dim
        if self.do_train_2:
            D_pre = mypickle(self.init_dir + '/_D_pre')
            init_D = self._get_init_table(D_pre)
            D_table = tf.get_variable('description_table', [self.n_E, D],
                      initializer = tf.constant_initializer(init_D))
        else:
            D_table = tf.get_variable('description_table', [self.n_E, D],
                      initializer = self.initializer)   
        D_table = tf.nn.l2_normalize(tf.reshape(D_table, [-1, D, 1, 1]), 1)
        h_d = tf.gather(D_table, self.X[:, 0])
        t_d = tf.gather(D_table, self.X[:, 1]) 
        return h_d, t_d
        
    
    def _get_init_table(self, pre_table):
        """
        Initialized E_table or C_table or D_table by numpy.
        If the in E1, then initialize it by cancer's embedding;
        Else, initialize it by random truncated normal distribution.
        """        
                
        D = self.dim
        m = np.zeros((self.n_E, D))
        for e in self.E:
            i = self.E_dict[e]
            if e in self.E1:
                m[i, :] = pre_table[self.cE_dict[e], :]
            else:
                for j in range(D):
                    while True:
                        x = np.random.normal(0, 0.02)
                        if np.abs(x) <= 0.04:
                            break
                    m[i, j] = x
        return m

    
    def _data_2(self):
        """Load pc 2 classification data."""
                
        for key in ['train', 'dev', 'test']:
            df = pd.read_csv(self.dir + key + '.csv')
            n_T = df.shape[0]
            H, Ta = list(df['H']), list(df['T'])
            X = np.array([[self.E_dict[H[i]], self.E_dict[Ta[i]]] 
                          for i in range(n_T)])
            n_E = len(set(X[:, 0]) | set(X[:, 1]))
            exec('self.{}_X_2 = X'.format(key))
            exec('self.n_{}_2 = n_T'.format(key))
            print('    #{:5}  : {:>6} ({:>5} E)'.format(key.title(), n_T, n_E))
            
        self.pool = {tuple(x) for x in self.train_X_2.tolist() +
                     self.dev_X_2.tolist() + self.test_X_2.tolist()}
    

    def _init_2(self):
        """Initialize pc structure trainable variables - 2."""
        
        shape = {re.match('^(.*):\\d+$', v.name).group(1):
                 v.shape.as_list() for v in tf.trainable_variables()}
        tvs = [re.match('^(.*):\\d+$', v.name).group(1)
               for v in tf.trainable_variables() if 'loss_12' not in v.name]
        
        if not self.do_train_2:
            p = self.out_dir + '/model_2.ckpt'           
            ivs = {v[0]: v[0] for v in tf.train.list_variables(p) 
                   if v[0] in tvs}
            tf.train.init_from_checkpoint(p, ivs)
        else:
            if self.pre_train:
                ivs = [v for v in tvs if 'structure' in v]
            else:
                ivs = []                

        print('\n>>  {} of {} trainable variables initialized.'. \
              format(len(ivs), len(tvs)))  
        for v in tvs:
            print('    {}{} : {}'.format('*' if v in ivs else '-',
                                         v, shape[v]))
            

    def _train_2(self, sess):  
        """
        (1) Training and Evalution process of prediction for link or nolink.
        (2) Evaluate for dev dataset totally 10 breakpoints during training,
            evaluate for train dataset lastly.
        
        Args:
            sess: tf.Session
        """

        bs, n_train, eps = self.batch_size, self.n_train_2, self.epoches_2
        n_batch = n_train // bs
        print('\n>>  Training Process. ({} EPOCHES) '.format(eps))
        bps = list(range(eps // 10 - 1, eps, eps // 10))
        print('EPOCH LOS_TR LOS_DE **ALL NoLik *Link  time   Time')
        
        t0 = t1 = timeit.default_timer()
        for ep in range(eps):
            sample = random.sample(range(n_train), n_train)
            idxes = [sample[i * bs: (i + 1) * bs] for i in range(n_batch)]
            Loss = 0.0
            for idx in idxes:     
                X_2, Y_2 = self._get_neg('train', idx)
                feed_dict = {self.X: X_2, self.Y_2: Y_2, self.keep_prob: 0.9}
                loss, _ = sess.run([self.loss_2, self.train_op_2], feed_dict)
                Loss += loss           
      
            if ep in bps:
                print('{:^5} {:>6.4f} '. 
                      format(ep + 1, Loss / n_batch), end = '')
                self._classification_2(sess, 'dev')                               
                _t = timeit.default_timer()
                print('{:^6.2f} {:^6.2f}'. \
                      format((_t - t1) / 60, (_t - t0) / 60))
                t1 = _t
                
        tf.train.Saver().save(sess, self.out_dir + '/model_2.ckpt')
    
    
    def _predict_2(self, sess):
        """
        Predict for test dataset - 2.
        
        Args:
            sess: tf.Session.
        """
        
        print('\n>>  Test Classification Result.')
        print('LOS_TE **ALL NoLik *Link')
        self._classification_2(sess, 'test')
        
        
    def _get_neg(self, key, idx = None):
        """
        Get negative X_2 and Y_2 for training.
        
        Args:
            key: 'train' or 'dev' or 'test'
            idx(None): if None: all; else: random idex.
        """
        
        if idx:
            pos_X_2 = eval('self.{}_X_2[idx]'.format(key))
        else:
            pos_X_2 = eval('self.{}_X_2'.format(key))
        
        bs = len(pos_X_2)
        neg_X_2 = []
        for i in range(bs):
            while True:
                new_X = (random.choice(range(self.n_E)), 
                          random.choice(range(self.n_E)))
                if new_X not in self.pool:
                    neg_X_2.append(new_X)
                    break
        
        Y_2 = [1] * bs + [0] * bs
        
        return np.vstack([pos_X_2, np.array(neg_X_2)]), np.array(Y_2)
        
    
    def _classification_2(self, sess, key):
        """
        Classification process - 2, cal loss and accuracy.
        
        Args:
            sess: tf.Session
            key: 'dev' or 'test'
        """
        
        X_2, Y_2 = self._get_neg(key)
        loss, Y_2_, = sess.run([self.loss_2, self.pre_2], {self.X: X_2,
                                self.Y_2: Y_2, self.keep_prob: 1.0})
        
        N = len(Y_2) // 2
        c_m_2 = np.zeros((2, 2), dtype = np.int32)
        for i in range(N):
            c_m_2[0, Y_2_[i + N]] += 1
            c_m_2[1, Y_2_[i]] += 1
        acc_2 = {'all': round((c_m_2[0, 0] + c_m_2[1, 1]) / N / 2, 3),
                  'nolink' : round(c_m_2[0, 0] / sum(c_m_2[0, :]), 3),
                  'link': round(c_m_2[1, 1] / sum(c_m_2[1, :]), 3)}
        result = {'ACC-2': acc_2}
        print('{:>6.4f} {:>5.3f} {:>5.3f} {:>5.3f} '.format(loss,
              acc_2['all'], acc_2['nolink'], acc_2['link']), end = '')
        
        if key == 'test':
            print('\n\nConfusion Matrix:')
            print(c_m_2)
            result.update(self.args)
            if self.do_train_2:
                myjson(self.out_dir + '/result', result)
    
    
    def _data_12(self, sess):
        """
        Load pc 12 classification data.
        
        Args:
            sess: tf.Session.
        """
        
        Y_dict = {'CAI': 0, 'EF': 1, 'ER': 2, 'MI': 3, 'SR': 4, 'SRP': 5,
                  'Ep': 6, 'TRc': 7, 'cA': 8, 'sE': 9, 'sp': 10, 'rw': 11}
        self.inv_Y_dict = {value: key for key, value in Y_dict.items()}
        
        M_dict = {}
        for i in range(6):
            M_dict[i] = [1.0] * 6 + [0.0] * 6
        for i in range(6, 8):
            M_dict[i] = [0.0] * 6 + [1.0] * 2 + [0.0] * 4
        for i in range(8, 10):
            M_dict[i] = [0.0] * 8 + [1.0] * 2 + [0.0] * 2
        for i in range(10, 12):
            M_dict[i] = [0.0] * 10 + [1.0] * 2
        
        for key in ['train', 'dev', 'test']:
            df = pd.read_csv(self.dir + key + '.csv')
            _n_T = df.shape[0]
            _X, _Y = eval('self.{}_X_2'.format(key)), list(df['Y'])
            
            if key == 'train':
                label = [1] * _n_T
            else:
                label = sess.run(self.pre_2, {self.X: _X, self.keep_prob: 1.0})
            
            X, Y, M = [], [], []
            for i in range(_n_T):
                if label[i] == 1:
                    X.append(_X[i])
                    Y.append(Y_dict[_Y[i]])
                    M.append(M_dict[Y[-1]])
            X, Y, M = np.array(X), np.array(Y), np.array(M)
            n_T, n_E = len(X), len(set(X[:, 0]) | set(X[:, 1]))
            exec('self.{}_X_12 = X'.format(key))
            exec('self.{}_Y_12 = Y'.format(key))
            exec('self.{}_M = M'.format(key))
            exec('self.n_{}_12 = n_T'.format(key))
            print('    #{:5}  : {:>6} ({:>5} E)'.format(key.title(), n_T, n_E))
    
    
    def _init_12(self):
        """Initialize pc structure trainable variables - 12."""
        
        shape = {re.match('^(.*):\\d+$', v.name).group(1):
                  v.shape.as_list() for v in tf.trainable_variables()}
        tvs = [re.match('^(.*):\\d+$', v.name).group(1)
                for v in tf.trainable_variables() if 'loss_2' not in v.name]
        
        if not self.do_train_12:
            p = self.out_dir + '/model_12.ckpt'           
            ivs = {v[0]: v[0] for v in tf.train.list_variables(p) 
                    if v[0] in tvs}
            tf.train.init_from_checkpoint(p, ivs)
        else:
            ivs = [v for v in tvs if 'loss_12' not in v]               

        print('\n>>  {} of {} trainable variables initialized.'. \
              format(len(ivs), len(tvs)))  
        for v in tvs:
            print('    {}{} : {}'.format('*' if v in ivs else '-',
                                          v, shape[v]))
    
    
    def _train_12(self, sess):  
        """
        (1) Training and Evalution process of prediction for 12 relations.
        (2) Evaluate for dev dataset totally 10 breakpoints during training,
            evaluate for train dataset lastly.
        
        Args:
            sess: tf.Session
        """

        bs, n_train, eps = self.batch_size, self.n_train_12, self.epoches_12
        n_batch = n_train // bs
        print('\n>>  Training Process. ({} EPOCHES) '.format(eps))
        bps = list(range(eps // 10 - 1, eps, eps // 10))
        print('EPOCH LOS_TR LOS_DE **ALL ', end = '')
        for j in range(12):
            print('{:^5} '.format(self.inv_Y_dict[j]), end = '')
        print(' time   TIME')
        
        t0 = t1 = timeit.default_timer()
        for ep in range(eps):
            sample = random.sample(range(n_train), n_train)
            idxes = [sample[i * bs: (i + 1) * bs] for i in range(n_batch)]
            Loss = 0.0
            for idx in idxes:     
                feed_dict = {self.X: self.train_X_12[idx], 
                              self.Y_12: self.train_Y_12[idx],
                              self.M: self.train_M[idx], self.keep_prob: 0.9}
                loss, _ = sess.run([self.loss_12, self.train_op_12], feed_dict)
                Loss += loss           
      
            if ep in bps:
                print('{:^5} {:>6.4f} '. 
                      format(ep + 1, Loss / n_batch), end = '')
                self._classification_12(sess, 'dev')                                 
                _t = timeit.default_timer()
                print('{:^6.2f} {:^6.2f}'. \
                      format((_t - t1) / 60, (_t - t0) / 60))
                t1 = _t
                
        tf.train.Saver().save(sess, self.out_dir + '/model_12.ckpt')
            
    
    def _predict_12(self, sess):
        """
        Predict for test dataset - 12.
        
        Args:
            sess: tf.Session.
        """
        
        print('\n>>  Test Classification Result.')
        print('LOS_TE **ALL ', end = '')
        for j in range(12):
            print('{:^5} '.format(self.inv_Y_dict[j]), end = '')
        print('')
        self._classification_12(sess, 'test')
        
        
    def _classification_12(self, sess, key):
        """
        Classification process - 12, cal loss and accuracy.
        
        Args:
            sess: tf.Session
            key: 'dev' or 'test'
        """
        
        X_12 = eval('self.{}_X_12'.format(key))
        Y_12 = eval('self.{}_Y_12'.format(key))
        M = eval('self.{}_M'.format(key))
        loss, Y_12_, = sess.run([self.loss_12, self.pre_12],
                                {self.X: X_12, self.Y_12: Y_12, self.M: M,
                                  self.keep_prob: 1.0})
        
        N = len(Y_12)
        c_m_12 = np.zeros((12, 12), dtype = np.int32)
        for i in range(N):
            c_m_12[Y_12[i], Y_12_[i]] += 1
        acc_12, K = {}, 0
        for j in range(12):
            k = c_m_12[j, j]
            acc_12[self.inv_Y_dict[j]] = round(k / sum(c_m_12[j, :]), 3)
            K += k
        acc_12['all'] = round(K / sum(sum(c_m_12)), 3)
        result = myjson(self.out_dir + '/result')
        result['ACC-12'] = acc_12
        print('{:>6.4f} {:>5.3f} '.format(loss, acc_12['all']), end = '')
        for j in range(12):
            print('{:>5.3f} '.format(acc_12[self.inv_Y_dict[j]]), end = '')
        
        if key == 'test':
            print('\n\nConfusion Matrix:')
            print(c_m_12)
            if self.do_train_12:
                myjson(self.out_dir + '/result', result)
        
    
    def run(self, config):
        """
        Running Process.
        
        Args:
            config: tf.ConfigProto
        """

        print('\n' + '- ' * 5 + ' < Classification - 2 >' + ' -' * 5) 
        self._data_2()
        self._init_2()
        with tf.Session(config = config) as sess:
            tf.global_variables_initializer().run()   
            if self.do_train_2:
                self._train_2(sess)
            self._predict_2(sess)
            
            print('\n' + '- ' * 5 + ' < Classification - 12 >' + ' -' * 5) 
            self._data_12(sess)
        self._init_12()
        with tf.Session(config = config) as sess:
            tf.global_variables_initializer().run()   
            if self.do_train_12:
                self._train_12(sess)
            self._predict_12(sess)
        

def main():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['CUDA_VISIBLE_DEVICES'] = '1' 
    config = tf.ConfigProto() 
    config.gpu_options.allow_growth = True
    
    args = \
        {'pre_train'   : None,
         'dim'         : 256,
         'n_filter'    : 4,
         'batch_size'  : 16384,
         'l_r_2'       : 1e-4,
         'epoches_2'   : 100,
         'do_train_2'  : True,
         'l_r_12'      : 1e-4,
         'epoches_12'  : 100,
         'do_train_12' : True
         }
    
    model = PathwayCommons(args) 
    model.run(config)

if __name__ == '__main__':
    main()
    
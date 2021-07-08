import re
import json
import pickle
import timeit
import random
import numpy as np
import pandas as pd
import tensorflow.compat.v1 as tf
from os import makedirs
from os.path import exists


class KGE():
    """
    A class of processing and tool functions for Knowledge Graph Embedding.
    Abbreviation:
        c(C) : Category(s) 
        d(D) : Description(s)  
        s(S) : Structure(s)
        t(T) : Triple(s)
        r(R) : Relation(s)
        e(E) : Entity(ies)
        h(H) : Head(s)
        t(T)a: Tail(s)
        p    : path
    """
    
    def __init__(self, args):
        """
        (1) Initialize KGE with args dict.
        (2) Named model dir and out dir.
        (2) Load entity, relation and triple.
        """
        
        self.args = args
        for key, value in self.args.items():
            if type(value) == str:
                exec('self.{} = "{}"'.format(key, value))
            else:
                exec('self.{} = {}'.format(key, value))
                        
        self.dir = '../Dataset/{}/'.format(self.disease)
        self.out_dir = self.dir + self.model + '/' + ''.join(['S',  
                       '+C' if self.lanta_c else '',
                       '+D' if self.lanta_d else '']) + '/'
        if not exists(self.out_dir):
            makedirs(self.out_dir)
            
        self.initializer = tf.truncated_normal_initializer(stddev = 0.02)
        
        print('\n' + '==' * 4 + ' < {} > && < {} >'.format(self.disease,
              self.model) + '==' * 4)        
        self._em_data()
        self._em_structure()
    
              
    def _em_data(self):
        """
        (1) Get entity mapping dict (E_dict).
        (2) Get relation mapping dict (R_dict), and the index for activate 
            and inhibit.
        (3) Get train, dev and test dataset for embedding.
        (4) Get replace_h_prob dict and triple pool for negative 
            sample's generation.
        """
        
        self.E = list(pd.read_csv(self.dir + 'entity.csv')['E'])
        self.n_E = len(self.E)
        self.E_dict = dict(zip(self.E, range(self.n_E)))
        self.E_reverse_dict = dict(zip(range(self.n_E), self.E))
              
        self.R = list(pd.read_csv(self.dir + 'relation.csv')['R'])
        self.n_R = len(self.R)
        self.R_dict = dict(zip(self.R, range(self.n_R)))
        self.R_reverse_dict = dict(zip(range(self.n_R), self.R))
    
        p = self.dir + '_T.data'            
        if exists(p):
            self.T = mypickle(p)
        else:
            self.T = {}
            for key in ['train', 'dev', 'test']: 
                df = pd.read_csv(self.dir + key + '.csv')
                H, R, Ta = list(df['H']), list(df['R']), list(df['T'])
                t = [[self.E_dict[H[i]], self.R_dict[R[i]], self.E_dict[Ta[i]]]
                     for i in range(df.shape[0])]                            
                self.T[key] = np.array(t)
            mypickle(p, self.T)
                    
        for key in ['train', 'dev', 'test']:
            T = self.T[key]
            n_T = len(T)
            exec('self.{} = T'.format(key))
            exec('self.n_{} = n_T'.format(key))
            print('    #{:5} : {:6} ({:>5} E + {:>3} R)'.format( \
                  key.title(), n_T, len(set(T[:, 0]) | set(T[:, 2])),
                  len(set(T[:, 1]))))
                            
        rpc_h_prob = {}
        for r in range(self.n_R):
            idx = np.where(self.train[:, 1] == r)[0]
            t_per_h = len(idx) / len(set(self.train[idx, 0]))
            h_per_t = len(idx) / len(set(self.train[idx, 2]))
            rpc_h_prob[r] = t_per_h / (t_per_h + h_per_t)
        self.rpc_h = lambda r : np.random.binomial(1, rpc_h_prob[r])
        
        self.pool = {tuple(x) for x in self.train.tolist() +
                     self.dev.tolist() + self.test.tolist()}
    
    
    def _projector(self, s, p):
        """
        A projection function for C and D layer.
        
        Args:
            s : score vector
            p : project normal vector
        """
        
        p = tf.nn.l2_normalize(p, 1)
        return s - tf.reduce_sum(p * s, 1, keepdims = True) * s
    
    
    def _C_layer(self):
        """Category layer"""
        
        all_C = mypickle('C&D/C_dict')
        E_index = myjson('C&D/E_index')
        C_dict = [all_C[E_index[e]] for e in self.E]
                
        c_list = []
        for c in C_dict:
            c_list.extend(c)
        
        c_list = sorted(set(c_list))
        n_C = len(c_list)
        c_dict = dict(zip(c_list, range(n_C))) 
        
        C_dict = tf.to_int32([[c_dict[x] for x in y] for y in C_dict])
        C_table = tf.nn.l2_normalize(tf.get_variable('category_table',
                  [n_C, self.dim], initializer = self.initializer), 1)            
        
        h_c_pos = tf.reduce_mean(tf.gather(C_table,
                  tf.gather(C_dict, self.T_pos[:, 0])), 1)
        t_c_pos = tf.reduce_mean(tf.gather(C_table,
                  tf.gather(C_dict, self.T_pos[:, -1])), 1)        
        h_c_neg = tf.reduce_mean(tf.gather(C_table,
                  tf.gather(C_dict, self.T_neg[:, 0])), 1)
        t_c_neg = tf.reduce_mean(tf.gather(C_table,
                  tf.gather(C_dict, self.T_neg[:, -1])), 1)
        
        self.C_pre = tf.reduce_mean(tf.gather(C_table,
                     tf.gather(C_dict, self.T_pos[:, 0])), 1)
        
        return h_c_pos - t_c_pos, h_c_neg - t_c_neg         


    def _D_layer(self):
        """Description layer"""
        
        all_D = mypickle('C&D/D_table')
        E_index = myjson('C&D/E_index')

        init_D = np.array([all_D[E_index[e]] for e in self.E])        
        _D_table = tf.nn.l2_normalize( \
                   tf.get_variable('description_table', [self.n_E, 768],
                   initializer = tf.constant_initializer(init_D)), 1)
        d_w = tf.get_variable('d_weight', [768, self.dim],
                              initializer = self.initializer)
        d_b = tf.get_variable('d_bias', [self.dim],
                              initializer = tf.zeros_initializer())
        D_table = tf.nn.dropout(tf.matmul(_D_table, d_w) + d_b, self.keep_prob)
        
        h_d_pos = tf.gather(D_table, self.T_pos[:, 0])
        t_d_pos = tf.gather(D_table, self.T_pos[:, -1])                                                  
        h_d_neg = tf.gather(D_table, self.T_neg[:, 0])
        t_d_neg = tf.gather(D_table, self.T_neg[:, -1])
        
        self.D_pre = tf.gather(D_table, self.T_pos[:, 0])
        
        return h_d_pos - t_d_pos, h_d_neg - t_d_neg


    def _em_init(self):
        """Initialize embedding trainable variables."""
        
        shape = {re.match('^(.*):\\d+$', v.name).group(1):
                 v.shape.as_list() for v in tf.trainable_variables()}
        tvs = [re.match('^(.*):\\d+$', v.name).group(1)
               for v in tf.trainable_variables()]
                       
        if not self.do_train:
            p = self.out_dir + 'model.ckpt'
        else:
            if self.lanta_c or self.lanta_d:
                p = self.dir + self.model + '/S/model.ckpt'
            else:
                p = None
                ivs = {}
        if p:
            ivs = {v[0]: v[0] for v in tf.train.list_variables(p) 
                   if v[0] in tvs}
            tf.train.init_from_checkpoint(p, ivs)
            
        print('\n>>  {} of {} trainable variables initialized.'. \
              format(len(ivs), len(tvs)))  
        for v in tvs:
            print('    {}{} : {}'.format('*' if v in ivs else '-', v, 
                                         shape[v]))
            
        
    def _em_train(self, sess):  
        """
        (1) Training and Evalution process of embedding.
        (2) Evaluate for dev dataset totally 10 breakpoints during training,
            evaluate for train dataset lastly.
        
        Args:
            sess: tf.Session
        """

        bs, n_train, eps = self.batch_size, self.n_train, self.epoches
        n_batch = n_train // bs
        bps = list(range(eps // 10 - 1, eps, eps // 10))
        print('\n>>  Training Process. ({} EPOCHES) '.format(eps))
        if self.do_predict:
            print('    EPOCH  LOSS    time   TIME')
        else:
            print('    EPOCH  LOSS   E-MR  E-MRR R-MR R-MRR  time   TIME')  
            
        t0 = t1 = timeit.default_timer()
        for ep in range(eps):
            sample = random.sample(range(n_train), n_train)
            idxes = [sample[i * bs: (i + 1) * bs] for i in range(n_batch)]
        
            Loss = 0.0
            for idx in idxes:     
                T_pos = self.train[idx]
                T_neg = self._get_T_neg(T_pos)
                feed_dict = {self.T_pos: T_pos, self.T_neg: T_neg,
                             self.keep_prob: 0.9}
                loss, _ = sess.run([self.loss, self.train_op], feed_dict)
                Loss += loss         
      
            if ep in bps:
                print('    {:^5} {:>6.4f} '. \
                      format(ep + 1, Loss / n_batch / bs), end = '')
                if not self.do_predict:
                    if ep / eps > 0.2:
                        _ = self._link_prediction(sess, self.dev)
                    else:
                        print('  --    ---   --   --- ', end = '')                            
                _t = timeit.default_timer()
                print(' {:^6.2f} {:^6.2f}'. \
                      format((_t - t1) / 60, (_t - t0) / 60))
                t1 = _t
        
        if self.do_predict:
            tf.train.Saver().save(sess, self.out_dir + 'model.ckpt')
        else:
            print('\n    Train:  --   ', end = '')
            _ = self._link_prediction(sess, self.train[random.sample( \
                range(n_train), self.n_dev)])
            _t = timeit.default_timer()
            print(' {:^6.2f} {:^6.2f}'. \
                  format((_t - t1) / 60, (_t - t0) / 60))
               
            
    def _em_predict(self, sess):
        """
        Predict for test dataset.
        
        Args:
            sess: tf.Session
        """
                
        print('\n>>  Test Link Prediction Result.')
        t0 = timeit.default_timer()
        print('    E-MR  E-MRR R-MR R-MRR  TIME\n   ', end = '')
        result = self._link_prediction(sess, self.test)
        print('  {:^6.2f}'.format((timeit.default_timer() - t0) / 60))
        result.update(self.args)
        myjson(self.out_dir + 'result', result)
        
        if self.disease in ['_cancer', '_all'] and self.model == 'ConvKB':
            E_pre = sess.run(self.E_table)
            mypickle(self.out_dir + '_E_pre', E_pre)
            
            if self.lanta_c or self.lanta_d:
                E_T = np.array([[i, 0, 0] for i in range(self.n_E)])
                if self.lanta_c:
                    C_pre = sess.run(self.C_pre, {self.T_pos: E_T})
                    mypickle(self.out_dir + '_C_pre', C_pre)
                if self.lanta_d:
                    D_pre = sess.run(self.D_pre, {self.T_pos: E_T,
                                                  self.keep_prob: 1.0})
                    mypickle(self.out_dir + '_D_pre', D_pre)
                    
                    
    def _get_T_neg(self, T_pos):
        """
        (1) Get negative triple (T_neg) for training.
        (2) Replace relation with probability of 10%, else replace head or
            tail depends on replace_h_prob.
        
        Args:
            T_pos: positive triples
        """
        
        T_neg = []
        for h, r, ta in T_pos.tolist():
            while True:    
                if np.random.binomial(1, 0.1):
                    new_T = (h, random.choice(range(self.n_R)), ta)
                else:
                    new_e = random.choice(range(self.n_E))
                    new_T = (new_e, r, ta) if self.rpc_h(r) else \
                            (h, r, new_e)
                if new_T not in self.pool:
                    T_neg.append(new_T)
                    break
        return np.array(T_neg)
    
    
    def _link_prediction(self, sess, T_pos):   
        """
        Linking Prediction of knowledge graph embedding.
        Return entity MR, entity MRR, relation MR, relation MRR.
        [ MR (mean rank), MRR (mean reciprocal rank) ]
        
        Args:
            sess: tf.Session
            T_pos: positive triple to predict
        """
        
        E_rank, R_rank = [], []
        for T in T_pos.tolist():      
            rpc_h = np.array([T for i in range(self.n_E)])
            rpc_h[:, 0] = range(self.n_E)
            score_h = sess.run(self.score_pos, {self.T_pos: rpc_h, 
                                                self.keep_prob: 1.0})
            
            rpc_t = np.array([T for i in range(self.n_E)])
            rpc_t[:, 2] = range(self.n_E)
            score_t = sess.run(self.score_pos, {self.T_pos: rpc_t,
                                                self.keep_prob: 1.0})
            
            rpc_r = np.array([T for i in range(self.n_R)])
            rpc_r[:, 1] = range(self.n_R)
            score_r = sess.run(self.score_pos, {self.T_pos: rpc_r,
                                                self.keep_prob: 1.0})

            E_rank.extend([self._cal_ranks(score_h, T, 0), 
                           self._cal_ranks(score_t, T, 2)])    
            R_rank.append(self._cal_ranks(score_r, T, 1))
        
        E_MR = round(np.mean(E_rank), 1)
        E_MRR = round(np.mean([1 / x for x in E_rank]), 3)
        R_MR = round(np.mean(R_rank), 1)
        R_MRR = round(np.mean([1 / x for x in R_rank]), 3)
        
        print('{:>6.1f} {:>5.3f} {:>4.1f} {:>5.3f}'. \
              format(E_MR, E_MRR, R_MR, R_MRR), end = '')
        
        return {'E-MR': E_MR, 'E_MRR': E_MRR, 'R-MR': R_MR, 'R_MRR': R_MRR}
                
    
    def _cal_ranks(self, score, T, idx):
        """
        Cal link prediction rank for a single triple.
        
        Args:
            score: replace an entity (a relation) by all the entity (relation)
                   of an real triple, shape of [n_E, 3] ([n_R, 3])
            T: raw triple
            idx: the replace place of the triple
        """
        
        rank = np.argsort(score)
        out = 1 
        for x in rank:
            if x == T[idx]:
                break
            else:
                new_T = T.copy()
                new_T[idx] = x
                if tuple(new_T) not in self.pool:
                    out += 1
        return out
    
    
    def run(self, config):
        """
        Running Process.
        
        Args:
            config: tf.ConfigProto
        """
        
        with tf.Session(config = config) as sess:
            tf.global_variables_initializer().run()   
            if self.do_train:
                self._em_train(sess)
            if self.do_predict:
                self._em_predict(sess)
        
        
def myjson(p, data = None):
    """
    Read (data is None) or Dump (data is not None) a json file.    
    
    Args:
        p: file path
        data(None): json data
    """

    if '.json' not in p:
        p += '.json'
    if data is None:
        with open(p) as file:
            return json.load(file)
    else:
        with open(p, 'w') as file:
            json.dump(data, file) 
            

def mypickle(p, data = None):
    """
    Read (data is None) or Dump (data is not None) a pickle file.    
    
    Args:
        p: file path
        data(None): pickle data
    """

    if '.data' not in p:
        p += '.data'
    if data is None:
        with open(p, 'rb') as file:
            return pickle.load(file)
    else:
        with open(p, 'wb') as file:
            pickle.dump(data, file)     
            
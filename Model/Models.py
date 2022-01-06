from KGE import KGE
import numpy as np
import tensorflow as tf


class TransE(KGE):
    """Translating Embeddings for Modeling Multi-relational Data"""
    
    def __init__(self, args):
        super().__init__(args)
            
    
    def kge_variables(self):
        pass
    
        
    def em_structure(self, h, r, t, key = 'pos'):
        return h + r - t
    
    
    def cal_score(self, s):
        return tf.reduce_sum(s ** 2, -1)
            
    
    def cal_lp_score(self, h, r, t):        
        s_rpc_h = self.E_table + tf.expand_dims(r - t, 1)  
        s_rpc_t = tf.expand_dims(h + r, 1) - self.E_table
        lp_h, lp_t = self.cal_score(s_rpc_h), self.cal_score(s_rpc_t)
        if self.lanta_c:
            c_rpc_h = self.projector(s_rpc_h, self.C_table - \
                                     tf.expand_dims(self.t_c_pos, 1))
            c_rpc_t = self.projector(s_rpc_t, tf.expand_dims( \
                                     self.h_c_pos, 1) - self.C_table)
            lp_h -= self.lanta_c * self.cal_score(c_rpc_h)
            lp_t -= self.lanta_c * self.cal_score(c_rpc_t)
        if self.lanta_d:
            d_rpc_h = self.projector(s_rpc_h, self.D_table - \
                                     tf.expand_dims(self.t_d_pos, 1))
            d_rpc_t = self.projector(s_rpc_t, tf.expand_dims( \
                                     self.h_d_pos, 1) - self.D_table)
            lp_h -= self.lanta_d * self.cal_score(d_rpc_h)
            lp_t -= self.lanta_d * self.cal_score(d_rpc_t)
        
        return lp_h, lp_t
        
            

class TransH(KGE):
    """Knowledge Graph Embedding by Translating on Hyperplanes."""
    
    def __init__(self, args):
        super().__init__(args)
        
            
    def kge_variables(self):
        P_table = tf.get_variable('projection_table', initializer = \
                  tf.random_uniform([self.n_R, self.dim], -self.K, self.K))
        P_table = tf.nn.l2_normalize(P_table, 1)
        self.p_pos = tf.gather(P_table, self.T_pos[:, 1])
        self.p_neg = tf.reshape(tf.tile(tf.expand_dims(self.p_pos, 1), 
                     [1, self.n_neg, 1]), [-1, self.dim])
        
        self.l2_kge.append(self.p_pos)
        
        
    def em_structure(self, h, r, t, key = 'pos'):       
        self.transfer = lambda s, p: \
            s - tf.reduce_sum(p * s, -1, keepdims = True) * p    
            
        if key == 'pos':
            p = self.p_pos
        elif key == 'neg':
            p = self.p_neg
            
        h = self.transfer(h, p)
        t = self.transfer(t, p)
        return h + r - t
    
    
    def cal_score(self, s):
        return tf.reduce_sum(s ** 2, -1)
    
    
    def cal_lp_score(self, h, r, t):        
        p = self.p_pos
        p_E_table = self.transfer(self.E_table, tf.expand_dims(p, 1))
        s_rpc_h = p_E_table + tf.expand_dims(r - self.transfer(t, p), 1)
        s_rpc_t = tf.expand_dims(self.transfer(h, p) + r, 1) - p_E_table    
        lp_h, lp_t = self.cal_score(s_rpc_h), self.cal_score(s_rpc_t)
        if self.lanta_c:
            c_rpc_h = self.projector(s_rpc_h, self.C_table - \
                                     tf.expand_dims(self.t_c_pos, 1))
            c_rpc_t = self.projector(s_rpc_t, tf.expand_dims( \
                                     self.h_c_pos, 1) - self.C_table)
            lp_h -= self.lanta_c * self.cal_score(c_rpc_h)
            lp_t -= self.lanta_c * self.cal_score(c_rpc_t)
        if self.lanta_d:
            d_rpc_h = self.projector(s_rpc_h, self.D_table - \
                                     tf.expand_dims(self.t_d_pos, 1))
            d_rpc_t = self.projector(s_rpc_t, tf.expand_dims( \
                                     self.h_d_pos, 1) - self.D_table)
            lp_h -= self.lanta_d * self.cal_score(d_rpc_h)
            lp_t -= self.lanta_d * self.cal_score(d_rpc_t)
        
        return lp_h, lp_t
    
    
    
class ConvKB(KGE):
    """
    A Novel Embedding Model for Knowledge Base Completion Based on 
    Convolutional Neural Network.
    """
    
    def __init__(self, args):  
        super().__init__(args)
        
        
    def kge_variables(self):
        init_f = np.array([[[np.random.normal(0.1, 0.01, self.n_filter)],  
                            [np.random.normal(0.1, 0.01, self.n_filter)], 
                            [np.random.normal(-0.1, 0.01, self.n_filter)]]])
        self.F = tf.get_variable('filter', [1, 3, 1, self.n_filter],
                                 initializer = tf.constant_initializer(init_f))
        
        K = np.sqrt(6.0 / (self.dim * self.n_filter + 1))
        self.W = tf.get_variable('weight', initializer = \
                 tf.random_uniform([self.dim * self.n_filter, 1], -K, K))
        
        self.l2_kge.append(self.W)
    
        
    def em_structure(self, h, r, t, key = 'pos'):
        #(B, D, 3, 1) conv (1, 3, 1, F) ==> (B, D, 1, F)
        h = tf.reshape(h, [-1, self.dim, 1, 1])
        r = tf.reshape(r, [-1, self.dim, 1, 1])
        t = tf.reshape(t, [-1, self.dim, 1, 1])
        return tf.nn.conv2d(tf.concat([h, r, t], 2), self.F,
                            strides = [1, 1, 1, 1], padding = 'VALID') 
    
    
    def cal_score(self, s):
        #((B, D, 1, F) ==> (B, D * F)) * (D * F, 1) ==> (B, 1)
        return tf.matmul(tf.nn.relu(tf.reshape(s, [-1, self.dim * \
                                                   self.n_filter])), self.W)
      
        
    def cal_lp_score(self, h, r, t):      
        h = tf.reshape(h, [-1, self.dim, 1, 1])
        r = tf.reshape(r, [-1, self.dim, 1, 1])
        t = tf.reshape(t, [-1, self.dim, 1, 1])
        E_table = tf.reshape(self.E_table, [-1, self.dim, 1, 1])        
        s_rpc_h = tf.nn.conv2d(tf.concat([E_table, tf.tile( \
                  tf.concat([r, t], 2), [self.n_E, 1, 1, 1])], 2), self.F,
                  strides = [1, 1, 1, 1], padding = 'VALID')
        s_rpc_t = tf.nn.conv2d(tf.concat([tf.tile(tf.concat([h, r], 2),
                  [self.n_E, 1, 1, 1]), E_table], 2), self.F,
                  strides = [1, 1, 1, 1], padding = 'VALID')
        lp_h, lp_t = self.cal_score(s_rpc_h), self.cal_score(s_rpc_t)
        
        if self.lanta_c:
            c_rpc_h = self.projector(s_rpc_h, tf.reshape(self.C_table - 
                                     self.t_c_pos, [-1, self.dim, 1, 1]))
            c_rpc_t = self.projector(s_rpc_t, tf.reshape(self.h_c_pos - 
                                     self.C_table, [-1, self.dim, 1, 1]))
            lp_h -= self.lanta_c * self.cal_score(c_rpc_h)
            lp_t -= self.lanta_c * self.cal_score(c_rpc_t)
        if self.lanta_d:
            d_rpc_h = self.projector(s_rpc_h, tf.reshape(self.D_table - 
                                     self.t_d_pos, [-1, self.dim, 1, 1]))
            d_rpc_t = self.projector(s_rpc_t, tf.reshape(self.h_d_pos - 
                                     self.D_table, [-1, self.dim, 1, 1]))
            lp_h -= self.lanta_d * self.cal_score(d_rpc_h)
            lp_t -= self.lanta_d * self.cal_score(d_rpc_t)
        
        return tf.reshape(lp_h, [1, -1]), tf.reshape(lp_t, [1, -1])
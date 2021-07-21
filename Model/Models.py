import numpy as np
from KGE import KGE
import tensorflow.compat.v1 as tf


class TransE(KGE):
    """A TransE model attaching category and description annotation."""
    
    def __init__(self, args):
        super().__init__(args)
            
                         
    def _em_structure(self):
        """Main structure of TransE's embedding."""
        if not self.margin:
            raise 'Lack margin value!'
        D = self.dim
        
        with tf.variable_scope('structure'): #(B, D)
            E_table = tf.nn.l2_normalize(tf.get_variable('entity_table',
                      [self.n_E, D], initializer = self.initializer), 1)           
            h_s_pos = tf.gather(E_table, self.T_pos[:, 0])
            t_s_pos = tf.gather(E_table, self.T_pos[:, -1])
            h_s_neg = tf.gather(E_table, self.T_neg[:, 0])
            t_s_neg = tf.gather(E_table, self.T_neg[:, -1])
            
            R_table = tf.nn.l2_normalize(tf.get_variable('relation_table',
                      [self.n_R, D], initializer = self.initializer), 1)
            r_pos = tf.gather(R_table, self.T_pos[:, 1])
            r_neg = tf.gather(R_table, self.T_neg[:, 1])
            
            s_pos = h_s_pos + r_pos - t_s_pos
            s_neg = h_s_neg + r_neg - t_s_neg

        with tf.variable_scope('score'): #(B, 1)
            self.score_pos = tf.reduce_sum(s_pos ** 2, 1)
            self.score_neg = tf.reduce_sum(s_neg ** 2, 1)
            
            if self.lanta_c:
                with tf.variable_scope('category'): #(B, D)
                    c_pos, c_neg = self._C_layer()
                    s_c_pos = self._projector(s_pos, c_pos)
                    s_c_neg = self._projector(s_neg, c_neg)
                self.score_pos -= self.lanta_c * tf.reduce_sum(s_c_pos ** 2, 1)
                self.score_neg -= self.lanta_c * tf.reduce_sum(s_c_neg ** 2, 1)
                
            if self.lanta_d:             
                with tf.variable_scope('description'): #(B, D)
                    d_pos, d_neg = self._D_layer()
                    s_d_pos = self._projector(s_pos, d_pos)
                    s_d_neg = self._projector(s_neg, d_neg)
                self.score_pos -= self.lanta_d * tf.reduce_sum(s_d_pos ** 2, 1)
                self.score_neg -= self.lanta_d * tf.reduce_sum(s_d_neg ** 2, 1)
                
        with tf.variable_scope('loss'): #(1)
            self.loss = tf.reduce_sum(tf.nn.relu(self.margin + 
                                      self.score_pos - self.score_neg))
            self.train_op = tf.train.AdamOptimizer(self.l_r). \
                            minimize(self.loss)
                                              
    
    
class TransH(KGE):
    """A TransH model attaching category and description annotation."""
    
    def __init__(self, args):
        super().__init__(args)
            
                         
    def _em_structure(self):
        """Main structure of TransH's embedding."""
        if not self.margin:
            raise 'Lack margin value!'
        D = self.dim
        
        with tf.variable_scope('structure'): #(B, D)        
            E_table = tf.nn.l2_normalize(tf.get_variable('entity_table',
                      [self.n_E, D], initializer = self.initializer), 1)           
            _h_s_pos = tf.gather(E_table, self.T_pos[:, 0])
            _t_s_pos = tf.gather(E_table, self.T_pos[:, -1])
            _h_s_neg = tf.gather(E_table, self.T_neg[:, 0])
            _t_s_neg = tf.gather(E_table, self.T_neg[:, -1])
            
            R_table = tf.nn.l2_normalize(tf.get_variable('relation_table',
                      [self.n_R, D], initializer = self.initializer), 1)
            r_pos = tf.gather(R_table, self.T_pos[:, 1])
            r_neg = tf.gather(R_table, self.T_neg[:, 1])
            
            P_table = tf.nn.l2_normalize(tf.get_variable('projection_table',
                      [self.n_R, D], initializer = self.initializer), 1)
            p_pos = tf.gather(P_table, self.T_pos[:, 1])
            p_neg = tf.gather(P_table, self.T_neg[:, 1])
            
            h_s_pos = self._projector(_h_s_pos, p_pos)
            t_s_pos = self._projector(_t_s_pos, p_pos)
            h_s_neg = self._projector(_h_s_neg, p_neg)
            t_s_neg = self._projector(_t_s_neg, p_neg)
                        
            s_pos = h_s_pos + r_pos - t_s_pos
            s_neg = h_s_neg + r_neg - t_s_neg

        with tf.variable_scope('score'): #(B, 1)
            self.score_pos = tf.reduce_sum(s_pos ** 2, 1)
            self.score_neg = tf.reduce_sum(s_neg ** 2, 1)
            
            if self.lanta_c:
                with tf.variable_scope('category'): #(B, D)
                    c_pos, c_neg = self._C_layer()
                    s_c_pos = self._projector(s_pos, c_pos)
                    s_c_neg = self._projector(s_neg, c_neg)
                self.score_pos -= self.lanta_c * tf.reduce_sum(s_c_pos ** 2, 1)
                self.score_neg -= self.lanta_c * tf.reduce_sum(s_c_neg ** 2, 1)
                
            if self.lanta_d:             
                with tf.variable_scope('description'): #(B, D)
                    d_pos, d_neg = self._D_layer()
                    s_d_pos = self._projector(s_pos, d_pos)
                    s_d_neg = self._projector(s_neg, d_neg)
                self.score_pos -= self.lanta_d * tf.reduce_sum(s_d_pos ** 2, 1)
                self.score_neg -= self.lanta_d * tf.reduce_sum(s_d_neg ** 2, 1)
                
        with tf.variable_scope('loss'): #(1)
            self.loss = tf.reduce_sum(tf.nn.relu(self.margin + 
                                      self.score_pos - self.score_neg))
            self.train_op = tf.train.AdamOptimizer(self.l_r). \
                            minimize(self.loss)
                            
                            
                        
class ConvKB(KGE):
    """A ConcKB model attaching category and description annotation."""
    
    def __init__(self, args):  
        super().__init__(args)
        

    def _em_structure(self):
        """Main structure of ConvKB."""
        if not self.n_filter:
            raise 'Lack n_filter value!'
        D, F = self.dim, self.n_filter
        N = D * F
          
        with tf.variable_scope('structure'):  #(B, D, 1, F)       
            self.E_table = tf.nn.l2_normalize(tf.get_variable('entity_table',
                           [self.n_E, D], initializer = self.initializer), 1)
            self.E_table = tf.reshape(self.E_table, [-1, D, 1, 1])
            h_pos = tf.gather(self.E_table, self.T_pos[:, 0])
            t_pos = tf.gather(self.E_table, self.T_pos[:, -1])
            h_neg = tf.gather(self.E_table, self.T_neg[:, 0])
            t_neg = tf.gather(self.E_table, self.T_neg[:, -1])
            
            R_table = tf.nn.l2_normalize(tf.get_variable('relation_table',
                      [self.n_R, D], initializer = self.initializer), 1)
            R_table = tf.reshape(R_table, [-1, D, 1, 1])
            r_pos = tf.gather(R_table, self.T_pos[:, 1])
            r_neg = tf.gather(R_table, self.T_neg[:, 1])
            
            init_f = np.array([[[np.random.normal(0.1, 0.01, F)],  
                                [np.random.normal(0.1, 0.01, F)], 
                                [np.random.normal(-0.1, 0.01, F)]]])
            f = tf.get_variable('filter', [1, 3, 1, F],
                initializer = tf.constant_initializer(init_f))
            
            #(B, D, 3, 1) conv (1, 3, 1, F) ==> (B, D, 1, F)
            s_pos = tf.nn.conv2d(tf.concat([h_pos, r_pos, t_pos], 2), 
                    f, strides = [1, 1, 1, 1], padding = 'VALID') 
            s_neg = tf.nn.conv2d(tf.concat([h_neg, r_neg, t_neg], 2), 
                    f, strides = [1, 1, 1, 1], padding = 'VALID')
        
        with tf.variable_scope('score'): #(B, 1)  
            w = tf.get_variable('weight', [N, 1], 
                                initializer = self.initializer)
            
            #((B, D, 1, F) ==> (B, D * F)) * (D * F, 1)
            self.score_pos = tf.squeeze(tf.matmul(tf.nn.relu(tf.reshape( \
                              s_pos, [-1, N])), w))
            self.score_neg = tf.squeeze(tf.matmul(tf.nn.relu(tf.reshape( \
                              s_neg, [-1, N])), w))

            #(B, D, 1, F) - (B, D, 1, 1) * (B, D, 1, F) .* (B, D, 1, 1)
            if self.lanta_c:            
                with tf.variable_scope('category'): 
                    c_pos, c_neg = self._C_layer()
                    c_pos = tf.reshape(c_pos, [-1, D, 1, 1])
                    c_neg = tf.reshape(c_neg, [-1, D, 1, 1])
                    s_c_pos = self._projector(s_pos, c_pos)
                    s_c_neg = self._projector(s_neg, c_neg)   
                    
                self.score_pos -= self.lanta_c * \
                    tf.squeeze(tf.matmul(tf.nn.dropout(tf.reshape(s_c_pos,
                    [-1, N]), self.keep), w))
                self.score_neg -= self.lanta_c * \
                    tf.squeeze(tf.matmul(tf.nn.dropout(tf.reshape(s_c_neg,
                    [-1, N]), self.keep), w))
                
            if self.lanta_d:
                with tf.variable_scope('description'): 
                    d_pos, d_neg = self._D_layer()
                    d_pos = tf.reshape(d_pos, [-1, D, 1, 1])
                    d_neg = tf.reshape(d_neg, [-1, D, 1, 1])
                    s_d_pos = self._projector(s_pos, d_pos)
                    s_d_neg = self._projector(s_neg, d_neg)
                
                self.score_pos -= self.lanta_d * \
                    tf.squeeze(tf.matmul(tf.nn.dropout(tf.reshape(s_d_pos,
                    [-1, N]), self.keep), w))
                self.score_neg -= self.lanta_d * \
                    tf.squeeze(tf.matmul(tf.nn.dropout(tf.reshape(s_d_neg,
                    [-1, N]), self.keep), w))
            
        with tf.variable_scope('loss'): #(1)
            self.loss = tf.reduce_sum(tf.nn.softplus(self.score_pos) + \
                                      tf.nn.softplus(- self.score_neg)) + \
                        0.001 / 2 * tf.reduce_sum(w ** 2)
            self.train_op = tf.train.AdamOptimizer(self.l_r). \
                            minimize(self.loss)
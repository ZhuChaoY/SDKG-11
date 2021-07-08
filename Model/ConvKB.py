import os
import numpy as np
from KGE import KGE
import tensorflow.compat.v1 as tf


class ConvKB(KGE):
    """A ConcKB model attaching category and description annotation."""
    
    def __init__(self, args):
        """
        (1) Initializes ConvKB with arg dict.
        (2) Named model out_dir with S, C and D.
        (3) If don't use neither category nor description annotation,
            initialized model with TransE of same dim.
        (4) If use category or description annotation, initialized model with
            ConvKB of same dim and n_filter.
            
        Args:
            'disease'    : knowledge graph embedding model for which disease.
            'dim'        : embedding dim.
            'n_filter'   : number of filter used in conv layer.
            'lanta_c'    : the weight hyperparameter of category annotation.
            'lanta_d'    : the weight hyperparameter of description annotation.
            'l_r'        : learning rate.
            'epoches'    : training epoches.
            'batch_size' : batch size for training.
            'do_train'   : whether to train the model.
            'do_predict' : whether to predict for test dataset.
        """
        
        super().__init__(args)
        

    def _em_structure(self):
        """Main structure of ConvKB."""
        
        print('\n    *Dim           : {}'.format(self.dim))
        print('    *N_filter      : {}'.format(self.n_filter))
        print('    *Lanta_C       : {}'.format(self.lanta_c))
        print('    *Lanta_D       : {}'.format(self.lanta_d))
        print('    *Learning_Rate : {}'.format(self.l_r))
        print('    *Batch_Size    : {}'.format(self.batch_size))
        print('    *Epoches       : {}'.format(self.epoches))
        
        tf.reset_default_graph()
        self.T_pos = tf.placeholder(tf.int32, [None, 3])
        self.T_neg = tf.placeholder(tf.int32, [None, 3])
        self.keep_prob = tf.placeholder(tf.float32)
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
                    [-1, N]), self.keep_prob), w))
                self.score_neg -= self.lanta_c * \
                    tf.squeeze(tf.matmul(tf.nn.dropout(tf.reshape(s_c_neg,
                    [-1, N]), self.keep_prob), w))
                
            if self.lanta_d:
                with tf.variable_scope('description'): 
                    d_pos, d_neg = self._D_layer()
                    d_pos = tf.reshape(d_pos, [-1, D, 1, 1])
                    d_neg = tf.reshape(d_neg, [-1, D, 1, 1])
                    s_d_pos = self._projector(s_pos, d_pos)
                    s_d_neg = self._projector(s_neg, d_neg)
                
                self.score_pos -= self.lanta_d * \
                    tf.squeeze(tf.matmul(tf.nn.dropout(tf.reshape(s_d_pos,
                    [-1, N]), self.keep_prob), w))
                self.score_neg -= self.lanta_d * \
                    tf.squeeze(tf.matmul(tf.nn.dropout(tf.reshape(s_d_neg,
                    [-1, N]), self.keep_prob), w))
            
        with tf.variable_scope('loss'): #(1)
            self.loss = tf.reduce_sum(tf.nn.softplus(self.score_pos) + \
                                      tf.nn.softplus(- self.score_neg)) + \
                        0.001 / 2 * tf.reduce_sum(w ** 2)
            self.train_op = tf.train.AdamOptimizer(self.l_r). \
                            minimize(self.loss)
          
        self._em_init()   
         

def main():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['CUDA_VISIBLE_DEVICES'] = '2' 
    config = tf.ConfigProto() 
    config.gpu_options.allow_growth = True
       
    diseases = \
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
         'can' : ['_cancer', 4096],
         'all' : ['_all', 8192]}
        
    disease, batch_size = diseases['can']
    args = \
        {'model'      : 'ConvKB',
         'disease'    : disease,
         'batch_size' : batch_size,
         'dim'        : 256,
         'n_filter'   : 8,
         'lanta_c'    : 0.0,
         'lanta_d'    : 0.0,
         'l_r'        : 1e-4,
         'epoches'    : 200,
         'do_train'   : True,
         'do_predict' : True}
    
    model = ConvKB(args)
    model.run(config)
        

if __name__ == '__main__':     
    main()

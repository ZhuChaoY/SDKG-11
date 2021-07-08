import os
from KGE import KGE
import tensorflow.compat.v1 as tf


class TransE(KGE):
    """A TransE model attaching category and description annotation."""
    
    def __init__(self, args):
        """
        Initializes TransE with arg dict.
            
        Args:
            'disease'    : knowledge graph embedding model for which disease.
            'dim'        : embedding dim.
            'margin'     : margin hyperparameter.
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
        """Main structure of TransE's embedding."""
        
        print('\n    *Dim           : {}'.format(self.dim))
        print('    *Margin        : {}'.format(self.margin))
        print('    *Lanta_C       : {}'.format(self.lanta_c))
        print('    *Lanta_D       : {}'.format(self.lanta_d))
        print('    *Learning_Rate : {}'.format(self.l_r))
        print('    *Batch_Size    : {}'.format(self.batch_size))
        print('    *Epoches       : {}'.format(self.epoches))
        
        tf.reset_default_graph()
        self.T_pos = tf.placeholder(tf.int32, [None, 3])
        self.T_neg = tf.placeholder(tf.int32, [None, 3])
        self.keep_prob = tf.placeholder(tf.float32)
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
                                               
        self._em_init()
    

def main():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0' 
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
         'can' : ['_cancer', 4096]}
        
    disease, batch_size = diseases['can']
    args = \
        {'model'      : 'TransE',
         'disease'    : disease,
         'batch_size' : batch_size,
         'dim'        : 256,
         'margin'     : 1.0,
         'lanta_c'    : 0.0,
         'lanta_d'    : 0.0,
         'l_r'        : 1e-3,
         'epoches'    : 800,
         'do_train'   : True,
         'do_predict' : True}
        
    model = TransE(args)
    model.run(config)
    
                 
if __name__ == '__main__':      
    main()
        
 
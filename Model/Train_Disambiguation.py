import os
import re
import timeit
import random
import numpy as np
import pandas as pd
import tensorflow as tf 
import Tokenization as tz
from KGE import myjson, mypickle
from Optimization import AdamWeightDecayOptimizer


class Train_Disambiguation():
    """Performe entity disambiguation task."""
    
    def __init__(self, args):
        """
        (1) Initialized Disambiguation as args dict.
        (2) Load train, dev and test examples.
        (3) Construct BERT model.
        """
        
        for key, value in args.items():
            if type(value) == str:
                exec('self.{} = "{}"'.format(key, value))
            else:
                exec('self.{} = {}'.format(key, value))
            
        self.out_dir = 'Disambiguation/' + self.model + '/'
        self.initializer = tf.truncated_normal_initializer(stddev = 0.02)    
        self.tokenizer = tz.FullTokenizer(self.model + '/vocab.txt')
        
        self._dis_data()
        self._dis_structure()
        

    def _dis_data(self):
        """Load disambuguation dataset"""
        
        print('\n' + '# ' * 3 + ' Loading Dataset ' + ' #' * 3)
        self.n_train, self.train = self._load_train_or_dev('train')
        self.n_dev, self.dev = self._load_train_or_dev('dev')
        self.n_test, self.test = self._load_test()
        

    def _load_train_or_dev(self, _set):        
        """ 
        Process raw train, dev data as examples with shape of (n_train, 4) 
        or (n_dev, 4).
        
        Args:
            _set: 'train' or 'dev'.
        """
        
        t0 = timeit.default_timer()
        p = self.out_dir + '_' + _set + '.data'
        if os.path.exists(p):
            examples = mypickle(p)
        else:
            raw = pd.read_csv('Disambiguation/' + _set + '.csv')
            text_as, text_bs = raw['S1'], raw['S2']
            labels = raw['label']
            examples = [self._convert_example( \
                        tz.convert_to_unicode(text_as[i]),
                        tz.convert_to_unicode(text_bs[i]),
                        tz.convert_to_unicode(str(labels[i]))) 
                        for i in range(raw.shape[0])]
            mypickle(p, examples)
                
        n_example = len(examples)
        print('>>  Loading {} {} examples. ({:.2f} min)'.format(n_example,
              _set, (timeit.default_timer() - t0) / 60))
        return n_example, examples
        
    
    def _load_test(self):
        """ 
        Process raw test data as examples with length of n_test.
        Each example contains it's candidate entities' description.
        """
        
        t0 = timeit.default_timer()
        p = self.out_dir + '_test.data'
        if os.path.exists(p):
            examples = mypickle(p) 
        else:
            data = myjson('Disambiguation/test')
            examples = []
            for key, value in data.items():
                _, text_a = eval(key)
                text_a = tz.convert_to_unicode(text_a)
                batch_data = []
                for text_b in value:
                    text_b = tz.convert_to_unicode(text_b)
                    batch_data.append(self._convert_example(text_a, text_b))
                ids = np.array([x[0] for x in batch_data])
                mask = np.array([x[1] for x in batch_data])
                segment = np.array([x[2] for x in batch_data])
                examples.append((key, ids, mask, segment))
            mypickle(p, examples)
            
        n_example = len(examples)            
        print('>>  Loading {} test examples. ({:.2f} min)'.format(n_example,
                   (timeit.default_timer() - t0) / 60)) 
                                                          
        return n_example, examples
    
    
    def _convert_example(self, text_a, text_b, label = None):
        """
        Convert an raw data into an example with ids, mask, segment and label.
        
        Args:
            text_a: sentence1.
            text_b: sentence2.
            label (None): positive (1) or negative (0) or None (test).
        """
        
        tokens_a = self.tokenizer.tokenize(text_a)
        tokens_b = self.tokenizer.tokenize(text_b)
        while True:
            if len(tokens_a) + len(tokens_b) <= self.len_d - 3:
                break
            if len(tokens_a) > len(tokens_b):
                tokens_a.pop()
            else:
                tokens_b.pop()
            
        tokens = ['[CLS]'] + tokens_a + ['[SEP]'] + tokens_b + ['[SEP]']
        ids = self.tokenizer.convert_tokens_to_ids(tokens)
        segment = [0] * (len(tokens_a) + 2) + [1] * (len(tokens_b) + 1)
        mask = [1] * len(ids)
        while len(ids) < self.len_d:
            ids.append(0)
            mask.append(0)
            segment.append(0)
            
        if label:
            return ids, mask, segment, {'0': 0, '1': 1}[label]
        else:
            return ids, mask, segment
    
    
    def _dis_structure(self):
        """Main structure of training disambuguation."""
        
        print('\n' + '# ' * 3 + ' Constructing Model ' + ' #' * 3)
        print('    *Model         : {}'.format(self.model))
        print('    *len_d         : {}'.format(self.len_d))
        print('    *Learning_Rate : {}'.format(self.l_r))
        print('    *Batch_Size    : {}'.format(self.batch_size))
        print('    *Epoches       : {}'.format(self.epoches))
        
        tf.reset_default_graph()
        with tf.variable_scope('bert'):
            self._bert_layer()
        with tf.variable_scope('loss'):
            self._finetune_layer()
            
        self._dis_init()
        

    def _bert_layer(self):
        """BERT layer, initialized with biobert."""
        
        self.label = tf.placeholder(tf.int32, [None]) #(B)
        self.keep_prob = tf.placeholder(tf.float32) #(1)
        #(B, L) ==> (B * L)
        self.ids = tf.placeholder(tf.int32, [None, self.len_d])
        ids = tf.reshape(self.ids, [-1])
        #(B, L) ==> (B * L) ==> (B * L, 2)
        self.segment = tf.placeholder(tf.int32, [None, self.len_d]) 
        segment = tf.one_hot(tf.reshape(self.segment, [-1]), depth = 2)
        #(1, 1, L, 1) * [(B, L) ==> (B, 1, 1, L)] ==> (B, 1, L, L)
        self.mask = tf.placeholder(tf.int32, [None, self.len_d])
        mask = tf.cast(tf.reshape(self.mask, [-1, 1, 1, self.len_d]),
                       tf.float32)
        att_mask = -10000.0 * (1.0 - tf.ones([1, 1, self.len_d, 1]) * mask) 
        
        with tf.variable_scope('embeddings'): #(B, L, 768)
            #(28996 | 30522, 768) ==> (B * L, 768) ==> (B, L, 768)
            word_table = tf.get_variable('word_embeddings', 
                         [28996 if self.model == 'Biobert' else 30522, 768],
                         initializer = self.initializer)
            em_out = tf.reshape(tf.gather(word_table, ids), 
                                [-1, self.len_d, 768]) 
            #(B * L, 2) * (2, 768) == > (B * L, 768) ==> (B, L, 768)
            token_table = tf.get_variable('token_type_embeddings', [2, 768],
                          initializer = self.initializer)
            em_out += tf.reshape(tf.matmul(segment, token_table), 
                                 [-1, self.len_d, 768]) 
            #(B, L, 768) + [(512, 768) ==> (1, L, 768)] ==> (B, L, 768)
            position_table = tf.get_variable('position_embeddings', [512, 768], 
                             initializer = self.initializer)
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
                        mid_out = \
                        tf.layers.dense(att_out, 3072, activation = gelu,
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
        """Attention layer for BERT layer"""
        
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
            att_out = layer_norm(tf.nn.dropout(tf.layers.dense(self_out, 768,
            kernel_initializer = self.initializer), self.keep_prob) + prev_out)
        
        return att_out        


    def _finetune_layer(self):
        """Finetune layer for entity disambiguation task."""
        
        w = tf.get_variable('output_weights', [2, 768], 
                            initializer = self.initializer)
        b = tf.get_variable('output_bias', [2], 
                            initializer = tf.zeros_initializer())
        label = tf.one_hot(self.label, depth = 2, dtype = tf.float32)
        logits = tf.nn.bias_add(tf.matmul(tf.nn.dropout(self.bert_out,
                 self.keep_prob), w, transpose_b = True), b)
        self.probility = tf.nn.softmax(logits, axis = -1)
        self.prediction = tf.argmax(self.probility, 1)
        self.loss = tf.reduce_sum(-tf.reduce_sum(label *
                    tf.nn.log_softmax(logits, axis = -1), axis = -1)) 
        
        n_step = (self.n_train // self.batch_size + 1) * self.epoches
        self.train_op = AdamWeightDecayOptimizer(self.loss, self.l_r,
                                                 n_step, n_step // 10)
        
        
    def _dis_init(self):
        """Initialize disambiguation structure trainable variables."""
                
        tvs = [re.match('^(.*):\\d+$', v.name).group(1)
               for v in tf.trainable_variables()]
        
        if not self.do_train:
            p = self.out_dir + 'model.ckpt'           
            ivs = {v[0]: v[0] for v in tf.train.list_variables(p) 
                   if v[0] in tvs}
        else:
            p = self.model + '/bert_model.ckpt'          
            ivs = {v[0]: v[0] for v in tf.train.list_variables(p) 
                   if v[0] in tvs and 'bert' in v[0]}
        tf.train.init_from_checkpoint(p, ivs)            
        
        print('\n>>  {} of {} trainable variables initialized.'. \
              format(len(ivs), len(tvs)))  
        
        
    def _dis_train(self, sess):
        """
        (1) Training process of entity disambiguation task.
        (2) Evaluate for train and dev dataset each epoch.
        
        Args:
            sess: tf.Session.
        """
        
        print('\n' + '# ' * 6 + ' Training ({} EPOCHES) '. \
              format(self.epoches) + ' #' * 6)
        t0 = t1 = timeit.default_timer()
        print('              Train         Dev')
        print('    EPOCH  LOSS   ACC   LOSS   ACC   time   TIME')
                 
        for epoch in range(self.epoches):
            print('    {:^5}'.format(epoch + 1), end = '')
            
            for _set in ['train', 'dev']:
                batches = self._get_batches(_set)
                Loss, Pre = 0.0, None
                for ids, mask, segment, label in batches:
                    feed_dict = \
                        {self.ids: ids, self.mask: mask,
                         self.segment: segment, self.label: label,
                         self.keep_prob: 1.0 - 0.1 * (_set == 'train')}
                    if _set == 'train':        
                        loss, pre, _ = sess.run([self.loss, self.prediction,
                                       self.train_op], feed_dict)
                    else:
                        loss, pre = sess.run([self.loss, self.prediction],
                                             feed_dict)
                    Loss += loss
                    if Pre is None:
                        Pre = pre
                        Label = label
                    else:
                        Pre = np.hstack((Pre, pre))
                        Label = np.hstack((Label, label))
                acc = sum(Pre == Label) / Pre.shape[0]
                print(' {:>6.4f} {:^5.3f}'.format(Loss / self.n_train, acc),
                      end = '')
            
            _t = timeit.default_timer()
            print(' {:^6.2f} {:^6.2f}'.format((_t - t1) / 60, (_t - t0) / 60))
            t1 = _t
            
        tf.train.Saver().save(sess, self.out_dir + 'model.ckpt')
        
    
    def _get_batches(self, _set):
        """
        Get batch example for train and dev examples with args' batch_size.
        
        Args:
            _set: 'train' or 'dev'.
        """
        
        bs = self.batch_size
        data = eval('self.' + _set + '.copy()')
        random.shuffle(data)                    
        n_batch = len(data) // bs   
        idxes = [data[i * bs: (i + 1) * bs] for i in range(n_batch - 1)]
        idxes.append(data[(n_batch - 1) * bs: ])
        batches = []
        for idx in idxes:
            ids = np.array([x[0] for x in idx])
            mask = np.array([x[1] for x in idx])
            segment = np.array([x[2] for x in idx])
            label = np.array([x[3] for x in idx])
            batches.append((ids, mask, segment, label))
        
        return batches
    
     
    def _dis_predict(self, sess):
        """
        Prediction process of entity disambiguation task.
        
        Args:
            sess: tf.Session.
        """
        
        print('\n' + '# ' * 6 + ' Prediction ' + ' #' * 6)
        t0 = timeit.default_timer()

        result = {}
        for key, ids, mask, segment in self.test:
            feed_dict = {self.ids: ids, self.mask: mask,
                         self.segment: segment, self.keep_prob: 1.0}
            probility = sess.run(self.probility, feed_dict)
            result[key] = [round(x, 6) for x in probility[:, 1].tolist()]
        myjson(self.out_dir + 'prediction', result)
                
        print('>>  Total Time: {:.1f}s'.format(timeit.default_timer() - t0))
        
        
    def run(self, config):
        """
        Running Process.
        
        Args:
            config: tf.ConfigProto
        """
        
        with tf.Session(config = config) as sess:
            tf.global_variables_initializer().run()   
            if self.do_train:
                self._dis_train(sess)
            if self.do_predict:
                self._dis_predict(sess)
        
        
def gelu(x):
    return 0.5 * x * (1.0 + tf.tanh((0.797884 * (x + 0.044715 * x * x * x))))


def layer_norm(inputs):
    return tf.contrib.layers.layer_norm(inputs = inputs,
           begin_norm_axis = -1, begin_params_axis = -1) 

def main():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['CUDA_VISIBLE_DEVICES'] = '1' 
    config = tf.ConfigProto() 
    config.gpu_options.allow_growth = True

    args = \
        {'model'      : ['Biobert', 'Bert'][0], 
         'len_d'      : 128,
         'epoches'    : 5,
         'batch_size' : 16,
         'l_r'        : 1e-5,
         'do_train'   : True,
         'do_predict' : True}
    
    model = Train_Disambiguation(args)
    model.run(config)


if __name__ == '__main__':
    main()
    
    

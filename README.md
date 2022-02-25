# SDKG-11
Multimodal Reasoning based on Knowledge Graph Embedding for Specific Diseases

## Abstract
**Motivation**: Knowledge Graph (KG) is becoming increasingly important in the biomedical field. Deriving new and reliable knowledge from existing knowledge by knowledge graph embedding technology is a cutting-edge method. Some add a variety of additional information to aid reasoning, namely multimodal reasoning. However, few works based on the existing biomedical KGs are focused on specific diseases.    
**Results**: This work develops a construction and multimodal reasoning process of Specific Disease Knowledge Graphs (SDKGs). We construct SDKG-11, a SDKG set including five cancers, six non-cancer diseases, a combined Cancer5, and a combined Diseases11, aiming to discover new reliable knowledge and provide universal pre-trained knowledge for that specific disease field. SDKG-11 is obtained through original triplet extraction, standard entity set construction, entity linking, and relation linking. We implement multimodal reasoning by reverse-hyperplane projection for SDKGs based on structure, category, and description embeddings. Multimodal reasoning improves pre-existing models on all SDKGs using entity prediction task as the evaluation protocol. We verify the model's reliability in discovering new knowledge by manually proofreading predicted drug-gene, gene-disease, and disease-drug pairs. Using embedding results as initialization parameters for the biomolecular interaction classification, we demonstrate the universality of embedding models.   

## Paper Link
https://academic.oup.com/bioinformatics/advance-article/doi/10.1093/bioinformatics/btac085/6527626?login=true

## Files
### Annotation/
**E_dict_0.json** ~ **E_dict_5.json**   
**get_E_dict.py** : Run it first to get complete E_dict.json   

### Dataset/ 
**5 Cancers** : Including colon_cancer, gallbladder_cancer, gastric_cancer, liver_cancer, lung_cancer          
**6 NonCancer** : Including alzheimer_disease, copd, coronary_heart_disease, diabetes, heart_failure, rheumatoid_arthritis        
**Cancer5**   
**Disease11**     

### Model/ 
**KGE.py** : Class of processing and tool functions for Knowledge Graph Embedding    
**Models.py** : TransE, TransH, ConvKB structure    
**Run_KGE.py** : Run KGE.py        
#### C&D/
**C_dict.data** : Dict of entity category annotation  
**D_table.data** : Table of entity description annotation  
**E_index.json** : Entity index dict for C_dict and D_table  
**get_C_dict.py** : Run it to get C_dict.data and E_index.data    
**D_Table.py** : Structure for training description table        
**Optimization.py** : Training optimization of BioBERT     
**Tokenization.py** : Tokenization function of BioBERT     
**Run_D_Table.py** : Run it to get D_table.data  
#### Pretrained BioBERT/  
**bert_config.json**  
**bert_model.ckpt.data-00000-of-00001**  
**bert_model.ckpt.index**  
**bert_model.ckpt.meta**  
**vocab.txt**  
Self download from https://github.com/dmis-lab/biobert     

### Supplementary Table/  
**Supplementary Table S1 (statistical analysis of entity prediction)**  
**Supplementary Table S2 (drug-gene new inferred knowledge)**   
**Supplementary Table S3 (gene-disease new inferred knowledge)**   
**Supplementary Table S4 (disease-drug new inferred knowledge)**   
**Supplementary Table S5 (closed-triplets)**    

## Reference
(1) **TransE**: [Translating Embeddings for Modeling Multi-relational Data](https://www.cs.sjtu.edu.cn/~li-fang/deeplearning-for-modeling-multi-relational-data.pdf)   
(2) **TransH**: [Knowledge Graph Embedding by Translating on Hyperplanes](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.486.2800&rep=rep1&type=pdf)   
(3) **ConvKB**: [A Novel Embedding Model for Knowledge Base Completion Based on Convolutional Neural Network](https://arxiv.org/pdf/1712.02121.pdf)   
(4) **BERT**: [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/pdf/1810.04805.pdf) (Code: https://github.com/google-research/bert)    
(5) **BioBERT**: [BioBERT: a pre-trained biomedical language representation model for biomedical text mining](https://arxiv.org/pdf/1901.08746v2.pdf)

## Operating Instructions
(1) Run get_E_dict.py to get E_dict.json in **Annotation/**     
```
python get_E_dict.py
```

(2) Run get_C_dict.py to get C_dict.data and E_index.data in **Model/C&D/** (Already in the folder, you can not run)    
```
python get_C_dict.py   
```

(3) Run Run_D_Table.py to get D_table.data in **Model/C&D/**     
```
python Run_D_Table.py --len_d 150 --dim 200 --l_r 1e-5 --batch_size 8 --epoches 10 --earlystop 1   
```

(4) Run Run_KGE.py to train TransE, TransH, and ConvKB in **Model/**
#### Parameter Interpretation  
lanta_c == 0 and lanta_d == 0 : S  
lanta_c != 0 and lanta_d == 0 : S + C  
lanta_c == 0 and lanta_d != 0 : S + D  
lanta_c != 0 and lanta_d != 0 : S + C + D  

**[disease]** from the abbreviation of disease names as follow      
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

**TransE**:   
```
python Run_KGE.py --model TransE --disease [disease] --dim 200 --margin 0.6 --lanta_c 0.0 --lanta_d 0.0 --l_r 5e-3 --epoches 1000
```
**TransH**:  
```
python Run_KGE.py --model TransH --disease [disease] --dim 200 --margin 0.6 --lanta_c 0.0 --lanta_d 0.0 --l_r 5e-3 --epoches 1000
```
**ConvKB**:  
```
python Run_KGE.py --model ConvKB --disease [disease] --dim 200 --n_filter 10 --lanta_c 0.0 --lanta_d 0.0 --l_r 1e-3 --epoches 200
```

The above are the parameters for S.  
For S + C, S + D, and S + C + D, (l_r = 5e-4,  epoches = 200) is recommended.  

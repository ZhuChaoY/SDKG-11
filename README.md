# KGE-SDKG
Multimodal Reasoning based on Knowledge Graph Embedding for Specific Diseases

## Abstract
**Motivation**: Knowledge Graph (KG) is becoming increasingly important in the biomedical field. Deriving new and reliable knowledge from existing knowledge by knowledge graph embedding technology is a cutting-edge method. Some add a variety of additional information to aid reasoning, namely multimodal reasoning. However, most of the works based on the existing biomedical KGs are rarely focus on specific diseases.   
**Results**: This work proposes a construction and multimodal reasoning process of Specific Disease Knowledge Graphs (SDKGs), considering five cancers and six non-cancer diseases, aiming to discover new reliable knowledge and provide universal pre-trained knowledge for that specific disease field. The SDKGs are obtained through original triplet extraction, standard entity set construction, entity linking, and relation linking. We implement multimodal reasoning for SDKGs based on structure, category, and description embeddings. Multimodal reasoning improves pre-existing models on all SDKGs using entity prediction task as the evaluation protocol. We verify the model's reliability in discovering new knowledge by manual proofreading predicted drug-gene, gene-disease, and disease-drug entity pairs. Using embedding results as initialization parameters for the biomolecular interaction classification, we demonstrate the universality of embedding models.

## Files
### Dataset/
**12** special disease knowledge graph from original data：  
**5** Cancer (colon_cancer, gallbladder_cancer, gastric_cancer, liver_cancer, lung_cancer)    
**6** NonCancer (alzheimer_disease, chronic_obstructive_pulmonary, coronary_heart_disease, diabetes, heart_failure, rheumatoid_arthritis)    
**1** Cancer-Generic   
**entity_name_dict.json** : Key-Entity_name mapping dict.   

### inferred new knowledge/  
Inferred new knowledge of drug-gene, gene-disease, and disease-drug entity pairs of special diseases.   
Closed triplets of {drug, gene, disease}.   

### Model/ 
**KGE.py** : Class of processing and tool functions for Knowledge Graph Embedding.  
**Models.py** : TransE, TransH, ConvKB structure.  
**Run_KGE.py** : Train KGE model.  
**Tokenization.py** : Tokenization function of BERT and BioBERT.  
**Optimization.py** : Training optimization of BERT and BioBERT.  
**Train_D_Table.py** : Training function for description table.   
**C&D/** : Category table and Description table (Run Train_D_Table.py to get).     
**Biobert** : BioBERT parameters, self download from https://github.com/dmis-lab/biobert.    

## Reference
(1) **TransE**: [Translating Embeddings for Modeling Multi-relational Data](https://www.cs.sjtu.edu.cn/~li-fang/deeplearning-for-modeling-multi-relational-data.pdf)   
(2) **TransH**: [Knowledge Graph Embedding by Translating on Hyperplanes](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.486.2800&rep=rep1&type=pdf)   
(3) **ConvKB**: [A Novel Embedding Model for Knowledge Base Completion Based on Convolutional Neural Network](https://arxiv.org/pdf/1712.02121.pdf)   
(4) **BERT**: [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/pdf/1810.04805.pdf) (Code: https://github.com/google-research/bert)    
(5) **BioBERT**: [BioBERT: a pre-trained biomedical language representation model for biomedical text mining](https://arxiv.org/pdf/1901.08746v2.pdf)

## Operating Instructions
(1) Run Train_D_Table.py to get D_table in Model/C&D/.   
```
python Train_D_Table.py --len_d 128 --l_r 1e-5 --batch_size 8 --epoches 5 --do_train True --do_predict True  
```

(2) Run Run_KGE.py to train TransE, TransH, and ConvKB.  
Note that we use the abbreviation of disease nameas as follow.
{'ad'  : 'alzheimer_disease',
 'cop' : 'chronic_obstructive_pulmonary',
 'cc'  : 'colon_cancer',
 'chd' : 'coronary_heart_disease',
 'dia' : 'diabetes',
 'gbc' : 'gallbladder_cancer',
 'gsc' : 'gastric_cancer',
 'hf'  : 'heart_failure',
 'lic' : 'liver_cancer',
 'luc' : 'lung_cancer',
 'ra'  : 'rheumatoid_arthritis',
 'can' : '_cancer'}  
**TransE**:   
```
python Run_KGE.py --model TransE --disease can --dim 256 --margin 1.0 --dropout 0.0 --lanta_c 0.0 --lanta_d 0.0 --l_r 1e-3 --epoches 800 --do_train True --save_model False --do_predict True --do_evaluate False
```
**TransH**:  
```
python Run_KGE.py --model TransH --disease can --dim 256 --margin 1.0 --dropout 0.0 --lanta_c 0.0 --lanta_d 0.0 --l_r 1e-3 --epoches 400 --do_train True --save_model False --do_predict True --do_evaluate False   
```
**ConvKB**:  
```
python Run_KGE.py --model ConvKB --disease can --dim 256 --n_filter 8 --dropout 0.1 --lanta_c 0.0 --lanta_d 0.0 --l_r 1e-4 --epoches 200 --do_train True --save_model False --do_predict True --do_evaluate False   
```







# KGE-SDKG
Multimodal Reasoning based on Knowledge Graph Embedding for Specific Diseases

## Abstract
Motivation: Knowledge Graph (KG) is becoming increasingly important in the biomedical field. In addition to being a way of knowledge storage, KG can also provide a valuable reference for disciplinary research. Deriving new and reliable knowledge from existing knowledge by knowledge graph embedding technology is a popular method. Some add a variety of additional information to aid reasoning, namely multimodal reasoning. However, most of the existing biomedical KGs are constructed around an entity type, and they rarely focus on one specific disease.  
Results: We propose a complete Specific Disease Knowledge Graph (SDKG) construction and multimodal reasoning process, considering five cancer diseases and six non-cancer diseases currently, aiming to discover new reliable knowledge and provide effective pre-training prior knowledge for that specific disease field. The final used SDKGs are obtained through original triplet extraction, standard entity set construction, impact factor screening, entity linking, and relation linking. We implement multimodal reasoning for SDKGs based on structure embedding, category embedding, and description embedding. Multimodal reasoning improves published embedding models on all SDKGs using link prediction task as the evaluation protocol. We use embedding results as initialization parameters to provide pre-trained knowledge for biomolecule interaction classification task. The result shows that these pre-trained embeddings can improve 1.5% overall classification accuracy. 

# Files
## Model
KGE.py : A class of processing and tool functions for Knowledge Graph Embedding.  
Tokenization.py : A tokenization function of BERT and BioBERT.  
Optimization.py : A training optimization of BERT and BioBERT.  
TransE.py       :  



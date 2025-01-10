# This repository is the implementation of HMFQA: Health-oriented Multimodal Food Question Answering.



## Updates
January 10, 2025: Update the main code of the Tomm work

October 25, 2022: HMFQA is accepted by MMM2023 (Oral)!

## Abstract
Health-oriented food analysis has become a research hotspot in recent years because it can help people keep away from unhealthy diets. Remarkable advancements have been made in recipe retrieval, food recommendation, nutrition and calorie estimation. However, existing works still cannot well balance the individual preference and the health. Multimodal food question and answering (MFQA) presents substantial promise for practical applications, yet it remains underexplored. In this paper, we introduce a health-oriented MFQA dataset with 9,000 Chinese question-answer pairs based on a multimodal food knowledge graph (MFKG) collected from a food-sharing website. Additionally, we propose a novel framework for MFQA in the health domain that leverages implicit general knowledge and explicit domain-specific knowledge. The framework comprises four key components: implicit general knowledge injection module (IGKIM), explicit domain-specific knowledge retrieval module (EDKRM), ranking module and answer module. The IGKIM facilitates knowledge acquisition at both the feature and text levels. The EDKRM retrieves the most relevant candidate knowledge from the knowledge graph based on the given question. The ranking module sorts the results retrieved by EDKRM and further retrieve candidate knowledge relevant to the problem. Subsequently, the answer module thoroughly analyzes the multimodal information in the query along with the retrieved relevant knowledge to predict accurate answers. Extensive experimental results on the MFQA dataset demonstrate the effectiveness of our proposed method.

For an overview of the Model, please refere to the following picture:
![Model](/Resource/framework.png)

## Main Results

|         Method         |  F1-Score   | Accuracy |
|:----------------------:|:-----------:| :---: | 
|         BAMnet         |    0.681    | 0.546 | 
|      ConceptBert       |    0.664    | 0.662 | 
|          HAN           |    0.668    | 0.664 | 
|          BAN           |    0.676    | 0.671 | 
| Hypergraph Transformer |    0.693    | 0.642 | 
|     HMFQA（MMM2023）     |    0.710    | 0.682 | 
|     KB-HMFQA（Tomm）     |  0.739 | 0.739 | 


## Getting Started


## Prepare：  
Program Structure:

    /HMFQA  
        /Code  
        /Config
        /Resource
            /Dataset
                /dataset_file
            /MFKG
                /KG_pth
            /Model
                /Contrast
            /Output
        /Util
        
1, Download Resource and Util from:
https://pan.baidu.com/s/1N1c8l83i5YDl9hDoJWg8yA?pwd=xgki
and merge the downloaded files into project according to the Directory Structure.

2, Decompress the files in MFKG:
MFQA and MFKG are stored in /HMFQA/Resource/Dataset and /HMFQA/Resource/MFKG respectively.
Note that the MFQA and MFKG provided here has already been transformed into tensors.
The Structure of MFKG:
![MFKG](/Resource/MFKG.png)

3, For original text and images:
We add in https://pan.baidu.com/s/1N1c8l83i5YDl9hDoJWg8yA?pwd=xgki, /HMFQA/Original_Dataset.

Decompress the files in /HMFQA/Original_Dataset.
You may refer to the following instructions:(Linux)
cat filename.tar.gz.*|tar -ZXV

The recipe image is named by the corresponding recipe id.
The process image is named by the corresponding recipe id and the number of processing steps.


## Train and test

Run the main function of the Python files in /HMFQA/Code.

"# tr.Main()" if you only want to test model file.

Different configuration files load different models.Please adjust the configuration file before training.

### Contrast experiment: 
ContrastTrainer.py for BAN,HAN,ConceptBert and Hypergraph.  

ContrastTrainerBAM.py for BAM.

MainTrainer.py for HMFQA.



# This repository is the implementation of HMFQA: Health-oriented Multimodal Food Question Answering.

For an overview of the Model, please refere to the following picture:
![Model](/Resource/FoodQA.png)

### Directory Structure：  

    /HMFQA  
        /Code  
        /Config
        /Resource
            /Dataset
            /MFKG
            /Model
            /Output
        /Util
        
Download Resource and Util from:
https://pan.baidu.com/s/1N1c8l83i5YDl9hDoJWg8yA?pwd=xgki
and merge the downloaded files into project according to the Directory Structure.
Decompress the files in MFKG.


MFQA and MFKG are stored in /HMFQA/Resource/Dataset and /HMFQA/Resource/MFKG respectively.
The Structure of MFKG:
![MFKG](/Resource/MFKG.png)


### For train and test, run the main function of the Python files in /HMFQA/Code.

Baseline train and test: 
ContrastTrainer.py for BAN,HAN,ConceptBert and Hypergraph.
ContrastTrainerBAM.py for BAM.

HMFQA train and test:
MainTrainer.py for HMFQA.
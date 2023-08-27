# This repository is the implementation of HMFQA: Health-oriented Multimodal Food Question Answering.

For an overview of the Model, please refere to the following picture:
![Model](/Resource/FoodQA.png)


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



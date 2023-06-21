# RCC: CT Image Classification
## Introduction

We used CT images and masks of renal tumors to extract radiomics features, and then used a random forest classification model to classify tumors as benign or malignant.

CT image
<figure >
    <img src="./img/picture2.png" width="200"/>
</figure >
Mask
<figure>
    <img src="./img/picture3.png" width="202"/>
</figure >

The flow as shown below.
![flow](./img/picture1.png)


# Installation python package
[Pyradiomics](https://pyradiomics.readthedocs.io/en/latest/) needs to be installed before using this classification model. 
* Use the following command to install pyradiomics or enter their [website](https://pyradiomics.readthedocs.io/en/latest/installation.html) for other installation methods.
    ```
    python -m pip install pyradiomics
    ```
* We calculate radiomics feature using three different filiters(original, wavelet and LoG). Finally, a total of 1316 features will be obtained.

# Usage
We provide some [sample cases](./samplecase/) from [TCGA](https://portal.gdc.cancer.gov/projects/TCGA-KIRC) to familiarize you with this classification model before using your own dataset. 

## 1. Clone repository
You need to clone our repository and then enter the folder named KidneyCancerClassification.
```
git clone https://github.com/LinYuXuan-judy/KidneyTumorclassification.git
cd KidneyCancerClassification
```
## 2. Feature extraction
You need to run [featureExtract.py](./featureExtract.py) with the following command to extraction feature. We also provide an environment file ([exampleCT.yaml](./exampleCT.yaml)) to set some parameters in pyradiomics, so please don't modify it.   
* The most important thing is that the label of the tumor in the mask file must be **2**.

Example: 
```
python3 featureExtract.py samplecase/image samplecase/mask
```

The result of feature extraction will be stored in a folder named featureResult. We provide two different types of files, one is a csv file and the other is a pkl file, as shown below. 
```
featureResult
├── feature.csv
└── feature.pkl
```

If you want to use your own dataset, you need to provide two floders, one with all CT images, the other with all mask files of CT images. Then they need to be arranged into a specific structure, as shown below. 

* You need to notice that the **names** of the same case images and masks need to be the **same** and the **data types** of images and masks need to be **nii.gz** files.

```
samplecase
├── image
│   ├── case_00000.nii.gz
│   ├── case_00001.nii.gz
│   ├── case_00002.nii.gz
│   ├── ...
│
└── mask
    ├── case_00000.nii.gz
    ├── case_00001.nii.gz
    ├── case_00002.nii.gz
    ├── ...
```
And running [featureExtract.py](./featureExtract.py) with the following command.
```
python3 featureExtract.py path/to/the/image/directory path/to/the/mask/directory
```
## 3. Classification - benign vs. low grade vs. high grade
You need to run [classification.py](./classification.py) with the following command to classify tumors as benign, low grade or high grade. 

* You don't need to enter the path to the radiomics feature file as we have programmed to fetch its path, *so please do not move the location of the radiomics results file*. 

```
python3 classification.py
```

The result of classification will be stored in a folder named classificationResult. 
```
classificationResult
└── classificationResult.pkl
```
We store the results using predict_proba, with the order of predict_proba being benign, low grade, high grade.

Example: 

| case_name | predict_proba |
| --------- | :-----------: |
| case_00000| [0.56158, 0.25373, 0.38673]      |
| case_00001| [0.23156, 0.61989, 0.15849]      |
| case_00002| [0.89561, 0.15698, 0.43575]      |

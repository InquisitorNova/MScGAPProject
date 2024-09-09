# Generative Accumulation of Photons: Developing Advanced Models for Generative Modelling and Denoising:

## Project Summary:
In this project we proposed, implemented and evaluated several different neural network architectures for use in the Generative Accumulation of Photons Model, a novel self-supervised
approach to denoising images affected by shot noise. In this experiment we took 14 different neural network architectures and trained them on the FFHQ256 dataset, a large and diverse set 
of 70,000 web scraped flicker images downsampled to 256x256 image size. Data augmentation through horizontal flipping was performed to increase the effective sample dataset size
to 140,000 training instances. To train the models we divided the sample dataset using the 80(train)/20(valid) rule. To further reduce the time taken to train an individual epoch, 
a virtual batch size of 512 was used for the train and a for the validation a virtual batch size of 128 was used. With these settings a single epoch averaged around 1-3minutes 
for the models depending on complexity. The python and Jupyter files in this repository were ran on an NVIDIA AD102 (GeForce RTX 4090) with 24 GB VRAM with the training of an individual
GAP model taking about 1-3 hours depending on the complexity of the mmodel. To train a cascade of 9 GAP Models as used in this experiment took a training time of around 10-18 hours 
depending on the complexity of the models used in the cascade.

Once trained, the models were evaluated and compared against one another to produce the final performance metrics and statistics provided in the report. In this github for sake of ease,
the results of the experiments are stored in the data folder and can be reloaded in the jupyter notebook files without the need to retrain all the models. The figures produced by the
notebooks are also stored in a folder for sake of easy access and retrival. It is recommended due to the length it takes to train the model and the amount of memory these models take up 
on the machine to simply load the prestored files to obtain the figures and only retrain the models if absolutely necessary.

## Requirements:
The Python package requirements required to run these files is provided in the requirements.txt file. It
is advised that you set up a virtual anaconda environment andf then run them in visual studio code to best
recreate the environment these files were created in.

## Setup Instructions:

### 1. Clone the Github Repository: extract the folder from the github first.
'''bash 
git clone https://github.com/InquisitorNova/MScGAPProject/tree/main
'''
Then cd into the project folder

### 2. Create/initialise a conda environment and download dependencies:
For example say we intialise the environment as GAPProject,

'''bash
conda create --name GAPProject python=3.11 anaconda
pip install -r requirements.txt
'''
### 3. Run the DataPreprocessing File first:
The rest of code, requires the FFHQ256 dataset to run. The DataProcessing file makes an api call to the kaggle datasets where the FFHQ 256 dataset is stored. The dataset is downloaded 
as a compresssed zip file and then uncompressed and stored locally in a folder called "ffhq256_dataset". All other jupyter notebooks assume the existence of this file. For all subsequent notebooks
check that the file paths point to where the dataset is being stored to enable proper functionality of the notebooks. The file paths were defined in the notebook to be relative and based off of a 
linux operating system, so bear this in mind if runing it on windows. 

### 4. Run the Demonstration Face Generation File Second:
This noteboook generates the original single model and the original 9-model cascade model producing the summary statistics and corresponding figures used as the benchmark in the report. Run this file to ensure
that the necessary model weights for the subsequent jupyter notebook files are created as well as the necessary numpy statistics. 

### 5. Run the Demonstration Face Progressive File Third:
This notebook generates the original curriculum trained model producing the sumamry statistics and corresponding figures for the curriculum learning based approach in the report. Run this file to ensure 
that the necessary models weights for the subsequent jupyter notebook files are created as well as the necessary numpy statistics. 

### 6. Run the Demonstration Student Test File Fourth:
This notebook generates the Dynamic and Hierarchical UNet models, producing the summary statistics and corresponding figures for the dynamic experiments. Run this file to ensure that the results for the
dynamic models are available for the remaining notebooks.

### 7. Run the Demonstration Gated Test File Fith:
This notebook generates the Curriculum Based Hierarchical and Dynamic UNets, producing summary statistics and corresponding figures for the dynamic experiments. This file is ran to ensure that the results for the
advanced curriculum trained dynamic models are available for the remaining notebooks.

### 8. Run the Demonstration Face Custom UNets last:
This notebook takes the results stored as numpy arrays from the last couple of files and combines it with additional results from the reamining architectures which weren't trained in the 
previous notebook files. It generates from the combined results summary plots showcasing the performance of the different UNets across the PSNR range. These plots alongisde other summary statistics
provide the basis for the analysis of the different UNets in the report.

### Evaluation:
The 5 jupyter notebooks produce quantitative results and visualisations of the performances of the different GAP models across the PSNR Spectrum. The plots produced demonstrate the capabilities of each of the developed
architectures at different noise levels. 

### Dataset:
The FFHQ256 dataset is a dataset known as the FlickerFaces-HQ dataset is a rich and diverse set of 70,000 png faces downsampled to 256x256 spatial resolution. The dataset contains faces from a diverse range of ages, ethinicities, 
image backgrounds, and accessories. The facial dataset provides a benchmark for image generation models such as GANs, Diffusion Models and VAEs. This dataset was introduced as part of academic research introduced by the StyleGAN, a popular
generative model. It is a dowmsampled verison of a much larger 1024x1024 verison of the FFHQ dataset which itself was constructed from web scrapping from Flickr. The dataset can be found at: https://www.kaggle.com/datasets/denislukovnikov/ffhq256-images-only

### Model Names:
Original_Model_m40_32 = Single GAP Model covering the whole PSNR range from -40 PSNR to 32 PSNR = OriginalGAPUNet_Final.py
Deep Supervision Model = Single GAP Model covering the whole PSNR range with auxiliary loss functions providing intermediate supervision = DeepSupervisionUNet_Final_2.py
Gated Model = Single GAP Model with convolutional and attention-based gates = ResidualGatedUNet_Final.py
PSNR_Model = Single GAP Model conditioned on the input psnr level of the feature maps = PSNRUNet_Final_2.py
HybridDilated_Model = Single GAP model which utilises hybrid dilation blocks = HybridDilatedResidualUNet_Final2.py  
AdaptiveUNet_Final = Single GAP model which utilises FiLM and AdaIN layers = AdaptiveResidualUNet_Final2.py
CBAMUNet_Final = Single GAP model which utilises Convolutional Block Attention Modules (CBAM) = CBAM_ResidualUNet_Final.py
Attention_Model = Single GAP Model which utilises multiheaded self-attention mechanisms = AttentionResidualUNet_Final.py
Cross_Attention_Model = Single GAP Model which utilises conditioned self-attention mechanisms = CrossAttentionUNet_Final2.py
DenseUNet_Model = Single GAP Model which utilises densely connected convolutional layers = DenseUNet_Final2.py
Hierarchical_Model = Single GAP Model which utilises the hierarchical learning mechanism = AdvancedGatedUNet_Final.py
Dynamic Convolutional Model = Single GAP which uitilises Dynamic Convolutions = GatedDynamicUNet_Final.py
Hierarhical_Dynamic_Mode = Single GAP Model which utilises both Dynamic and Hierarchical Based Learning = HierarchicalUNet_Final.py




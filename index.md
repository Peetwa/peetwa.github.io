## Automatic Machine Learning Architecture Selection for Breast MRI Classification

## Abstract
Convolutional neural networks (CNN) are increasingly used for image classification tasks.  In general, the architecture of these networks are set ad hoc with little rationale for selected various components such as number of layers, layer depth, and convolution settings. In this work, we develop a structured approach to explore and select architectures that provide optimal classification performance.  This was constructed with an IRB-approved data set containing 10,924 2-D maximum intensity projection (MIP) breast MRI images containing breast cancer lesion present or lesion absent classes. The architecture search method employs a genetic algorithm to generate CNN-based classifiers, representing as strings and mutating them.  During architecture updates, each classifier goes through supervised machine learning on the training set. The search method identifies the method with the highest validation accuracy. In initial testing, we built an optimal CNN that classifies lesion present images with 75% accuracy and achieves an AUC score of 83%. This approach offers a rational framework for architecture exploration, potentially leading to more efficient and generalizable CNN-based classifiers.


## Method

To find 

## Automatic Machine Learning Architecture Selection for Breast MRI Classification

## Abstract
Convolutional neural networks (CNN) are increasingly used for image classification tasks.  In general, the architecture of these networks are set ad hoc with little rationale for selected various components such as number of layers, layer depth, and convolution settings. In this work, we develop a structured approach to explore and select architectures that provide optimal classification performance.  This was constructed with an IRB-approved data set containing 10,924 2-D maximum intensity projection (MIP) breast MRI images containing breast cancer lesion present or lesion absent classes. The architecture search method employs a genetic algorithm to generate CNN-based classifiers, representing as strings and mutating them.  During architecture updates, each classifier goes through supervised machine learning on the training set. The search method identifies the method with the highest validation accuracy. In initial testing, we built an optimal CNN that classifies lesion present images with 75% accuracy and achieves an AUC score of 83%. This approach offers a rational framework for architecture exploration, potentially leading to more efficient and generalizable CNN-based classifiers.

## Genetic Algorithm
### Mutations
1. Remove block
2. Randomize paramaters of block
3. Insert new block

### Generating Offspring
1. Take 4 individuals at random from the parent population Pt. Select the two with the highest fitnesses as parent_1 and parent_2. 
2. Split parent_1 and parent_2 at random indeces r_1 and r_2 into head_1, tail_1, head_2, tail_2.
3. Generate new models q1 = head_1+tail_2 and q2 = head_2+tail_1
4. Add q1 and q2 to Qt.
5. Repeat until len(Qt) == len(Pt)

### Environmental Selection
1. Pick two random indviduals from Pt U Qt
2. Kill the individual with the lower fitness.
3. Repeat until len(Pt U Qt) == len(Pt)

### Model Generation

Models are generated from a base model consisting of an input layer, a convolutional layer and a pooling layer and between 1 and 10 blocks and a dense output layer. Each block can either be a identity block consisting of three convolutional layers and a skip connection, or a convolutional block consiting of three convolutional layers and a skip layer that passes through a single convolutional layer.




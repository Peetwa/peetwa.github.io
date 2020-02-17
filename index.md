## Genetic Algorithm for Machine Learning Architecture Selection for Breast MRI Classification

## Abstract
Convolutional neural networks (CNN) are increasingly used for image classification tasks. In general, the architectures of these networks are set ad hoc with little justification for selecting various components, such as the number of layers, layer depth, and convolution settings. In this work, we develop a structured approach to explore and select architectures that provide optimal classification performance. This was developed with an IRB-approved data set with 9,216 2-D maximum intensity projection (MIP) MRI breast images, containing breast cancer malignant or benign classes. This data set was divided into 7,372 training, 922 validation, and 922 test images. The architecture search method employs a genetic algorithm to find optimal ResNet-based classification architectures through crossover and mutation. Each generation, new model architectures are created from the genetic algorithm and trained with supervised machine learning. This search method identifies the model with the highest area under the ROC curve (AUC). The genetic algorithm produced an optimal model architecture which classifies malignancy in images with 76% accuracy and achieves an AUC score of .83. This approach offers a rational framework for automatic architecture exploration, potentially leading to a set of more efficient and generalizable CNN-based classifiers.

## Genetic Algorithm
### Mutations
1. Remove block
2. Randomize paramaters of block
3. Insert new block

### Generating Offspring
1. Take 4 individuals at random from the parent population Pt. Select the two with the highest fitnesses as parent_1 and parent_2. 
2. Split parent_1 and parent_2 at random indices r_1 and r_2 into head_1, tail_1, head_2, tail_2.
3. Generate new models q1 = head_1+tail_2 and q2 = head_2+tail_1
4. Add q1 and q2 to Qt.
5. Repeat until len(Qt) == len(Pt)

### Environmental Selection
1. Pick two random indviduals from Pt U Qt
2. Kill the individual with the lower fitness.
3. Repeat until len(Pt U Qt) == len(Pt)

### Model Generation
Models are generated from a base model consisting of an input layer, a convolutional layer and a pooling layer and between 1 and 10 blocks and a dense output layer. Each block can either be a identity block consisting of three convolutional layers and a skip connection, or a convolutional block consiting of three convolutional layers and a skip layer that passes through a single convolutional layer.

![Identity Block](https://miro.medium.com/max/2916/1*uyXEvYztiv3fGGCCPbm8Jg.png)
Identity Block

![Convolutional Block](https://miro.medium.com/max/2588/1*U5wkA4O1IpY-ekXqFh0tUQ.png)
Convolutional Block

### Example Images
![Classified Images](/test.svg)



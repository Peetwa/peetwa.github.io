## Genetic Algorithm for Machine Learning Architecture Selection for Breast MRI Classification

## Abstract
Convolutional neural networks (CNN) are increasingly used for image classification tasks. In general, the architectures of these networks are set ad hoc with little justification for selecting various components, such as the number of layers, layer depth, and convolution settings. In this work, we develop a structured approach to explore and select architectures that provide optimal classification performance. This was developed with an IRB-approved data set with 9,216 2-D maximum intensity projection (MIP) MRI breast images, containing breast cancer malignant or benign classes. This data set was divided into 7,372 training, 922 validation, and 922 test images. The architecture search method employs a genetic algorithm to find optimal ResNet-based classification architectures through crossover and mutation. Each generation, new model architectures are created from the genetic algorithm and trained with supervised machine learning. This search method identifies the model with the highest area under the ROC curve (AUC). The genetic algorithm produced an optimal model architecture which classifies malignancy in images with 76% accuracy and achieves an AUC score of .83. This approach offers a rational framework for automatic architecture exploration, potentially leading to a set of more efficient and generalizable CNN-based classifiers.

## Genetic Algorithm
### Mutations
There are three possible mutations: modification, removal of a block at a random location, or inserting a block at a random location between the base layers and the output layers. When a model is selected for mutation a copy of the model is created and then mutated.

```python
def mutate(q):
    mutation_index = np.random.randint(len(q))
    mutation_type = np.random.randint(3)

    if mutation_type == 0:
        return remove_block(q,mutation_index)
    elif mutation_type == 1:
        return modify_block(q,mutation_index)
    elif mutation_type == 2:
        return insert_block(q,mutation_index)
```
```python
def remove_block(q, mutation_index):
    return q[0:mutation_index] + q[mutation_index+1:]
```
```python
def modify_block(q, mutation_index):
    q[mutation_index]['F1'] = np.random.randint(1,33)*8
    q[mutation_index]['F2'] = q[mutation_index]['F1']
    q[mutation_index]['F3'] = q[mutation_index]['F1']
    q[mutation_index]['kernal'] = np.random.randint(1,4)
    
    return q
```
```python
def insert_block(q, mutation_index):
    return q[0:mutation_index] + [generate_block()] + q[mutation_index:]
```

### Crossover
In the crossover function, we select two unique parents that have a 50% chance to create two new models through crossover or mutation that become the offspring. If the parents do not crossover, there is a 25% chance one of them will produce an offspring through mutation. To produce two new models through crossover, we split both parents at random points between the base layers and the output layers of the models. Then, stitch the head of the first parent to the tail of the second parent and the tail of the first parent to the head of the second parent. This yields two new offspring. Both of the offspring will have a 50% chance to be mutated before the model architectures are saved and ready for training

```python
def crossover(p1, p2):
    p1_head, p1_tail, p2_head, p2_tail = split_model(p1, p2)

    q1_params = p1_head + p2_tail
    q2_params = p2_head + p1_tail

    return q1_params, q2_params
```

### Environmental Selection
1. Pick two random indviduals from the union of the parent and offspring population
2. Remove the individual with the lower fitness.
3. Repeat until len(Pt U Qt) == len(Pt)
 
### Model Generation
For every model generated in the genetic algorithm, we start from the same base input layers and output layers, with a random combination of one to ten identity blocks and convolutional blocks that make up the bulk of the model. In ResNet-50, each conv2D layer would have a different number of filters, however, in this implementation we keep this value uniform within the block to reduce the search space of our algorithm. Blocks are also assigned a random kernel size of 1x1, 2x2, or 3x3. This only changes the kernel size of the middle conv2D layer, leaving the others at the default 1x1 size. Once the models are built, the parameters of their architectures are stored in JSON files so they can be reconstructed later for training and evaluation.

![Identity Block](https://miro.medium.com/max/2916/1*uyXEvYztiv3fGGCCPbm8Jg.png)
Identity Block

![Convolutional Block](https://miro.medium.com/max/2588/1*U5wkA4O1IpY-ekXqFh0tUQ.png)
Convolutional Block

### Example Images
![Classified Images](/test.svg)

### Best Model
<a href="https://peetwa.github.io/Models">Best Model</a>
![Final Model](/Models.md)

# Artificial Intelligence
Created in IntelliJ for simplicity when running, make sure you follow the .idea file configurations.

Video showing a walkthrough of the programs can be found here

In javaai.aann the class ML1 extends  BaseML. It uses BaseML to load the data from the CSV file then using that data, 
populates a hashmap for classification purposes.

The NNSMap class extends Java’s HashMap to work in a nearest neighbor fashion using the square Euclidean distance metric.

ML2 extends the previous ML1 but now is able to test the Nearest Neighbor Map.

Here are the edge cases tested
```java
new Measure(5.1, 3.5, 1.4, 0.2),
new Measure(16.0, 4.2, 3.8, 11.0),
new Measure(4.2, 0.8, 0.0, 2.0),
new Measure(5.8,2.7,4.2,1.3),
new Measure(11.0, 16.0, 4.2, 3.8),
new Measure(5.1,3.6,1.5,0.4),
new Measure(0.0,0.0, 0.0, 0.0),
new Measure(100.0,100.0, 100.0, 100.0),
new Measure(-1.0, -1.0, -1.0, -1.0)));
```

ML3 proves that empirically we first put the NNS map to the test on data it has not been trained on.

Within the javaai.ann package, XorHelloWorld.java Trains the XOR network.

Circuit1.java is our own network that we want to train according to the eightfold permutations of X1, X2, and X3 inputs.

Within the javaai.ann.input package, you will find NormalizedIris.java which uses a simple unsupervised learning algorithm to return the low and high range of the data as a 2-tuple array.
This allows all of the data to become normalized between -1 and 1.

Within the javaai.ann.output package, HoangEquilateralDecoding does the encoding and tests it using this minimum distance method by adding Gaussian noise to simulate hypothetical ANN output.
That is, we take the ideal values and perturb them with normally distributed noise with mean 0 and standard deviation 1 scaled by some perturbance value.

Within the javaai.ann.basic package, HoangzIris builds a multilayer perceptron for the iris data. It does this by developing normalized input for the MLP, 
encoding the output values which the multilayer perceptron generates, and tests the multilayer perceptron I designed and trained. 

# Project

Goal is to use a GA to train an MLP for the XOR gate (Ciruit1), implement and test the feedforward equations, then integrate the GA support to learn the XOR gate (Circuit1).

XorObjective.java in javaai.metah.ga generates uniform random deviates for the weights within the range of the constraints.
It then returns double-precision value representing the “fitness” of the interneuron weights based on batch learning.
Lastly, it deploys the DoubleArrayGenome class properly.

XorGa.java is the final driver program that makes the GA run, converge, and learn the XOR gate.
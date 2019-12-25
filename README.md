# Sequential-Forward-Feature-Selection
Python implementation of Sequential Forward Feature Selection from scratch.

* The program will take one input: a dataset where the last column is the class variable. 
* The program will load the dataset and then use the wrapper approach with a sequential forward selection strategy to find a set of essential features. 
* Stratified 5-fold cross-validation was used for measuring accuracy. 
* The program will keep adding the features as long as there is some improvement in the classification accuracy. 
* The output of the program will be the set of important features on the console.

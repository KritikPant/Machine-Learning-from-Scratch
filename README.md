This is a series of scripts written while following the Machine Learning from Scratch Tutorials available on the channel named Python Engineer and available at https://www.youtube.com/playlist?list=PLqnslRFeH2Upcrywf-u2etjdxxkL8nl7E. n this tutorial, the instructor teaches how to implement popular machine learning algorithms while only using python and numpy without the use of additional libraries.

What I learnt:

K Nearest Neigbours:
- A sample is classified by a popularity vote of its nearest neighbours. i.e. If k=3 and 2 of the 3 nearest points on a graph belong to the same class then the sample will be labelled as part of tht class.
- For this to work, training samples must be provided (i.e. multiple different classes plotted on the graph)
- In order to calculate distances we use Euclidiean Distance
<p align="center">
  <img src="https://github.com/KritikPant/Machine-Learning-from-Scratch/blob/master/Images/KNN_Demo.png?raw=true" alt="KNN Demo from tutorial video"/>
</p>

Linear Regression:
- ŷ = wx + b (w = weights, b = bias)
- To find weights and biases, a cost function is used:
<p align="center">
  <img src="https://github.com/KritikPant/Machine-Learning-from-Scratch/blob/master/Images/MSE.png?raw=true" alt="MSE"/>
</p>

- Since this is the error, we want to minimize this so we need to find the minimum of this function. To do this we need to find the derivative:
<p align="center">
  <img src="https://github.com/KritikPant/Machine-Learning-from-Scratch/blob/master/Images/MSE_Derivative.png?raw=true" alt="MSE Derivative"/>
</p>

- This calculates the gradient of the cost function with respect to _w_ and respect to _b_
- Now we use gradient descent which is an iterative technique to find the minimum point:
<p align="center">
  <img src="https://github.com/KritikPant/Machine-Learning-from-Scratch/blob/master/Images/GradientDescent.png?raw=true" alt="Gradient Descent"/>
</p>

- _"So we have some initialization of the weights and the bias and then we want to go into the direction of the steepest descent and the steepest descent is also the gradient so we want to go into the direction of the into the negative direction of the gradient and we do this iteratively until we finally reached the minimum"_
- To do this iteratively, we need some update rules:
<p align="center">
  <img src="https://github.com/KritikPant/Machine-Learning-from-Scratch/blob/master/Images/UpdateRules%2BDerivativesLinearRegression.png?raw=true" alt="Linear Reression - Update rules and derivatives"/>
</p>

- The learning rate is a very important parameter as a small learning rate might be slower but more accurate but a large learning rate might be faster but at the same time never find the minimum point.

<p align="center">
  <img src="https://github.com/KritikPant/Machine-Learning-from-Scratch/blob/master/Images/ComparisonOfLearningRates.png?raw=true" alt="Comparison of learning rates"/>
</p>

Logistic Regression:
- _"In statistics, the logistic model is used to model the probability of a certain class or event existing such as pass/fail, win/lose, alive/dead or healthy/sick. This can be extended to model several classes of events such as determining whether an image contains a cat, dog, lion, etc"_
- In linear regression, we use the formula `f(w,b) = wx + b` which outputs continuous values. To change this into a probability, we use the sigmoid function:
<p align="center">
  <img src="https://github.com/KritikPant/Machine-Learning-from-Scratch/blob/master/Images/Sigmoid.png?raw=true" alt="Sigmoid Function"/>
</p>

- The approximations are as follows due to applying the sigmoid function f(w,b) = wx + b (w = weights, b = bias):
<p align="center">
  <img src="https://github.com/KritikPant/Machine-Learning-from-Scratch/blob/master/Images/LogisticRegressionApproximations.png?raw=true" alt="Logistic Regression Approximations"/>
</p>

- This will output a probability between 0 and 1
- This is the cost function we use:
<p align="center">
  <img src="https://github.com/KritikPant/Machine-Learning-from-Scratch/blob/master/Images/LogisticRegressionCostFunction.png?raw=true" alt="Logistic Regression + Cost Function"/>
</p>

- To optimize this formula, we use gradient descent again. These are the update rules and derivatives for the logistic regression algorithm:
<p align="center">
  <img src="https://github.com/KritikPant/Machine-Learning-from-Scratch/blob/master/Images/UpdateRules%2BDerivativesLogisticRegression.png?raw=true" alt="Logistic Regression - Update rules + Derivatives"/>
</p>

Naive Bayes Classifier

- Based on the Bayes Theorem which states _If we have two events A and B then the probability of event A given that B has already happened is equal to the probability of B given that A has happened times the probability of A divided by the probability of B_:
<p align="center">
  <img src="https://github.com/KritikPant/Machine-Learning-from-Scratch/blob/master/Images/BayesTheorem.png?raw=true" alt="Bayes Theorem"/>
</p>

- In our case, we use this like so:
<p align="center">
  <img src="https://github.com/KritikPant/Machine-Learning-from-Scratch/blob/master/Images/UsageOfBayesTheorem.png?raw=true" alt="How we will use the bayes theorem"/>
</p>

- We then use the chain rule to get the following:
<p align="center">
  <img src="https://github.com/KritikPant/Machine-Learning-from-Scratch/blob/master/Images/UsageOfBayesTheorem2.png?raw=true" alt="How we will use the bayes theorem"/>
</p>

- Terminology:
  - P(y|X) is called the posterior probability 
  - P(X|y) is called the class conditional probability
  - P(y) is called the prior probability of Y
  - P(X) is called the prior probability of X

- It is called _Naive_ Bayes because it assumes that all features (factors contributing to overall probability) are mutually independent which is unlikely in the real world
- _"For example if you want to predict the probability that a person is going out for a run given the feature that the sun is shining and also given the feature that the person is healthy, then both of these features might be independent but both contribute to this probability that the person goes out. In real life a lot of features are not mutually independent but this assumption works fine for a lot of problems"_
- We then have to select the class with the highest probability. We can therefore use the first formula given below. However, since we are only interested in y, we can ignore P(X). We then must use logarithms to get to the third formula provided below. We do this as all the probabilities will be between 0 and 1 so the final calculation will result in a very small number which could lead to overflow errors.
<p align="center">
  <img src="https://github.com/KritikPant/Machine-Learning-from-Scratch/blob/master/Images/BayesSelectClass.png?raw=true" alt="How to select the class with the highest probability"/>
</p>

- In the end, P(y) = frequency
- The class conditional probability is calculated as follows:
<p align="center">
  <img src="https://github.com/KritikPant/Machine-Learning-from-Scratch/blob/master/Images/BayesClassConditionalP.png?raw=true" alt="Class Conditional Probability Calculation"/>
</p>
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
  <img src="https://github.com/KritikPant/Machine-Learning-from-Scratch/blob/master/Images/UpdateRules%2BDerivatives.png?raw=true" alt="Update rules and derivatives"/>
</p>

- The learning rate is a very important parameter as a small learning rate might be slower but more accurate but a large learning rate might be faster but at the same time never find the minimum point.

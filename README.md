This is a series of scripts written while following the Machine Learning from Scratch Tutorials available on the channel named Python Engineer and available at https://www.youtube.com/playlist?list=PLqnslRFeH2Upcrywf-u2etjdxxkL8nl7E. n this tutorial, the instructor teaches how to implement popular machine learning algorithms while only using python and numpy without the use of additional libraries.

What I learnt:

K Nearest Neigbours:
- A sample is classified by a popularity vote of its nearest neighbours. i.e. If k=3 and 2 of the 3 nearest points on a graph belong to the same class then the sample will be labelled as part of tht class.
- For this to work, training samples must be provided (i.e. multiple different classes plotted on the graph)
- In order to calculate distances we use Euclidiean Distance
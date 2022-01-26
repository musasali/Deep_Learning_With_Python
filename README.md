# Experiment 1 Newton Raphson Method
1. What's the Newton-Raphson Method?
1. Can you find the root of the Equation $(x-3)^3 + 2x^2 = 0$ in this way?
1. What happens if you give a different initial guess value (e.g 4.0 or 300000.0) and end condition (e.g 1e-2 or 1e-6) respectively?
1. Write the Python code and record the solution.

# Experiment 2 Data Visualization
1. Create a DataFrame frome the spx.csv file.
1. Clip the data from '2007-01-01' to '2010-12-31' and select the SPX column. 
1. Draw the clipped data on the left. Add annotations and arrows using the annotate functions and the data of crisis_data.
1. Set the xlimit from ’1/1/2007’ to ’1/1/2011’, and ylimit from 600 to 1800. Set the title as
’Important dates in the 2008-2009 financial crisis’. 
1. Calculate the histogram of the data by numpy
function and draw the horizon bar on the right. Then set the title as ’hisogram in the 2008-2010
financial crisis’.

# Experiment 3 Linear regression - a simple NeuralNetwork
1. A very simple neural network
1. Concepts such as target function and cost function
1. Gradient descent optimisation

# Experiment 4 A typical workflow of Keras
1. Define your training data: input tensors and target tensors.
1. Define a network of layers (or model ) that maps your inputs to your targets.
1. Configure the learning process by choosing a loss function, an optimizer, and some metrics to monitor.
1. Iterate on your training data by calling the fit() method of your model.

# Experiment 5 Classifying movie reviews - a binary classification example
Understand the IMDB database and construct a binary classification network.
<ol> 
    <li> Try using one or three hidden layers, and see how doing so affects validation and test accuracy.
    <li> Try using layers with more hidden units or fewer hidden units: 32 units, 64 units, and so on.
    <li> Try using the mse loss function instead of binary_crossentropy.
    <li> Try using the tanh activation (an activation that was popular in the early days of neural networks) instead of relu.
</ol>

# Experiment 6 handwritten digits classification using CNN
Create a neural networks (convnets) for grayscale image classification. Using data augmentation to mitigate overfitting, fine-tuning a pretrained convnet and visualizing what convnets learn
<ol> 
    <li> Understanding convolutional neural networks (convnets)
    <li> Using data augmentation to mitigate overfitting
    <li> Using a pretrained convnet to do feature extraction
    <li> Fine-tuning a pretrained convnet
    <li> Visualizing what convnets learn 
</ol>

# Experiment 7 Time Series Prediction - Advanced use of recurrent neural networks
Implement the time series prediction with jena climate dataset using Recurrent Network Layers, try to pass different arguments to observe
the result, plot the results and analyze the differences. There are many other things to try, in order to improve performance on the
temperature-forecasting problem:
<ol>
    <li> Adjust the number of units in each recurrent layer in the stacked setup
    <li> Adjust the learning rate used by the RMSprop optimizer.
    <li> Try using LSTM layers instead of GRU layers.
    <li> Try using a bigger densely connected regressor on top of the recurrent layers: Don't forget to eventually run the best-performing models on the test set! Otherwise, you'll develop architectures that are overfitting to the validation set.
</ol>

# Comprehensive Experiment Training a convnet on a small dataset
1. Train an image-classification model using very little data
1. Try to use data augmentation to mitigate overfitting
    ## Steps
    1. Download the data from the kaggle.com website
    2. Descript the data information
    3. Make directories of training dataset
    4. Copying images to training, validation, and test directories
    5. Count how many pictures are in each training split (train/validation/test)
    6. Building Convolutional network
    7. Configuring the model for training
    8. Image-processing
    9. Fitting the model using a batch generator
    10. Displaying curves of loss and accuracy during training
    11. Predict the class of several picture
    12. Conclusion and discussion


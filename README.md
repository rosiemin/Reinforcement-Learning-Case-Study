# Deep learning case study

The goal of this case study is to get some practice applying deep learning models to data.  Please select a case study from one of the options below.
<br/>
### Revisit an old case study with new tools
You used linear regression to predict [heavy equipment auction sale prices](https://github.com/gSchool/ds-case-study-linear-models/blob/master/predict_auction_price/README.md), and non-parametric models to predict if someone would [churn from a ride-sharing company](https://github.com/gSchool/dsi-ml-case-study).  Choose one of these old case-studies and use a neural net to get a solution instead.  Compare and contrast your neural network solution with the more standard machine learning models.  Where did it do well?  Where did it do worse?  Was it as easy to train?  How do you interpret your results?  This case case should give you perspective on the advantages and disadvantages of neural networks.
<br/>
<br/>
### Definitely *not* MNIST
The [MNIST dataset](http://yann.lecun.com/exdb/mnist/) is so 1998.  In 2011 Yaroslav Bulatov made the [notMNIST dataset](http://yaroslavvb.blogspot.com/2011/09/notmnist-dataset.html).  There are still 10 classes, but instead of numbers 0 - 9 you are classifying letters A - J of many different font styles.  Note that there are large and small tarball compressed (.tar.gz) datasets of .png files for you process (starting small!).  
<br/>
<br/>
### A part of history: CIFAR-10
From Wikipedia:  
*The CIFAR-10 dataset (Canadian Institute For Advanced Research) is a collection of images that are commonly used to train machine learning and computer vision algorithms. It is one of the most widely used datasets for machine learning research.  The CIFAR-10 dataset contains 60,000 32x32 color images in 10 different classes.  The 10 different classes represent airplanes, cars, birds, cats, deer, dogs, frogs, horses, ships, and trucks. There are 6,000 images of each class.* 
<br/>
You can download the dataset [here](https://www.cs.toronto.edu/~kriz/cifar.html).  Be advised that you may need to subset this dataset to get a trained model in a reasonable amount of time.
<br/>
<br/>
### So you want to predict stocks, don't you?
Well then, Kaggle has data for you.  Have one team member join [Kaggle](https://www.kaggle.com/) (it's free) and then go to Kaggle's [Dow Jones Industrial Average 30 company stock prediciton competion](https://www.kaggle.com/szrlee/stock-time-series-20050101-to-20171231), click on the "Data" tab, scroll down the page until you see the blue "Download All" button and click it to get all the data.  
<br/>
<br/>
### Machine translation
Using guidance from [this blog](https://blog.keras.io/a-ten-minute-introduction-to-sequence-to-sequence-learning-in-keras.html) and bilingual sentence pairs [here](http://www.manythings.org/anki/) make a machine learning model that will translate sentences from one language to another.  You may need to scope this project severely to get timely results. 

At the end of the day your group will be expected to present for 5-10
minutes on your findings.  You can do this directly from your Jupyter
notebooks (but realize that the audience is not going to read your code - use figures/plots and make results clear!)

Cover the following in your presentation.

   1. Talk about what you planned to accomplish
   2. How you organized yourselves as a team
   3. What architecture you chose and why
   4. What final architecture you chose and why (how did you pick your hyperparameters?)
   5. How measured your model's performance
   6. Things learned along the way.


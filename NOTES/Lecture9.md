### 10/9/24
* classification

* our data can have many classes (finite) and predictors/features
* we can use classification to build a model which is methematical model thata llows us to classify the data
* classification: assigning numeric points to a specific class/category that we try to assign our data points to
* what property/combinationo of age and tumor size is unique to malignant tumors?
    * from the class example, it could be malignant if tumor size > 1/2 age
        * this condition makes up our boundary line between categories?
    ![image info](./assets/classification.png)
* once we classify, we are able to distinguish points in one class from points in another and find that threshold
* sometimes we can distinguish clearly, but toher times it may not be as easy to distinguish two classes from another since there might just not be a solution
    * like if we tried to classify animals s rhinos or elephans by weight, there may be some animals where we don't know for sure but have a good guess
* the feasibility fo a classificationt ask completely depends on the relationship between the attributes/predictors and the class
    * like in the case of rhino vs. elephants and weight, there may not be a great relationship between the animal and weight, so it might not be a super feasibly classification
* so like how do we decide a good predictor?
    * i.e. can we use review lengths and number of stars (1-3) as a relationship nd use the review length as a predictor for review quality? or vice versa
    * ![image info](./assets/boobies.png)
        * so yes, in the third, review length is a good predictor of quality, since very little overlap of data and bigger different
        * in the second, a little bit, but the overlap might make it now as good
* correlation is not causation
    * jsut because they related doesn;t mean causation..
* how do we know we've done a good job at classification?
    * testign without cheating
        * split up our data into a trainign set and a separate testign set
        * use the training set to find patterns and create model
        * use the testing case ot evaluate the model on datat its never seen before
    * testing w/o cheating allows us to check that we have not learned a mdoel too specific to the dataset
        * no underfitting or overfitting
            * under = too loose to data
            * over = too tight to training
* that's why we need test and train
    * less mistakes made by model as we keep ttraiing and learning, but we don;t ant to learn too much (liek by trainign all the data) in case we learn somethign too specific
* if we do train on all the data, then we learn trainign too well, and test is low
* if we train not with a lot of data, then no accuracy anywhere
* outliers and noise?
* we can make mistakes during our classifications
* types of mistaes:
    * say we have a rare disease, out of 1000 data piints, only 10 have diseas, a modeel that simply tells peopel theyd ont have a disease, then accuracy is 99%, even though the model not actually be doing anything 
* lazy classification tool
* classification
    * training step: create model based on exaples and training set
    * testing steo: use model to fill in blanks of testing set and compare results of model to the tru values
* instance-based classifiers
    * use the stored trainign records to predict the class label of unseen cases
        * basiclaly liek if we've seen this data before, give it the same classsification as the previous one
    * rote-learners: perform classifgication onlu if the attributies of the unseen record exactly match a record inour trainign set
* nearest neighbor classifier
    * if there is no existing record with same attributes as our unseen record, then we can use similar records for our classification
* K nearest enighbor classifier
    * need
        * training set
        * distance function
        * value for k
    1. compute distance of unseen record to all training records
    2. identify the k nearest neighbors
    3. aggregate the labels of these k neighbors to predice the unseen record class
        * i.e. majority rule
* how do we aggregate?
    * majority rule
        * using weighted majority based on distance
            * if you are closr then you get greater weight in the majority rule
* chanigng k changes the coplexity of the model since we can look at mroe data as k increases
* we can scall attributes to get a better set of data...?
* implications of size of k
    * if k i stoo small -> sensitive to noise points/doesn't generalize wlel
    * if  is too big -> neighborhood may include points from other classes
* pros and cons of KNN
    * pros
        * simple to undersand why a given unseen record was given a particular class
    * cons
        * expensive to classify new points
        * KNN can be problematic in high dimensions
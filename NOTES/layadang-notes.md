# Lecture 2 (09/16)

## Additional Git information
**git fork**: one moment in time of a repo

**remote "upstream"**: main repo you forked from
* precedence over your changes

**remote "origin"**: your forked repo
* git push to your origin
* then git pull (request) your updates to upstream

**To sync:**
```
git pull upstream main

git push origin main
```
The main branch is supposed to be stable, not where you do development (needs to be in isolated area)
* always use a new branch!

## Data Science notes

**Confirmation bias**: subset of examples is not representative of the "truth", tendency to look for more positive examples rather than negative

_"all models are wrong but some are useful"_
* we have a finite set of points, but we're finding a continuous relationship 
* data changes over time (monitor that model over time, are we able to refresh and relearn?)
* what if tehre are infinite number of rules?

Hypothesis is a subset of the rule — do we need more positive/negative examples to learn about the rule?

Sometimes things are random, sometimes we just need more information (another feature is needed to find the relationship)

**Data science workflow**: Process data, explore data, extract features, create model

The data is everything, putting it in the right format is important 

Exploring data is important, visualize it, become an expert: if you don't understand it, how are models supposed to?

Extract features you think affects the outcome: transform the data into what you think is useful

At the very end is when you try to create the model.
* a rather small part of data science

## Worksheet questions
### _a) What property must a hypothesis have?_

Be a subset of the rule, or get closer to the rule

### _b) What examples would you have wanted to try?_

(8, 4, 2) does it need to be in ascending order?

### _c) Given the hypothesis (x, 2x, 3x), for each of the following, determine whether they are positive or negative examples:_

- (2, 4, 6)
- (6, 8 , 10)
- (1, 3, 5)

Positive, negative, negative

### _d) Describe steps of a Data Science Workflow_

Process the data

Explore the data: process again if necessary

Extract features: which ones seem relevant to the task? process and explore again if necessary

Create model: evaluate the model and see which steps to redo

### _e) Give a real world example for each of the following data types:_

- record: (name, age, balance)
- graph: supply + demand
- image: a big matrix of pixels
- text: list of words

### _f) Give a real world example of unsupervised learning_

Given a bunch of mass + size for various coins (unlabeled types), see the clusters and determine how many types of coins are there

What structures are there that are naturally occuring? Clusters?

### _g) Give a real world example of supervised learning_

Given a bunch of mass + size for various coins (labeled types: dimes, quarters, pennies, etc), see the clusters and predict coin type given mass and size


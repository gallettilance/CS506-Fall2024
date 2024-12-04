### 10/21/24
* different classifiers so far
    * nearest neighbor
    * decision treee split on an attribute
    * naive bayes
* now learn support vector machines (SVMs)
    * find the best line (best split) for splitting up +s and -s
* when we do splits with SVM, add 2 lines on each side of the boundary to show the distance from the data to the boundary line
    * we want to find the street (boundary + 2 side lines) that is the widest, so that way we find the widest separation between the classes of data
    ![image info](./assets/street.png)
* how do we actually define a street?
    * line => $y = mx + b$, so let's apply that to this:
        * $X_2 = mX_1 + b$ (then rearrange and accout for some scalars)
        * $w_1x_1 + w_2x_2 + b = 0$
* then how do we actually utlize this lilne defintion to classify an unknown point, u?
    * just plug u into the equation, and if negativbe number, than negative class, positive
* how do we define the lines of the sides of the street?
    * one line is set to -1, the other set to 1
* so the entire street definition
    ![image info](./assets/fullstreet.png)
* if you multiply by a constant than thr decision boundary doesn;t change!
    * but doesn;t this mean infinite solutions/possible definitions?
        * no, since we have the side lines of the streets as part of out defintion, since the actual street middle won't change, but the width of the streect changes
* so if magnitude of w is large, then more narrow street
* if magnitude of w is small, then wider street
    * i think this is basically c
* so how to make the street as wide as possible? -> find the smallest w as possible
* how to move street in the direction of a point
    * pick a step size a (to move a steps in the direction of x)
        * $w_{new} = w_{old} + y_i * x * a$
        * $b_{new} = b_{old} + y_i * a$
* so now the street is moved, so now we have to readjust the size (since the position of the street is different now)
* the full algorithm -> perceptron algorithm
    * start with random line $w_1x_1 + w_z2x_2 + b = 0$
    * define
        * total number of iterations
        * learning rate a
        * expanding rate c
    * repeat number of iterations times
        * pick a point (xi,yi) from dataset
        * if correctly classified, do nothign
        * if incorrectly classified
            * adjust by...
        * expand or rretract width by c
* in short
    * a => steps, so like moc=ving the actual street position, so learning
    * c => width, so like changign thee width of street in postion
* what if we don;t know the width of the street... then we should find it!
    * let's say we have variable y_i, where it is 1 if it's a positive example of -1 if it's a negative example
    * we know all our points should be outside of the street
    * so 
    ![image info](./assets/how.png)
    ![image info](./assets/how2.png)
    ![image info](./assets/how3.png)
    * and we can also see how width and w are inversely proportional
* remember that w is just the vector [w1, w2] 
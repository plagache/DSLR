# multi-classifier : one-vs-all #

[Page wikipedia of a Multiclass classification](https://en.wikipedia.org/wiki/Multiclass_classification)

this strategy involves training a single classifier per class
in our case with have 4 class / houses / classifier


## Loss function ##

our dataset is compose of a 2 layer matrix

    j = column / lectures
    i = index / line
    x = Values associate with > i and j
    m = i max
y represent if our the validity ?
`y = 1` >> the element is of the class
`y = 0` >> the element is not of the class

    J(θ) = (−1/m) m ∑ i=1 yi log( hθ(xi)) + (1 − yi) log(1 − hθ(xi))

we have 2 part in this equation
if y = 1
it activate this part of the equation

    yi log( hθ(xi))

and if y = 0

    (1 − yi) log(1 − hθ(xi))

this is the hypothesis function for the input x
it calculate the probability that the input data x belongs to the class

    x = input data
    h = hypothesis

theta = θ = this is the parameter we are tweaking during training >

    hθ(x) = g(θT x)

probably of this form : θ0 * x0 + θ1 * x1 + θn * xn

[Page wikipedia of a sigmoide function](https://en.wikipedia.org/wiki/Sigmoid_function)

this is a sigmoide function

    L = Limit
    g(z) = L/(1 + e^−z)

this is our activating function
the activation function is usually an abstraction representing the probability/(activation) of a single neuron
it will normalize our data between 0 and 1
[Page wikipedia of a Logistic function](https://en.wikipedia.org/wiki/Logistic_function)
this is a logistique function, its a sigmoide with L = 1

    g(z) = 1/(1 + e^−z)


## Partial derivative ##


    ∂/∂θj J(θ) = 1/m m∑ i=1 (hθ(xi) − yi)xij

∂/∂θj J(θ) : This expression represents the partial derivative of the cost function J(θ) with respect to the j-th parameter θj.

i eme element of the j eme feature
xij

this is the difference between our model prediction and the actual target
i being an index
it is the difference between the hypothesis hθ(xi) and the actual target yi
` diff = (hθ(xi) − yi) `

our goal is to make the sum of all the element(prediction * i do not know) in our dataset
`m∑ i=1`

rapporter au nombre d'element dans notre dataset
`1/m`

## Methods ##

### Train model
tweaking thetas during training to produce weight

### Predict model
takes as parameters the weight of the previous model
create a houses.csv

### Test accuracy
we test houses.csv with the scikit-learn lib accuracy score
[wiki python scikit function accuracy](https://scikit-learn.org/stable/modules/model_evaluation.html#accuracy-score)

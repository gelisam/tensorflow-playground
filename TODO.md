# Baby Singular Learning Theory

## [ ] Replace the "Parity Bot" title with "Baby Singular Learning Theory" everywhere.

Including in the README, the `<title>`, possibly more.

## [ ] Replace the explanation everywhere.

Write something short, I will polish the explanation later.

The new project is about validating the part of Singular Learning Theory which claims that we might be able to control which algorithm the model will learn only by carefully selecting the data. I will be demonstrating this using a very tiny model, as usual.

## [ ] Delete the static analysis

## [ ] Reshape the model

Only 2 inputs, then a layer with two ReLU neurons, then the output layer, only a single linear neuron (this is not a classification task anymore).

## [ ] Different problem

The two inputs are numbers between -1.0 and 1.0, not bits.

One is called the "flag" and the other is called the "payload".

The goal is simply to compute `output = payload`. Easy!

## [ ] Different visualization

Visualize the output as an `output = f payload` graph on the range -2.0 to 2.0. The goal is to see how the function generalizes outside of the dataset, which only uses -1.0 to 1.0.

## [ ] Visualize the loss landscape

Add a section with a bunch of `y = f x` graphs, showing how the loss changes as we vary each of the weights and biases in the tiny model.

## [ ] Visualize and edit the output neuron's bias

Currently, we can see and edit the bias of every neuron except the output neuron. Let's fix that by adding a small square (like the square which shows the other biases) but no medium square (like the squares where we show the output of each neuron)


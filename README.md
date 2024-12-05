# My experiments on tensorflow-playground

This is a fork of google's tensorflow playground. I like to use it for my
experiments so I can easily visualize the results.

## The Parity experiment

This experiment shows that neural networks really can get stuck in a local
minima.

The goal of the neural network is to learn the parity function (whether the
number of 1 bits in the input is odd). Upon startup, the weights are hardcoded
to a human-understandable solution: the first hidden layer's nth node
calculates if the number of bits is at least n, the second hidden layer's nth
node calculates if the number of bits is exactly n, and the output node checks
if the number of bits is 1, 3, 5, or 7. This demonstrates that a solution
exists.

If you click the reset button and train the network, the fixed point will
usually be almost perfect, but not quite: one or two inputs will be
wrongly-classified. This demonstrates that the network is stuck in a local
minima: if it wasn't stuck, then it would be able to improve its weight and
find a fully-working solution.

## Future work

1.  How to get the network unstuck? Maybe refining on the incorrect inputs for a
    little bit, and then going back to train on every input?
2.  The human-understandable solution uses all the nodes of the first hidden
    layer. Maybe the network needs more room, so that it can experiment with
    alternative solutions without compromising its reward?
    Non-anthropomorphised version: after finding one quasi-solution, the
    network can further optimize its accuracy by calculating two or three
    quasi-solutions and averaging or majority-voting the results. this
    encourages using all the available nodes to calculate the same result in
    different ways. if one of those results happens to be even more accurate
    than the almost-perfect solution, then the network will naturally learn to
    ignore the almost-perfect results in favor of the perfect results.
    
    If so, then above a certain size for the first hidden layer with respect to
    the number of input bits, the probability of success should increase
    drastically.
    
    If that is not what the network chooses to do naturally, then maybe we can
    design an architecture which enforces it: divide the network in two, then
    once both halves stabilize, keep the best one, and randomize the other one.
    Keep going in the hope to find a randomized version which does better.
    
    If that works, try a deep version of that, where we alt-randomize a small
    portion of the network in order to give it the opportunity to fix small
    bugs without losing the overall structure of the solution.
3.  Another possibility is that having more nodes in the hidden layers gives
    the neural network more chances to get lucky during the initialization:
    maybe there are only 5 local minima and one global minima, so with enough
    nodes, there is a bigger chance that one of the competing solutions happens
    to be within the vicinity of the global minima. Then the network only has
    to hill-climb each competing solution towards its respective minima, and
    then to select the one with the global minima and ignore the other
    competitors.
    
    If so, then increasing the size of the first hidden layer with respect to
    the number of input bits should improve the probability of success, but it
    should do so linearly, not drastically.
    
    If that is what the network chooses to do naturally, then this explains why
    large neural networks perform better, but in a disappointing way. It also
    explains why once trained, they compress so well.

## Running an experiment

To run the visualization locally, run:
- `npm i` to install dependencies
- `npm run build` to compile the app and place it in the `dist/` directory
- `npm run serve` to serve from the `dist/` directory and open a page on your browser.

For a fast edit-refresh cycle when developing run `npm run serve-watch`.
This will start an http server and automatically re-compile the TypeScript,
HTML and CSS files whenever they change.

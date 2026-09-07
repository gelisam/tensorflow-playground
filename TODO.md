# Baby Singular Learning Theory

## [x] Replace the "Parity Bot" title with "Baby Singular Learning Theory" everywhere.

Including in the README, the `<title>`, possibly more.

## [x] Replace the explanation everywhere.

Write something short, I will polish the explanation later.

The new project is about validating the part of Singular Learning Theory which claims that we might be able to control which algorithm the model will learn only by carefully selecting the data. I will be demonstrating this using a very tiny model, as usual.

## [x] Delete the static analysis

## [x] Reshape the model

Only 2 inputs, then a layer with two ReLU neurons, then the output layer, only a single linear neuron (this is not a classification task anymore).

## [x] Different problem

The two inputs are numbers between -1.0 and 1.0, not bits.

One is called the "flag" and the other is called the "payload".

The goal is simply to compute `output = payload`. Easy!

## [x] Different visualization

Visualize the output as an `output = f payload` graph on the range -2.0 to 2.0. The goal is to see how the function generalizes outside of the dataset, which only uses -1.0 to 1.0.

## [x] Visualize the loss landscape

Add a section with a bunch of `y = f x` graphs, showing how the loss changes as we vary each of the weights and biases in the tiny model.

## [x] Visualize and edit the output neuron's bias

Currently, we can see and edit the bias of every neuron except the output neuron. Let's fix that by adding a small square (like the square which shows the other biases) but no medium square (like the squares where we show the output of each neuron)

## [x] Move the contents of the WEIGHTS panel to the right

So that the word "payload" isn't cropped.

## [x] Put the graphs in a new LANDSCAPE panel

They are currently to the right of the CODE section, outside of any panel.

## [x] Space out the graphs in the LANDSCAPE panel

So that the axis numbers aren't cropped. And so that the axis labels, axis numbers, and graph titles aren't on top of each other.

## [x] Remove the CODE panel and the green rectangle

And the code behind it. We don't need "range" functions, nor the code for "safe in theory" and "safe in practice".

## [x] Use an orthogonal dataset

Don't generate the points of the dataset randomly, put the points on a grid.

## [x] Fix the LANDSCAPE fold button

Clicking on it toggles the triangle but doesn't fold the contents of the panel.

## [ ] Make the LANDSCAPE axes thiner

They are extremely thick right now. They should be 1px or 2px max.

## [ ] Move the LANDSCAPE X axis labels lower

They still overlap a bit with the axis numbers.

## [ ] Move the payload/output graph to OUTPUT

Ditch the heatmap visualization. Instead, draw the `output = f(payload)` graph, but draw several lines, one for each of the 5 different values for `flag` in the dataset's grid. Use a different color for each line. Use this same visualization instead of a heatmap in the WEIGHTS section, including when hovering over a node. Keep the dashed line for the target function `output = payload`.

## [ ] Dataset sequences

Instead of picking a single dataset at the beginning of training and training on random elements of that dataset until the user pauses, a "dataset sequence" will specify a list of elements. When pressing play, we train on each element in sequence, and we stop automatically once we run out of elements.

Introduce the concept to the codebase using the trivial sequence which simply goes through all the elements of the grid several times. Make sure the training stops when we run out of elements, and that it is possible to pause and resume the training. Toggle back the play button automatically when we run out of elements and grey it out.

## [ ] Dataset sequence dropdown

Add a dropdown to the top-right, like in https://gelisam.com/sandbagging

The choices are:
1. flat on left
   initially train on `output = ReLU(payload)`, then slowly increase `flag` from -1.0 to 1.0 as the function transforms into `output = payload` by moving the flat part of the function up until it's outside the -1.0 to 1.0 range. Animate the dashed line to match.
2. flat on right
   same, but starting with `output = -ReLU(-payload)` and moving the flat part down.
3. flat on both sides
   same, but starting with `output = 0.0` and quickly becoming the shape
   ```
        ---
       /
      /
   ---
   ```
   , moving the bottom flat part further down and the top flat part further up.
4. never flat
   same, but starting with `output = 0.0` and quickly becoming the shape
   ```
         /
        /
     ---
    /
   /
   ```
   , shrinking the flat part until it has length 0.0.
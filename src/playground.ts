/* Copyright 2016 Google Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

import * as nn from "./nn";
import { Activations } from "./activation";
import {
  State,
  datasets,
  regDatasets,
  activations,
  // regularizations, // Removed
  getKeyFromValue
} from "./state";
import {Example2D, shuffle, xyToBits, classifyParityData, bitlength, DataGenerator} from "./dataset";
import {AppendingLineChart} from "./linechart";
import * as d3 from 'd3';

// Helper function for formatting numbers
function formatNumber(num: number): string {
    // Round to 1 decimal place, but show .0
    let fixed = num.toFixed(1);
    // Avoid -0.0
    if (fixed === "-0.0") {
        return "0.0";
    }
    return fixed;
}

const RECT_SIZE = 30;
const BIAS_SIZE = 5;

/** The flag values used in the dataset's grid, each drawn in its own color. */
const FLAG_VALUES = [-1, -0.5, 0, 0.5, 1];
const FLAG_COLORS = ["#e41a1c", "#ff7f00", "#4daf4a", "#377eb8", "#984ea3"];
/** Step size (in payload units) used when sampling the payload/output charts. */
const CHART_STEP = 0.05;

enum HoverType {
  BIAS, WEIGHT
}

interface InputFeature {
  f: (x: number, y: number) => number;
  label?: string;
}

let INPUTS: {[name: string]: InputFeature} = {
  "flag": {f: (x, y) => x, label: "flag"},
  "payload": {f: (x, y) => y, label: "payload"},
};

let HIDABLE_CONTROLS = [
  ["Show test data", "showTestData"],
  ["Play button", "playButton"],
  ["Step button", "stepButton"],
  ["Reset button", "resetButton"],
  ["Learning rate", "learningRate"],
  ["# of hidden layers", "numHiddenLayers"],
];

class Player {
  private timerIndex = 0;
  private isPlaying = false;
  private callback: (isPlaying: boolean) => void = null;

  /** Plays/pauses the player. */
  playOrPause() {
    if (this.isPlaying) {
      this.isPlaying = false;
      this.pause();
    } else {
      this.isPlaying = true;
      if (iter === 0) {
        simulationStarted();
      }
      this.play();
    }
  }

  onPlayPause(callback: (isPlaying: boolean) => void) {
    this.callback = callback;
  }

  play() {
    this.pause();
    this.isPlaying = true;
    if (this.callback) {
      this.callback(this.isPlaying);
    }
    this.start(this.timerIndex);
  }

  pause() {
    this.timerIndex++;
    this.isPlaying = false;
    if (this.callback) {
      this.callback(this.isPlaying);
    }
  }

  private start(localTimerIndex: number) {
    d3.timer(() => {
      if (localTimerIndex < this.timerIndex) {
        return true;  // Done.
      }
      oneStep();
      return false;  // Not done.
    }, 0);
  }
}

let state = State.deserializeState();

// Filter out inputs that are hidden.
state.getHiddenProps().forEach(prop => {
  if (prop in INPUTS) {
    delete INPUTS[prop];
  }
});

let selectedNodeId: string = null;
/** Maps each node/input id to a function computing its value for a given (flag, payload). */
let nodeGetValue: {[id: string]: (flag: number, payload: number) => number} = {};
// Domain used both for the flag/payload input space and the charted output range.
let xDomain: [number, number] = [-2, 2];
let linkWidthScale = d3.scale.linear()
  .domain([0, 5])
  .range([1, 10])
  .clamp(true);
let colorScale = d3.scale.linear<string, number>()
                     .domain([-1, 0, 1])
                     .range(["#f59322", "#e8eaeb", "#0877bd"])
                     .clamp(true);
let iter = 0;
let trainData: Example2D[] = [];
let testData: Example2D[] = [];
/** Number of times the trivial dataset sequence repeats the full grid before training stops automatically. */
const SEQUENCE_REPEATS = 200;
/** The predetermined list of training examples, consumed one epoch's worth at a time as training progresses. */
let datasetSequence: Example2D[] = [];
/** Index into datasetSequence of the next epoch to train on. */
let sequenceIndex = 0;
let network: nn.Node[][] = null;
let lossTrain = 0;
let lossTest = 0;
let player = new Player();
let lineChart = new AppendingLineChart(d3.select("#linechart"),
    ["#777"]); // Only one color for training loss
let recentTrainLosses: number[] = [];
const LEARNING_RATES = [10, 3, 1, 0.3, 0.1, 0.03, 0.01, 0.003, 0.001, 0.0001, 0.00001];

function enableFeaturesForDataset() {
  // Default toy-model configuration: two inputs, one ReLU layer with two neurons, one linear output.
  state.networkShape = [2];
  state.numHiddenLayers = state.networkShape.length;
}

function updateLearningRateDisplay(newRate: number, highlight = true) {
    const learningRateValues = document.getElementById('learning-rate-values');
    if (!learningRateValues) return;

    const learningRates = Array.from(learningRateValues.children).map(d => parseFloat(d.innerHTML));
    const newIndex = learningRates.indexOf(newRate);

    if (newIndex !== -1) {
        const currentTransform = learningRateValues.style.transform;
        const newTransform = `translateY(-${newIndex * 20}px)`;

        if (currentTransform === newTransform) {
            return;
        }

        learningRateValues.style.transform = newTransform;

        if (highlight) {
            learningRateValues.classList.add('highlight');
            setTimeout(() => {
                learningRateValues.classList.remove('highlight');
            }, 1200);
        }
    }
}

function makeGUI() {
  // Add collapsible section functionality
  initCollapsibleSections();

  d3.select("#reset-button").on("click", () => {
    // Main reset button now generates a new random seed
    state.seed = Math.floor(Math.random() * 900000 + 100000).toString();
    state.serialize();
    userHasInteracted();
    generateData(); // Uses the new random seed
    reset(); // Reset network (will use the new random seed, no hardcoded weights)
    d3.select("#play-pause-button");
  });

  d3.select("#play-pause-button").on("click", function () {
    // Change the button's content.
    userHasInteracted();
    player.playOrPause();
  });

  player.onPlayPause(isPlaying => {
    d3.select("#play-pause-button").classed("playing", isPlaying);
  });

  setPlayButtonEnabled(true);

  d3.select("#next-step-button").on("click", () => {
    player.pause();
    userHasInteracted();
    if (iter === 0) {
      simulationStarted();
    }
    oneStep();
  });

  // Add Enter key listener for the user seed input
  const userSeedInput = document.getElementById("userSeed") as HTMLInputElement;
  userSeedInput.addEventListener("keypress", function(event) {
    if (event.key === "Enter") {
      event.preventDefault(); // Prevent default form submission if any
      document.getElementById("applyUserSeed").click(); // Trigger click on apply button
    }
  });

  d3.select("#add-layers").on("click", () => {
    if (state.numHiddenLayers >= 6) {
      return;
    }
    state.networkShape[state.numHiddenLayers] = 2;
    state.numHiddenLayers++;
    parametersChanged = true;
    reset();
  });

  d3.select("#remove-layers").on("click", () => {
    if (state.numHiddenLayers <= 0) {
      return;
    }
    state.numHiddenLayers--;
    state.networkShape.splice(state.numHiddenLayers);
    parametersChanged = true;
    reset();
  });


  // Add scale to the gradient color map.
  let x = d3.scale.linear().domain([-1, 1]).range([0, 144]);
  let xAxis = d3.svg.axis()
    .scale(x)
    .orient("bottom")
    .tickValues([-1, 0, 1])
    .tickFormat(d3.format("d"));
  d3.select("#colormap g.core").append("g")
    .attr("class", "x axis")
    .attr("transform", "translate(0,10)")
    .call(xAxis);

  // Seed controls
  d3.select("#applyUserSeed").on("click", () => {
    const userSeedInput = document.getElementById("userSeed") as HTMLInputElement;
    const newSeed = userSeedInput.value;
    if (newSeed && newSeed.trim() !== "") {
      state.seed = newSeed.trim();
      state.serialize(); // Save the new seed
      userHasInteracted();
      Math.seedrandom(state.seed); // Seed the RNG with the user's input
      updateSeedDisplay(); // Show the user's seed
      generateDataPointsOnly(); // Generate new data points using this seed
      reset(); // Reset the network (it will use the now-seeded RNG)
    }
  });

  // Initial display of the seed
  updateSeedDisplay();
}

function updateSeedDisplay() {
  const userSeedInput = document.getElementById("userSeed") as HTMLInputElement;
  if (userSeedInput) {
    userSeedInput.value = state.seed;
  }
}

function updateBiasesUI(network: nn.Node[][]) {
  nn.forEachNode(network, true, node => {
    d3.select(`rect#bias-${node.id}`).style("fill", colorScale(node.bias));
  });
}

function updateWeightsUI(network: nn.Node[][], container) {
  for (let layerIdx = 1; layerIdx < network.length; layerIdx++) {
    let currentLayer = network[layerIdx];
    // Update all the nodes in this layer.
    for (let i = 0; i < currentLayer.length; i++) {
      let node = currentLayer[i];
      for (let j = 0; j < node.inputLinks.length; j++) {
        let link = node.inputLinks[j];
        container.select(`#link${link.source.id}-${link.dest.id}`)
            .style({
              "stroke-dashoffset": -iter / 3,
              "stroke-width": linkWidthScale(Math.abs(link.weight)),
              "stroke": colorScale(link.weight)
            })
            .datum(link);
      }
    }
  }
}

let isHovercardBeingEdited = false;

function drawNode(cx: number, cy: number, nodeId: string, isInput: boolean,
    container, node?: nn.Node, showCanvas = true) {
  let x = cx - RECT_SIZE / 2;
  let y = cy - RECT_SIZE / 2;

  let nodeGroup = container.append("g")
    .attr({
      "class": "node",
      "id": `node${nodeId}`,
      "transform": `translate(${x},${y})`
    });

  // Draw the main rectangle.
  nodeGroup.append("rect")
    .attr({
      x: 0,
      y: 0,
      width: RECT_SIZE,
      height: RECT_SIZE,
    });
  if (isInput) {
    let label = INPUTS[nodeId].label != null ?
        INPUTS[nodeId].label : nodeId;
    // Draw the input label.
    let text = nodeGroup.append("text").attr({
      class: "main-label",
      x: -10,
      y: RECT_SIZE / 2, "text-anchor": "end"
    });
    if (/[_^]/.test(label)) {
      let myRe = /(.*?)([_^])(.)/g;
      let myArray;
      let lastIndex;
      while ((myArray = myRe.exec(label)) != null) {
        lastIndex = myRe.lastIndex;
        let prefix = myArray[1];
        let sep = myArray[2];
        let suffix = myArray[3];
        if (prefix) {
          text.append("tspan").text(prefix);
        }
        text.append("tspan")
        .attr("baseline-shift", sep === "_" ? "sub" : "super")
        .style("font-size", "9px")
        .text(suffix);
      }
      if (label.substring(lastIndex)) {
        text.append("tspan").text(label.substring(lastIndex));
      }
    } else {
      text.append("tspan").text(label);
    }
    nodeGroup.classed("active", true);
  }
  if (!isInput) {
    // Draw the node's bias.
    nodeGroup.append("rect")
      .attr({
        id: `bias-${nodeId}`,
        x: -BIAS_SIZE - 2,
        y: RECT_SIZE - BIAS_SIZE + 3,
        width: BIAS_SIZE,
        height: BIAS_SIZE,
      }).on("mouseenter", function() {
        if (!isHovercardBeingEdited) {
          updateHoverCard(HoverType.BIAS, node, d3.mouse(container.node()));
        }
      }).on("mouseleave", function() {
        if (!isHovercardBeingEdited) {
          updateHoverCard(null);
        }
      });
  }

  // Compute this node's value as a function of (flag, payload). For hidden
  // and output nodes this runs a forward pass; for input nodes it just reads
  // off the corresponding input feature.
  let getValue = node != null ?
      (flag: number, payload: number) => {
        nn.forwardProp(network, constructInput(flag, payload));
        return node.output;
      } :
      (flag: number, payload: number) => INPUTS[nodeId].f(flag, payload);
  nodeGetValue[nodeId] = getValue;

  if (!showCanvas) {
    return nodeGroup;
  }

  // Draw the node's mini payload/output chart.
  let div = d3.select("#network").insert("div", ":first-child")
    .attr({
      "id": `canvas-${nodeId}`,
      "class": "canvas"
    })
    .style({
      position: "absolute",
      left: `${x + 3}px`,
      top: `${y + 3}px`
    })
    .on("mouseenter", function() {
      selectedNodeId = nodeId;
      div.classed("hovered", true);
      nodeGroup.classed("hovered", true);
      drawPayloadOutputChart();
    })
    .on("mouseleave", function() {
      selectedNodeId = null;
      div.classed("hovered", false);
      nodeGroup.classed("hovered", false);
      drawPayloadOutputChart();
    });
  if (isInput) {
    div.classed("active", true);
  }
  let chartSvg = div.append("svg")
      .attr("width", RECT_SIZE)
      .attr("height", RECT_SIZE);
  div.datum({id: nodeId, chartSvg});
  drawMiniChart(chartSvg, getValue);
  return nodeGroup;
}

// Draw network
function drawNetwork(network: nn.Node[][]): void {
  let svg = d3.select("#svg");
  // Remove all svg elements.
  svg.select("g.core").remove();
  // Remove all div elements.
  d3.select("#network").selectAll("div.canvas").remove();
  d3.select("#network").selectAll("div.plus-minus-neurons").remove();

  // Use a fixed width instead of calculating from DOM positions
  // This prevents issues when sections are collapsed
  let padding = 3;
  let width = 600; // features.column's width in the CSS
  svg.attr("width", width);

  // Map of all node coordinates.
  let node2coord: {[id: string]: {cx: number, cy: number}} = {};
  nodeGetValue = {};
  let container = svg.append("g")
    .classed("core", true)
    .attr("transform", `translate(${padding},${padding})`);
  // Draw the network layer by layer.
  let numLayers = network.length;
  let featureWidth = 118;
  let layerScale = d3.scale.ordinal<number, number>()
      .domain(d3.range(1, numLayers - 1))
      .rangePoints([featureWidth, width - RECT_SIZE], 0.7);
  let nodeIndexScale = (nodeIndex: number) => nodeIndex * (RECT_SIZE + 25);


  let calloutThumb = d3.select(".callout.thumbnail").style("display", "none");
  let calloutWeights = d3.select(".callout.weights").style("display", "none");
  let idWithCallout = null;
  let targetIdWithCallout = null;

  // Draw the input layer separately.
  let cx = RECT_SIZE / 2 + 55;
  let nodeIds = Object.keys(INPUTS);
  let maxY = nodeIndexScale(nodeIds.length);
  nodeIds.forEach((nodeId, i) => {
    let cy = nodeIndexScale(i) + RECT_SIZE / 2;
    node2coord[nodeId] = {cx, cy};
    drawNode(cx, cy, nodeId, true, container);
  });

  // Draw the intermediate layers.
  for (let layerIdx = 1; layerIdx < numLayers - 1; layerIdx++) {
    let numNodes = network[layerIdx].length;
    let cx = layerScale(layerIdx) + RECT_SIZE / 2;
    maxY = Math.max(maxY, nodeIndexScale(numNodes));
    addPlusMinusControl(layerScale(layerIdx), layerIdx);
    for (let i = 0; i < numNodes; i++) {
      let node = network[layerIdx][i];
      let cy = nodeIndexScale(i) + RECT_SIZE / 2;
      node2coord[node.id] = {cx, cy};
      drawNode(cx, cy, node.id, false, container, node);

      // Show callout to thumbnails.
      let numNodes = network[layerIdx].length;
      let nextNumNodes = network[layerIdx + 1].length;
      if (idWithCallout == null &&
          i === numNodes - 1 &&
          nextNumNodes <= numNodes) {
        calloutThumb.style({
          display: null,
          top: `${20 + 3 + cy}px`,
          left: `${cx}px`
        });
        idWithCallout = node.id;
      }

      // Draw links.
      for (let j = 0; j < node.inputLinks.length; j++) {
        let link = node.inputLinks[j];
        let path: SVGPathElement = drawLink(link, node2coord, network,
            container, j === 0, j, node.inputLinks.length).node() as any;
        // Show callout to weights.
        let prevLayer = network[layerIdx - 1];
        let lastNodePrevLayer = prevLayer[prevLayer.length - 1];
        if (targetIdWithCallout == null &&
            i === numNodes - 1 &&
            link.source.id === lastNodePrevLayer.id &&
            (link.source.id !== idWithCallout || numLayers <= 5) &&
            link.dest.id !== idWithCallout &&
            prevLayer.length >= numNodes) {
          let midPoint = path.getPointAtLength(path.getTotalLength() * 0.7);
          calloutWeights.style({
            display: null,
            top: `${midPoint.y + 5}px`,
            left: `${midPoint.x + 3}px`
          });
          targetIdWithCallout = link.dest.id;
        }
      }
    }
  }

  // Draw the output node separately.
  cx = width + RECT_SIZE / 2;
  let node = network[numLayers - 1][0];
  let cy = nodeIndexScale(0) + RECT_SIZE / 2 + 30;
  node2coord[node.id] = {cx, cy};
  drawNode(cx, cy, node.id, false, container, node, false);
  // Draw links.
  for (let i = 0; i < node.inputLinks.length; i++) {
    let link = node.inputLinks[i];
    drawLink(link, node2coord, network, container, i === 0, i,
        node.inputLinks.length);
  }
  // Adjust the height of the svg, plus a bit for the "This is the output from
  // one neuron" tips.
  svg.attr("height", maxY + 100);

  // Adjust the height of the features column.
  let height = Math.max(
    getRelativeHeight(calloutThumb),
    getRelativeHeight(calloutWeights),
    getRelativeHeight(d3.select("#network"))
  );
  d3.select(".column.features").style("height", height + "px");
}

function getRelativeHeight(selection) {
  let node = selection.node() as HTMLAnchorElement;
  return node.offsetHeight + node.offsetTop;
}

function addPlusMinusControl(x: number, layerIdx: number) {
  let div = d3.select("#network").append("div")
    .classed("plus-minus-neurons", true)
    .style("left", `${x - 10}px`);

  let i = layerIdx - 1;
  let firstRow = div.append("div").attr("class", `ui-numNodes${layerIdx}`);
  firstRow.append("button")
      .attr("class", "mdl-button mdl-js-button mdl-button--icon")
      .on("click", () => {
        let numNeurons = state.networkShape[i];
        if (numNeurons >= 8) {
          return;
        }
        state.networkShape[i]++;
        parametersChanged = true;
        reset();
      })
    .append("i")
      .attr("class", "material-icons")
      .text("add");

  firstRow.append("button")
      .attr("class", "mdl-button mdl-js-button mdl-button--icon")
      .on("click", () => {
        let numNeurons = state.networkShape[i];
        if (numNeurons <= 1) {
          return;
        }
        state.networkShape[i]--;
        parametersChanged = true;
        reset();
      })
    .append("i")
      .attr("class", "material-icons")
      .text("remove");

  let suffix = state.networkShape[i] > 1 ? "s" : "";
  div.append("div").text(
    state.networkShape[i] + " neuron" + suffix
  );
}

function updateHoverCard(type: HoverType, nodeOrLink?: nn.Node | nn.Link,
    coordinates?: [number, number]) {
  let hovercard = d3.select("#hovercard");
  if (type == null) {
    isHovercardBeingEdited = false;
    hovercard.style("display", "none");
    d3.select("#svg").on("click", null);
    return;
  }
  d3.select("#svg").on("click", () => {
    isHovercardBeingEdited = true;
    hovercard.select(".value").style("display", "none");
    let input = hovercard.select("input");
    input.style("display", null);
    input.on("input", function() {
      if (this.value != null && this.value !== "") {
        if (type === HoverType.WEIGHT) {
          (nodeOrLink as nn.Link).weight = +this.value;
        } else {
          (nodeOrLink as nn.Node).bias = +this.value;
        }
        updateUI();
      }
    });
    input.on("keydown", () => {
      let keycode = (d3.event as any).keyCode;
      if (keycode === 13 || keycode === 27) {
        updateHoverCard(null, nodeOrLink, coordinates);
      }
    });
    let inputElement = input.node() as HTMLInputElement;
    inputElement.focus();
    inputElement.select();
  });
  let value = (type === HoverType.WEIGHT) ?
    (nodeOrLink as nn.Link).weight :
    (nodeOrLink as nn.Node).bias;
  let name = (type === HoverType.WEIGHT) ? "Weight" : "Bias";
  hovercard.style({
    "left": `${coordinates[0] + 20}px`,
    "top": `${coordinates[1]}px`,
    "display": "block"
  });
  hovercard.select(".type").text(name);
  hovercard.select(".value")
    .style("display", null)
    .text(value.toPrecision(2));
  hovercard.select("input")
    .property("value", value.toPrecision(2))
    .style("display", "none");
}

function drawLink(
    input: nn.Link, node2coord: {[id: string]: {cx: number, cy: number}},
    network: nn.Node[][], container,
    isFirst: boolean, index: number, length: number) {
  let line = container.insert("path", ":first-child");
  let source = node2coord[input.source.id];
  let dest = node2coord[input.dest.id];
  let datum = {
    source: {
      y: source.cx + RECT_SIZE / 2 + 2,
      x: source.cy
    },
    target: {
      y: dest.cx - RECT_SIZE / 2,
      x: dest.cy + ((index - (length - 1) / 2) / length) * 12
    }
  };
  let diagonal = d3.svg.diagonal().projection(d => [d.y, d.x]);
  line.attr({
    "marker-start": "url(#markerArrow)",
    class: "link",
    id: "link" + input.source.id + "-" + input.dest.id,
    d: diagonal(datum, 0)
  });

  // Add an invisible thick link that will be used for
  // showing the weight value on hover.
  container.append("path")
    .attr("d", diagonal(datum, 0))
    .attr("class", "link-hover")
    .on("mouseenter", function() {
      if (!isHovercardBeingEdited) {
        updateHoverCard(HoverType.WEIGHT, input, d3.mouse(this));
      }
    }).on("mouseleave", function() {
      if (!isHovercardBeingEdited) {
        updateHoverCard(null);
      }
    });
  return line;
}

function getLoss(network: nn.Node[][], dataPoints: Example2D[]): number {
  let loss = 0;
  for (let i = 0; i < dataPoints.length; i++) {
    let dataPoint = dataPoints[i];
    let input = constructInput(dataPoint.x, dataPoint.y);
    let outputNode = nn.forwardProp(network, input);
    loss += nn.Errors.SQUARE.error(outputNode.output, dataPoint.label);
  }
  return loss / dataPoints.length;
}

function updateUI() {
  // Update the links visually.
  updateWeightsUI(network, d3.select("g.core"));
  // Update the bias values visually.
  updateBiasesUI(network);

  // Redraw each node's mini payload/output chart.
  d3.select("#network").selectAll("div.canvas")
      .each(function(data: {id: string, chartSvg: any}) {
    drawMiniChart(data.chartSvg, nodeGetValue[data.id]);
  });

  function zeroPad(n: number): string {
    let pad = "000000";
    return (pad + n).slice(-pad.length);
  }

  function addCommas(s: string): string {
    return s.replace(/\B(?=(\d{3})+(?!\d))/g, ",");
  }

  function humanReadable(n: number): string {
    return n.toFixed(3);
  }

  // Update loss and iteration number.
  d3.select("#loss-train").text(humanReadable(lossTrain));
  d3.select("#iter-number").text(addCommas(zeroPad(iter)));
  drawPayloadOutputChart();
  drawLossLandscape();

  lineChart.addDataPoint([lossTrain]); // Only add training loss
}

function constructInputIds(): string[] {
  let result: string[] = [];
  for (let inputName in INPUTS) {
    result.push(inputName);
  }
  return result;
}

function constructInput(x: number, y: number): number[] {
  let input: number[] = [];
  for (let inputName in INPUTS) {
    input.push(INPUTS[inputName].f(x, y));
  }
  return input;
}

function updateLearningRate(loss: number) {
  if (loss <= 0.10) {
    state.learningRate = 0.01;
  } else if (loss <= 0.25) {
    state.learningRate = 0.03;
  } else if (loss <= 0.40) {
    state.learningRate = 0.1;
  } else {
    state.learningRate = 0.3;
  }
  updateLearningRateDisplay(state.learningRate);
}

function oneStep(): void {
  if (sequenceIndex >= datasetSequence.length) {
    // The dataset sequence has run out of elements; there's nothing left to train on.
    return;
  }

  // Consume one epoch's worth of elements from the dataset sequence.
  let epoch = datasetSequence.slice(sequenceIndex, sequenceIndex + trainData.length);
  sequenceIndex += epoch.length;

  iter++;
  // The training is done in batches of 10.
  let batchSize = 10;
  for (let i = 0; i < epoch.length / batchSize; i++) {
    let batch = epoch.slice(i * batchSize, (i + 1) * batchSize);
    batch.forEach(point => {
      let input = constructInput(point.x, point.y);
      nn.forwardProp(network, input);
      nn.backProp(network, point.label, nn.Errors.SQUARE);
    });

    // Update weights
    nn.updateWeights(network, state.learningRate);
  }

  // Compute the loss.
  lossTrain = getLoss(network, trainData);
  lossTest = getLoss(network, testData);
  updateLearningRate(lossTrain);

  updateUI();

  if (sequenceIndex >= datasetSequence.length) {
    // Ran out of elements: stop training, toggle the play button back off, and grey it out.
    player.pause();
    setPlayButtonEnabled(false);
  }
}

/**
 * Builds the trivial dataset sequence: the full grid, shuffled and repeated
 * SEQUENCE_REPEATS times, then rewinds the sequence index back to the start.
 */
function generateDatasetSequence() {
  datasetSequence = [];
  for (let repeat = 0; repeat < SEQUENCE_REPEATS; repeat++) {
    let epoch = trainData.slice();
    shuffle(epoch);
    datasetSequence.push(...epoch);
  }
  sequenceIndex = 0;
  setPlayButtonEnabled(true);
}

/** Enables/disables (and visually greys out) the play button. */
function setPlayButtonEnabled(enabled: boolean) {
  d3.select("#play-pause-button")
    .attr("disabled", enabled ? null : true);
}

export function getOutputWeights(network: nn.Node[][]): number[] {
  let weights: number[] = [];
  for (let layerIdx = 0; layerIdx < network.length - 1; layerIdx++) {
    let currentLayer = network[layerIdx];
    for (let i = 0; i < currentLayer.length; i++) {
      let node = currentLayer[i];
      for (let j = 0; j < node.outputLinks.length; j++) {
        let link = node.outputLinks[j];
        weights.push(link.weight);
      }
    }
  }
  return weights;
}

function reset(onStartup=false, hardcodeWeightsOption?:boolean) { // hardcodeWeightsOption is now optional
  lineChart.reset();
  state.serialize();
  if (!onStartup) {
    userHasInteracted();
  }
  player.pause();

  // Set learning rate to 0.3 and update UI
  state.learningRate = 0.3;
  updateLearningRateDisplay(state.learningRate, false);
  recentTrainLosses = []; // Also clear recent losses on reset

  // Rebuild the dataset sequence and rewind to its start, re-enabling the play button.
  generateDatasetSequence();

  // Determine if weights should be hardcoded
  // Priority:
  // 1. Explicit hardcodeWeightsOption if provided (e.g. during initial parity setup)
  // 2. If state.seed is "0"
  // 3. Default to false if neither of the above
  const shouldUseHardcodedWeights = hardcodeWeightsOption !== undefined ? hardcodeWeightsOption :
                                   (state.seed === "0");

  let suffix = state.numHiddenLayers !== 1 ? "s" : "";
  d3.select("#layers-label").text("Hidden layer" + suffix);
  d3.select("#num-layers").text(state.numHiddenLayers);

  // Make a simple network.
  iter = 0;
  let numInputs = constructInput(0 , 0).length;
  let shape = [numInputs].concat(state.networkShape).concat([1]);
  // Output is now a single linear neuron for the payload regression task.
  let outputActivation = Activations.LINEAR;
  network = nn.buildNetwork(shape, Activations.RELU, outputActivation, constructInputIds());

  if (shouldUseHardcodedWeights) {
    network[1][0].inputLinks[0].weight = 0.0;
    network[1][0].inputLinks[1].weight = 1.0;
    network[1][0].bias = 0.0;
    network[1][1].inputLinks[0].weight = 0.0;
    network[1][1].inputLinks[1].weight = 0.0;
    network[1][1].bias = 0.0;
    network[2][0].inputLinks[0].weight = 0.0;
    network[2][0].inputLinks[1].weight = 1.0;
    network[2][0].bias = 0.0;
  }

  lossTrain = getLoss(network, trainData);
  lossTest = getLoss(network, testData);
  updateLearningRate(lossTrain);
  drawNetwork(network);
  updateUI();
  updateSeedDisplay(); // Ensure seed display is current
}

/**
 * Sets up the RNG, updates the seed display, and regenerates data points
 * using the current `state.seed`.
 * This function does NOT modify `state.seed` itself.
 */
function generateData() {
  // state.seed must be set by the caller if a change is intended.
  Math.seedrandom(state.seed);
  updateSeedDisplay(); // Update the displayed seed based on current state.seed
  generateDataPointsOnly(); // Generate points using the now-seeded RNG
}

/**
 * Generates data points (train and test) based on the current state settings
 * (dataset, noise, numSamples) and populates trainData and testData.
 * Assumes Math.random has already been seeded.
 */
function generateDataPointsOnly() {
  const values = FLAG_VALUES;
  const data: Example2D[] = [];
  for (let i = 0; i < values.length; i++) {
    for (let j = 0; j < values.length; j++) {
      const flag = values[i];
      const payload = values[j];
      data.push({x: flag, y: payload, label: payload});
    }
  }
  trainData = data;
  testData = data;
}

let firstInteraction = true;
let parametersChanged = false;

function userHasInteracted() {
  if (!firstInteraction) {
    return;
  }
  firstInteraction = false;
  let page = 'index';
  if (state.tutorial != null && state.tutorial !== '') {
    page = `/v/tutorials/${state.tutorial}`;
  }
}

function simulationStarted() {
  parametersChanged = false;
}

makeGUI();
// state.seed is initialized to "0" by state.ts on first load if not in hash.
// generateData() will use this seed.
generateData();
reset(true, true); // true for onStartup, true for hardcodeWeights (because seed is "0")

function initCollapsibleSections() {
  // Initialize collapse buttons for each section
  const sections = [
    { selector: '.instructions.column', stateKey: 'instructionsCollapsed' },
    { selector: '.ai-safety.column', stateKey: 'aiSafetyCollapsed' },
    { selector: '.features.column', stateKey: 'featuresCollapsed' },
    { selector: '.output.column', stateKey: 'outputCollapsed' },
    { selector: '.landscape.column', stateKey: 'landscapeCollapsed' },
    { selector: '.code.column', stateKey: 'codeCollapsed' }
  ];

  sections.forEach(section => {
    const column = d3.select(section.selector);
    const collapseButton = column.select('.collapse-button');
    const headerWrapper = column.select('.header-wrapper');

    // Add click handler to the header
    headerWrapper.on('click', () => {
      toggleSection(section.selector, section.stateKey);
    });

    // Apply initial collapsed state
    if (state[section.stateKey]) {
      column.classed('collapsed', true);
      collapseButton.style('transform', 'rotate(0deg)');
    } else {
      column.classed('collapsed', false);
      collapseButton.style('transform', 'rotate(90deg)');
    }
  });
}

function toggleSection(selector: string, stateKey: string) {
  const column = d3.select(selector);
  const collapseButton = column.select('.collapse-button');
  const isCollapsed = column.classed('collapsed');

  // Update state
  state[stateKey] = !isCollapsed;
  state.serialize();

  // Toggle collapsed class
  column.classed('collapsed', !isCollapsed);

  if (!isCollapsed) {
    collapseButton.style('transform', 'rotate(0deg)');
  } else {
    collapseButton.style('transform', 'rotate(90deg)');
  }
}

function drawLossLandscape() {
  const container = d3.select("#loss-landscape");
  container.selectAll("*").remove();

  const paramEntries: Array<{label: string, getter: () => number, setter: (value: number) => void}> = [];

  for (let i = 0; i < network[1].length; i++) {
    const node = network[1][i];
    for (let j = 0; j < node.inputLinks.length; j++) {
      const link = node.inputLinks[j];
      paramEntries.push({
        label: `h${i}.w${j}`,
        getter: () => link.weight,
        setter: (value: number) => { link.weight = value; }
      });
    }
    paramEntries.push({
      label: `h${i}.b`,
      getter: () => node.bias,
      setter: (value: number) => { node.bias = value; }
    });
  }

  const outputNode = network[network.length - 1][0];
  for (let i = 0; i < outputNode.inputLinks.length; i++) {
    const link = outputNode.inputLinks[i];
    paramEntries.push({
      label: `out.w${i}`,
      getter: () => link.weight,
      setter: (value: number) => { link.weight = value; }
    });
  }
  paramEntries.push({
    label: "out.b",
    getter: () => outputNode.bias,
    setter: (value: number) => { outputNode.bias = value; }
  });

  const chartWidth = 190;
  const chartHeight = 110;
  const margin = {top: 12, right: 10, bottom: 28, left: 32};

  paramEntries.forEach((entry, index) => {
    const svg = container.append("svg")
      .attr("width", chartWidth)
      .attr("height", chartHeight);

    const xScale = d3.scale.linear().domain([-2, 2]).range([margin.left, chartWidth - margin.right]);
    const yMin = Math.min(0, lossTrain * 0.8);
    const yMax = Math.max(0.5, lossTrain * 1.5, 0.5);
    const yScale = d3.scale.linear().domain([yMin, yMax]).range([chartHeight - margin.bottom, margin.top]);

    const currentValue = entry.getter();
    const lineData: {x: number, y: number}[] = [];
    for (let x = currentValue - 2; x <= currentValue + 2; x += 0.1) {
      entry.setter(x);
      lineData.push({x, y: getLoss(network, trainData)});
    }
    entry.setter(currentValue);

    const line = d3.svg.line<{x: number, y: number}>()
      .x(d => xScale(d.x))
      .y(d => yScale(d.y));

    svg.append("path")
      .datum(lineData)
      .attr("d", line)
      .attr("fill", "none")
      .attr("stroke", "#0877bd")
      .attr("stroke-width", 1.5);

    svg.append("g")
      .attr("class", "x axis")
      .attr("transform", `translate(0,${chartHeight - margin.bottom})`)
      .call(d3.svg.axis().scale(xScale).orient("bottom").ticks(3));

    svg.append("g")
      .attr("class", "y axis")
      .attr("transform", `translate(${margin.left},0)`)
      .call(d3.svg.axis().scale(yScale).orient("left").ticks(2));

    svg.append("text")
      .attr("x", chartWidth / 2)
      .attr("y", chartHeight - 2)
      .attr("text-anchor", "middle")
      .style("font-size", "9px")
      .text(entry.label);

    if (index % 3 === 2) {
      container.append("br");
    }
  });
}

/**
 * Computes, for each of the dataset's flag values, the sequence of
 * (payload, value) points obtained by scanning payload across the chart's
 * domain. `getValue` typically runs a forward pass of the network.
 */
function computeFlagCurves(
    getValue: (flag: number, payload: number) => number
): {x: number, y: number}[][] {
  return FLAG_VALUES.map(flag => {
    const points: {x: number, y: number}[] = [];
    for (let payload = xDomain[0]; payload <= xDomain[1] + 1e-9; payload += CHART_STEP) {
      points.push({x: payload, y: getValue(flag, payload)});
    }
    return points;
  });
}

/**
 * Draws the dashed "output = payload" target line, plus one colored line
 * per flag value, into the given svg using the given scales.
 */
function drawFlagCurves(svg, xScale, yScale,
    getValue: (flag: number, payload: number) => number, strokeWidth: number) {
  const line = d3.svg.line<{x: number, y: number}>()
    .x(d => xScale(d.x))
    .y(d => yScale(d.y));

  svg.append("path")
    .datum([{x: xDomain[0], y: xDomain[0]}, {x: xDomain[1], y: xDomain[1]}])
    .attr("d", line)
    .attr("fill", "none")
    .attr("stroke", "#d0d0d0")
    .attr("stroke-dasharray", "4,4");

  computeFlagCurves(getValue).forEach((points, i) => {
    svg.append("path")
      .datum(points)
      .attr("d", line)
      .attr("fill", "none")
      .attr("stroke", FLAG_COLORS[i])
      .attr("stroke-width", strokeWidth);
  });
}

/** Draws a small, axis-less version of the payload/output chart. */
function drawMiniChart(svg, getValue: (flag: number, payload: number) => number) {
  svg.selectAll("*").remove();
  const xScale = d3.scale.linear().domain(xDomain).range([0, RECT_SIZE]);
  const yScale = d3.scale.linear().domain(xDomain).range([RECT_SIZE, 0]);
  drawFlagCurves(svg, xScale, yScale, getValue, 1);
}

function drawPayloadOutputChart() {
  const container = d3.select("#payload-output-chart");
  container.selectAll("*").remove();

  const width = 560;
  const height = 320;
  const margin = {top: 12, right: 16, bottom: 32, left: 40};
  const svg = container.append("svg")
    .attr("width", width)
    .attr("height", height);

  const xScale = d3.scale.linear().domain(xDomain).range([margin.left, width - margin.right]);
  const yScale = d3.scale.linear().domain(xDomain).range([height - margin.bottom, margin.top]);

  svg.append("g")
    .attr("class", "x axis")
    .attr("transform", `translate(0,${height - margin.bottom})`)
    .call(d3.svg.axis().scale(xScale).orient("bottom").tickValues([-2, -1, 0, 1, 2]));

  svg.append("g")
    .attr("class", "y axis")
    .attr("transform", `translate(${margin.left},0)`)
    .call(d3.svg.axis().scale(yScale).orient("left").tickValues([-2, -1, 0, 1, 2]));

  const selectedId = selectedNodeId != null ? selectedNodeId : nn.getOutputNode(network).id;
  drawFlagCurves(svg, xScale, yScale, nodeGetValue[selectedId], 2);

  svg.append("text")
    .attr("x", width / 2)
    .attr("y", height - 4)
    .attr("text-anchor", "middle")
    .style("font-size", "11px")
    .text("payload");

  svg.append("text")
    .attr("transform", "rotate(-90)")
    .attr("x", -height / 2)
    .attr("y", 14)
    .attr("text-anchor", "middle")
    .style("font-size", "11px")
    .text("output");

  const legend = container.append("div").style({
    display: "flex",
    "justify-content": "center",
    gap: "14px",
    "margin-top": "-4px"
  });
  FLAG_VALUES.forEach((flag, i) => {
    legend.append("span")
      .style("font-size", "11px")
      .style("color", FLAG_COLORS[i])
      .text(`flag=${flag}`);
  });
}

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
import {ActivationFunction, Activations} from "./activation";

/**
 * A node in a neural network. Each node has a state
 * (pre-activation, output, and their respectively derivatives) which changes
 * after every forward and back propagation run.
 */
export class Node {
  id: string;
  bias = 0.1;
  /** List of input links. */
  inputLinks: Link[] = [];
  /** List of output links. */
  outputLinks: Link[] = [];
  /** Activation function that takes pre-activation and returns node's output */
  activation: ActivationFunction;

  /** The sum of the inputs * weights + bias. */
  preActivation: number;

  /** relu(preActivation), or whatever is the activation function */
  output: number;

  /** To minimize the squared error, we would like to change the output in this direction. */
  desiredDeltaOutput: number;

  /** Sum of 'desiredDeltaPreActivation' for many samples. */
  accDesiredDeltaPreActivation = 0;

  /** Number of samples that have contributed to 'accDesiredDeltaPreActivation'. */
  numDesiredDeltaPreActivations = 0;

  /**
   * Creates a new node with the provided id and activation function.
   */
  constructor(id: string, activation: ActivationFunction) {
    this.id = id;
    this.activation = activation;
  }

  /** Recomputes the node's output and returns it. */
  updateOutput(): number {
    // Stores pre-activation into the node.
    this.preActivation = this.bias;
    for (let j = 0; j < this.inputLinks.length; j++) {
      let link = this.inputLinks[j];
      this.preActivation += link.weight * link.source.output;
    }
    this.output = this.activation.output(this.preActivation);
    return this.output;
  }
}

/**
 * An error function and its derivative.
 */
export interface ErrorFunction {
  /** f(output - target) */
  error: (output: number, target: number) => number;
  /** f'(output - target) */
  der: (output: number, target: number) => number;
}

/** Built-in error functions */
export class Errors {
  public static SQUARE: ErrorFunction = {
    error: (output: number, target: number) =>
               0.5 * Math.pow(output - target, 2),
    der: (output: number, target: number) => output - target
  };
}

/**
 * A link in a neural network. Each link has a weight and a source and
 * destination node. Also it has an internal state (error derivative
 * with respect to a particular input) which gets updated after
 * a run of back propagation.
 */
export class Link {
  id: string;
  source: Node;
  dest: Node;
  weight = Math.random() - 0.5; // Will use seedrandom via playground.ts
  isDead = false;

  /** To minimize the squared error, we would like to change the input in this direction. */
  desiredDeltaInput: number;

  /** Sum of 'desiredDeltaWeight' for many downstream nodes and many samples. */
  accDesiredDeltaWeight = 0;

  /** Number of samples that have contributed to 'accDesiredDeltaWeight'. */
  numDesiredDeltaWeights = 0;

  /**
   * Constructs a link in the neural network initialized with random weight.
   *
   * @param source The source node.
   * @param dest The destination node.
   */
  constructor(source: Node, dest: Node) {
    this.id = source.id + "-" + dest.id;
    this.source = source;
    this.dest = dest;
  }
}

/**
 * Builds a neural network.
 *
 * @param networkShape The shape of the network. E.g. [1, 2, 3, 1] means
 *   the network will have one input node, 2 nodes in first hidden layer,
 *   3 nodes in second hidden layer and 1 output node.
 * @param activation The activation function of every hidden node.
 * @param outputActivation The activation function for the output nodes.
 * @param inputIds List of ids for the input nodes.
 */
export function buildNetwork(
    networkShape: number[], activation: ActivationFunction,
    outputActivation: ActivationFunction,
    inputIds: string[]): Node[][] {
  let numLayers = networkShape.length;
  let id = 1;
  /** List of layers, with each layer being a list of nodes. */
  let network: Node[][] = [];
  for (let layerIdx = 0; layerIdx < numLayers; layerIdx++) {
    let isOutputLayer = layerIdx === numLayers - 1;
    let isInputLayer = layerIdx === 0;
    let currentLayer: Node[] = [];
    network.push(currentLayer);
    let numNodes = networkShape[layerIdx];
    for (let i = 0; i < numNodes; i++) {
      let nodeId = id.toString();
      if (isInputLayer) {
        nodeId = inputIds[i];
      } else {
        id++;
      }
      let node = new Node(nodeId, isOutputLayer ? outputActivation : activation);
      currentLayer.push(node);
      if (layerIdx >= 1) {
        // Add links from nodes in the previous layer to this node.
        for (let j = 0; j < network[layerIdx - 1].length; j++) {
          let prevNode = network[layerIdx - 1][j];
          let link = new Link(prevNode, node);
          prevNode.outputLinks.push(link);
          node.inputLinks.push(link);
        }
      }
    }
  }
  return network;
}

/**
 * Runs a forward propagation of the provided input through the provided
 * network. This method modifies the internal state of the network - the
 * pre-activation and output of each node in the network.
 *
 * @param network The neural network.
 * @param inputs The input array. Its length should match the number of input
 *     nodes in the network.
 * @return The output node.
 */
export function forwardProp(network: Node[][], inputs: number[]): Node {
  const inputLayer = network[0];
  if (inputs.length !== inputLayer.length) {
    throw new Error("The number of inputs must match the number of nodes in" +
      " the input layer");
  }
  // Update the input layer.
  for (let i = 0; i < inputLayer.length; i++) {
    const node = inputLayer[i];
    node.output = inputs[i];
  }

  for (let layerIdx = 1; layerIdx < network.length; layerIdx++) {
    const currentLayer = network[layerIdx];
    // Update all the nodes in this layer.
    for (let i = 0; i < currentLayer.length; i++) {
      const node = currentLayer[i];
      node.updateOutput();
    }
  }
  const outputNode = network[network.length - 1][0];
  return outputNode;
}

/**
 * Runs a backward propagation using the provided target and the
 * computed output of the previous call to forward propagation.
 * This method modifies the internal state of the network - the error
 * derivatives with respect to each node, and each weight
 * in the network.
 */
export function backProp(network: Node[][], target: number,
    errorFunc: ErrorFunction): void {
  // The output node is a special case. We use the user-defined error
  // function for the derivative.
  const outputNode = network[network.length - 1][0];
  const outputValue = outputNode.output;
  const desiredDeltaSquaredError = -1; // We want to reduce the error.
  const squaredErrorWrtOutput = errorFunc.der(outputValue, target);
  outputNode.desiredDeltaOutput = desiredDeltaSquaredError * squaredErrorWrtOutput;

  // Go through the layers backwards.
  for (let layerIdx = network.length - 1; layerIdx >= 1; layerIdx--) {
    const currentLayer = network[layerIdx];
    // At this point we know how much we want to change the output of each node
    // in this layer. Use it to calculate how much we want to change the
    // pre-activation of each node.
    for (let i = 0; i < currentLayer.length; i++) {
      const node = currentLayer[i];
      const outputWrtPreActivation = node.activation.der(node.preActivation);
      const desiredDeltaPreActivation = node.desiredDeltaOutput * outputWrtPreActivation;
      node.accDesiredDeltaPreActivation += desiredDeltaPreActivation;
      node.numDesiredDeltaPreActivations++;

      // Now that we know how much we want to change the pre-activation the node,
      // we can calculate how much we want to change each weight and each input.
      for (let j = 0; j < node.inputLinks.length; j++) {
        const link = node.inputLinks[j];
        if (link.isDead) {
          continue;
        }
        // contribution = weight * input
        const input = link.source.output;
        const desiredDeltaContribution = desiredDeltaPreActivation;
        const contributionWrtWeight = input;
        const contributionWrtInput = link.weight;
        const desiredDeltaWeight = desiredDeltaContribution * contributionWrtWeight;
        link.desiredDeltaInput = desiredDeltaContribution * contributionWrtInput;
        link.accDesiredDeltaWeight += desiredDeltaWeight;
        link.numDesiredDeltaWeights++;
      }
    }
    if (layerIdx === 1) {
      continue;
    }
    const prevLayer = network[layerIdx - 1];
    for (let i = 0; i < prevLayer.length; i++) {
      const node = prevLayer[i];
      // Sum up the desired delta inputs from all downstream links in order to
      // compute this node's desired delta output.
      node.desiredDeltaOutput = 0;
      for (let j = 0; j < node.outputLinks.length; j++) {
        const downstreamLink = node.outputLinks[j];
        node.desiredDeltaOutput += downstreamLink.desiredDeltaInput;
      }
    }
  }
}

function safeDivide(numerator: number, denominator: number): number {
  if (denominator === 0) {
    return 0;
  }
  return numerator / denominator;
}

/**
 * Update the weights of the network using the accumulated desired deltas.
 */
export function updateWeights(network: Node[][], learningRate: number) {
  for (let layerIdx = 1; layerIdx < network.length; layerIdx++) {
    let currentLayer = network[layerIdx];
    for (let i = 0; i < currentLayer.length; i++) {
      let node = currentLayer[i];
      // Update the node's bias.
      const accDesiredDeltaBias = node.accDesiredDeltaPreActivation;
      const numDesiredDeltaBias = node.numDesiredDeltaPreActivations;
      const practicalDesiredDeltaBias = safeDivide(accDesiredDeltaBias, numDesiredDeltaBias);

      const totalDesiredDeltaBias = practicalDesiredDeltaBias;
      node.bias += learningRate * totalDesiredDeltaBias;

      node.accDesiredDeltaPreActivation = 0;
      node.numDesiredDeltaPreActivations = 0;

      // Update the weights coming into this node.
      for (let j = 0; j < node.inputLinks.length; j++) {
        let link = node.inputLinks[j];
        if (link.isDead) {
          continue;
        }

        const practicalDesiredDeltaWeight = safeDivide(link.accDesiredDeltaWeight, link.numDesiredDeltaWeights);

        const totalDesiredDeltaWeight = practicalDesiredDeltaWeight;
        link.weight += learningRate * totalDesiredDeltaWeight;

        link.accDesiredDeltaWeight = 0;
        link.numDesiredDeltaWeights = 0;
      }
    }
  }
}

/** Iterates over every node in the network/ */
export function forEachNode(network: Node[][], ignoreInputs: boolean,
    accessor: (node: Node) => any) {
  for (let layerIdx = ignoreInputs ? 1 : 0;
      layerIdx < network.length;
      layerIdx++) {
    let currentLayer = network[layerIdx];
    for (let i = 0; i < currentLayer.length; i++) {
      let node = currentLayer[i];
      accessor(node);
    }
  }
}

/** Returns the output node in the network. */
export function getOutputNode(network: Node[][]) {
  return network[network.length - 1][0];
}


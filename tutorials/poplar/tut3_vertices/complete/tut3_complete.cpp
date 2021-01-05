// Copyright (c) 2018 Graphcore Ltd. All rights reserved.

/* This file contains the completed version of Poplar tutorial 3.
   See the Poplar user guide for details.
*/

#include <iostream>
#include <poplar/Engine.hpp>
#include <poplar/Graph.hpp>
#include <poplar/IPUModel.hpp>

using namespace poplar;
using namespace poplar::program;

int main() {
  // Create the IPU model device
  IPUModel ipuModel;
  Device device = ipuModel.createDevice();
  Target target = device.getTarget();

  // Create the Graph object
  Graph graph(target);

  // Add codelets to the graph
  graph.addCodelets("tut3_codelets.cpp");

  // Add variables to the graph
  Tensor v1 = graph.addVariable(FLOAT, {4}, "v1");
  Tensor v2 = graph.addVariable(FLOAT, {4}, "v2");
  for (unsigned i = 0; i < 4; ++i) {
    graph.setTileMapping(v1[i], i);
    graph.setTileMapping(v2[i], i);
  }

  // Create a control program that is a sequence of steps
  Sequence prog;

  // Add steps to initialize the variables
  Tensor c1 = graph.addConstant<float>(FLOAT, {4}, {1.0, 1.5, 2.0, 2.5});
  graph.setTileMapping(c1, 0);
  prog.add(Copy(c1, v1));

  ComputeSet computeSet = graph.addComputeSet("computeSet");
  for (unsigned i = 0; i < 4; ++i) {
    VertexRef vtx = graph.addVertex(computeSet, "SumVertex");
    graph.connect(vtx["in"], v1.slice(i, 4));
    graph.connect(vtx["out"], v2[i]);
    graph.setTileMapping(vtx, i);
    graph.setCycleEstimate(vtx, 20);
  }

  // Add step to execute the compute set
  prog.add(Execute(computeSet));

  // Add step to print out v2
  prog.add(PrintTensor("v2", v2));

  // Create the engine
  Engine engine(graph, prog);
  engine.load(device);

  // Run the control program
  std::cout << "Running program\n";
  engine.run(0);
  std::cout << "Program complete\n";

  return 0;
}

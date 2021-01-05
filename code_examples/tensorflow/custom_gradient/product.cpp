// Copyright 2020 Graphcore Ltd.
#include <poplar/Graph.hpp>
#include <poplin/MatMul.hpp>
#include <popops/ElementWise.hpp>
#include <poputil/exceptions.hpp>

/// We are using a stateless op which requires
/// API level 1 or higher.
extern "C" {
  int32_t custom_op_api_level = 1;
}

/// Meta data function sets properties of the forward op.
extern "C"
void Build_metadata(std::vector<std::int64_t>& allocating_indices,
                    std::uint32_t& num_inplace,
                    bool& is_elementwise,
                    bool& is_stateless,
                    std::uint32_t num_inputs) {
  allocating_indices.clear();
  num_inplace = 0;
  is_elementwise = false;
  is_stateless = false;
}

extern "C" poplar::program::Program Build(
  poplar::Graph& graph, const std::vector<poplar::Tensor>& inputs,
  std::vector<poplar::Tensor>& outputs, const std::string& debug_prefix) {

  if (inputs.size() != 2) {
    throw poputil::poplibs_error("product requires 2 inputs.");
  }

  auto input = inputs[0];
  auto weights = inputs[1];
  if (input.rank() != 2 && weights.rank() != 2) {
    throw poputil::poplibs_error("Both inputs must be matrices.");
  }

  if (input.dim(1) != weights.dim(0)) {
    throw poputil::poplibs_error("Product shapes incompatible.");
  }

  poplar::program::Sequence prog;
  auto result = poplin::matMul(graph, input, weights, prog,
                               debug_prefix + "/product");
  outputs.push_back(result);

  return prog;
}

/// The gradient op requires its own meta data. In this case we mark the op
/// as stateless so that only one instance of the op is compiled even when
/// we ask for the gradient multiple times (e.g. we use tf.gradients() in
/// the python code).
extern "C"
void Build_grad_metadata(std::vector<std::int64_t>& allocating_indices,
                    std::uint32_t& num_inplace,
                    bool& is_elementwise,
                    bool& is_stateless,
                    std::uint32_t num_inputs) {
  allocating_indices.clear();
  num_inplace = 0;
  is_elementwise = false;
  is_stateless = true;
}

extern "C"
poplar::program::Program Build_grad(
    poplar::Graph& graph, int input_grad_index,
    const std::vector<poplar::Tensor>& gradients,
    const std::vector<poplar::Tensor>& fwd_inputs,
    const std::vector<poplar::Tensor>& fwd_outputs,
    std::vector<poplar::Tensor>& outputs,
    const std::string& debug_prefix) {

  poplar::program::Sequence prog;
  auto inputsTransposed = fwd_inputs[0].dimShuffle({1, 0});
  auto weightsTransposed = fwd_inputs[1].dimShuffle({1, 0});
  auto gradOfLossWrtWeights =
    poplin::matMul(graph, inputsTransposed, gradients[0],
    prog, debug_prefix + "/dLdW");
  auto gradOfLossWrtInput =
    popops::mul(graph,
              gradients[0].broadcast(fwd_inputs[1].dim(0), 1),
              weightsTransposed.broadcast(gradients[0].dim(0), 0),
              prog,
              debug_prefix + "/dLdX");
  
  outputs.push_back(gradOfLossWrtInput);
  outputs.push_back(gradOfLossWrtWeights);

  return prog;
}

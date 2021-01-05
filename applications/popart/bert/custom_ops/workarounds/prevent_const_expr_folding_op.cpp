// Copyright 2020 Graphcore Ltd.
#include <popart/op.hpp>
#include <popart/names.hpp>
#include <popart/opmanager.hpp>
#include <popart/region.hpp>
#include <popart/popx/opx.hpp>
#include <popart/popx/opxmanager.hpp>
#include <popart/popx/devicex.hpp>

namespace CustomOperators
{
    const popart::OperatorIdentifier PreventConstFolding = {"ai.graphcore", "PreventConstFolding", 1};
} // namespace CustomOperators
namespace CustomGradOperators {
  const popart::OperatorIdentifier PreventConstFoldingGrad = {"ai.graphcore", "PreventConstFoldingGrad", 1};
} // namespace CustomGradOperators

class PreventConstFoldingOp;
class PreventConstFoldingGradOp;
class PreventConstFoldingOpx;
class PreventConstFoldingGradOpx;

// By default, const expressions ops get folded to optimise the graph and remove unnessary ops
// at the start. However, in this case, it causes the word embedding to exist in both its 
// original and transposed form. By adding this op, the constant expression folding transform
// can't fold through it, so we prevent folding after this point.

class PreventConstFoldingOp : public popart::Op
{
public:
    PreventConstFoldingOp(const popart::OperatorIdentifier &_opid, const Op::Settings &settings_)
        : Op(_opid, settings_) {}

    void setup() final { outInfo(0) = inInfo(0); }

    std::unique_ptr<Op> clone() const {
        return std::make_unique<PreventConstFoldingOp>(*this);
    }

    std::vector<std::unique_ptr<Op>> getGradOps() {
        std::vector<std::unique_ptr<Op>> upops;
        upops.emplace_back(std::make_unique<PreventConstFoldingGradOp>(*this));
        return upops;
    }

    float getSubgraphValue() const final { return getLowSubgraphValue(); }
};

static popart::OpDefinition PreventConstFoldingOpDef({});

static popart::OpCreator<PreventConstFoldingOp> PreventConstFoldingOpCreator(
    popart::OpDefinitions({{CustomOperators::PreventConstFolding, PreventConstFoldingOpDef}}),
    [](const popart::OperatorIdentifier &_opid,
       const popart::Op::Settings &settings,
       const popart::Attributes &attr) -> std::unique_ptr<popart::Op> {
        return std::unique_ptr<PreventConstFoldingOp>(new PreventConstFoldingOp(_opid, settings));
    },
    true);

class PreventConstFoldingOpx : public popart::popx::Opx {
public:
    PreventConstFoldingOpx(popart::Op *op, popart::popx::Devicex *devicex) : popart::popx::Opx(op, devicex)
    { verifyOp<PreventConstFoldingOp>(op, CustomOperators::PreventConstFolding); }

    popart::popx::InputCreatorType getInputCreatorType(popart::InIndex) const {
        return popart::popx::InputCreatorType::CanUnwind;
    }

    poplar::Tensor unwindTensorLayout(poplar::Tensor tensor, popart::InIndex, popart::OutIndex) const {
        return tensor;
    }

    popart::view::RegMap unwindRegion(popart::InIndex, popart::OutIndex) const {
        return [this](const popart::view::Region &r) {
            return popart::view::Regions(1, r);
        };
    }

    void grow(poplar::program::Sequence &prog) const final {
        insert(outId(0), getInTensor(0));
    }
};

class PreventConstFoldingGradOp : public PreventConstFoldingOp
{
public:
    PreventConstFoldingGradOp(const PreventConstFoldingOp &fwdOp)
        : PreventConstFoldingOp(CustomGradOperators::PreventConstFoldingGrad, fwdOp.getSettings()) {}

    PreventConstFoldingGradOp(const popart::Op::Settings &settings)
        : PreventConstFoldingOp(CustomGradOperators::PreventConstFoldingGrad, settings) {}

    std::unique_ptr<popart::Op> clone() const final {
        return std::make_unique<PreventConstFoldingGradOp>(*this);
    }

    const std::vector<popart::GradInOutMapper> &gradInputInfo() const {
        static const std::vector<popart::GradInOutMapper> inInfo = {
            {0, 0, popart::GradOpInType::GradOut}};

        return inInfo;
    }
    const std::map<int, int> &gradOutToNonGradIn() const {
        static const std::map<int, int> outInfo = {{0, 0}};
        return outInfo;
    }
};

class PreventConstFoldingGradOpx : public popart::popx::Opx {
public:
  PreventConstFoldingGradOpx(popart::Op *op, popart::popx::Devicex *devicex)
      : popart::popx::Opx(op, devicex) {
    verifyOp<PreventConstFoldingGradOp>(op, CustomGradOperators::PreventConstFoldingGrad);
  }

  void grow(poplar::program::Sequence &prog) const final {
      setOutTensor(0, getInTensor(0));
  }
};

static popart::popx::OpxCreator<PreventConstFoldingOpx>
    preventConstFoldingOpxCreator(CustomOperators::PreventConstFolding);
static popart::popx::OpxCreator<PreventConstFoldingGradOpx>
    preventConstFoldingGradOpxCreator(CustomGradOperators::PreventConstFoldingGrad);

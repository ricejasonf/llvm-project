//===------------------------ Passes.cpp ------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <heavy/Dialect.h>
#include <heavy/Dialect/Passes.h>
#include <mlir/IR/Operation.h>

namespace heavy {
#define GEN_PASS_DEF_STRIPGLOBALBINDINGSPASS
// TODO
//#define GEN_PASS_DEF_STRIPLOCALBINDINGSPASS
#include <heavy/Dialect/HeavyPasses.h.inc>
}

using namespace heavy;

namespace {
struct StripGlobalBindingsPass
  : public heavy::impl::StripGlobalBindingsPassBase<StripGlobalBindingsPass> {
  void runOnOperation() override;
  bool isSetFound(mlir::Value OpResult);
};
}  // namespace


void StripGlobalBindingsPass::runOnOperation() {
  mlir::ModuleOp ModuleOp = getOperation();
  for (mlir::Operation& TopLevelOp : ModuleOp.getBody()->getOperations()) {
    if (auto GlobalOp = dyn_cast<heavy::GlobalOp>(&TopLevelOp)) {
      if (GlobalOp.isExternal())
        continue;
      mlir::Block& InitBlock = GlobalOp.getInitializer().front();
      auto ContOp = dyn_cast<heavy::ContOp>(InitBlock.back());
      if (!ContOp || ContOp.getArgs().empty())
        continue;
      if (!isa<heavy::HeavyBindingType>(ContOp.getArgs().front().getType()))
        continue;

      bool IsSetFound = false;
      ModuleOp->walk([&](heavy::LoadGlobalOp LG) -> mlir::WalkResult {
          if (LG.getName() == GlobalOp.getSymName() &&
              isSetFound(LG.getResult()))
            IsSetFound = true;
          return mlir::WalkResult::advance();
        });

      // Find any SetOp that uses the result of said LoadGlobalOp.
      if (!IsSetFound)
        llvm::errs() << "Found global binding to be stripped: "
                     << GlobalOp.getSymName() << "\n";
    }
  }
}

// Recursively chase operation results to see if we find a "use" by a SetOp.
bool StripGlobalBindingsPass::isSetFound(mlir::Value OpResult) {
  // Because globals can be loaded anywhere, any SetOp requires
  // the variable to exist as a binding.
  for (mlir::OpOperand& Use : OpResult.getUses()) {
    if (isa<heavy::SetOp>(Use.getOwner()))
      return true;
    if (mlir::Value Capture = getCapture(Use.getOwner())) {
      if (isSetFound(Capture))
        return true;
    }
  }
  return false;
}

// If Operand is being captured, get the result of the
// LoadRefOp that loads the captured Operand.
// Otherwise, return mlir::Value().
mlir::Value HeavyModulePass::getCapture(mlir::OpOperand Operand) {
  mlir::Operation* Op = Operand.getOwner();
  unsigned Index = Operand.getOperandNumber();

  heavy::FuncOp FuncOp = lookupCapturingFunction(Op);
  if (!FuncOp)  // Not a capture
    return mlir::Value();

  // Find the LoadRefOp with the corresponding Index.
  for (mlir::Operation& Op : FuncOp.getBody().front()) {
    if (auto LoadRefOp = dyn_cast<heavy::LoadRefOp>(&Op)) {
      if (LoadRefOp.getIndex() == Index)
        return LoadRefOp.getResult();
    }
  }

  Op->emitOpError("capture operation does not have "
                  "corresponding heavy.load_ref");
  return mlir::Value();
}

// If Op is a capturing operation, then get the function name
// and perform lookup in the current ModuleOp.
heavy::FuncOp HeavyModulePass::lookupCapturingFunction(mlir::Operation* Op) {
  llvm::StringRef FuncName;
  if (auto PC = dyn_cast<heavy::PushContOp>(Op))
    FuncName = PC.getName();
  else if (auto L = dyn_cast<heavy::LambdaOp>(Op))
    FuncName = L.getName();

  if (FuncName.empty())  // Not a capture
    return heavy::FuncOp();

  mlir::ModuleOp M = getOperation();
  mlir::Operation* FoundOp = M.lookupSymbol(FuncName);
  auto FuncOp = llvm::dyn_cast_if_present<heavy::FuncOp>(FoundOp);
  if (!FuncOp || FuncOp.getBody().empty()) {
    Op->emitOpError("operation refers to capturing function "
                    "with no definition in the current module");
    return heavy::FuncOp();
  }
  return FuncOp;
}

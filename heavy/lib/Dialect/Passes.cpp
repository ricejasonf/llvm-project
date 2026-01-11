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

// Strip any heavy.unbox whose input is
// no longer possibly a binding.
// (which is invalid.)
void stripUnboxes(mlir::ModuleOp Op) {
  Op->walk([](heavy::UnboxOp UnboxOp) {
      mlir::Value Input = UnboxOp.getBinding();
      if (!isa<HeavyBindingType, HeavyUnknownType>(Input.getType())) {
        UnboxOp.getResult().replaceAllUsesWith(Input);
        UnboxOp.erase();
      }
      return mlir::WalkResult::advance();
    });
}

// Strip heavy.match_type that maps to the same type.
void stripMatchTypes(mlir::ModuleOp Op) {
  Op->walk([](heavy::MatchTypeOp MatchTypeOp) {
      mlir::Value Input = MatchTypeOp.getArg();
      if (Input.getType() == MatchTypeOp.getResult().getType()) {
        MatchTypeOp.getResult().replaceAllUsesWith(Input);
        MatchTypeOp.erase();
      }
      return mlir::WalkResult::advance();
    });
}
}  // namespace


void StripGlobalBindingsPass::runOnOperation() {
  mlir::ModuleOp ModuleOp = getOperation();
  for (mlir::Operation& TopLevelOp : ModuleOp.getBody()->getOperations()) {
    if (auto GlobalOp = dyn_cast<heavy::GlobalOp>(&TopLevelOp)) {
      mlir::Type Type = GlobalOp.getType();
      if (GlobalOp.isExternal() ||
          !isa<heavy::HeavyBindingType>(Type))
        continue;

      bool IsSetFound = false;
      ModuleOp->walk([&](heavy::LoadGlobalOp LG) -> mlir::WalkResult {
          if (LG.getName() == GlobalOp.getSymName() &&
              isSetFound(LG.getResult()))
            IsSetFound = true;
          return mlir::WalkResult::advance();
        });

      // Find any SetOp that uses the result of said LoadGlobalOp.
      if (IsSetFound)
        continue;

      // Strip the binding.
      GlobalOp->walk([&](heavy::InitGlobalOp IG) -> mlir::WalkResult {
          if (auto B = IG.getVar().getDefiningOp<heavy::BindingOp>()) {
            Type = B.getInput().getType();
            IG.setOperand(B.getInput());
            B.erase();
          } else {
            IG->emitOpError("heavy.init_global input type did not match "
                            "annotated type of heavy.global_op");
          }
          // There should be only one InitGlobalOp.
          return mlir::WalkResult::interrupt();
        });

      // Update the GlobalOp annotated type.
      GlobalOp.setType(Type);

      // Update the type of the heavy.load_globals.
      ModuleOp->walk([&](heavy::LoadGlobalOp LG) -> mlir::WalkResult {
          if (LG.getName() == GlobalOp.getSymName()) {
            LG.getResult().setType(Type);
            // Update any captures. (Yes, capturing globals is useful.)
            updateCaptureTypes(LG.getResult());
          }
          return mlir::WalkResult::advance();
        });

      stripUnboxes(ModuleOp);
      stripMatchTypes(ModuleOp);
    }
  }
}

// Recursively chase operation results to see if we find a "use" by a SetOp.
bool StripGlobalBindingsPass::isSetFound(mlir::Value OpResult) {
  assert(OpResult);
  // Because globals can be loaded anywhere, any SetOp requires
  // the variable to exist as a binding.
  for (mlir::OpOperand& Use : OpResult.getUses()) {
    assert(Use.getOwner());
    if (isa<heavy::SetOp>(Use.getOwner()))
      return true;
    if (mlir::Value Capture = getCapture(Use)) {
      if (isSetFound(Capture))
        return true;
    }
  }
  return false;
}

// If Operand is being captured, get the result of the
// LoadRefOp that loads the captured Operand.
// Otherwise, return mlir::Value().
mlir::Value HeavyModulePass::getCapture(mlir::OpOperand& Operand) {
  assert(Operand.get());
  mlir::Operation* Op = Operand.getOwner();
  unsigned Index = Operand.getOperandNumber();

  heavy::FuncOp FuncOp = lookupCapturingFunction(Op);
  if (!FuncOp)  // Not a capture
    return mlir::Value();

  // Find the LoadRefOp with the corresponding Index.
  mlir::Value Result;
  FuncOp->walk([&](heavy::LoadRefOp LoadRefOp) {
      if (LoadRefOp.getIndex() == Index) {
        Result = LoadRefOp.getResult();
        return mlir::WalkResult::interrupt();
      }
      return mlir::WalkResult::advance();
    });

  if (!Result)
    Op->emitOpError("capture operand #") << Index <<
                    "does not have corresponding heavy.load_ref";
  return Result;
}

// Recursively, chase captures of a value updating the type
// the corresponding heavy.load_refs.
void HeavyModulePass::updateCaptureTypes(mlir::Value Value) {
  for (mlir::OpOperand& Use : Value.getUses()) {
    assert(Use.getOwner());
    if (mlir::Value Capture = getCapture(Use)) {
      Capture.setType(Value.getType());
      updateCaptureTypes(Capture);
    }
  }
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

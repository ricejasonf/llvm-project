//===------------------------ Passes.h --------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef HEAVY_DIALECT_PASSES_H
#define HEAVY_DIALECT_PASSES_H

#include <heavy/Dialect.h>
#include <mlir/Pass/Pass.h>

namespace heavy {
// Base class for passes need to chase continuation
// and lambda functions within a module.
class HeavyModulePass : public mlir::OperationPass<mlir::ModuleOp> {
  using Base = mlir::OperationPass<mlir::ModuleOp>;

public:
  using Base::OperationPass;
  mlir::Value getCapture(mlir::OpOperand& Operand);
  heavy::FuncOp lookupCapturingFunction(mlir::Operation* Op);
  void updateCaptureTypes(mlir::Value);
};
}

// Generated stuff
namespace heavy {
#define GEN_PASS_DECL
#include "heavy/Dialect/HeavyPasses.h.inc"

#define GEN_PASS_REGISTRATION
#include "heavy/Dialect/HeavyPasses.h.inc"
}  // namespace heavy

#endif  // HEAVY_DIALECT_PASSES_H

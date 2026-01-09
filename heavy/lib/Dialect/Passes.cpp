//===------------------------ Passes.cpp ------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <heavy/Value.h>
#include <heavy/Dialect.h>
#include <heavy/Dialect/Passes.h>

namespace heavy {
#define GEN_PASS_DEF_STRIPGLOBALBINDINGSPASS
// TODO
//#define GEN_PASS_DEF_STRIPLOCALBINDINGSPASS
#include <heavy/Dialect/HeavyPasses.h.inc>
}

namespace {
struct StripGlobalBindingsPass
  : public heavy::impl::StripGlobalBindingsPassBase<StripGlobalBindingsPass> {
  void runOnOperation() override;
};
}  // namespace


void StripGlobalBindingsPass::runOnOperation() {
  mlir::ModuleOp ModuleOp = getOperation();
  for (mlir::Operation& TopLevelOp : ModuleOp.getBody()->getOperations()) {
    if (auto GlobalOp = heavy::dyn_cast<heavy::GlobalOp>(&TopLevelOp)) {
      llvm::errs() << "Found operation: " << GlobalOp.getSymName() << "\n";
    }
  }
}

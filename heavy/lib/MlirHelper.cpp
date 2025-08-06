//===------------ MlirHelper.cpp - MlirHelper stuff -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  Defines helper functions for mlir scheme bindings.
//
//===----------------------------------------------------------------------===//

#include <heavy/Context.h>
#include <heavy/Mlir.h>
#include <heavy/MlirHelper.h>
#include <heavy/Value.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/Operation.h>

namespace heavy::mlir_helper {
mlir::MLIRContext* getCurrentContext(heavy::Context& C) {
  return any_cast<mlir::MLIRContext*>(
      mlir_bind_var::current_context.get(C));
}

mlir::OpBuilder* getBuilder(heavy::Context& C, heavy::Value V) {
  if (auto* OpBuilder = any_cast<mlir::OpBuilder>(&V))
    return OpBuilder;

  C.RaiseError("current-builder is invalid");
  return nullptr;
}

mlir::OpBuilder* getCurrentBuilder(heavy::Context& C) {
  heavy::Value V = mlir_bind_var::current_builder.get(C);
  return getBuilder(C, V);
}

mlir::Operation* getSingleOpArg(heavy::Context& C, heavy::ValueRefs Args) {
  mlir::Operation* Op = nullptr;
  if (Args.size() == 1)
    Op = heavy::dyn_cast<mlir::Operation>(Args[0]);
  if (!Op)
    C.RaiseError("expecting mlir operation");
  return Op;
}

void with_builder_impl(Context& C, mlir::OpBuilder const& Builder,
                       heavy::Value Thunk) {
  heavy::Value PrevBuilder = C.CreateBinding(heavy::Empty());
  heavy::Value NewBuilder = C.CreateAny(Builder);

  heavy::Value Before = C.CreateLambda(
    [](heavy::Context& C, heavy::ValueRefs Args) {
      // Save the previous state and instate the new... state.
      // (ie Builder)
      auto* PrevBuilder = cast<heavy::Binding>(C.getCapture(0));
      heavy::Value NewBuilder = C.getCapture(1);

      // Set the Binding.
      PrevBuilder->setValue(mlir_bind_var::current_builder.get(C));

      // Set the "current-builder" value.
      mlir_bind_var::current_builder.set(C, NewBuilder);
      C.Cont();
    }, CaptureList{PrevBuilder, NewBuilder});

  heavy::Value After = C.CreateLambda(
    [](heavy::Context& C, heavy::ValueRefs Args) {
      // Restore previous state
      auto* PrevBuilder = cast<heavy::Binding>(C.getCapture(0));
      mlir_bind_var::current_builder.set(C, PrevBuilder->getValue());
      C.Cont();
    }, CaptureList{PrevBuilder});

  C.DynamicWind(Before, Thunk, After);
}
}  // end namespace heavy::mlir_helper

//===---- MlirHelper.h - Mlir binding helper functions ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file declares helper functions for binding mlir objects in heavy scheme.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_HEAVY_MLIR_HELPER_H
#define LLVM_HEAVY_MLIR_HELPER_H

#include <heavy/Context.h>
#include <heavy/Mlir.h>
#include <heavy/Value.h>
#include <mlir/IR/Value.h>
#include <llvm/ADT/StringRef.h>

namespace mlir {
  class OpBuilder;
  class Operation;
}

namespace heavy::mlir_bind_var {
extern heavy::ContextLocal current_context;
extern heavy::ContextLocal current_builder;
}

namespace heavy::mlir_helper {
  namespace kind {
    // Omit mlir.op since mlir::Operation* is already embedded in heavy::Value.
    constexpr char const* mlir_context  = "mlir.context";
    constexpr char const* mlir_type     = "mlir.type";
    constexpr char const* mlir_attr     = "mlir.attr";
    constexpr char const* mlir_region   = "mlir.region";
    constexpr char const* mlir_block    = "mlir.block";
    constexpr char const* mlir_value    = "mlir.value";
    constexpr char const* mlir_builder  = "mlir.builder"; // OpBuilder
  }

// Create OpaquePtr tagged with a string for mlir objects.
template <typename T>
heavy::Value createTagged(heavy::Context& C, llvm::StringRef Kind, T Obj) {
  return C.CreateTagged(C.CreateSymbol(Kind), Obj);
}

// Get mlir Type/Attribute from tagged OpaquePtr.
template <typename T>
T getTagged(heavy::Context& C, llvm::StringRef Kind, heavy::Value Value) {
  if (auto* Tagged = heavy::dyn_cast<heavy::Tagged>(Value)) {
    heavy::Symbol* KindSym = C.CreateSymbol(Kind);
    if (Tagged->isa(KindSym))
      return Tagged->cast<T>();
  }

  return T(nullptr);
}

mlir::MLIRContext* getCurrentContext(heavy::Context& C);
mlir::OpBuilder* getBuilder(heavy::Context& C, heavy::Value V);
mlir::OpBuilder* getCurrentBuilder(heavy::Context& C);
mlir::Operation* getSingleOpArg(heavy::Context& C, heavy::ValueRefs Args);
void with_builder_impl(Context& C, mlir::OpBuilder const& Builder,
                       heavy::Value Thunk);

}  // end namespace heavy::mlir_helper


#endif

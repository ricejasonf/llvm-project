//===--- Mlir.cpp - Mlir binding syntax for HeavyScheme ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file defines syntax mlir bindings for HeavyScheme.
//
//===----------------------------------------------------------------------===//

#include "heavy/Mlir.h"
#include "heavy/Context.h"
#include "heavy/OpGen.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Casting.h"
#include "memory"

bool HEAVY_MLIR_IS_LOADED = false;

heavy::ExternSyntax<> HEAVY_MLIR_VAR(create_op);
heavy::ExternSyntax<> HEAVY_MLIR_VAR(region_op);
heavy::ExternSyntax<> HEAVY_MLIR_VAR(results);
heavy::ExternSyntax<> HEAVY_MLIR_VAR(result);
heavy::ExternSyntax<> HEAVY_MLIR_VAR(block_begin);
heavy::ExternSyntax<> HEAVY_MLIR_VAR(block_end);
heavy::ExternSyntax<> HEAVY_MLIR_VAR(block_ops);
heavy::ExternSyntax<> HEAVY_MLIR_VAR(insert_before);
heavy::ExternSyntax<> HEAVY_MLIR_VAR(insert_after);
heavy::ExternSyntax<> HEAVY_MLIR_VAR(with_insertion_point);
heavy::ExternSyntax<> HEAVY_MLIR_VAR(type);
heavy::ExternSyntax<> HEAVY_MLIR_VAR(attr);


// Create operation. Argument are lists values or attributes or result type
//  specified by auxilary keywords: arguments, attributes, result-types.
void create_op(Context& C, ValueRefs Args) {
  mlir::MlirContext& MlirContext = C.MlirContext.get();
  C.RaiseError("TODO not implemented");
}

// Get an operation region by name.
void region(Context& C, ValueRefs Args) {
  C.RaiseError("TODO not implemented");
}

// Get list of blocks in a region.
void region_blocks(Context& C, ValueRefs Args) {
  C.RaiseError("TODO not implemented");
}

// Get list of results of op.
void results(Context& C, ValueRefs Args) {
  C.RaiseError("TODO not implemented");
}

// Get operation result by index (default = 0).
void result(Context& C, ValueRefs Args) {
  C.RaiseError("TODO not implemented");
}

// Get insertion point to prepend current block.
void block_begin(Context& C, ValueRefs Args) {
  C.RaiseError("TODO not implemented");
}

// Get insertion point to append current block.
void block_end(Context& C, ValueRefs Args) {
  C.RaiseError("TODO not implemented");
}

// Get argument from block by index.
void block_arg(Context& C, ValueRefs Args) {
  C.RaiseError("TODO not implemented");
}

// Get list of operations from current block.
void block_ops(Context& C, ValueRefs Args) {
  C.RaiseError("TODO not implemented");
}

// Get insertion point to prepend.
// Argument can be Operation, Region, or Block.
//  Operation - inserts before operation in containing block.
//  Region    - inserts before operation in first block.
//  Block     - inserts before operation in block.
void insert_before(Context& C, ValueRefs Args) {
  C.RaiseError("TODO not implemented");
}

// Get insertion point to append similar to insert_before.
void insert_after(Context& C, ValueRefs Args) {
  C.RaiseError("TODO not implemented");
}

// Dynamic wind with an insertion point.
void with_insertion_point(Context& C, ValueRefs Args) {
  C.RaiseError("TODO not implemented");
}

// Get a type by parsing a string.
void type(Context& C, ValueRefs Args) {
  mlir::MlirContext& MlirContext = C.MlirContext.get();
  auto* String = dyn_cast<heavy::String>(Args[0]);
  if (!String)
    return C.RaiseError("expecting string");
  llvm::StringRef TypeStr = String->getView();

  mlir::Type Type = mlir::parseType(TypeStr, MlirContext
                                    TypeStr.size(),
                                    /*isKnownNullTerminated*/true);
  if (!Type)
    return C.RaiseError("mlir type parse failed");

  C.Cont(C.CreatePair(C.CreateSymbol("mlir.type"),
                      C.CreateOpaquePtr(Type.getImpl());
}

// Get an attribute by parsing a string.
void attr(Context& C, ValueRefs Args) {
  mlir::MlirContext& MlirContext = C.MlirContext.get();
  auto* String = dyn_cast<heavy::String>(Args[0]);
  if (!String)
    return C.RaiseError("expecting string");
  llvm::StringRef AttrStr = String->getView();

  mlir::Attr Attr = mlir::parseAttr(AttrStr, MlirContext
                                    AttrStr.size(),
                                    /*isKnownNullTerminated*/true);
  if (!Attr)
    return C.RaiseError("mlir attribute parse failed");

  C.Cont(C.CreatePair(C.CreateSymbol("mlir.attr"),
                      C.CreateOpaquePtr(Attr.getImpl());
}

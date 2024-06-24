//===- Mlir.h - Mlir binding functions for HeavyScheme ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file declares values and functions for the (heavy mlir) scheme library
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_HEAVY_MLIR_H
#define LLVM_HEAVY_MLIR_H

#include "heavy/Context.h"
#include "heavy/Value.h"
#include "llvm/ADT/SmallVector.h"

#define HEAVY_MLIR_LIB                _HEAVYL5SheavyL4Smlir
#define HEAVY_MLIR_LIB_(NAME)         _HEAVYL5SheavyL4Smlir ## NAME
#define HEAVY_MLIR_LIB_STR            "_HEAVYL5SheavyL4Smlir"
#define HEAVY_MLIR_LOAD_MODULE        HEAVY_MLIR_LIB_(_load_module)
#define HEAVY_MLIR_INIT               HEAVY_MLIR_LIB_(_init)

#define HEAVY_MLIR_VAR(NAME)          HEAVY_MLIR_VAR__##NAME
#define HEAVY_MLIR_VAR__export        HEAVY_MLIR_LIB_(V6Sexport)

namespace heavy {

class Context;
class Value;
class OpGen;
class OpEval;
class Pair;
using ValueRefs = llvm::MutableArrayRef<heavy::Value>;

}

namespace heavy::mlir {
// syntax (top level, continuable)

// TODO Support creating custom MLIR context. Currently
//      we default to the current context that scheme is
//      compiling under.

// Create operation. Argument are lists values or attributes or result type
//  specified by auxilary keywords: arguments, attributes, result-types.
void create_op(Context& C, ValueRefs Args);

// Get an operation region by name.
void region(Context& C, ValueRefs Args);

// Get list of blocks in a region.
void region_blocks(Context& C, ValueRefs Args);

// Get list of results of op.
void results(Context& C, ValueRefs Args);

// Get operation result by index (default = 0).
void result(Context& C, ValueRefs Args);

// Get insertion point to prepend current block.
void block_begin(Context& C, ValueRefs Args);

// Get insertion point to append current block.
void block_end(Context& C, ValueRefs Args);

// Get argument from block by index.
void block_arg(Context& C, ValueRefs Args);

// Get list of operations from current block.
void block_ops(Context& C, ValueRefs Args);

// Get insertion point to prepend.
// Argument can be Operation, Region, or Block.
//  Operation - inserts before operation in containing block.
//  Region    - inserts before operation in first block.
//  Block     - inserts before operation in block.
void insert_before(Context& C, ValueRefs Args);

// Get insertion point to append similar to insert_before.
void insert_after(Context& C, ValueRefs Args);

// Dynamic wind with an insertion point.
void with_insertion_point(Context& C, ValueRefs Args);

// Get a type by parsing a string.
void type(Context& C, ValueRefs Args);

// Get an attribute by parsing a string.
void attr(Context& C, ValueRefs Args);
}
/*
create_op
region_op
results
result
block_begin
block_end
block_ops
insert_before
insert_after
with_insertion_point
type
attr
*/

extern heavy::ExternSyntax<> HEAVY_BASE_VAR(create_op);
extern heavy::ExternSyntax<> HEAVY_BASE_VAR(region);
extern heavy::ExternSyntax<> HEAVY_BASE_VAR(region_blocks);
extern heavy::ExternSyntax<> HEAVY_BASE_VAR(results);
extern heavy::ExternSyntax<> HEAVY_BASE_VAR(result);
extern heavy::ExternSyntax<> HEAVY_BASE_VAR(block_begin);
extern heavy::ExternSyntax<> HEAVY_BASE_VAR(block_end);
extern heavy::ExternSyntax<> HEAVY_BASE_VAR(block_arg);
extern heavy::ExternSyntax<> HEAVY_BASE_VAR(block_ops);
extern heavy::ExternSyntax<> HEAVY_BASE_VAR(insert_before);
extern heavy::ExternSyntax<> HEAVY_BASE_VAR(insert_after);
extern heavy::ExternSyntax<> HEAVY_BASE_VAR(with_insertion_point);
extern heavy::ExternSyntax<> HEAVY_BASE_VAR(type);
extern heavy::ExternSyntax<> HEAVY_BASE_VAR(attr);

extern "C" {
// initialize the module for run-time independent of the compiler
inline void HEAVY_BASE_INIT(heavy::Context& Context) {
  // syntax
  HEAVY_BASE_VAR(create_op) = heavy::base::create_op;
    HEAVY_MLIR_VAR(region_op) = heavy::mlir::region_op;
    HEAVY_MLIR_VAR(results) = heavy::mlir::results;
    HEAVY_MLIR_VAR(result) = heavy::mlir::result;
    HEAVY_MLIR_VAR(block_begin) = heavy::mlir::block_begin;
    HEAVY_MLIR_VAR(block_end) = heavy::mlir::block_end;
    HEAVY_MLIR_VAR(block_ops) = heavy::mlir::block_ops;
    HEAVY_MLIR_VAR(insert_before) = heavy::mlir::insert_before;
    HEAVY_MLIR_VAR(insert_after) = heavy::mlir::insert_after;
    HEAVY_MLIR_VAR(with_insertion_point) = heavy::mlir::with_insertion_point;
    HEAVY_MLIR_VAR(type) = heavy::mlir::type;
    HEAVY_MLIR_VAR(attr) = heavy::mlir::attr;
}

inline void HEAVY_BASE_LOAD_MODULE(heavy::Context& Context) {
  HEAVY_BASE_INIT(Context);
  heavy::initModule(Context, HEAVY_BASE_LIB_STR, {
    {"create_op", HEAVY_MLIR_VAR(create_op)},
    {"region_op", HEAVY_MLIR_VAR(region_op)},
    {"results", HEAVY_MLIR_VAR(results)},
    {"result", HEAVY_MLIR_VAR(result)},
    {"block_begin", HEAVY_MLIR_VAR(block_begin)},
    {"block_end", HEAVY_MLIR_VAR(block_end)},
    {"block_ops", HEAVY_MLIR_VAR(block_ops)},
    {"insert_before", HEAVY_MLIR_VAR(insert_before)},
    {"insert_after", HEAVY_MLIR_VAR(insert_after)},
    {"with_insertion_point", HEAVY_MLIR_VAR(with_insertion_point)},
    {"type", HEAVY_MLIR_VAR(type)},
    {"attr", HEAVY_MLIR_VAR(attr)}
  });

#endif  // LLVM_HEAVY_MLIR_H

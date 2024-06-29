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

#include <heavy/Context.h>
#include <heavy/Mlir.h>
#include <heavy/OpGen.h>
#include <heavy/Value.h>
#include <mlir/AsmParser/AsmParser.h>
#include <llvm/ADT/StringExtras.h>
#include <llvm/Support/Casting.h>
#include <memory>

heavy::ContextLocal   HEAVY_MLIR_VAR(current_mlir_context);
heavy::ContextLocal   HEAVY_MLIR_VAR(current_mlir_builder);
heavy::ExternSyntax<> HEAVY_MLIR_VAR(create_op);
heavy::ExternSyntax<> HEAVY_MLIR_VAR(region_op);
heavy::ExternSyntax<> HEAVY_MLIR_VAR(results);
heavy::ExternSyntax<> HEAVY_MLIR_VAR(result);
heavy::ExternSyntax<> HEAVY_MLIR_VAR(at_block_begin);
heavy::ExternSyntax<> HEAVY_MLIR_VAR(at_block_end);
heavy::ExternSyntax<> HEAVY_MLIR_VAR(at_block_terminator);
heavy::ExternSyntax<> HEAVY_MLIR_VAR(insert_before);
heavy::ExternSyntax<> HEAVY_MLIR_VAR(insert_after);
heavy::ExternSyntax<> HEAVY_MLIR_VAR(with_insertion_before);
heavy::ExternSyntax<> HEAVY_MLIR_VAR(with_insertion_after);
heavy::ExternSyntax<> HEAVY_MLIR_VAR(type);
heavy::ExternSyntax<> HEAVY_MLIR_VAR(attr);

// TODO load_dialect

namespace {
  namespace kind {
    // Omit mlir.op since mlir::Operation* is already embedded in heavy::Value.
    constexpr char const* mlir_type     = "mlir.type";
    constexpr char const* mlir_attr     = "mlir.attr";
    constexpr char const* mlir_region   = "mlir.region";
    constexpr char const* mlir_block    = "mlir.block";
    constexpr char const* mlir_value    = "mlir.value";
    constexpr char const* mlir_builder  = "mlir.builder";
  }

  // Create OpaquePtr tagged with a string for mlir objects.
  template <typename Context, typename T>
  heavy::Value CreateTagged(Context& C, llvm::StringRef Kind, T Obj) {
    assert(bool(Obj) && "disallow nullptr for tagged mlir objects");
    return C.CreateTagged(C.CreateSymbol(Kind), Obj);
  }

  // Get mlir Type/Attribute from tagged OpaquePtr.
  template <typename T>
  T GetTagged(heavy::Context& C, llvm::StringRef Kind, heavy::Value Value) {
    if (auto* Tagged = heavy::dyn_cast<heavy::Tagged>(Value)) {
      heavy::Symbol* KindSym = C.CreateSymbol(Kind);  
      if (Tagged->isa(KindSym))
        return Tagged->cast<T>();
    }

    return T(nullptr);
  }
}  // namespace

namespace heavy::mlir_bind {
// Create operation. Argument are lists values or attributes or result type
//  specified by auxilary keywords: arguments, attributes, result-types.
void create_op(Context& C, ValueRefs Args) {
  //mlir::MLIRContext* MLIRContext = C.MLIRContext.get();
  C.RaiseError("TODO not implemented");
}

// Get an operation region by index (defaulting to 0).
// Usage: (region op) or (region op index)
void region(Context& C, ValueRefs Args) {
  if (Args.size() != 1 && Args.size() != 2)
    return C.RaiseError("invalid arity");

  mlir::Operation* Op = heavy::dyn_cast<mlir::Operation>(Args[1]);
  if (!Op)
    return C.RaiseError("expecting mlir.op");

  if (Args.size() > 1 && !heavy::isa<heavy::Int>(Args[1]))
    return C.RaiseError("expecting index");

  int32_t Index = heavy::isa<heavy::Int>(Args[1]) ?
                    int32_t{heavy::cast<heavy::Int>(Args[1])} : 0;
  // Regions are part of the Ops TrailingObjects so
  // we can expect the pointers to be stable.
  mlir::Region* Region = &(Op->getRegion(Index));
  if (!Region)
    return C.RaiseError("invalid mlir.region");
  C.Cont(CreateTagged(C, kind::mlir_region, Region));
}

// Get entry block from region/op by index.
// If an op is provided the first region is used.
void entry_block(Context& C, ValueRefs Args) {
  if (Args.size() != 1)
    return C.RaiseError("invalid arity");

  mlir::Region* Region = nullptr;
  if (mlir::Operation* Op = dyn_cast<mlir::Operation>(Args[0])) {
    if (Op->getNumRegions() < 1)
      return C.RaiseError("mlir.op has no regions");
    Region = &Op->getRegion(0);
  } else {
    Region = GetTagged<mlir::Region*>(C, kind::mlir_region, Args[0]);
  }

  if (!Region)
    return C.RaiseError("expecting mlir.op/mlir.region");
  if (Region->empty())
      return C.RaiseError("mlir.region has no entry block");
  mlir::Block* Block = &(Region->front());
  if (!Block)
    return C.RaiseError("invalid mlir.block");
  C.Cont(CreateTagged(C, kind::mlir_block, Block));
}

// Get list of results of op.
void results(Context& C, ValueRefs Args) {
  // This might be useful for applying to operations
  // via quasiquote splicing.
  C.RaiseError("TODO not implemented");
}

// Get operation result by index (default = 0).
void result(Context& C, ValueRefs Args) {
  if (Args.size() != 1 && Args.size() != 2)
    return C.RaiseError("invalid arity");

  mlir::Operation* Op = heavy::dyn_cast<mlir::Operation>(Args[1]);
  if (!Op)
    return C.RaiseError("expecting mlir.op");

  if (Args.size() > 1 && !heavy::isa<heavy::Int>(Args[1]))
    return C.RaiseError("expecting index");

  int32_t Index = heavy::isa<heavy::Int>(Args[1]) ?
                    int32_t{heavy::cast<heavy::Int>(Args[1])} : 0;
  mlir::Value Result = Op->getResult(Index);
  if (!Result)
    return C.RaiseError("invalid mlir.op result");
  C.Cont(CreateTagged(C, kind::mlir_value, Result));
}

void with_builder(Context& C, ValueRefs Args) {
  C.RaiseError("TODO not implemented");
}

void at_block_begin(Context& C, ValueRefs Args) {
  C.RaiseError("TODO not implemented");
}

void at_block_end(Context& C, ValueRefs Args) {
  C.RaiseError("TODO not implemented");
}

void at_block_terminator(Context& C, ValueRefs Args) {
  C.RaiseError("TODO not implemented");
}

// Get insertion point to prepend.
// Argument can be Operation, Region, or Block.
//  Operation - inserts before operation in containing block.
//  Region    - inserts before operation in first block.
//  Block     - inserts before operation in block.
void with_insertion_before(Context& C, ValueRefs Args) {
  C.RaiseError("TODO not implemented");
}

// Get insertion point to append similar to insert_before.
void with_insertion_after(Context& C, ValueRefs Args) {
  C.RaiseError("TODO not implemented");
}

// Get a type by parsing a string.
void type(Context& C, ValueRefs Args) {
  mlir::MLIRContext* MLIRContext = C.MLIRContext.get();
  llvm::StringRef TypeStr = Args[0].getStringRef();
  if (TypeStr.empty())
    return C.RaiseError("expecting string");

  mlir::Type Type = mlir::parseType(TypeStr, MLIRContext,
                                    nullptr, heavy::String::IsNullTerminated);
  if (!Type)
    return C.RaiseError("mlir type parse failed");

  C.Cont(CreateTagged(C, kind::mlir_type, Type.getImpl()));
}

// Get an attribute by parsing a string.
//  Usage: (attr type attr_str)
//    type - a string or a mlir.type object
//    attr_str - the string to be parsed
void attr(Context& C, ValueRefs Args) {
  mlir::MLIRContext* MLIRContext = C.MLIRContext.get();
  if (Args.size() != 2)
    return C.RaiseError("invalid arity");

  mlir::Type Type;
  llvm::StringRef TypeStr = Args[1].getStringRef();
  if (!TypeStr.empty()) {
    Type = mlir::parseType(TypeStr, MLIRContext, nullptr,
                           heavy::String::IsNullTerminated);
    if (!Type)
      return C.RaiseError("mlir type parse failed");
  }
  else {
    Type = GetTagged<mlir::Type>(C, kind::mlir_type, Args[1]);
    if (!Type)
      return C.RaiseError("invalid mlir type");
  }

  llvm::StringRef AttrStr = Args[1].getStringRef();
  if (AttrStr.empty())
    return C.RaiseError("expecting string");

  mlir::Attribute Attr = mlir::parseAttribute(AttrStr, MLIRContext,
                                              Type, nullptr,
                                              heavy::String::IsNullTerminated);
  if (!Attr)
    return C.RaiseError("mlir attribute parse failed");

  C.Cont(CreateTagged(C, kind::mlir_attr, Attr.getImpl()));
}

}  // namespace heavy::mlir_bind

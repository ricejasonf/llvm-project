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
#include <mlir/IR/Verifier.h>
#include <mlir/Support/LogicalResult.h>
#include <llvm/ADT/StringExtras.h>
#include <llvm/Support/Casting.h>
#include <memory>

heavy::ContextLocal   HEAVY_MLIR_VAR(current_context);
heavy::ContextLocal   HEAVY_MLIR_VAR(current_builder);
heavy::ExternSyntax<> HEAVY_MLIR_VAR(create_op);
heavy::ExternSyntax<> HEAVY_MLIR_VAR(region);
heavy::ExternSyntax<> HEAVY_MLIR_VAR(results);
heavy::ExternSyntax<> HEAVY_MLIR_VAR(result);
heavy::ExternSyntax<> HEAVY_MLIR_VAR(at_block_begin);
heavy::ExternSyntax<> HEAVY_MLIR_VAR(at_block_end);
heavy::ExternSyntax<> HEAVY_MLIR_VAR(at_block_terminator);
heavy::ExternSyntax<> HEAVY_MLIR_VAR(block_op);
heavy::ExternSyntax<> HEAVY_MLIR_VAR(set_insertion_point);
heavy::ExternSyntax<> HEAVY_MLIR_VAR(set_insertion_after);
heavy::ExternSyntax<> HEAVY_MLIR_VAR(type);
heavy::ExternSyntax<> HEAVY_MLIR_VAR(attr);
heavy::ExternSyntax<> HEAVY_MLIR_VAR(with_new_context);
heavy::ExternSyntax<> HEAVY_MLIR_VAR(with_builder);
heavy::ExternSyntax<> HEAVY_MLIR_VAR(load_dialect);
heavy::ExternSyntax<> HEAVY_MLIR_VAR(parent_op);
heavy::ExternSyntax<> HEAVY_MLIR_VAR(op_next);
heavy::ExternSyntax<> HEAVY_MLIR_VAR(dump);
heavy::ExternSyntax<> HEAVY_MLIR_VAR(verify);

namespace {
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
  heavy::Value CreateTagged(heavy::Context& C, llvm::StringRef Kind, T Obj) {
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

  mlir::MLIRContext* getCurrentContext(heavy::Context& C) {
    return GetTagged<mlir::MLIRContext*>(C, kind::mlir_context,
        HEAVY_MLIR_VAR(current_context).get(C));
  }

  mlir::OpBuilder* getBuilder(heavy::Context& C, heavy::Value V) {
    if (auto* Tagged = heavy::dyn_cast<heavy::Tagged>(V)) {
      heavy::Symbol* KindSym = C.CreateSymbol(kind::mlir_builder);
      if (Tagged->isa(KindSym))
        return &Tagged->cast<mlir::OpBuilder>();
    }

    C.RaiseError("current-builder is invalid");
    return nullptr;
  }

  mlir::OpBuilder* getCurrentBuilder(heavy::Context& C) {
    heavy::Value V = HEAVY_MLIR_VAR(current_builder).get(C);
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

}  // namespace

namespace heavy::mlir_bind {
// Create operation. Argument are
//  specified by any of the following auxilary keywords:
//    attributes - takes list of name-value pairs
//                 where the value is either mlir.attr or
//                 heavy::Value is lifted to mlir.attr.
//    operands - takes list of mlir.values
//    regions - takes a single integer for the number of regions
//    result-types - takes list of mlir.types
//    successors - takes a list of mlir.blocks
//  (create-op _name_
//    (attributes   _attrs_ ...)
//    (operands     _values_ ...)
//    (regions      _regions_ ...)
//    (result-types _types_ ...)
//    (successors   _blocks_ ...))
//
void create_op(Context& C, ValueRefs Args) {
  heavy::SourceLocation Loc = {};
  llvm::StringRef OpName = "";
  heavy::Value Attributes   = heavy::Empty();
  heavy::Value Operands     = heavy::Empty();
  heavy::Value Regions      = heavy::Empty();
  heavy::Value ResultTypes  = heavy::Empty();
  heavy::Value Successors   = heavy::Empty();

  if (!Args.empty() && isa<SourceValue>(Args[0])) {
    Loc = Args[0].getSourceLocation();
    Args = Args.drop_front();
  }

  if (Args.empty())
    return C.RaiseError("invalid arity");

  if (llvm::StringRef Str = Args[0].getStringRef(); !Str.empty())
    OpName = Str;
  else
    return C.RaiseError("expecting operation name");

  Args = Args.drop_front();

  // Parse the remaining arguments which should be
  // lists starting with the appropriate aux. keyword.
  for (heavy::Value Val : Args) {
    auto* Pair = dyn_cast<heavy::Pair>(Val);
    if (!Pair)
      return C.RaiseError("expecting list for argument");
    auto* Symbol = dyn_cast<heavy::Symbol>(Pair->Car);

    if (!Symbol)
      return C.RaiseError("expecting keyword: "
                          "attributes, operands, regions, "
                          "result-types, successors");
    if (!isa<heavy::Pair, heavy::Empty>(Pair->Cdr))
      return C.RaiseError("expecting list");

    llvm::StringRef Keyword = Symbol->getStringRef();
    // Allow overriding duplicates. (might as well)
    if (Keyword == "attributes")
      Attributes = Pair->Cdr;
    if (Keyword == "operands")
      Operands = Pair->Cdr;
    if (Keyword == "result-types")
      ResultTypes = Pair->Cdr;
  }
  mlir::OpBuilder* Builder = getCurrentBuilder(C);
  if (!Builder) return;

  mlir::Location MLoc = mlir::OpaqueLoc::get(Loc.getOpaqueEncoding(),
                                             Builder->getContext());
  auto OpState = mlir::OperationState(MLoc, OpName);

  // attributes
  for (heavy::Value V : Attributes) {
    llvm::StringRef Name;
    if (auto* P = dyn_cast<heavy::Pair>(V)) {
      Name = P->Car.getStringRef();
      if (auto* P2 = dyn_cast<heavy::Pair>(P->Cdr)) {
        V = P2->Car;
        if (!isa<heavy::Empty>(P2->Cdr))
          Name = "";  // Clear the name so we raise error below.
      }
    }
    if (Name.empty())
      return C.RaiseError("expecting name-value pair for attribute",
                          Attributes);

    auto Attr = GetTagged<mlir::Attribute>(C, kind::mlir_attr, V);

    // If the object is not a mlir::Attribute, simply make
    // the heavy::Value into one.
    if (!Attr)
      Attr = HeavyValueAttr::get(Builder->getContext(), V);

    mlir::NamedAttribute NamedAttr = Builder->getNamedAttr(Name, Attr);
    OpState.attributes.push_back(NamedAttr);
  }

  // operands
  for (heavy::ListIterator Itr = Operands.begin();
       Itr != Operands.end();
       ++Itr) {
    auto MVal = GetTagged<mlir::Value>(C, kind::mlir_value, *Itr);
    if (!MVal)
      return C.RaiseError("expecting mlir.value", {Itr.Current, Operands});
    OpState.operands.push_back(MVal);
  }

  // regions
  {
    int Num = -1;
    auto* P = dyn_cast<heavy::Pair>(Regions);
    if (P && isa<heavy::Int>(P->Car))
      Num = cast<heavy::Int>(P->Car);

    if (!P || !isa<heavy::Empty>(P->Cdr) || Num < 0)
      return C.RaiseError("expecting positive integer");
    for (int I = 0; I < Num; ++I)
      OpState.regions.push_back(std::make_unique<mlir::Region>());
  }

  // result-types
  for (heavy::ListIterator Itr = ResultTypes.begin();
       Itr != ResultTypes.end();
       ++Itr) {
    auto MType = GetTagged<mlir::Type>(C, kind::mlir_type, *Itr);
    if (!MType)
      return C.RaiseError("expecting mlir.type", {Itr.Current, ResultTypes});
    OpState.types.push_back(MType);
  }

  // successors
  for (heavy::ListIterator Itr = Successors.begin();
       Itr != Successors.end();
       ++Itr) {
    mlir::Block* Block = GetTagged<mlir::Block*>(C, kind::mlir_block, *Itr);
    if (Block == nullptr)
      return C.RaiseError("expecting mlir.block", {Itr.Current, Successors});
    OpState.successors.push_back(Block);
  }

  // Create the operation using the Builder
  mlir::Operation* Op = Builder->create(OpState);
  C.Cont(heavy::Value(Op));
}

// Get an operation region by index (defaulting to 0).
// (region _op_)
// (region op _index_)
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
// (result _op_)
// (result _op_ _index_)
void result(Context& C, ValueRefs Args) {
  if (Args.size() != 1 && Args.size() != 2)
    return C.RaiseError("invalid arity");

  mlir::Operation* Op = heavy::dyn_cast<mlir::Operation>(Args[0]);
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

// Get the nth block argument from a block.
// (block-arg _block_ _index_)
void block_arg(Context& C, ValueRefs Args) {
  if (Args.size() != 2)
    return C.RaiseError("invalid arity");
  mlir::Block* Block = GetTagged<mlir::Block*>(C, kind::mlir_block, Args[0]);
  if (!Block || !isa<heavy::Int>(Args[1]))
    return C.RaiseError("expecting block and index");
  int Index = cast<heavy::Int>(Args[1]);
  // A mlir::BlockArgument is a mlir::Value 
  mlir::Value Val = Block->getArgument(Index);
  heavy::Value MVal = CreateTagged(C, kind::mlir_value, Val);
  C.Cont(MVal);
}

static void with_builder_impl(Context& C, mlir::OpBuilder const& Builder,
                              heavy::Value Thunk) {
  mlir::MLIRContext* MLIRContext = getCurrentContext(C);
  heavy::Value PrevBuilder = C.CreateBinding(heavy::Empty());
  heavy::Value NewBuilder = CreateTagged(C, kind::mlir_builder,
                              mlir::OpBuilder(MLIRContext));

  heavy::Value Before = C.CreateLambda(
    [](heavy::Context& C, heavy::ValueRefs Args) {
      // Save the previous state and instate the new... state.
      // (ie Builder)
      auto* PrevBuilder = cast<heavy::Binding>(C.getCapture(0));
      heavy::Value NewBuilder = C.getCapture(1);

      // Set the Binding
      PrevBuilder->setValue(HEAVY_MLIR_VAR(current_builder).get(C));

      // Set the "current-builder" value
      HEAVY_MLIR_VAR(current_builder).set(C, NewBuilder);
      C.Cont();
    }, CaptureList{PrevBuilder, NewBuilder});

  heavy::Value After = C.CreateLambda(
    [](heavy::Context& C, heavy::ValueRefs Args) {
      // Restore previous state
      auto* PrevBuilder = cast<heavy::Binding>(C.getCapture(0));
      HEAVY_MLIR_VAR(current_builder).set(C, PrevBuilder->getValue());
      C.Cont();
    }, CaptureList{PrevBuilder});

  C.DynamicWind(Before, Thunk, After);
}

// (with-builder _builder_ _thunk_)
void with_builder(Context& C, ValueRefs Args) {
  if (Args.size() != 2)
    return C.RaiseError("expecting 2 arguments");
  mlir::OpBuilder* Builder = getBuilder(C, Args[0]);
  if (!Builder)
    return;
  return with_builder_impl(C, *Builder, Args[1]);
}

static mlir::Block* get_arg_block(Context& C, ValueRefs Args) {
  mlir::Block* Block = nullptr;
  if (Args.size() == 1)
    Block = GetTagged<mlir::Block*>(C, kind::mlir_block, Args[0]);
  if (Block == nullptr)
    C.RaiseError("expecting mlir.block");
  return Block;
}

// Alter current-builder to insert at beginning of block.
// (at-block-begin _block_)
void at_block_begin(Context& C, ValueRefs Args) {
  if (mlir::Block* Block = get_arg_block(C, Args)) {
    mlir::OpBuilder* Builder = getCurrentBuilder(C);
    if (!Builder) return;
    Builder->atBlockBegin(Block);
    C.Cont();
  }
  // Note: error raised in get_arg_block
}

// Alter current-builder to insert at end to block
// (at-block-end _block_)
void at_block_end(Context& C, ValueRefs Args) {
  if (mlir::Block* Block = get_arg_block(C, Args)) {
    mlir::OpBuilder* Builder = getCurrentBuilder(C);
    if (!Builder) return;
    Builder->atBlockEnd(Block);
    C.Cont();
  }
  // Note: error raised in get_arg_block
}

// Alter current-builder to insert before terminator in block.
// (at-block-terminator _block_)
void at_block_terminator(Context& C, ValueRefs Args) {
  if (mlir::Block* Block = get_arg_block(C, Args)) {
    mlir::OpBuilder* Builder = getCurrentBuilder(C);
    if (!Builder) return;
    Builder->atBlockTerminator(Block);
    C.Cont();
  }
  // Note: error raised in get_arg_block
}

// Set insertion point to prepend. (alters current-builder)
// (set-insertion-point _op_or_region_)
// Argument can be Operation, Region, or Block.
//  Operation - Insert before operation in containing block.
//  Region    - Insert before operation in first block.
void set_insertion_point(Context& C, ValueRefs Args) {
  if (Args.size() != 1)
    return C.RaiseError("invalid arity");

  mlir::OpBuilder* Builder = getCurrentBuilder(C);
  if (!Builder) return;

  if (auto* Op = dyn_cast<mlir::Operation>(Args[0]))
    Builder->setInsertionPoint(Op);
  else if (mlir::Region* R = GetTagged<mlir::Region*>(C, kind::mlir_region,
                                                      Args[0])) {
    // Automatically add block if needed.
    if (R->empty())
      R->emplaceBlock();
    mlir::Block* Block = &R->front();
    Builder->setInsertionPointToStart(Block);
  }
  C.Cont();
}

// Set insertion point to after the operation. (alters current-builder)
// (set-insertion-after _op_)
void set_insertion_after(Context& C, ValueRefs Args) {
  if (Args.size() != 1)
    return C.RaiseError("invalid arity");
  mlir::OpBuilder* Builder = getCurrentBuilder(C);
  if (!Builder) return;

  mlir::Operation* Op = dyn_cast<mlir::Operation>(Args[0]);
  if (!Op)
    return C.RaiseError("expecting mlir_operation");

  Builder->setInsertionPoint(Op);
  C.Cont();
}

// Get a type by parsing a string.
// (type _string_)
void type(Context& C, ValueRefs Args) {
  mlir::MLIRContext* MLIRContext = getCurrentContext(C);
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
//  Usage: (attr _type_ _attr_str)
//    type - a string or a mlir.type object
//    attr_str - the string to be parsed
void attr(Context& C, ValueRefs Args) {
  mlir::MLIRContext* MLIRContext = getCurrentContext(C);
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

// (with-new-context _thunk_)
void with_new_context(heavy::Context& C, heavy::ValueRefs Args) {
  // Create a new context, and call a
  // thunk with it as the current-context.
  if (Args.size() != 1)
    return C.RaiseError("expecting thunk");

  auto* Thunk = dyn_cast<heavy::Lambda>(Args[0]);

  if (!Thunk)
    return C.RaiseError("expecting thunk");

  auto NewContextPtr = std::make_unique<mlir::MLIRContext>();
  heavy::Value NewMC = CreateTagged(C, kind::mlir_context,
                                    NewContextPtr.get());
  heavy::Value NewBuilder = CreateTagged(C, kind::mlir_builder,
                              mlir::OpBuilder(NewContextPtr.get()));

  // The Prev values are Bindings because we can enter
  // this via escape procedure from anywhere.
  // (If that was not obvious)
  heavy::Value PrevMC = C.CreateBinding(heavy::Empty());
  heavy::Value PrevBuilder = C.CreateBinding(heavy::Empty());

  heavy::Value Before = C.CreateLambda(
    [](heavy::Context& C, heavy::ValueRefs Args) {
      // Save the previous state and instate the new... state.
      // (ie MLIRContext and Builder)
      auto* PrevMC = cast<heavy::Binding>(C.getCapture(0));
      auto* PrevBuilder = cast<heavy::Binding>(C.getCapture(1));
      heavy::Value NewMC = C.getCapture(2);
      heavy::Value NewBuilder = C.getCapture(3);

      // Set the Bindings
      PrevMC->setValue(HEAVY_MLIR_VAR(current_context).get(C));
      PrevBuilder->setValue(HEAVY_MLIR_VAR(current_builder).get(C));

      // Set the "current" values
      HEAVY_MLIR_VAR(current_context).set(C, NewMC);
      HEAVY_MLIR_VAR(current_builder).set(C, NewBuilder);
      C.Cont();
    }, CaptureList{PrevMC, PrevBuilder, NewMC, NewBuilder});

  heavy::Value After = C.CreateLambda(
    [](heavy::Context& C, heavy::ValueRefs Args) {
      // Restore previous state
      auto* PrevMC = cast<heavy::Binding>(C.getCapture(0));
      auto* PrevBuilder = cast<heavy::Binding>(C.getCapture(1));
      HEAVY_MLIR_VAR(current_context).set(C, PrevMC->getValue());
      HEAVY_MLIR_VAR(current_builder).set(C, PrevBuilder->getValue());
      C.Cont();
    }, CaptureList{PrevMC, PrevBuilder});

  C.DynamicWind(std::move(NewContextPtr), Before, Thunk, After);
}

// Dynamically load a mlir dialect by its name.
// (load-dialect _name_)
void load_dialect(Context& C, heavy::ValueRefs Args) {
  if (Args.size() != 1)
    return C.RaiseError("expecting dialect name");

  llvm::StringRef Name = Args[0].getStringRef();

  if (Name.empty())
    return C.RaiseError("expecting dialect name");

  mlir::MLIRContext* MLIRContext = getCurrentContext(C);
  mlir::Dialect* Dialect = MLIRContext->getOrLoadDialect(Name);
  if (Dialect == nullptr)
    return C.RaiseError(C.CreateString("failed to load dialect: ", Name), {});

  C.Cont();
}

// Get parent operation of block.
// (block-op _block_)
void block_op(Context& C, heavy::ValueRefs Args) {
  mlir::Block* Block = nullptr;
  if (Args.size() == 1)
    Block = GetTagged<mlir::Block*>(C, kind::mlir_block, Args[0]);

  if (!Block)
    return C.RaiseError("expecting block");

  mlir::Operation* ParentOp = Block->getParentOp();
  heavy::Value Result = ParentOp != nullptr ? heavy::Value(ParentOp) :
                                              heavy::Empty();
  C.Cont(Result);
}

// Get the next operation from an operation
// or () if we find the tail of the list.
// (op-parent _op_)
void op_next(Context& C, heavy::ValueRefs Args) {
  if (mlir::Operation* Op = getSingleOpArg(C, Args)) {
    Op = Op->getNextNode();
    heavy::Value Result = Op != nullptr ? heavy::Value(Op) :
                                          heavy::Empty();
    C.Cont(Result);
  }
}

// Get the parent operation of an operation.
// or () if the operation is unlinked
// (parent-op _op_)
void parent_op(Context& C, heavy::ValueRefs Args) {
  if (mlir::Operation* Op = getSingleOpArg(C, Args)) {
    Op = Op->getParentOp();
    heavy::Value Result = Op != nullptr ? heavy::Value(Op) :
                                          heavy::Empty();
    C.Cont(Result);
  }
}

// Return true if operation verify is success otherwise
// false. Diagnostics are output to stderr by default via MLIR.
// (verify _op_)
void verify(Context& C, heavy::ValueRefs Args) {
  if (mlir::Operation* Op = getSingleOpArg(C, Args)) {
    bool Success = mlir::verify(Op).succeeded();
    C.Cont(heavy::Bool(Success));
  }
}

// (dump _op_)
void dump(Context& C, heavy::ValueRefs Args) {
  if (mlir::Operation* Op = getSingleOpArg(C, Args)) {
    Op->dump();
    C.Cont();
  }
}

}  // namespace heavy::mlir_bind

extern "C" {
// initialize the module for run-time independent of the compiler
void HEAVY_MLIR_INIT(heavy::Context& C) {
  mlir::MLIRContext* MC = C.MLIRContext.get();
  heavy::Value MC_Val = CreateTagged(C, kind::mlir_context, MC);
  heavy::Value BuilderVal = CreateTagged(C, kind::mlir_builder,
                                         mlir::OpBuilder(MC));
  HEAVY_MLIR_VAR(current_context).init(C, MC_Val);
  HEAVY_MLIR_VAR(current_builder).init(C, BuilderVal);

  // syntax
  HEAVY_MLIR_VAR(create_op) = heavy::mlir_bind::create_op;
  HEAVY_MLIR_VAR(region) = heavy::mlir_bind::region;
  HEAVY_MLIR_VAR(results) = heavy::mlir_bind::results;
  HEAVY_MLIR_VAR(result) = heavy::mlir_bind::result;
  HEAVY_MLIR_VAR(at_block_begin) = heavy::mlir_bind::at_block_begin;
  HEAVY_MLIR_VAR(at_block_end) = heavy::mlir_bind::at_block_end;
  HEAVY_MLIR_VAR(block_op) = heavy::mlir_bind::block_op;
  HEAVY_MLIR_VAR(op_next) = heavy::mlir_bind::op_next;
  HEAVY_MLIR_VAR(parent_op) = heavy::mlir_bind::parent_op;
  HEAVY_MLIR_VAR(set_insertion_point) = heavy::mlir_bind::set_insertion_point;
  HEAVY_MLIR_VAR(set_insertion_after) = heavy::mlir_bind::set_insertion_after;
  HEAVY_MLIR_VAR(type) = heavy::mlir_bind::type;
  HEAVY_MLIR_VAR(attr) = heavy::mlir_bind::attr;
  HEAVY_MLIR_VAR(load_dialect) = heavy::mlir_bind::load_dialect;
  HEAVY_MLIR_VAR(with_builder) = heavy::mlir_bind::with_builder;
  HEAVY_MLIR_VAR(with_new_context) = heavy::mlir_bind::with_new_context;
  HEAVY_MLIR_VAR(verify) = heavy::mlir_bind::verify;
  HEAVY_MLIR_VAR(dump) = heavy::mlir_bind::dump;
}

void HEAVY_MLIR_LOAD_MODULE(heavy::Context& C) {
  HEAVY_MLIR_INIT(C);
  heavy::initModule(C, HEAVY_MLIR_LIB_STR, {
    {"create-op", HEAVY_MLIR_VAR(create_op)},
    {"region", HEAVY_MLIR_VAR(region)},
    {"results", HEAVY_MLIR_VAR(results)},
    {"result", HEAVY_MLIR_VAR(result)},
    {"at-block-begin", HEAVY_MLIR_VAR(at_block_begin)},
    {"at-block-end", HEAVY_MLIR_VAR(at_block_end)},
    {"block-op", HEAVY_MLIR_VAR(block_op)},
    {"op-next", HEAVY_MLIR_VAR(op_next)},
    {"parent-op", HEAVY_MLIR_VAR(parent_op)},
    {"set-insertion-point", HEAVY_MLIR_VAR(set_insertion_point)},
    {"set-insertion-after", HEAVY_MLIR_VAR(set_insertion_after)},
    {"type", HEAVY_MLIR_VAR(type)},
    {"attr", HEAVY_MLIR_VAR(attr)},
    {"with-new-context", HEAVY_MLIR_VAR(with_new_context)},
    {"with-builder", HEAVY_MLIR_VAR(with_builder)},
    {"load-dialect", HEAVY_MLIR_VAR(load_dialect)},
    {"verify", HEAVY_MLIR_VAR(verify)},
    {"dump", HEAVY_MLIR_VAR(dump)}
  });
}
}

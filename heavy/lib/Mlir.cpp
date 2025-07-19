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
#include <heavy/Dialect.h>
#include <heavy/Mlir.h>
#include <heavy/MlirHelper.h>
#include <heavy/OpGen.h>
#include <heavy/Value.h>
#include <mlir/AsmParser/AsmParser.h>
#include <mlir/IR/Verifier.h>
#include <mlir/Support/LogicalResult.h>
#include <llvm/ADT/StringExtras.h>
#include <llvm/Support/Casting.h>
#include <memory>

#define HEAVY_MLIR_VAR(NAME) ::heavy::mlir_bind_var::NAME

// Manually provide mangled names as needed.
#define HEAVY_MLIR_VAR_STR__create_op_impl \
                        HEAVY_MLIR_LIB_(Vrm6Screatemi2Sop)
#define HEAVY_MLIR_VAR_STR_(NAME) HEAVY_MLIR_VAR_STR__##NAME
#define HEAVY_MLIR_VAR_STR(X) HEAVY_MLIR_VAR_STR_STR(HEAVY_MLIR_VAR_STR_(X))
#define HEAVY_MLIR_VAR_STR_STR(X) HEAVY_MLIR_VAR_STR_STR_STR(X)
#define HEAVY_MLIR_VAR_STR_STR_STR(X) #X

namespace heavy::mlir_bind_var {
heavy::ContextLocal   current_context;
heavy::ContextLocal   current_builder;
heavy::ExternSyntax<> create_op;
heavy::ExternFunction create_op_impl;
heavy::ExternFunction region;
heavy::ExternFunction entry_block;
heavy::ExternFunction results;
heavy::ExternFunction result;
heavy::ExternFunction at_block_begin;
heavy::ExternFunction at_block_end;
heavy::ExternFunction at_block_terminator;
heavy::ExternFunction block_op;
heavy::ExternFunction set_insertion_point;
heavy::ExternFunction set_insertion_after;
heavy::ExternFunction type;
heavy::ExternFunction function_type_impl;
heavy::ExternFunction attr;
heavy::ExternFunction type_attr;
heavy::ExternFunction value_attr;
template <typename AttrTy>
heavy::ExternFunction string_attr;
heavy::ExternFunction with_new_context;
heavy::ExternFunction with_builder;
heavy::ExternFunction load_dialect;
heavy::ExternFunction parent_op;
heavy::ExternFunction op_next;
heavy::ExternFunction verify;
heavy::ExternFunction module_lookup;
}

using namespace heavy::mlir_helper;

namespace heavy::mlir_bind {
// Provide function to support create_op syntax.
// Require type checking as a precondition.
// (%create-op _name_
//             _attrs_        : vector?
//             _operands_     : vector?
//             _regions_      : number?
//             _result_types_ : vector?
//             _successors_   : vector?)
void create_op_impl(Context& C, ValueRefs Args) {
  if (Args.size() != 6)
    return C.RaiseError("invalid arity");

  heavy::Vector* Attributes   = cast<heavy::Vector>(Args[1]);
  heavy::Vector* Operands     = cast<heavy::Vector>(Args[2]);
  int NumRegions              = cast<heavy::Int>(Args[3]);
  heavy::Vector* ResultTypes  = cast<heavy::Vector>(Args[4]);
  heavy::Vector* Successors   = cast<heavy::Vector>(Args[5]);

  llvm::StringRef OpName = Args[0].getStringRef();
  if (OpName.empty())
    return C.RaiseError("expecting operation name");

  Args = Args.drop_front();

  mlir::OpBuilder* Builder = getCurrentBuilder(C);
  if (!Builder) return;

  heavy::SourceLocation Loc = C.getLoc();
  mlir::Location MLoc = mlir::OpaqueLoc::get(Loc.getOpaqueEncoding(),
                                             Builder->getContext());
  auto OpState = mlir::OperationState(MLoc, OpName);

  // attributes
  for (heavy::Value V : Attributes->getElements()) {
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
      return C.RaiseError("expecting name-value pair for attribute", V);

    auto Attr = getTagged<mlir::Attribute>(C, kind::mlir_attr, V);

    // If the object is not a mlir::Attribute, simply make
    // the heavy::Value into one.
    if (!Attr)
      Attr = HeavyValueAttr::get(Builder->getContext(), V);

    mlir::NamedAttribute NamedAttr = Builder->getNamedAttr(Name, Attr);
    OpState.attributes.push_back(NamedAttr);
  }

  // operands
  for (heavy::Value V : Operands->getElements()) {
    // FIXME
    // Implicitly unwrapping lists here because
    // unquote-splicing is not implemented.
    if (isa<heavy::Pair, heavy::Empty>(V)) {
      for (heavy::Value V2 : V) {
        auto MVal = getTagged<mlir::Value>(C, kind::mlir_value, V2);
        if (!MVal)
          return C.RaiseError("expecting mlir.value", V2);
        OpState.operands.push_back(MVal);
      }
      break;
    }
    auto MVal = getTagged<mlir::Value>(C, kind::mlir_value, V);
    if (!MVal)
      return C.RaiseError("expecting mlir.value", V);
    OpState.operands.push_back(MVal);
  }

  // regions
  for (int I = 0; I < NumRegions; ++I)
    OpState.regions.push_back(std::make_unique<mlir::Region>());

  // result-types
  for (heavy::Value V : ResultTypes->getElements()) {
    auto MType = getTagged<mlir::Type>(C, kind::mlir_type, V);
    if (!MType)
      return C.RaiseError("expecting mlir.type", V);
    OpState.types.push_back(MType);
  }

  // successors
  for (heavy::Value V : Successors->getElements()) {
    mlir::Block* Block = getTagged<mlir::Block*>(C, kind::mlir_block, V);
    if (Block == nullptr)
      return C.RaiseError("expecting mlir.block", V);
    OpState.successors.push_back(Block);
  }

  // Create the operation using the Builder
  mlir::Operation* Op = Builder->create(OpState);
  C.Cont(heavy::Value(Op));
}

// Create operation syntax. Argument are
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
//    (regions      _regions_ )
//    (result-types _types_ ...)
//    (successors   _blocks_ ...))
//
void create_op(Context& C, ValueRefs Args) {  // Syntax
  heavy::SourceLocation CallLoc = cast<Pair>(Args[0])
      ->Car.getSourceLocation();
  heavy::OpGen& OpGen = *C.OpGen;
  heavy::Value Attributes   = heavy::Empty();
  heavy::Value Operands     = heavy::Empty();
  heavy::Value NumRegions   = heavy::Int(0);
  heavy::Value ResultTypes  = heavy::Empty();
  heavy::Value Successors   = heavy::Empty();

  heavy::Pair* Input = dyn_cast<Pair>(cast<Pair>(Args[0])->Cdr);
  if (!Input)
    return C.RaiseError("expecting arguments");

  // Note that calls to GetSingleResult handle TailPos for us.

  // Require the _name_ argument.
  mlir::Value OpName = OpGen.GetSingleResult(Input->Car);
  if (OpGen.CheckError())
    return;

  // Process named arguments which are optional.
  for (auto [Loc, Arg] : WithSource(Input->Cdr)) {
    llvm::StringRef ArgName;
    if (heavy::Pair* P = dyn_cast<Pair>(Arg)) {
      ArgName = P->Car.getStringRef();
      Arg = P->Cdr;
    }

    if (ArgName == "attributes")
      Attributes = Arg;
    else if (ArgName == "operands")
      Operands = Arg;
    else if (ArgName == "regions") {
      // Support improper list.
      if (auto* PArg = dyn_cast<Pair>(Arg))
        NumRegions = PArg->Car;
      else
        NumRegions = Arg;
    }
    else if (ArgName == "result-types")
      ResultTypes = Arg;
    else if (ArgName == "successors")
      Successors = Arg;
    else
      return C.RaiseError("expecting named argument "
          "[attributes, operands, regions, result-types, successors]");
  }

  llvm::SmallVector<mlir::Value, 5> Vals;
  auto createInputVector = [&](heavy::Value Inputs) -> mlir::Value {
    for (auto [Loc, X] : WithSource(Inputs)) {
      C.setLoc(Loc);
      mlir::Value V = OpGen.GetSingleResult(X);
      if (OpGen.CheckError())
        return mlir::Value();
      Vals.push_back(V);
    }

    // Localize the values because we are doing this manually.
    for (auto& V : Vals)
      V = OpGen.LocalizeValue(V);

    mlir::Value InputVals = OpGen.create<heavy::VectorOp>(
                              Attributes.getSourceLocation(), Vals);
    Vals.clear();
    return InputVals;
  };

  mlir::Value AttrVals = createInputVector(Attributes);
  mlir::Value OperandVals = createInputVector(Operands);
  mlir::Value NumRegionsVal = OpGen.GetSingleResult(NumRegions);
  if (OpGen.CheckError())
    return;
  mlir::Value ResultTypeVals = createInputVector(ResultTypes);
  mlir::Value SuccessorVals = createInputVector(Successors);

  assert(Vals.empty());
  // Reuse Vals as arguments to %create-op
  Vals.push_back(OpName);
  Vals.push_back(AttrVals);
  Vals.push_back(OperandVals);
  Vals.push_back(NumRegionsVal);
  Vals.push_back(ResultTypeVals);
  Vals.push_back(SuccessorVals);

  // Create the call to %create-op (aka create_op_impl)
  mlir::Value Fn = OpGen.create<LoadGlobalOp>(CallLoc,
                        HEAVY_MLIR_VAR_STR(create_op_impl));
  mlir::Value Call = OpGen.createCall(CallLoc, Fn, Vals);
  C.Cont(OpGen.fromValue(Call));
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
  C.Cont(createTagged(C, kind::mlir_region, Region));
}

// Get entry block from region/op by index.
// If an op is provided the first region is used.
// Add block to region if empty.
void entry_block(Context& C, ValueRefs Args) {
  if (Args.size() != 1)
    return C.RaiseError("invalid arity");

  mlir::Region* Region = nullptr;
  if (mlir::Operation* Op = dyn_cast<mlir::Operation>(Args[0])) {
    if (Op->getNumRegions() < 1)
      return C.RaiseError("mlir.op has no regions");
    Region = &Op->getRegion(0);
  } else {
    Region = getTagged<mlir::Region*>(C, kind::mlir_region, Args[0]);
  }

  if (!Region)
    return C.RaiseError("expecting mlir.op/mlir.region");
  if (Region->empty())
    Region->emplaceBlock();
  mlir::Block* Block = &(Region->front());
  if (!Block)
    return C.RaiseError("invalid mlir.block");
  C.Cont(createTagged(C, kind::mlir_block, Block));
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
  Value IndexArg = Args.size() > 1 ? Args[1] : Value(Int{0});

  if (!Op)
    return C.RaiseError("expecting mlir.op");

  if (!heavy::isa<heavy::Int>(IndexArg))
    return C.RaiseError("expecting index");

  uint32_t Index = static_cast<uint32_t>(heavy::cast<heavy::Int>(IndexArg));

  if (Index >= Op->getNumResults())
    return C.RaiseError("result index is out of range");

  mlir::Value Result = Op->getResult(Index);
  if (!Result)
    return C.RaiseError("invalid mlir.op result");
  C.Cont(createTagged(C, kind::mlir_value, Result));
}

// Get the nth block argument from a block.
// (block-arg _block_ _index_)
void block_arg(Context& C, ValueRefs Args) {
  if (Args.size() != 2)
    return C.RaiseError("invalid arity");
  mlir::Block* Block = getTagged<mlir::Block*>(C, kind::mlir_block, Args[0]);
  if (!Block || !isa<heavy::Int>(Args[1]))
    return C.RaiseError("expecting block and index");
  int Index = cast<heavy::Int>(Args[1]);
  // A mlir::BlockArgument is a mlir::Value
  mlir::Value Val = Block->getArgument(Index);
  heavy::Value MVal = createTagged(C, kind::mlir_value, Val);
  C.Cont(MVal);
}

// Copy the builder. (ie Do not modify.)
// (with-builder [_builder_] _thunk_)
void with_builder(Context& C, ValueRefs Args) {
  if (Args.empty() || Args.size() > 2)
    return C.RaiseError("expecting 2 arguments");

  heavy::Value Thunk;
  mlir::MLIRContext* MLIRContext = getCurrentContext(C);
  mlir::OpBuilder Builder(MLIRContext);
  if (Args.size() == 2) {
    mlir::OpBuilder* BuilderPtr = getBuilder(C, Args[0]);
    if (!BuilderPtr)
      return;
    Thunk = Args[1];
    Builder = *BuilderPtr;
  } else {
    Thunk = Args[0];
  }

  return with_builder_impl(C, Builder, Thunk);
}

static mlir::Block* get_arg_block(Context& C, ValueRefs Args) {
  mlir::Block* Block = nullptr;
  if (Args.size() == 1)
    Block = getTagged<mlir::Block*>(C, kind::mlir_block, Args[0]);
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
    *Builder = mlir::OpBuilder::atBlockBegin(Block);
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
    *Builder = mlir::OpBuilder::atBlockEnd(Block);
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
    *Builder = mlir::OpBuilder::atBlockTerminator(Block);
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
  else if (mlir::Region* R = getTagged<mlir::Region*>(C, kind::mlir_region,
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

  C.Cont(createTagged(C, kind::mlir_type, Type.getImpl()));
}

// Create a function type (using vector literals.
// (%function-type #(<arg-types>...) #(<result-types>...))
void function_type_impl(Context& C, ValueRefs Args) {
  mlir::MLIRContext* MLIRContext = getCurrentContext(C);
  if (Args.size() != 2)
    return C.RaiseError("invalid arity");
  auto* ArgTypeVals = heavy::dyn_cast<heavy::Vector>(Args[0]);
  auto* ResultTypeVals = heavy::dyn_cast<heavy::Vector>(Args[1]);
  if (!ArgTypeVals || !ResultTypeVals)
    return C.RaiseError("expecting vectors");

  // Arg types
  llvm::SmallVector<mlir::Type, 8> ArgTypes;
  Args = Args.drop_front();
  for (heavy::Value Arg : ArgTypeVals->getElements()) {
    mlir::Type ArgType = getTagged<mlir::Type>(C, kind::mlir_type, Arg);
    if (!ArgType)
      return C.RaiseError("expecting mlir.type");
    ArgTypes.push_back(ArgType);
  }

  // Result types
  llvm::SmallVector<mlir::Type, 8> ResultTypes;
  for (heavy::Value Result : ResultTypeVals->getElements()) {
    mlir::Type ResultType = getTagged<mlir::Type>(C, kind::mlir_type, Result);
    if (!ResultType)
      return C.RaiseError("expecting mlir.type");
    ResultTypes.push_back(ResultType);
  }

  mlir::Type Type = mlir::FunctionType::get(MLIRContext, ArgTypes,
                                            ResultTypes);

  if (!Type)
    return C.RaiseError("mlir build function type failed");

  C.Cont(createTagged(C, kind::mlir_type, Type.getImpl()));
}

// Get an attribute by parsing a string.
//  Usage: (attr _attr_str [_type_])
//    attr_str - the string to be parsed
//    type - a string or a mlir.type object (defaults to NoneType)
void attr(Context& C, ValueRefs Args) {
  mlir::MLIRContext* MLIRContext = getCurrentContext(C);
  mlir::Attribute Attr;
  if (Args.size() > 2 || Args.empty())
    return C.RaiseError("invalid arity");

  heavy::Value AttrStrArg = Args[0];

  mlir::Type Type;
  if (Args.size() == 2) {
    heavy::Value TypeArg = Args[1];
    llvm::StringRef TypeStr = TypeArg.getStringRef();
    if (!TypeStr.empty()) {
      Type = mlir::parseType(TypeStr, MLIRContext, nullptr,
                             heavy::String::IsNullTerminated);
      if (!Type)
        return C.RaiseError("mlir type parse failed");
    }
    else {
      Type = getTagged<mlir::Type>(C, kind::mlir_type, TypeArg);
      if (!Type)
        return C.RaiseError("invalid mlir type");
    }
  }
  if (!Type)
    Type = mlir::NoneType::get(MLIRContext);

  llvm::StringRef AttrStr = AttrStrArg.getStringRef();
  if (AttrStr.empty())
    return C.RaiseError("expecting string");

  Attr = mlir::parseAttribute(AttrStr, MLIRContext,
                              Type, nullptr,
                              heavy::String::IsNullTerminated);
  if (!Attr)
    return C.RaiseError("mlir attribute parse failed");

  C.Cont(createTagged(C, kind::mlir_attr, Attr.getImpl()));
}

// (type-attr (type "!type-goes-here"))
void type_attr(Context& C, ValueRefs Args) {
  if (Args.size() != 1)
    return C.RaiseError("invalid arity");
  mlir::Type Type = getTagged<mlir::Type>(C, kind::mlir_type, Args[0]);
  if (!Type)
    return C.RaiseError("expecting a mlir.type");
  mlir::Attribute Attr = mlir::TypeAttr::get(Type);
  C.Cont(createTagged(C, kind::mlir_attr, Attr.getImpl()));
}

// Create a heavy scheme value attribute of type !heavy.value
void value_attr(Context& C, ValueRefs Args) {
  if (Args.size() != 1)
    return C.RaiseError("invalid arity");
  mlir::MLIRContext* MLIRContext = getCurrentContext(C);
  mlir::Attribute Attr = HeavyValueAttr::get(MLIRContext, Args[0]);
  C.Cont(createTagged(C, kind::mlir_attr, Attr.getImpl()));
}

template <typename AttrTy>
void string_attr(Context& C, ValueRefs Args) {
  if (Args.size() != 1)
    return C.RaiseError("invalid arity");
  if (!isa<heavy::String, heavy::Symbol>(Args[0]))
    return C.RaiseError("expecting string-like object");
  llvm::StringRef Str = Args[0].getStringRef(); 
  mlir::MLIRContext* MLIRContext = getCurrentContext(C);
  mlir::Attribute Attr = AttrTy::get(MLIRContext, Str);
  C.Cont(createTagged(C, kind::mlir_attr, Attr.getImpl()));
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

  auto NewContextPtr = std::make_unique<mlir::MLIRContext>(*C.DialectRegistry);
  heavy::Value NewMC = createTagged(C, kind::mlir_context,
                                    NewContextPtr.get());
  heavy::Value NewBuilder = createTagged(C, kind::mlir_builder,
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
  // Ensure the registry is up to date.
  MLIRContext->appendDialectRegistry(*C.DialectRegistry);
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
    Block = getTagged<mlir::Block*>(C, kind::mlir_block, Args[0]);

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

// (symbol-table-lookup _module-op_ _"symbolname"_)
void module_lookup(Context& C, heavy::ValueRefs Args) {
  if (Args.size() != 2)
    return C.RaiseError("invalid arity");

  mlir::ModuleOp ModuleOp = dyn_cast_or_null<mlir::ModuleOp>(
      dyn_cast<mlir::Operation>(Args[0]));
  llvm::StringRef SymbolName = Args[1].getStringRef();
  if (!ModuleOp)
    return C.RaiseError("expecting mlir::ModuleOp");
  if (SymbolName.empty())
    return C.RaiseError("expecting nonempty string-like object");

  mlir::Operation* Op = ModuleOp.lookupSymbol(SymbolName);
  heavy::Value Result = Op != nullptr ? heavy::Value(Op) : heavy::Empty();
  C.Cont(Result);
}

}  // namespace heavy::mlir_bind

extern "C" {
// initialize the module for run-time independent of the compiler
void HEAVY_MLIR_INIT(heavy::Context& C) {
  mlir::MLIRContext* MC = C.MLIRContext.get();
  heavy::Value MC_Val = createTagged(C, kind::mlir_context, MC);
  heavy::Value BuilderVal = createTagged(C, kind::mlir_builder,
                                         mlir::OpBuilder(MC));
  HEAVY_MLIR_VAR(current_context).init(C, C.CreateBinding(MC_Val));
  HEAVY_MLIR_VAR(current_builder).init(C, C.CreateBinding(BuilderVal));

  HEAVY_MLIR_VAR(create_op) = heavy::mlir_bind::create_op;
  HEAVY_MLIR_VAR(create_op_impl) = heavy::mlir_bind::create_op_impl;
  HEAVY_MLIR_VAR(region) = heavy::mlir_bind::region;
  HEAVY_MLIR_VAR(entry_block) = heavy::mlir_bind::entry_block;
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
  HEAVY_MLIR_VAR(function_type_impl) = heavy::mlir_bind::function_type_impl;
  HEAVY_MLIR_VAR(attr) = heavy::mlir_bind::attr;
  HEAVY_MLIR_VAR(type_attr) = heavy::mlir_bind::type_attr;
  HEAVY_MLIR_VAR(value_attr) = heavy::mlir_bind::value_attr;
  HEAVY_MLIR_VAR(string_attr<mlir::StringAttr>)
    = heavy::mlir_bind::string_attr<mlir::StringAttr>;
  HEAVY_MLIR_VAR(string_attr<mlir::FlatSymbolRefAttr>)
    = heavy::mlir_bind::string_attr<mlir::FlatSymbolRefAttr>;
  HEAVY_MLIR_VAR(load_dialect) = heavy::mlir_bind::load_dialect;
  HEAVY_MLIR_VAR(with_builder) = heavy::mlir_bind::with_builder;
  HEAVY_MLIR_VAR(with_new_context) = heavy::mlir_bind::with_new_context;
  HEAVY_MLIR_VAR(verify) = heavy::mlir_bind::verify;
  HEAVY_MLIR_VAR(module_lookup) = heavy::mlir_bind::module_lookup;
}

void HEAVY_MLIR_LOAD_MODULE(heavy::Context& C) {
  HEAVY_MLIR_INIT(C);
  heavy::initModuleNames(C, HEAVY_MLIR_LIB_STR, {
    {"create-op", HEAVY_MLIR_VAR(create_op)},
    {"%create-op", HEAVY_MLIR_VAR(create_op_impl)},
    {"current-builder", HEAVY_MLIR_VAR(current_builder).get_binding(C)},
    {"region", HEAVY_MLIR_VAR(region)},
    {"entry-block", HEAVY_MLIR_VAR(entry_block)},
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
    {"%function-type", HEAVY_MLIR_VAR(function_type_impl)},
    {"attr", HEAVY_MLIR_VAR(attr)},
    {"type-attr", HEAVY_MLIR_VAR(type_attr)},
    {"value-attr", HEAVY_MLIR_VAR(value_attr)},
    {"string-attr", HEAVY_MLIR_VAR(string_attr<mlir::StringAttr>)},
    {"flat-symbolref-attr",
      HEAVY_MLIR_VAR(string_attr<mlir::FlatSymbolRefAttr>)},
    {"with-new-context", HEAVY_MLIR_VAR(with_new_context)},
    {"with-builder", HEAVY_MLIR_VAR(with_builder)},
    {"load-dialect", HEAVY_MLIR_VAR(load_dialect)},
    {"verify", HEAVY_MLIR_VAR(verify)},
    {"module-lookup", HEAVY_MLIR_VAR(module_lookup)},
  });
}
}

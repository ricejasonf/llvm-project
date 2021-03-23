//===- Ops.td - MLIR Dialect Definitions for HeavyScheme ---*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  Defines operations for HeavyScheme dialect of MLIR
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_HEAVY_OPS_TD
#define LLVM_HEAVY_OPS_TD

include "mlir/IR/OpBase.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def HeavyDialect : Dialect {
  let name = "heavy";
}

def HeavyLambda   : OpaqueType<"heavy", "HeavyLambda",
                               "Same as heavy::Lambda*">;
def HeavyRest     : OpaqueType<"heavy", "HeavyRest",
                               "Same as heavy::Value but "
                               "denotes a rest parameter">;
def HeavyValue    : OpaqueType<"heavy", "HeavyValue",
                               "Same as heavy::Value">;

def HeavyValueAttr : Attr<CPred<"true">,
                          "Any Attribute convertible to heavy::Value"> {
  let storageType = "::mlir::heavy_mlir::HeavyValueAttr";
  let returnType = "::heavy::Value";
}

class HeavyOp<string mnemonic, list<OpTrait> traits = []> :
    Op<HeavyDialect, mnemonic, traits>;

def heavy_ApplyOp : HeavyOp<"apply"> {
  let summary = "apply";
  let description = [{
    Dynamically applies a set of operands to an operator which can
    be a user defined lambda, a builtin, or an FFI wrapper.
  }];

  let arguments = (ins HeavyValue:$fn, Variadic<HeavyValue>:$args,
                       BoolAttr:$isTailPos);
  let results   = (outs HeavyValue:$result);
  let builders = [
    OpBuilder<"::mlir::OpBuilder&, ::mlir::OperationState&, "
              "::mlir::Value, ArrayRef<::mlir::Value>, "
              "bool IsTailPos">
  ];
}

def heavy_BindingOp : HeavyOp<"binding", [Symbol]> {
  let summary = "binding";
  let description = [{
    BindingOp creates a value wrapper (heavy::Binding) that allows
    shared ownership of a reference. This has meaning with SetOp,
    and in the absence of such mutation, the binding may be
    elided and the result can refer directly to the enclosed value.
  }];

  let arguments = (ins HeavyValue:$input, BoolAttr:$isTopLevel);
  let results   = (outs HeavyValue:$result);
  let builders = [
    OpBuilder<"::mlir::OpBuilder&, ::mlir::OperationState&, "
              "::mlir::Value input, bool IsTopLevel">
  ];

  let extraClassDeclaration = [{
    bool isOptionalSymbol() { return true; }
    bool isSymbol() {
      return static_cast<bool>(getAttr(
        ::mlir::SymbolTable::getSymbolAttrName()));
    }
  }];
}

def heavy_BuiltinOp : HeavyOp<"builtin"> {
  let summary = "builtin";
  let description = [{
    This is a heavy::Value wrapper for a function
    of type heavy::ValueFn implemented in C++.
    This takes the callee name as a string as well
    as a constructed heavy::BuiltinFn for the tree
    evaluator.
  }];

  //let arguments = (ins FlatSymbolRefAttr:$calleeName,
                   //HeavyValueAttr:$builtinFn);
  let arguments = (ins HeavyValueAttr:$builtinFn);
  let results   = (outs HeavyValue:$result);
  let builders  = [
    OpBuilder<"::mlir::OpBuilder&, ::mlir::OperationState&, "
              "heavy::Builtin*">
  ];
}

def heavy_ConsOp : HeavyOp<"cons"> {
  let summary = "cons";
  let description = [{
    ConsOp constructs a pair from two values. That's it.
  }];

  let arguments = (ins HeavyValue:$a, HeavyValue:$b);
  let results   = (outs HeavyValue:$result);
  let builders = [
    OpBuilder<"::mlir::OpBuilder&, ::mlir::OperationState&, "
              "::mlir::Value A, ::mlir::Value B">
  ];
}

def heavy_ContOp : HeavyOp<"cont", [ReturnLike, Terminator]> {
  let summary = "cont";
  let description = [{
    ContOp models calling the current continuation. It can take
    an arbitrary amount of inputs and represents the result(s) of
    an expression.
  }];

  let arguments = (ins Variadic<HeavyValue>:$args);
  let results   = (outs); // no result
}

def heavy_GlobalOp : HeavyOp<"global", [Symbol]> {
  let summary = "global";
  let description = [{
    This indicates a module top level variable providing
    a symbol. The input would be a binding but it should be
    possible to reference a value directly.
  }];

  let arguments = (ins StrAttr:$sym_name,
                       HeavyValue:$input);
  let results   = (outs); // no result
}

def heavy_IfOp : HeavyOp<"if", []> {
  let summary = "if";
  let description = [{
    IfOp handles branching to one of two regions based on an
    input result.
  }];

  let arguments = (ins HeavyValue:$input);
  let results   = (outs HeavyValue:$result);

  let regions   = (region SizedRegion<1>:$thenRegion,
                          SizedRegion<1>:$elseRegion);
  let builders = [
    OpBuilder<"::mlir::OpBuilder&, ::mlir::OperationState&, "
              "::mlir::Value" >
  ];
}

def heavy_LambdaOp : HeavyOp<"lambda"> {
  let summary = "lambda";
  let description = [{
    Lambdas represent a closure. They take the name of a function
    and a list of values to capture.
  }];

  let arguments = (ins StrAttr:$name,
                       Variadic<HeavyValue>:$captures);
  let results = (outs HeavyValue:$result);

  let builders = [
    OpBuilder<"::mlir::OpBuilder&, ::mlir::OperationState&, "
              "::llvm::StringRef Name, "
              "::llvm::ArrayRef<::mlir::Value> Captures">
  ];
}

def heavy_LiteralOp : HeavyOp<"literal", [NoSideEffect]> {
  let summary = "literal";
  let description = [{
    Literals in scheme can be primitives or quoted scheme values that are
    known statically and should be immutable.
    For starters this should simply lower to calling the corresponding
    factory function to dynamically allocate the object and return a
    run-time heavy::Value.
  }];

  let arguments = (ins HeavyValueAttr:$input);
  let results   = (outs HeavyValue:$result);
  let builders = [
    OpBuilder<"::mlir::OpBuilder&, ::mlir::OperationState&, ::heavy::Value">
  ];
}

def heavy_LoadClosureOp : HeavyOp<"load_closure"> {
  let summary = "load_closure";
  let description = [{
    Read an element from lambda's capture list.
  }];

  let arguments = (ins HeavyLambda:$closure, UI32Attr:$index);
  let results   = (outs HeavyValue:$result);
  
  let builders = [
    OpBuilder<"::mlir::OpBuilder&, ::mlir::OperationState&,"
              "::mlir::Value Closure, uint32_t Index">
  ];
}

def heavy_LoadGlobalOp : HeavyOp<"load_global"> {
  let summary = "load_global";
  let description = [{
    Read the value from a global.
  }];

  let arguments = (ins FlatSymbolRefAttr:$name);
  let results   = (outs HeavyValue:$result);
  
  let builders = [
    OpBuilder<"::mlir::OpBuilder&, ::mlir::OperationState&,"
              "StringRef Name">
  ];
}

def heavy_SetOp : HeavyOp<"set"> {
  let summary = "set";
  let description = [{
    SetOp sets the enclosed value of a binding. This is used
    by the `set` syntax as well as lazy initialization of local
    variables needed by `letrec` and similar binding constructs.
  }];

  let arguments = (ins HeavyValue:$binding, HeavyValue:$input);
  let results   = (outs HeavyValue:$result);
  let builders = [
    OpBuilder<"::mlir::OpBuilder&, ::mlir::OperationState&, "
              "::mlir::Value binding, ::mlir::Value input">
  ];
}

def heavy_SpliceOp : HeavyOp<"splice"> {
  let summary = "splice";
  let description = [{
    SpliceOp splices two lists or vectors. The result type
    should be the same as second operand.
    TODO Vectors and types are not supported yet.
  }];

  let arguments = (ins HeavyValue:$a, HeavyValue:$b);
  let results   = (outs HeavyValue:$result);
  let builders = [
    OpBuilder<"::mlir::OpBuilder&, ::mlir::OperationState&, "
              "::mlir::Value A, ::mlir::Value B">
  ];
}

def heavy_UndefinedOp : HeavyOp<"undefined", [NoSideEffect]> {
  let summary = "undefined";
  let description = [{
    This is a singleton literal for heavy::Undefined
  }];

  let arguments = (ins);
  let results   = (outs HeavyValue:$result);

  let builders = [
    OpBuilder<"::mlir::OpBuilder&, ::mlir::OperationState&">
  ];
}



#endif

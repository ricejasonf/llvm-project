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

def HeavyValue : OpaqueType<"heavy", "HeavyValue",
                       "HeavyValue represents a heavy::Value*">;

def HeavyValueAttr : Attr<CPred<"true">,
                          "Any Attribute convertible to heavy::Value"> {
  let storageType = "::mlir::heavy_mlir::HeavyValueAttr";
  let returnType = "::heavy::Value *";
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

def heavy_LambdaOp : HeavyOp<"lambda", [FunctionLike]> {
  let summary = "lambda";
  let description = [{
    Lambdas represent function objects that have
    two significant properties:
      - Variables that are lexically defined in a parent scope are implicitly
        captured if they appear in the procedures scope. 
      - The object is callable with an arity that can be analyzed statically
        in call expression syntax, but must also be checked dynamically when
        using dynamic function application via `apply`
      - Note also that other syntaxes that bind local variables will transform
        into procedures during syntax expansion
  }];

  let arguments = (ins StrAttr:$name,
                       TypeAttr:$type,
                       BoolAttr:$hasRestParam,
                       Variadic<HeavyValue>:$captures);
  // $result is the instance of the closure/"function object"
  let results = (outs HeavyValue:$result);

  let builders = [
    OpBuilder<"::mlir::OpBuilder&, ::mlir::OperationState&,"
              "::llvm::StringRef Name, uint32_t Arity, bool HasRestParam,"
              "::llvm::ArrayRef<::mlir::Value> Captures">
  ];

  let extraClassDeclaration = [{
    // Hooks for FunctionLike
    unsigned getNumFuncArguments() {
      return FunctionLike::getType().getInputs().size();
    }
    unsigned getNumFuncResults() { return 1; }
  }];
}

def heavy_LiteralOp : HeavyOp<"literal", [NoSideEffect]> {
  let summary = "literal";
  let description = [{
    Literals in scheme can be primitives or quoted scheme values that are
    known statically and should be immutable.
    For starters this should simply lower to calling the corresponding
    factory function to dynamically allocate the object and return a
    run-time heavy::Value*.
  }];

  let arguments = (ins HeavyValueAttr:$input);
  let results   = (outs HeavyValue:$result);
  let builders = [
    OpBuilder<"::mlir::OpBuilder&, ::mlir::OperationState&, ::heavy::Value*">
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
    OpBuilder<"::mlir::OpBuilder&, ::mlir::OperationState&,"
              "::mlir::Value binding, ::mlir::Value input">
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

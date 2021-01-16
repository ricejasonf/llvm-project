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
include "mlir/Interfaces/SideEffectInterfaces.td"

def HeavyDialect : Dialect {
  let name = "heavy";
}

def HeavyValue : OpaqueType<"heavy", "HeavyValue",
                       "HeavyValue represents a heavy::Value*">;

def HeavyValueAttr : Attr<CPred<"::heavy::isConvertibleToValue($_self)">,
                          "Any Attribute convertible to heavy::Value"> {
  let storageType = "::mlir::Attribute";
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

  let arguments = (ins HeavyValue:$fn, Variadic<HeavyValue>:$args);
  let results   = (outs HeavyValue:$result);
  let builders = [
    OpBuilder<"::mlir::OpBuilder&, ::mlir::OperationState&,"
              "::mlir::Value, ArrayRef<::mlir::Value>">
  ];
}

def heavy_BindingOp : HeavyOp<"binding"> {
  let summary = "binding";
  let description = [{
    All DefineOps have initializer expressions that are
    wrapped in a BindingOp to create a mutable reference. In the
    absence of mutation or multiple captures, the binding may be
    elided and refer directly to the result of the initializer
    expression. A binding denotes potential shared ownership and is
    handled as such during garbage collection.
  }];

  let arguments = (ins HeavyValue:$input);
  let results   = (outs HeavyValue:$result);
  let builders = [
    OpBuilder<"::mlir::OpBuilder&, ::mlir::OperationState&, ::mlir::Value">
  ];
}

def heavy_DefineOp : HeavyOp<"define"> {
  let summary = "define";
  let description = [{
    This initializes top level and local variables.
    The result of the expression is always Undefined.
  }];

  let arguments = (ins HeavyValue:$input);
  let results   = (outs HeavyValue:$result);
  let builders = [
    OpBuilder<"::mlir::OpBuilder&, ::mlir::OperationState&, ::mlir::Value">
  ];
}

/*
def heavy_ProcedureOp : HeavyOp<"procedure", [FunctionLike]> {
  let summary = "procedure";
  let description = [{
    Procedures represent function objects that have
    two significant properties:
      - Variables that are lexically defined in a parent scope are implicitly
        captured if they appear in the procedures scope. 
      - The object is callable with an arity that can be analyzed statically
        in call expression syntax, but must also be checked dynamically when
        using dynamic function application via `apply`
      - Note also that other syntaxes that bind local variables will transform
        into procedures during syntax expansion
  }];

  let arguments = (ins ???);
  let result = (outs ???);
}
*/

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


#endif

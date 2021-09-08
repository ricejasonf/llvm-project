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

def heavy_ApplyOp : HeavyOp<"apply", [ReturnLike, Terminator]> {
  let summary = "apply";
  let description = [{
    ApplyOp represents a function call whose results are passed to
    a continuation. The initCont region should create a
    closure object to be used as the continuation, or the region
    should be empty in the case of non-tail calls.
  }];
  let regions   = (region AnyRegion:$initCont);
  let arguments = (ins HeavyValue:$fn, Variadic<HeavyValue>:$args);
  let results   = (outs);

  let extraClassDeclaration = [{
    bool isTailPos() {
      return initCont().empty();
    }
  }];
}

def heavy_BindingOp : HeavyOp<"binding", [Symbol]> {
  let summary = "binding";
  let description = [{
    BindingOp creates a value wrapper (heavy::Binding) that allows
    shared ownership of a reference. This has meaning with SetOp,
    and in the absence of such mutation, the binding may be
    elided and the result can refer directly to the enclosed value.
  }];

  let arguments = (ins HeavyValue:$input);
  let results   = (outs HeavyValue:$result);
  let builders = [
    OpBuilder<"::mlir::OpBuilder&, ::mlir::OperationState&, "
              "::mlir::Value input">
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

def heavy_CommandOp : HeavyOp<"command"> {
  let summary = "command";
  let description = [{
    Command is a node in a sequence that represents
    a single expression whose domain and effects
    would be discarded.

    Currently this is only used for top level expressions.
  }];

  let regions = (region AnyRegion:$body);
  let extraClassDeclaration = [{
    Block* addEntryBlock() {
      assert(body().empty() && "CommandOp already has an entry block");
      Block* entry = new Block();
      body().push_back(entry);
      return entry;
    }
  }];

  let arguments = (ins /*none*/);
  let results   = (outs /*none*/);
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
    a symbol. If there is no initializer then it is external.
  }];

  let regions = (region AnyRegion:$initializer);
  let extraClassDeclaration = [{
    bool isExternal() { return initializer().empty(); }
    Block* addEntryBlock() {
      assert(initializer().empty() && "GlobalOp already has an entry block");
      Block* entry = new Block();
      initializer().push_back(entry);
      return entry;
    }
  }];

  let arguments = (ins StrAttr:$sym_name);
  let results   = (outs HeavyValue:$result);
  let builders = [
    OpBuilder<"::mlir::OpBuilder&, ::mlir::OperationState&,"
              "StringRef Name">
  ];
}

def heavy_IfOp : HeavyOp<"if", []> {
  let summary = "if";
  let description = [{
    IfOp handles branching to one of two regions based on an
    input result. Neither regions may contain an ApplyOp.
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

def heavy_IfContOp : HeavyOp<"if_cont", [ReturnLike, Terminator]> {
  let summary = "if_cont";
  let description = [{
    IfContOp represents branching where one or more of the branches
    contain an ApplyOp so a continuation must be provided.
    The initCont region behaves the same as ApplyOp and should be
    empty if the operation is in tail position.
  }];

  let regions   = (region SizedRegion<1>:$thenRegion,
                          SizedRegion<1>:$elseRegion,
                          AnyRegion:$initCont);
  let arguments = (ins HeavyValue:$input);
  let results   = (outs);

  let extraClassDeclaration = [{
    bool isTailPos() {
      return initCont().empty();
    }
  }];
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

def heavy_PushContOp : HeavyOp<"push_cont", [ReturnLike, Terminator]> {
  let summary = "push_cont";
  let description = [{
    Creates a closure for a continuation object to be
    pushed to the continuation stack. This should only be
    used to terminate an operation's initCont region.
  }];

  let arguments = (ins StrAttr:$name,
                       Variadic<HeavyValue>:$captures);
  let results   = (outs);
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

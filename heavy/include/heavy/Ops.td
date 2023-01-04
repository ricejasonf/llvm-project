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
include "mlir/IR/RegionKindInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def HeavyDialect : Dialect {
  let name = "heavy";
}

// HeavyValueBase - Use for any type that is an alternative
//                  of Value and derives from ValueBase which
//                  implies the need for garbage collection.
class HeavyValueBase<string name> : Type<
  CPred<!strconcat("$_self.isa<::heavy::Heavy", name, "Ty>()")>,
  name, !strconcat("::heavy::", name)>;

def HeavyValue : Type<CPred<"$_self.isa<::heavy::HeavyValueTy>()">,
                  "value", "::heavy::Value">;

def HeavyValueAttr : Attr<CPred<"true">,
                          "Any Attribute convertible to heavy::Value"> {
  let storageType = "::heavy::HeavyValueAttr";
  let returnType = "::heavy::Value";
}

// Additional Types
def HeavyRest : HeavyValueBase<"Pair">; // For rest arguments.
def HeavyPair : HeavyValueBase<"Pair">;
def HeavySyntax : HeavyValueBase<"Syntax">;

def HeavyOpGen : Type<CPred<"$_self.isa<::heavy::HeavyOpGenTy>()">,
                  "OpGen", "::heavy::OpGen&">;
def HeavyMlirValue : Type<CPred<"$_self.isa<::heavy::HeavyMlirValueTy>()">,
                  "MlirValue", "::mlir::Value">;

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

def heavy_BindingOp : HeavyOp<"binding"> {
  let summary = "binding";
  let description = [{
    BindingOp creates a value wrapper (heavy::Binding) that allows
    shared ownership of a reference. This has meaning with SetOp,
    and in the absence of such mutation, the binding may be
    elided and the result can refer directly to the enclosed value.
  }];

  let arguments = (ins Optional<HeavyValue>:$input);
  let results   = (outs HeavyValue:$result);
  let builders  = [OpBuilder<(ins "::mlir::Value":$input)>];
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
    OpBuilder<(ins "::heavy::Builtin*":$builtinFn)>
  ];
}

def heavy_CommandOp : HeavyOp<"command", [IsolatedFromAbove]> {
  let summary = "command";
  let description = [{
    Command is a node in a sequence that represents
    a single expression whose domain and effects
    would be discarded.

    Currently this is only used for top level expressions.
  }];

  let regions = (region AnyRegion:$body);
  let extraClassDeclaration = [{
    mlir::Block* addEntryBlock() {
      assert(body().empty() && "CommandOp already has an entry block");
      mlir::Block* entry = new mlir::Block();
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
    OpBuilder<(ins "::mlir::Value":$a, "::mlir::Value":$b)>
  ];
}

def heavy_ContOp : HeavyOp<"cont", [ReturnLike, Terminator]> {
  let summary = "cont";
  let description = [{
    ContOp models calling the current continuation. It can take
    an arbitrary amount of inputs and represents the result(s) of
    an expression.
  }];

  let arguments = (ins Variadic<AnyTypeOf<[HeavyValue, HeavySyntax]>>:$args);
  let results   = (outs); // no result
}

def heavy_OpGenOp : HeavyOp<"opgen", [Terminator]> {
  let summary = "opgen";
  let description = [{
    OpGenOp compiles scheme into the HeavyScheme MLIR dialect.
    This operation is used for syntax code generation.
  }];

  let regions   = (region AnyRegion:$initCont);
  let arguments = (ins HeavyValue:$input);
  let results   = (outs);
  let extraClassDeclaration = [{
    bool isTailPos() {
      return initCont().empty();
    }
  }];
}

def heavy_GlobalOp : HeavyOp<"global", [Symbol, IsolatedFromAbove]> {
  let summary = "global";
  let description = [{
    This indicates a module top level variable providing
    a symbol. If there is no initializer then it is external.
  }];

  let regions = (region AnyRegion:$initializer);
  let extraClassDeclaration = [{
    bool isExternal() { return initializer().empty(); }
    mlir::Block* addEntryBlock() {
      assert(initializer().empty() && "GlobalOp already has an entry block");
      mlir::Block* entry = new mlir::Block();
      initializer().push_back(entry);
      return entry;
    }
  }];

  let arguments = (ins StrAttr:$sym_name);
  let results   = (outs HeavyValue:$result);
  let builders = [
    OpBuilder<(ins "::llvm::StringRef":$sym_name)>
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
  let builders  = [OpBuilder<(ins "::mlir::Value":$input)>];
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
    OpBuilder<(ins "::llvm::StringRef":$name,
              "::llvm::ArrayRef<::mlir::Value>":$captures)>
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
    OpBuilder<(ins "::heavy::Value":$input)>
  ];
}

def heavy_LoadClosureOp : HeavyOp<"load_closure"> {
  let summary = "load_closure";
  let description = [{
    Read an element from lambda's capture list.
  }];

  let arguments = (ins HeavyValue:$closure, UI32Attr:$index);
  let results   = (outs HeavyValue:$result);

  let builders = [
    OpBuilder<(ins "::mlir::Value":$closure, "uint32_t":$index)>
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
    OpBuilder<(ins "::llvm::StringRef":$name)>
  ];
}

def heavy_ExportOp : HeavyOp<"export", [Symbol, IsolatedFromAbove,
                                        NoTerminator]> {
  let summary = "export";
  let description = [{
    Hold ExportIdOp mappings of variable names to symbol names.
    This facilitates dynamic loading for a compiling context.
  }];
  let regions = (region AnyRegion:$body);
  let arguments = (ins StrAttr:$sym_name);
  let results   = (outs);
}

def heavy_ExportIdOp : HeavyOp<"export-id"> {
  let summary = "export-id";
  let description = [{
    Provide an identifier to associate with a symbol for a compiling context.
    All export-id operations are nested in a specially named FuncOp to be called
    by a compiling context when importing the module.
  }];
  let arguments = (ins FlatSymbolRefAttr:$symbolName, StrAttr:$id);
}

def heavy_MatchOp : HeavyOp<"match", [HasParent<"PatternOp">]> {
  let summary = "match";
  let description = [{
    As a component of pattern matching, this checks if a value
    is equal (in terms of `equal?`) to the input attribute.
    For scheme 'symbols', MatchOp will check for equivalent bindings.
    This attribute should be a leaf constant value.
  }];

  let arguments = (ins HeavyValueAttr:$val, HeavyValue:$input);
  let results   = (outs);

  let builders = [
    OpBuilder<(ins "::heavy::Value":$val, "::mlir::Value":$input)>
  ];
}

def heavy_MatchIdOp : HeavyOp<"match-id", [HasParent<"PatternOp">]> {
  let summary = "match-id";
  let description = [{
    Perform matching of a non-static value and input value such that they
    have the same binding or pointer address.
  }];

  let arguments = (ins HeavyValue:$val, HeavyValue:$input);
  let results   = (outs);

  let builders = [
    OpBuilder<(ins "::heavy::Value":$val, "::mlir::Value":$input)>
  ];
}

def heavy_MatchPairOp : HeavyOp<"match_pair", [HasParent<"PatternOp">]> {
  let summary = "match_pair";
  let description = [{
    As a component of pattern matching, this checks if a value
    is a pair and destructures it.
  }];

  let arguments = (ins HeavyValue:$input);
  let results   = (outs HeavyValue:$car, HeavyValue:$cdr);

  let builders = [
    OpBuilder<(ins "::mlir::Value":$input)>
  ];
}

def heavy_PatternOp : HeavyOp<"pattern", [HasParent<"SyntaxOp">]> {
  let description = [{
    PatternOp represents the pattern in a pattern/template
    pair as part of the syntax-rules transform specifier. The 
    region should contain matching operations on the input syntax,
    as well as ConsOps for the template, terminating with an OpGenOp
    on the constructed AST.
  }];

  let arguments = (ins);
  let results   = (outs);
  let regions   = (region SizedRegion<1>:$region);
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

def heavy_RenameOp : HeavyOp<"rename", [HasParent<"PatternOp">]> {
  let summary = "rename";
  let description = [{
    "Rename" a local variable by storing an opaque representation
    of its mlir::Value. The lifetime of this value expires with the
    MLIR Context so compilation of these operations should only be JIT.
    (Local syntax can never be exported)
  }];

  let arguments = (ins UI64Attr:$opaqueValue);
  let results   = (outs HeavyValue:$result);
  let builders  = [
    OpBuilder<(ins "::mlir::Value":$opaqueValue)>
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
    OpBuilder<(ins "::mlir::Value":$binding, "::mlir::Value":$input)>
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
    OpBuilder<(ins "::mlir::Value":$a, "::mlir::Value":$b)>
  ];
}

def heavy_SyntaxClosureOp : HeavyOp<"syntax_closure"> {
  let summary = "syntax_closure";
  let description = [{
    SyntaxClosureOp captures a syntax node with the current
    environment. This supports generating code during hygienic
    macro expansion.
  }];

  let arguments = (ins HeavyValue:$input);
  let results   = (outs HeavyValue:$result);
  let builders  = [
    OpBuilder<(ins "::mlir::Value":$input)>
  ];
}

def heavy_SyntaxOp: HeavyOp<"syntax", [IsolatedFromAbove, NoTerminator]> {
  let summary = "syntax";
  let description = [{
    SyntaxOp contains a linearly overloaded sequence of PatternOps that
    operate on the input AST.
    Any PatternOp that guarantees it cannot match can be removed.
    Any PatternOp that guarantees a match statically can have subsequent
    matches removed.
    A SyntaxOp must contain at least one PatternOp.
    Semantics are implied for chaining and failure to find a match.
  }];

  let regions   = (region SizedRegion<1>:$region);
  let arguments = (ins);
  let results   = (outs HeavySyntax:$result);
  let builders  = [OpBuilder<(ins)>];
}

def heavy_UndefinedOp : HeavyOp<"undefined", [NoSideEffect]> {
  let summary = "undefined";
  let description = [{
    This is a singleton literal for heavy::Undefined
  }];

  let arguments = (ins);
  let results   = (outs HeavyValue:$result);
  let builders = [
    OpBuilder<(ins)>
  ];
}

#endif

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
  CPred<!strconcat("mlir::isa<::heavy::Heavy", name, "Ty>($_self)")>,
  name, !strconcat("::heavy::Heavy", name, "Ty")>;

def HeavyValue : Type<CPred<"mlir::isa<::heavy::HeavyValueTy>($_self)">,
                  "value", "::heavy::HeavyValueTy">;

def HeavyValueAttr : AttrDef<HeavyDialect, "HeavyValue"> {
  let attrName = "heavy.value_attr";
  let storageType = "::heavy::HeavyValueAttr";
  let parameters = (ins "StringAttr":$expr);
  let genAccessors = 0;
}

// Additional Types
def HeavyContext : HeavyValueBase<"Context">;
def HeavyValueRefs : HeavyValueBase<"ValueRefs">; // For arguments array.
def HeavyRest : HeavyValueBase<"Rest">; // For rest arguments.
def HeavyPair : HeavyValueBase<"Pair">;
def HeavySyntax : HeavyValueBase<"Syntax">;
def HeavyProcedure : HeavyValueBase<"Procedure">;

def HeavyType : AnyTypeOf<[HeavyValue,
                           HeavyPair,
                           HeavyProcedure]>;

// If !heavy.value_refs is a function argument, the arguments should
// only be (!heavy.context, !heavy.value_refs).

def HeavyOpGen : Type<CPred<"mlir::isa<::heavy::HeavyOpGenTy>($_self)">,
                  "OpGen", "::heavy::OpGen&">;
def HeavyMlirValue : Type<CPred<"mlir::isa<::heavy::HeavyMlirValueTy>($_self)">,
                  "MlirValue", "::mlir::Value">;

class HeavyOp<string mnemonic, list<Trait> traits = []> :
    Op<HeavyDialect, mnemonic, traits>;

def heavy_ApplyOp : HeavyOp<"apply", [Terminator]> {
  let summary = "apply";
  let description = [{
    ApplyOp represents a function call whose results are passed to
    a continuation.

    An argument of type !heavy.value_refs represents splicing all
    of the current call arguments with arguments in that position.
  }];

  let arguments = (ins AnyTypeOf<[HeavyValue, HeavyProcedure]>:$fn,
    Variadic<AnyTypeOf<[HeavyValue, HeavyRest, HeavyValueRefs]>>:$args);
  let results   = (outs);
}

def heavy_BindingOp : HeavyOp<"binding"> {
  let summary = "binding";
  let description = [{
    BindingOp creates a value wrapper (heavy::Binding) that allows
    shared ownership of a reference. This has meaning with SetOp,
    and in the absence of such mutation, the binding may be
    elided and the result can refer directly to the enclosed value.
  }];

  let arguments = (ins Optional<AnyTypeOf<[HeavyValue,
                                          HeavyRest]>>:$input);
  let results   = (outs HeavyValue:$result);
  let builders  = [OpBuilder<(ins "::mlir::Value":$input)>];
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
      assert(getBody().empty() && "CommandOp already has an entry block");
      mlir::Block* entry = new mlir::Block();
      getBody().push_back(entry);
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

def heavy_ExpandPacksOp : HeavyOp<"expand_packs", []> {
  let description = [{
    Expand a list of packs from one or more subpatterns.
    This is equivalent to calling `map` on a list of packs
    with an input function that emits the output syntax
    as a list.
  }];

  let arguments = (ins HeavyValue:$cdr, Variadic<HeavyValue>:$packs);
  let results = (outs HeavyValue:$result);
  let regions = (region SizedRegion<1>:$body);

  let builders = [
    OpBuilder<(ins "::mlir::Value":$cdr,
                   "::mlir::ValueRange":$packs,
                   "std::unique_ptr<::mlir::Region>":$body)>
  ];
}

def heavy_OpGenOp : HeavyOp<"opgen", [Terminator]> {
  let summary = "opgen";
  let description = [{
    OpGenOp compiles scheme into the HeavyScheme MLIR dialect.
    This operation is used for syntax code generation.
  }];

  let arguments = (ins HeavyValue:$input);
  let results   = (outs);
}

def heavy_GlobalOp : HeavyOp<"global", [Symbol, IsolatedFromAbove]> {
  let summary = "global";
  let description = [{
    This indicates a module top level variable providing
    a symbol. If there is no initializer then it is external.
  }];

  let regions = (region AnyRegion:$initializer);
  let extraClassDeclaration = [{
    bool isExternal() { return getInitializer().empty(); }
    mlir::Block* addEntryBlock() {
      assert(getInitializer().empty() && "GlobalOp already has an entry block");
      mlir::Block* entry = new mlir::Block();
      getInitializer().push_back(entry);
      return entry;
    }
  }];

  let arguments = (ins StrAttr:$sym_name);
}

def heav_GlobalSyntaxOp : HeavyOp<"global_syntax",
                                  [Symbol, IsolatedFromAbove]> {
  let description = [{
    GlobalSyntaxOp denotes a module top level syntax object
    that is initialized at compile time.
    If there is not initializer then it is external.
  }];

  let regions = (region AnyRegion:$initializer);
  let extraClassDeclaration = [{
    bool isExternal() { return getInitializer().empty(); }
    mlir::Block* addEntryBlock() {
      assert(getInitializer().empty() && "GlobalOp already has an entry block");
      mlir::Block* entry = new mlir::Block();
      getInitializer().push_back(entry);
      return entry;
    }
  }];

  let arguments = (ins StrAttr:$sym_name);
}

def heavy_IfOp : HeavyOp<"if", []> {
  let summary = "if";
  let description = [{
    IfOp handles branching to one of two regions based on an
    input result.
  }];

  let regions   = (region SizedRegion<1>:$thenRegion,
                          SizedRegion<1>:$elseRegion);
  let arguments = (ins HeavyValue:$input);
  let results   = (outs HeavyValue:$result);

  let builders  = [OpBuilder<(ins "::mlir::Value":$input)>];
}

def heavy_IfContOp : HeavyOp<"if_cont", [ReturnLike, Terminator]> {
  let summary = "if_cont";
  let description = [{
    IfContOp represents branching where one or more of the branches
    contain an ApplyOp so a continuation must be provided.
  }];

  let regions   = (region SizedRegion<1>:$thenRegion,
                          SizedRegion<1>:$elseRegion);
  let arguments = (ins HeavyValue:$input);
  let results   = (outs);
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

def heavy_LiteralOp : HeavyOp<"literal", [Pure]> {
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

def heavy_LoadRefOp : HeavyOp<"load_ref"> {
  let summary = "load_ref";
  let description = [{
    Read an element from a ValueRefs object which can be
    the context's callee capture list or a variadic argument
    list which is primarily used for continuations.
  }];

  let arguments = (ins AnyTypeOf<[HeavyContext, HeavyValueRefs]>:$value_refs,
                       UI32Attr:$index);
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

def heavy_LoadModuleOp : HeavyOp<"load_module"> {
  let summary = "load_module";
  let description = [{
    Load a module by its mangled name and initialize it.
    This also enables the creation of a runtime dependency graph
    for exporting to bytecode files.
  }];

  let arguments = (ins FlatSymbolRefAttr:$name);
  let results   = (outs);

  let builders = [
    OpBuilder<(ins "::llvm::StringRef":$name)>
  ];
}

def heavy_ExportIdOp : HeavyOp<"export-id"> {
  let summary = "export-id";
  let description = [{
    Provide an identifier to associate with a target identifier $id
    with a mangled name $symbolName.
  }];
  let arguments = (ins FlatSymbolRefAttr:$symbolName, StrAttr:$id);
}

def heavy_MatchOp : HeavyOp<"match", []> {
  let summary = "match";
  let description = [{
    As a component of pattern matching, this checks if a value
    is equal (in terms of `equal?`) to the input attribute.
    The attribute should be a leaf constant value.
  }];

  let arguments = (ins HeavyValueAttr:$val, HeavyValue:$input);
  let results   = (outs);

  let builders = [
    OpBuilder<(ins "::heavy::Value":$val, "::mlir::Value":$input)>
  ];
}

def heavy_MatchIdOp : HeavyOp<"match-id", []> {
  let summary = "match-id";
  let description = [{
    Perform matching of a non-literal value and input value such that they
    have the same binding or pointer address.
    (Id means identifier.)
  }];

  let arguments = (ins HeavyValue:$val, HeavyValue:$input);
  let results   = (outs);

  let builders = [
    OpBuilder<(ins "::heavy::Value":$val, "::mlir::Value":$input)>
  ];
}

def heavy_MatchPairOp : HeavyOp<"match_pair", []> {
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

def heavy_MatchTailOp : HeavyOp<"match_tail", []> {
  let description = [{
    Destructure the last N lengthed list of the tail of the input list.
    The last cdr is counted in the length.
    (ie () or any element for improper list.)
    The tail will be matched against a pattern.
    If the tail is a pair, its address will serve as a sentinel
    for terminating the matching of elements against the subpattern.
  }];

  let arguments = (ins UI32Attr:$length,
                       HeavyValue:$input);
  let results   = (outs HeavyValue:$tail);

  let builders = [
    OpBuilder<(ins "uint32_t":$length,
                   "::mlir::Value":$input)>
  ];
}

def heavy_MatchVectorOp : HeavyOp<"match_vector", []> {
  let description = [{
    Destructure a vector literal with a single, optional subpattern.
    The attributes $head and $tail are the number of elements before
    and after the subpattern P_e. If P_e does not exist then $tail is 0.
    The number of results are $head + $tail + 1 or $head if $tail is 0.
    The results are ordered (H... E_e T....) where H and T are the
    corresponding head and tail elements and E_e is a list of the
    elements to match against the subpattern.
  }];

  let arguments = (ins UI32Attr:$head,
                       UI32Attr:$tail,
                       HeavyValue:$input);
  let results   = (outs Variadic<HeavyValue>:$results);

  let builders = [
    OpBuilder<(ins "uint32_t":$head,
                   "uint32_t":$tail,
                   "::mlir::Value":$input)>
  ];
}

def heavy_MatchArgsOp : HeavyOp<"match_args"> {
  let description = [{
    Destructure a dynamically lengthed array (ValueRefs) using the type
    of a function. This matches arity and types of its arguments
    and should support rest arguments.

    (Use for implementing case-lambda.)
  }];

  let arguments = (ins HeavyValueRefs:$valueRefs);
  let results   = (outs Variadic<AnyTypeOf<[HeavyValue,
                                            HeavyRest]>>:$results);

  let builders = [
    OpBuilder<(ins "::mlir::FunctionType":$functionType,
                   "::mlir::Value":$valueRefs)>
  ];
}

def heavy_MatchTypeOp : HeavyOp<"match_type", []> {
  let description = [{
    Perform dynamic cast resulting in a value with a specified type.
    Like other Match operations this will raise an error unless it
    appears in the body of a PatternOp.
  }];

  let arguments = (ins HeavyValue:$arg);
  let results   = (outs HeavyType:$result);
}

def heavy_SubpatternOp : HeavyOp<"subpattern", []> {
  let description = [{
    Match a list of elements over a subpattern that
    precedes the `...` ellipsis. Each pattern variable within
    the list is returned as a pack list that may only be expanded
    by a subtemplate.

    We stop at the elements that occur in the list after the
    subpattern by passing the tail value as a sentinel.
    It checks by the address of the pair or any cdr that is
    not a pair.
  }];

  let arguments = (ins HeavyValue:$input, HeavyValue:$tail);
  let results   = (outs Variadic<HeavyValue>:$packs);
  let regions   = (region SizedRegion<1>:$body);

  let builders = [
    OpBuilder<(ins "::mlir::Value":$input,
                   "::mlir::Value":$tail,
                   "std::unique_ptr<::mlir::Region>":$body,
                   "unsigned":$num_packs)>
  ];
}

def heavy_ResolveOp : HeavyOp<"resolve", [Terminator]> {
  let description = [{
    Terminate the region of a SubpatternOp or ExpandPacksOp.
    SubpatternOp takes the created pattern variables as $args.
    ExpandPacksOp creates the generated list as its single
    argument.
  }];

  let arguments = (ins Variadic<HeavyValue>:$args);
  let results   = (outs);
}

def heavy_PatternOp : HeavyOp<"pattern", []> {
  let description = [{
    PatternOp represents a pattern/template pair as part of the
    syntax-rules transform specifier. Operations in its region are
    evaluated in order. When a match operation fails within its region,
    evaluation will jump to the successor operation of the containing
    PatternOp (which is likely another pattern or error.)

    Successfully matched patterns will likely generate and compile a
    new scheme form via ConsOps and terminating with an OpGenOp.
    Alternatively they may terminate with an ErrorOp in the case
    of `syntax-error`.
  }];

  let arguments = (ins);
  let results   = (outs);
  let regions   = (region SizedRegion<1>:$region);
}

def heavy_PushContOp : HeavyOp<"push_cont", []> {
  let summary = "push_cont";
  let description = [{
    Creates a closure for a continuation object to be
    pushed to the continuation stack. This will typically
    precede an ApplyOp or other "non-tail" operation.
  }];

  let arguments = (ins StrAttr:$name,
                       Variadic<HeavyValue>:$captures);
  let results   = (outs);
}

def heavy_RenameOp : HeavyOp<"rename"> {
  let summary = "rename";
  let description = [{
    Capture a value and "rename" it using an identifier to use in the
    scope of a syntax object. These lookups can be shadowed by the
    environment created during template expansion (as I understand it.)

    The $capture mlir::Value which may belong to a parent scope is
    used to generate a localized value in the context of the template
    expansion.
  }];

  let arguments = (ins StrAttr:$id,
        AnyTypeOf<[HeavyValue, HeavySyntax]>:$capture);
  let results   = (outs HeavyValue:$result);
  let builders = [
    OpBuilder<(ins "::llvm::StringRef":$id, "::mlir::Value":$capture)>
  ];
}

def heavy_RenameGlobalOp : HeavyOp<"rename_global"> {
  let summary = "rename_global";
  let description = [{
    Similar to RenameOp, this RenameGlobalOp represents the possibility of
    expanding the use of a global by generating a LoadGlobalOp in the context
    of the template expansion.

    Because globals can be defined lazily within the module, it is possible
    that this operation could be generated when no global of the corresponding
    name was defined. In this case the RenameOp can be removed.
  }];

  let arguments = (ins StrAttr:$id, StrAttr:$sym);
  let results   = (outs HeavyValue:$result);
  let builders = [
    OpBuilder<(ins "::llvm::StringRef":$id, "::llvm::StringRef":$sym)>
  ];
}

def heavy_RenameEnvOp : HeavyOp<"rename_env"> {
  let description = [{
    RenameEnvOp creates a new EnvFrame with $bindings
    which serves as the tail of a new environment stack
    with the most local EnvFrame from $env on top.

    The local EnvFrame holds the local stack so any syntax closure
    created will be topped with any newly added local variables
    or EnvFrames automatically.
  }];

  let arguments = (ins HeavyValue:$env, Variadic<HeavyValue>:$bindings);
  let results   = (outs HeavyValue: $result);
  let builders = [
    OpBuilder<(ins "::mlir::Value":$env,
                   "::llvm::ArrayRef<::mlir::Value>":$args)>
  ];
}

def heavy_SourceLocOp : HeavyOp<"source-loc"> {
  let summary = "source-loc";
  let description = [{
    Get the source location from a HeavyValue object.
    This is primarily used to propagate source locations
    of the inputs to pattern variables.
  }];

  let arguments = (ins HeavyValue:$arg);
  let results   = (outs HeavyValue:$result);
  let builders = [
    OpBuilder<(ins "::mlir::Value":$arg)>
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
    SyntaxClosureOp captures a syntax node with the given
    environment. This supports generating code during hygienic
    macro expansion.
  }];

  let arguments = (ins HeavyValue:$sourceVal, HeavyValue:$input,
                       HeavyValue:$env);
  let results   = (outs HeavyValue:$result);
  let builders  = [
    OpBuilder<(ins "::mlir::Value":$sourceVal, "::mlir::Value":$input,
                   "::mlir::Value":$env)>
  ];
}

def heavy_SyntaxOp: HeavyOp<"syntax", []> {
  let summary = "syntax";
  let description = [{
    SyntaxOp creates a syntax function object that is callable like
    a lambda but has the specific purpose of generating new
    operations in the current builder context.

    The argument $name refers to the function name.
  }];

  let arguments = (ins StrAttr:$name);
  let results = (outs HeavySyntax:$result);

  let builders = [
    OpBuilder<(ins "::llvm::StringRef":$name)>
  ];
}

def heavy_ToVectorOp : HeavyOp<"to_vector"> {
  let summary = "to_vector";
  let description = [{
    ToVectorOp converts a list into a vector.
    This is used for the simplification of quasiquote
    of vector literals.
  }];

  let arguments = (ins HeavyValue:$input);
  let results   = (outs HeavyValue:$result);
  let builders = [
    OpBuilder<(ins "::mlir::Value":$input)>
  ];
}

def heavy_VectorOp : HeavyOp<"vector"> {
  let summary = "vector";
  let description = [{
    VectorOp creates a heavy::Vector from its arguments.
  }];

  let arguments = (ins Variadic<HeavyValue>:$args);
  let results   = (outs HeavyValue: $result);
  let builders = [
    OpBuilder<(ins "::llvm::ArrayRef<::mlir::Value>":$args)>
  ];
}

def heavy_UndefinedOp : HeavyOp<"undefined", [Pure]> {
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

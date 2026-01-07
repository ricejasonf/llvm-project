// Dialect/Types.td - MLIR Dialect Type Definitions for HeavyScheme
//
// Copyright Jason Rice 2025
//
#ifndef HEAVY_DIALECT_TYPE_TD
#define HEAVY_DIALECT_TYPE_TD

include "mlir/IR/AttrTypeBase.td"

def Heavy_Unknown : TypeDef<Heavy_Dialect, "HeavyUnknown", []> {
  let mnemonic = "unknown";
  let description = [{
    Define the opaquest of types.

    This is primarily used for datums when constructing syntax.
  }];
}

// Opaque value type that is not syntax or binding.
def Heavy_Value : TypeDef<Heavy_Dialect, "HeavyValue", []> {
  let mnemonic = "value";
  let description = [{
    Define the opaque value type that is not a binding or syntax.
  }];
}

def Heavy_Empty : TypeDef<Heavy_Dialect, "HeavyEmpty", []> {
  let mnemonic = "empty";
  let description = [{
    The empty list type analogous to heavy::Empty.
    In scheme it is known by `null?`.
  }];
}

def Heavy_Pair : TypeDef<Heavy_Dialect, "HeavyPair", []> {
  let mnemonic = "pair";
  let description = [{
    The pair type for linked lists
  }];
}

def Heavy_Procedure : TypeDef<Heavy_Dialect, "HeavyProcedure", []> {
  let mnemonic = "procedure";
  let description = [{
    The closure type aka lambda
  }];
}

def Heavy_Vector : TypeDef<Heavy_Dialect, "HeavyVector", []> {
  let mnemonic = "vector";
  let description = [{
    The vector type -- a run-time lengthed tuple-like type
  }];
}

def Heavy_Undefined : TypeDef<Heavy_Dialect, "HeavyUndefined", []> {
  let mnemonic = "undefined";
  let description = [{
    The unspecified type the report is always talking about
  }];
}

// Not value types

def Heavy_Context : TypeDef<Heavy_Dialect, "HeavyContext", []> {
  let mnemonic = "context";
  let description = [{
    The context object represents the current continuation
    along with the state of the entire scheme run-time.
  }];
}

def Heavy_ValueRefs : TypeDef<Heavy_Dialect, "HeavyValueRefs", []> {
  let mnemonic = "value_refs";
  let description = [{
    Represent a mutable contiguous array view to values that are typically
    the arguments to the current function call.
  }];
}

def Heavy_Rest : TypeDef<Heavy_Dialect, "HeavyRest", []> { // For rest arguments.
  let mnemonic = "rest";
  let description = [{
    Represent the dynamically lengthed linked list of arguments that
    form the rest parameter.
  }];
}

def Heavy_Syntax : TypeDef<Heavy_Dialect, "HeavySyntax", []> {
  let mnemonic = "syntax";
  let description = [{
    Represent a syntax object that can appear as a callee
    in a syntax.
  }];
}

def Heavy_Binding : TypeDef<Heavy_Dialect, "HeavyBinding", []> {
  let mnemonic = "binding";
  let description = [{
    A binding is an indirection on a value that must be "unboxed"
    for use as a normal scheme value.
  }];
}

#endif // HEAVY_DIALECT_TYPE_TD

// HeavyDialectPasses.td - MLIR Dialect Pass Definitions for HeavyScheme
//
// Copyright Jason Rice 2025
//
#ifndef HEAVY_DIALECT_PASSES_TD
#define HEAVY_DIALECT_PASSES_TD

include "mlir/Pass/PassBase.td"

def StripGlobalBindingsPass : Pass<"heavy-strip-global-bindings-pass",
                                   "mlir::ModuleOp"> {
  let summary = "Strip bindings for global variables if they are never mutated";
  let description = [{
    If a binding object of a global variable is never mutated via the
    `set!` syntax, then we can elide the creation and unboxing of the binding.
  }];

  let dependentDialects = ["heavy::HeavyDialect"];
}

def StripLocalBindingsPass : Pass<"heavy-strip-local-bindings-pass",
                                  "mlir::ModuleOp"> {
  let summary = "Strip binding for local variables if they are never mutated";
  let description = [{
    If a binding is never mutated via the `set!` syntax after it is read
    for the first time, then we can elide the creation and unboxing of the
    binding.
  }];

  let dependentDialects = ["heavy::HeavyDialect"];
}

#endif  // HEAVY_DIALECT_PASSES_TD

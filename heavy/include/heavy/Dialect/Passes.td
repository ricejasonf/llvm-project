// HeavyDialectPasses.td - MLIR Dialect Pass Definitions for HeavyScheme
//
// Copyright Jason Rice 2025
//
#ifndef HEAVY_DIALECT_PASSES_TD
#define HEAVY_DIALECT_PASSES_TD

include "mlir/Pass/PassBase.td"

class HeavyModulePass<string Name> : PassBase<Name, "heavy::HeavyModulePass">;

def StripGlobalBindingsPass : HeavyModulePass<"heavy-strip-global-bindings"> {
  let summary = "Strip bindings for global variables if they are never mutated";
  let description = [{
    If a binding object of a global variable is never mutated via the
    `set!` syntax, then we can elide the creation and unboxing of the binding.
  }];

  let dependentDialects = ["heavy::HeavyDialect"];
}

def StripLocalBindingsPass : HeavyModulePass<"heavy-strip-local-bindings"> {
  let summary = "Strip binding for local variables if they are never mutated";
  let description = [{
    If a binding is never mutated via the `set!` syntax after it is read
    for the first time, then we can elide the creation and unboxing of the
    binding.
  }];

  let dependentDialects = ["heavy::HeavyDialect"];
}

#endif  // HEAVY_DIALECT_PASSES_TD

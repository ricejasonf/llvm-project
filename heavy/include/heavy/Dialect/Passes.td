// HeavyDialectPasses.td - MLIR Dialect Pass Definitions for HeavyScheme
//
// Copyright Jason Rice 2025
//
#ifndef HEAVY_DIALECT_PASSES_TD
#define HEAVY_DIALECT_PASSES_TD

include "mlir/Pass/PassBase.td"

def MarkImmutableGlobalsPass : Pass<"heavy-mark-immutable-globals-pass"> {
  let summary = "Mark mutability of all globals";
  let description = [{
    Discern the mutability of all globals in a scheme module by
    finding any uses of set!.
  }];

  let dependentDialects = ["heavy::HeavyDialect"];
}

#endif  // HEAVY_DIALECT_PASSES_TD

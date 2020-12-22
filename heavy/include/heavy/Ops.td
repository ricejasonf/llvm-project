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
  let returnType = "::heavy::Value";
}

class HeavyOp<string mnemonic, list<OpTrait> traits = []> :
    Op<HeavyDialect, mnemonic, traits>;

def heavy_ConstantOp : HeavyOp<"constant", [NoSideEffect]> {
  let summary = "constant";
  let description = [{
    Imagine that picture of a dog at the keyboard looking at the camera
    with that face that says "I have no idea what I am doing!".
    That is the state of my existence while writing this.
  }];

  let arguments = (ins HeavyValueAttr:$input);
  let results   = (outs HeavyValue);
}

#endif

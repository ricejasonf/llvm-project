//===------------------------ Passes.h --------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

namespace {
struct MarkImmutableGlobalsPass
    : public heavy::impl::MarkImmutableGlobalsPassBase {
  void runOnOperation() override;
};
}  // namespace


void MarkImmutabeGlobalsPass::runOnOperation() {
  mlir::Operation* Op = getOperation();
}

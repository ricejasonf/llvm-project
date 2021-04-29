//===- EvaluationStack.h -  -------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file defines heavy::EvalutionStack
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_HEAVY_EVALUATION_STACK_H
#define LLVM_HEAVY_EVALUATION_STACK_H

#include "heavy/Dialect.h"
#include "heavy/Source.h"
#include "heavy/Value.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/TrailingObjects.h"
#include <algorithm>
#include <cassert>
#include <utility>

#ifndef HEAVY_STACK_SIZE
#define HEAVY_STACK_SIZE 1024 * 1024
#endif
#if LLVM_ADDRESS_SANITIZER_BUILD
#define HEAVY_STACK_RED_ZONE_SIZE 1
#else
#define HEAVY_STACK_RED_ZONE_SIZE 0
#endif

namespace heavy {

class Context;

struct RedZoneByte { };

class StackFrame final : public llvm::TrailingObjects<StackFrame, Value,
                                                RedZoneByte> {
  friend class llvm::TrailingObjects<StackFrame, Value>;

  mlir::Operation* Op;
  SourceLocation CallLoc;

  size_t numTrailingObjects(OverloadToken<Value> const) const {
    return getArgCount(Op);
  }

  size_t numTrailingObjects(OverloadToken<RedZoneByte> const) const {
    return HEAVY_STACK_RED_ZONE_SIZE;
  }

public:
  StackFrame(mlir::Operation* Op, SourceLocation Loc)
    : Op(Op),
      CallLoc(Loc)
  { }

  ~StackFrame() {
    // "Zero fill" everything except the red zone bytes
    CallLoc = {};
    Value* Vs = getTrailingObjects<Value>();
    std::fill(&Vs[0], &Vs[getArgCount(Op)], nullptr);
    Op = nullptr;
  }

  static unsigned getArgCount(mlir::Operation* Op) {
    if (ApplyOp A = llvm::dyn_cast_or_null<ApplyOp>(Op)) {
      return A.args().size() + 1; // includes callee
    }
    return 0;
  }

  static size_t sizeToAlloc(mlir::Operation* Op) {
    // this could potentially include a "red zone" of trailing bytes
    return totalSizeToAlloc<Value, RedZoneByte>(
        getArgCount(Op), HEAVY_STACK_RED_ZONE_SIZE);
  }

  bool isInvalid() const {
    return Op == nullptr;
  }

  // Gets the size in bytes that we allocated
  // minus the red zone bytes
  unsigned getMemLength() const {
    return sizeToAlloc(Op) - HEAVY_STACK_RED_ZONE_SIZE;
  }

  // "instruction pointer" Op
  mlir::Operation* getOp() { return Op; }

  SourceLocation getCallLoc() { return CallLoc; }

  // Returns the value for callee or nullptr
  // if the StackFrame is invalid
  Value getCallee() const {
    if (getArgCount(Op) == 0) return nullptr;
    return getTrailingObjects<Value>()[0];
  }

  llvm::MutableArrayRef<heavy::Value> getArgs() {
    return llvm::MutableArrayRef<heavy::Value>(
        getTrailingObjects<Value>(), getArgCount(Op));
  }

  // Get the previous stack frame. This assumes that
  // the previous stack frame exists in valid memory
  // bounds immediately after the *this* StackFrame
  // including red zone bytes.
  // The client is responsible for ensuring that the
  // previous stack frame is valid and in bounds
  StackFrame* getPrevious() {
    char* PrevPtr = reinterpret_cast<char*>(this) + sizeToAlloc(Op);
    return reinterpret_cast<StackFrame*>(PrevPtr);
  }
};

// EvaluationStack
//
//      - performs setup and tear down of function
//        calls
//      - manages a stack for unwinding TODO
//      - grows downward in memory
class EvaluationStack {
  static_assert(HEAVY_STACK_SIZE > 0, "HEAVY_STACK_SIZE must be valid");
  heavy::Context& Context;
  std::vector<char> Storage;
  heavy::StackFrame* Top;

  heavy::StackFrame* getStartingPoint() {
    // The bottom, invalid frame is pushed on top of this.
    // This frame should never be accessed.
    return reinterpret_cast<heavy::StackFrame*>(&Storage.back());
  }

  void EmitStackSpaceError();


public:
  EvaluationStack(heavy::Context& C)
    : Context(C),
      Storage(HEAVY_STACK_SIZE, 0),
      Top(getStartingPoint())
  {
    // push an invalid StackFrame as the bottom
    push(nullptr, {});
  }

  StackFrame* push(mlir::Operation* Op, heavy::SourceLocation CallLoc) {
    unsigned ByteLen = StackFrame::sizeToAlloc(Op);

    char* CurPtr = reinterpret_cast<char*>(Top);
    char* NewPtr = CurPtr - ByteLen;
    if (NewPtr < &Storage.front()) {
      EmitStackSpaceError();
      return nullptr;
    }

    StackFrame* New = new (NewPtr) StackFrame(Op, CallLoc);
    Top = New;
    return New;
  }

  // Args here includes the callee
  StackFrame* push(mlir::Operation* Op, heavy::SourceLocation CallLoc,
                   llvm::ArrayRef<Value> Args) {
    assert(StackFrame::getArgCount(Op) == Args.size() &&
        "operation arity mismatch");
    StackFrame* Frame = push(Op, CallLoc);
    if (!Frame) return nullptr;

    if (!Args.empty()) {
      auto DestArgs = Frame->getArgs();
      for (unsigned i = 0; i < Args.size(); ++i) {
        DestArgs[i] = Args[i];
      }
    }

    return Frame;
  }

  void pop() {
    StackFrame* Prev = Top->getPrevious();
    Top->~StackFrame();
    Top = Prev;
  }

  StackFrame* top() {
    return Top;
  }
};

}

#endif

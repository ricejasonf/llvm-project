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

  heavy::Value Caller;
  unsigned ArgCount;
  SourceLocation CallLoc;

  size_t numTrailingObjects(OverloadToken<Value> const) const {
    return ArgCount;
  }

  size_t numTrailingObjects(OverloadToken<RedZoneByte> const) const {
    return HEAVY_STACK_RED_ZONE_SIZE;
  }

public:
  StackFrame(heavy::Value Caller, unsigned ArgCount, SourceLocation Loc)
    : Caller(Caller),
      ArgCount(ArgCount),
      CallLoc(Loc)
  { }

  ~StackFrame() {
    // "Zero fill" everything except the red zone bytes
    CallLoc = {};
    Value* Vs = getTrailingObjects<Value>();
    std::fill(&Vs[0], &Vs[ArgCount], nullptr);
    Caller = nullptr;
  }

  static size_t sizeToAlloc(unsigned ArgCount) {
    // this could potentially include a "red zone" of trailing bytes
    return totalSizeToAlloc<Value, RedZoneByte>(
        ArgCount, HEAVY_STACK_RED_ZONE_SIZE);
  }

  bool isInvalid() const {
    return Caller == nullptr;
  }

  // Gets the size in bytes that we allocated
  // minus the red zone bytes
  unsigned getMemLength() const {
    return sizeToAlloc(ArgCount) - HEAVY_STACK_RED_ZONE_SIZE;
  }

  // may return nullptr
  heavy::Value getCaller() {
    return Caller;
  }

  // "instruction pointer" Op for IR OpEval
  // may return nullptr
  mlir::Operation* getOp() {
    if (auto* Op = dyn_cast_or_null<mlir::Operation>(Caller)) {
      return Op;
    }
    return nullptr;
  }

  SourceLocation getCallLoc() { return CallLoc; }

  // Returns the value for callee or nullptr
  // if the StackFrame is invalid
  Value getCallee() const {
    if (ArgCount == 0) return nullptr;
    return getTrailingObjects<Value>()[0];
  }

  llvm::MutableArrayRef<heavy::Value> getArgs() {
    return llvm::MutableArrayRef<heavy::Value>(
        getTrailingObjects<Value>(), ArgCount);
  }

  // Get the previous stack frame. This assumes that
  // the previous stack frame exists in valid memory
  // bounds immediately after the *this* StackFrame
  // including red zone bytes.
  // The client is responsible for ensuring that the
  // previous stack frame is valid and in bounds
  StackFrame* getPrevious() {
    char* PrevPtr = reinterpret_cast<char*>(this) + sizeToAlloc(ArgCount);
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

  StackFrame* push_helper(heavy::Value Caller, unsigned ArgCount,
                   heavy::SourceLocation CallLoc) {
    unsigned ByteLen = StackFrame::sizeToAlloc(ArgCount);

    char* CurPtr = reinterpret_cast<char*>(Top);
    char* NewPtr = CurPtr - ByteLen;
    if (NewPtr < &Storage.front()) {
      EmitStackSpaceError();
      return nullptr;
    }

    StackFrame* New = new (NewPtr) StackFrame(Caller, ArgCount, CallLoc);
    Top = New;
    return New;
  }

public:
  EvaluationStack(heavy::Context& C)
    : Context(C),
      Storage(HEAVY_STACK_SIZE, 0),
      Top(getStartingPoint())
  {
    // push an invalid StackFrame as the bottom
    push_helper(Value(nullptr), 0, {});
  }

  // Args here includes the callee
  StackFrame* push(heavy::Value Caller, heavy::SourceLocation CallLoc,
                   llvm::ArrayRef<Value> Args) {
    StackFrame* Frame = push_helper(Caller, Args.size(), CallLoc);
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

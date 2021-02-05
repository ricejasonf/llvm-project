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
class Value;

struct RedZoneByte { };

class StackFrame final : public llvm::TrailingObjects<StackFrame, Value*,
                                                RedZoneByte> {
  friend class llvm::TrailingObjects<StackFrame, Value*>;

  // The first argument will be the callee
  // An ArgCount of zero will indicate the bottom most StackFrame
  unsigned ArgCount;
  SourceLocation CallLoc;

  size_t numTrailingObjects(OverloadToken<Value*> const) const {
    return ArgCount;
  }

  size_t numTrailingObjects(OverloadToken<RedZoneByte> const) const {
    return HEAVY_STACK_RED_ZONE_SIZE;
  }

public:
  StackFrame(unsigned ArgN, SourceLocation Loc)
    : ArgCount(ArgN),
      CallLoc(Loc)
  { }

  ~StackFrame() {
    // "Zero fill" everything except the red zone bytes
    ArgCount = 0;
    CallLoc = {};
    Value** Vs = getTrailingObjects<Value*>();
    std::fill(&Vs[0], &Vs[ArgCount], nullptr);
  }

  static size_t sizeToAlloc(unsigned ArgCount) {
    // this could potentially include a "red zone" of trailing bytes
    return totalSizeToAlloc<Value*, RedZoneByte>(
        ArgCount, HEAVY_STACK_RED_ZONE_SIZE);
  }

  bool isInvalid() const {
    return ArgCount > 0;
  }

  // Gets the size in bytes that we allocated
  // minus the red zone bytes
  unsigned getMemLength() const {
    return sizeToAlloc(ArgCount) - HEAVY_STACK_RED_ZONE_SIZE;
  }

  SourceLocation getCallLoc() const {
    return CallLoc;
  }

  // Returns the value for callee or nullptr
  // if the StackFrame is invalid
  Value* getCallee() const {
    if (isInvalid()) return nullptr;
    return getTrailingObjects<Value*>()[0];
  }

  void setCallee(heavy::Value* X) {
    if (isInvalid()) return;
    getTrailingObjects<Value*>()[0] = X;
  }

  llvm::MutableArrayRef<heavy::Value*> getArgs() {
    return llvm::MutableArrayRef<heavy::Value*>(
        getTrailingObjects<Value*>() + 1, ArgCount - 1);
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

public:
  EvaluationStack(heavy::Context& C)
    : Context(C),
      Storage(HEAVY_STACK_SIZE, 0),
      Top(getStartingPoint())
  {
    // push an invalid StackFrame as the bottom
    push(0, {});
  }

  // ArgCount includes the callee
  // The client is responsible for setting the argument values
  // Returns the new StackFrame* or nullptr on error
  StackFrame* push(unsigned ArgCount, SourceLocation CallLoc) {
    unsigned ByteLen = StackFrame::sizeToAlloc(ArgCount);

    char* CurPtr = reinterpret_cast<char*>(Top);
    char* NewPtr = CurPtr - ByteLen;
    if (CurPtr <= NewPtr) {
      EmitStackSpaceError();
      return nullptr;
    }

    return new (NewPtr) StackFrame(ArgCount, CallLoc);
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

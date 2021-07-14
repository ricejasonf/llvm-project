//===- ContinuationStack.h -  -------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file defines heavy::ContinuationStack
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_HEAVY_CONTINUATION_STACK_H
#define LLVM_HEAVY_CONTINUATION_STACK_H

#include "heavy/Value.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/TrailingObjects.h"
#include <algorithm>
#include <cassert>
#include <utility>

#ifndef HEAVY_STACK_SIZE
#define HEAVY_STACK_SIZE 1024 * 1024
#endif

namespace heavy {

class Context;


// ContinuationStack
//      - Stores the continuations that are to be called in
//        order top to bottom
//      - Grows downward in memory
//      - Functions with non-tail calls are responsible for
//        pushing to the continuation stack (via Context)
//      - Functions are responsible for writing results via
//        Cont et al
class ContinuationStack {
  static_assert(HEAVY_STACK_SIZE > 0, "HEAVY_STACK_SIZE must be valid");
  heavy::Context& Context;
  std::vector<char> Storage;
  llvm::SmallVector<Value, 8> ApplyArgs; // includes the callee
  heavy::Lambda* Top;
  heavy::Lambda* Bottom; // not a valid ValueBase

  // Returns a pointer to an invalid ValueBase
  // used for Bottom and the initial Top
  heavy::Lambda* getStartingPoint() {
    return allocate(1);
  }

  void EmitStackSpaceError();

  Lambda* allocate(size_t size) {
    uintptr_t Cur = reinterpret_cast<uintptr_t>(Top);
    uintptr_t New = Cur - size;
    unsigned AlignmentPadding = New % alignof(Lambda);
    New -= AlignmentPadding;
    char* NewPtr = reinterpret_cast<char*>(New);
    if (NewPtr < &Storage.front()) {
      EmitStackSpaceError();
      return nullptr;
    }
    return reinterpret_cast<Lambda*>(NewPtr);
  }

  // PopCont
  //    - Pops the topmost Lambda zeroing its memory
  //      to prevent weird stuff
  void PopCont() {
    char* begin = reinterpret_cast<char*>(Top);
    char* end = begin + Top->getObjectSize();
    std::fill(begin, end, 0);
    Top = reinterpret_cast<Lambda*>(end);
  }

public:
  ContinuationStack(heavy::Context& C)
    : Context(C),
      Storage(HEAVY_STACK_SIZE, 0),
      ApplyArgs(1),
      Top(getStartingPoint()),
      Bottom(Top)
  {
    ApplyArgs[0] = Bottom;
  }

  ContinuationStack(ContinuationStack const&) = delete;

  heavy::Value Run(heavy::Context& Context) {
    while (Value Callee = ApplyArgs[0]) {
      if (Callee == Bottom) break;
      ValueRefs Args = ValueRefs(ApplyArgs).drop_front();
      switch (Callee.getKind()) {
      case ValueKind::Lambda: {
        Lambda* L = cast<Lambda>(Callee);
        L->call(Context, Args);
        if (L == Top) PopCont();
        break;
      }
      case ValueKind::Builtin: {
        Builtin* F = cast<Builtin>(Callee);
        F->Fn(Context, Args);
        break;
      }
      default:
        llvm_unreachable(
            "TODO raise error for invalid operator to function call");
#if 0
        String* Msg = Context.CreateString(
          "invalid operator for call expression: ",
          getKindName(Callee.getKind())
        );
        return SetError(CallLoc, Msg, Callee);
#endif
      }
    }
    return ApplyArgs[1];
  }

  // PushCont
  //    - Creates and pushes a temporary closure to the stack
  void PushCont(heavy::Lambda::FunctionDataView FnData, ValueRefs Captures) {
    size_t size = Lambda::sizeToAlloc(FnData, Captures.size());

    void* Mem = allocate(size);
    if (!Mem) {
      llvm_unreachable("TODO catastrophic failure or something");
    }

    Lambda* New = new (Mem) Lambda(FnData, Captures);
    Top = New;
  }

  //  Apply
  //    - Prepares a call without affecting the stack
  //    - This can be used for tail calls or, when used
  //      in conjunction with PushCont, non-tail calls
  //    - Args should include the callee
  void Apply(ValueRefs Args) {
    std::fill(ApplyArgs.begin(), ApplyArgs.end(), nullptr);
    ApplyArgs.resize(Args.size() + 1);
    std::copy(Args.begin(), Args.end(), ApplyArgs.begin());
  }

  // Cont
  //    - Prepares a call to the topmost continuation
  //    - Args should not include the callee
  void Cont(ValueRefs Args) {
    std::fill(ApplyArgs.begin(), ApplyArgs.end(), nullptr);
    ApplyArgs.resize(Args.size() + 1);
    ApplyArgs[0] = Top;
    auto Itr = ApplyArgs.begin();
    ++Itr;
    std::copy(Args.begin(), Args.end(), Itr);
  }

  //  RestoreStack
  //    - Restores the stack from a String that was saved by CallCC
  void RestoreStack(heavy::String Buffer) {
    // clear the current stack
    llvm_unreachable("TODO");
  }

#if 0 // TODO move this to Builtins I think
  //  CallCC
  //    - The lambda, its captures, and the entire stack buffer
  //      must be saved as an object on the heap as a new lambda
  //      that when invoked restores the stack buffer.
  void CallCC(Value InputProc) {
    char* begin = reinterpret_cast<char*>(Top);
    char* end = &(Storage.back());
    size_t size = end - begin;

    // SavedStack is kept alive by the heavy::Lambda capture
    Value SavedStack = Context.CreateString(llvm::StringRef(begin, size));
    auto Fn = [this, SavedStack](Context& C, ValueRefs Args) {
      // TODO have this function unwind/wind the dynamic points

      // TODO Make a way to include the callee in the function's parameters
      //      Saving it twice isn't the end of the world
      //      heavy::String SavedStack = Callee->getCapture(0);
      this->RestoreStack(SavedStack);
      this->Cont(Args);
    };

    PushCont(InputProc);
    Cont(Context.CreateLambda(Fn, SavedStack));
  }
#endif
};

}

#endif

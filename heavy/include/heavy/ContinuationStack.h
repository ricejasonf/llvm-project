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
#include "llvm/ADT/ScopeExit.h"
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

// ContinuationStack
//      - CRTP base class to add the "run-time" functionality
//        to a context class
//      - Stores the continuations that are to be called in
//        order top to bottom
//      - Grows downward in memory
//      - Functions with non-tail calls are responsible for
//        pushing to the continuation stack
//      - Functions are responsible for writing results via
//        Cont(Args...) et al
template <typename Derived>
class ContinuationStack {
  static_assert(HEAVY_STACK_SIZE > 0, "HEAVY_STACK_SIZE must be valid");
  std::vector<char> Storage;
  llvm::SmallVector<Value, 8> ApplyArgs; // includes the callee
  heavy::Lambda* Top;
  heavy::Lambda* Bottom; // not a valid Lambda but still castable to Value

  bool DidCallContinuation = false; // debug info

  Derived& getDerived() {
    return *static_cast<Derived*>(this);
  }

  void PrintStackSize() {
    size_t size = reinterpret_cast<uintptr_t>(&(Storage.back())) -
                  reinterpret_cast<uintptr_t>(Top);
    llvm::errs() << "STACK SIZE: " << size << '\n';
  }

  // Returns a pointer to an invalid Lambda*
  // used for Bottom and the initial Top
  heavy::Lambda* getStartingPoint() {
    uintptr_t Start = reinterpret_cast<uintptr_t>(&Storage.back());
    unsigned AlignmentPadding = Start % alignof(Lambda);
    Start -= AlignmentPadding;
    Start -= sizeof(heavy::Value);
    return reinterpret_cast<Lambda*>(Start);
  }

  Lambda* allocate(size_t size) {
    uintptr_t Cur = reinterpret_cast<uintptr_t>(Top);
    uintptr_t New = Cur - size;
    unsigned AlignmentPadding = New % alignof(Lambda);
    New -= AlignmentPadding;
    char* NewPtr = reinterpret_cast<char*>(New);
    if (NewPtr < &Storage.front()) {
      getDerived().EmitStackSpaceError();
      return nullptr;
    }
    return reinterpret_cast<Lambda*>(NewPtr);
  }

  // PopCont
  //  - Note that a call to PushCont will invalidate
  //    the returned Lambda*
  Lambda* PopCont() {
    if (Top == Bottom) return Bottom;
    Lambda* OldTop = Top;
    char* begin = reinterpret_cast<char*>(Top);
    char* end = begin + Top->getObjectSize();
    Top = reinterpret_cast<Lambda*>(end);
    return OldTop;
  }

  void ApplyHelper(Value Callee, ValueRefs Args) {
    DidCallContinuation = true; // debug mode only
    if (Args.data() != ApplyArgs.data()) {
      ApplyArgs.resize(Args.size() + 1);
      std::copy(Args.begin(), Args.end(), ApplyArgs.begin() + 1);
    }
    ApplyArgs[0] = Callee;
  }

public:
  ContinuationStack()
    : Storage(HEAVY_STACK_SIZE, 0),
      ApplyArgs(1),
      Top(getStartingPoint()),
      Bottom(Top)
  {
    ApplyArgs[0] = Bottom;
  }

  ContinuationStack(ContinuationStack const&) = delete;

  heavy::Value getCallee() {
    return ApplyArgs[0];
  }

  // Yield
  //  - Breaks the run loop yielding a value to serve
  //    as the result.
  //    (ie for a possibly nested call to eval)
  void Yield(ValueRefs Results) {
    Apply(Bottom, Results);
  }
  void Yield(Value Result) {
    Yield(ValueRefs(Result));
  }

  // PushBreak
  //  - Schedules a yield to be called so any
  //    evaluation that occurs on top can finish
  void PushBreak() {
    PushCont([](Derived& C, ValueRefs Args) {
      Yield(Args);
    });
  }

  // Begins evaluation by calling what is set
  // in ApplyArgs
  heavy::Value Resume() {
    Derived& Context = getDerived();
    if (Context.CheckError()) return Undefined();

    while (Value Callee = ApplyArgs[0]) {
      if (Callee == Bottom) break;

      // debug mode only
      DidCallContinuation = false;

      ValueRefs Args = ValueRefs(ApplyArgs).drop_front();
      switch (Callee.getKind()) {
      case ValueKind::Lambda: {
        Lambda* L = cast<Lambda>(Callee);
        L->call(Context, Args);
        break;
      }
      case ValueKind::Builtin: {
        // TODO make the interface for calling Builtins
        //      consistent with Lambda
        Builtin* F = cast<Builtin>(Callee);
        F->Fn(Context, Args);
        break;
      }
      default:
        String* Msg = Context.CreateString(
          "invalid operator for call expression: ",
          getKindName(Callee.getKind())
        );
        return Context.SetError(Callee.getSourceLocation(), Msg, Callee);
      }

      // this means a C++ function was not written correctly
      assert(DidCallContinuation && "function failed to call continuation");
    }
    if (ApplyArgs.size() > 1) return ApplyArgs[1];
    return Undefined{};
  }

  // PushCont
  //    - Creates and pushes a temporary closure to the stack
  template <typename Fn>
  void PushCont(Fn const& F, ValueRefs Captures) {
    auto FnData = heavy::Lambda::createFunctionDataView(F);
    size_t size = Lambda::sizeToAlloc(FnData, Captures.size());

    void* Mem = allocate(size);
    if (!Mem) {
      llvm_unreachable("TODO catastrophic failure or something");
    }

    Lambda* New = new (Mem) Lambda(FnData, Captures);
    Top = New;
  }

  void PushCont(heavy::Value Callable) {
    PushCont([](Derived& Context, ValueRefs Args) {
      Lambda* Self = cast<Lambda>(Context.getCallee());
      heavy::Value Callable = Self->getCapture(0);
      Context.Apply(Callable, Args.drop_front());
    }, Callable);
  }

  //  Apply
  //    - Prepares a call without affecting the stack
  //    - This can be used for tail calls or, when used
  //      in conjunction with PushCont, non-tail calls
  void Apply(Value Callee, ValueRefs Args) {
    ApplyHelper(Callee, Args);
  }

  // Cont
  //    - Prepares a call to the topmost continuation
  //    - Args should not include the callee
  void Cont(ValueRefs Args) {
    ApplyHelper(PopCont(), Args);
  }
  void Cont(Value Arg) { Cont(ValueRefs(Arg)); }

  //  RestoreStack
  //    - Restores the stack from a String that was saved by CallCC
  void RestoreStack(heavy::String* Buffer) {
    char* begin = reinterpret_cast<char*>(Top);
    char* end = &(Storage.back());
    llvm::StringRef BufferView = Buffer->getView();
    std::fill(begin, end, char(0));
    std::copy(BufferView.begin(), BufferView.end(), begin);
  }

  //  CallCC
  //    - The lambda, its captures, and the entire stack buffer
  //      must be saved as an object on the heap as a new lambda
  //      that when invoked restores the stack buffer.
  void CallCC(Value InputProc) {
    Derived& Context = getDerived();
    char* begin = reinterpret_cast<char*>(Top);
    char* end = &(Storage.back());
    size_t size = end - begin;

    auto Fn = [this](Derived& Ctx, ValueRefs Args) -> Value {
      // TODO unwind/wind the dynamic points
      Lambda* Callee = cast<Lambda>(Ctx.getCallee());
      String* SavedStack = cast<String>(Callee->getCapture(0));
      this->RestoreStack(SavedStack);
      this->Cont(Args);
      return Undefined{};
    };

    // SavedStack is kept alive by the heavy::Lambda capture
    Value SavedStack = Context.CreateString(llvm::StringRef(begin, size));
    Value Proc = Context.CreateLambda(Fn, SavedStack);
    Apply(InputProc, Proc);
  }
};

}

#endif

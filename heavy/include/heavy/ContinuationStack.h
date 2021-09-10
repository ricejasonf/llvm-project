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
#include <initializer_list>
#include <utility>

#ifndef HEAVY_STACK_SIZE
#define HEAVY_STACK_SIZE 1024 * 1024
#endif

namespace heavy {
using CaptureList = std::initializer_list<Value>;

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

  class DWind {
  public:
    heavy::Vector* Vec = nullptr;
    DWind() = default;
    DWind(Vector* V) : Vec(V) { }
    DWind(Value V) : Vec(cast_or_null<Vector>(V)) { }
    operator Value() { return Value(Vec); }

    int getDepth() {
      if (!Vec) return 0;
      return cast<Int>(Vec->get(0));
    }
    Value getBeforeFn() { return Vec->get(1); }
    Value getAfterFn() { return Vec->get(2); }
    Value getParent() { return Vec->get(3); }

    bool operator==(DWind Other) const { return Vec == Other.Vec; }
  };

  DWind CurDW;

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
  Value PopCont() {
    if (Top == Bottom) return Bottom;
    assert(Top < Bottom && "top is out of bounds");
    Lambda* OldTop = Top;
    uintptr_t begin = reinterpret_cast<uintptr_t>(Top);
    unsigned size = Top->getObjectSize();
    uintptr_t end = begin + size;
    unsigned A = alignof(Lambda);
    unsigned AlignmentPadding = (A - (end % A)) % A;
    end += AlignmentPadding;
    Top = reinterpret_cast<Lambda*>(end);
    return Value(OldTop);
  }

  void ApplyHelper(Value Callee, ValueRefs Args) {
    assert(!DidCallContinuation &&
        "continuation should be specified only once");
    DidCallContinuation = true; // debug mode only
    if (Args.data() != ApplyArgs.data()) {
      ApplyArgs.resize(Args.size() + 1);
      std::copy(Args.begin(), Args.end(), ApplyArgs.begin() + 1);
    }
    ApplyArgs[0] = Callee;
  }

  // TraverseWindings
  //  - Allows escape procedures to call their
  //    dynamic-wind points when entering or exiting
  //    a "dynamic extent".
  //  - Do not modify CurDW in this function
  //  - Inspired by travel-to-point!
  void TraverseWindings(DWind Src, DWind Dest) {
#if HEAVY_DYNAMIC_WIND_DEBUG
    llvm::errs() << "Src  depth: " << Src.getDepth()
                 << " (Vector* " << reinterpret_cast<uintptr_t>(Src.Vec)
                 << ")\n";
    llvm::errs() << "Dest depth: " << Dest.getDepth()
                 << " (Vector* " << reinterpret_cast<uintptr_t>(Dest.Vec)
                 << ")\n";
#endif
    Derived& C = getDerived();
    if (Src == Dest) {
      C.Cont(Undefined());
      return;
    }
    if (Src.getDepth() < Dest.getDepth()) {
      C.PushCont([](Derived& C, ValueRefs) {
        DWind Dest = C.getCapture(0);
        llvm::errs() << "calling before\n";
        C.Apply(Dest.getBeforeFn(), llvm::None);
      }, CaptureList{Dest});
      C.TraverseWindings(Src, Dest.getParent());
    } else {
      C.PushCont([](Derived& C, ValueRefs) {
        DWind Src  = C.getCapture(0);
        DWind Dest = C.getCapture(1);
        C.TraverseWindings(Src.getParent(), Dest);
      }, CaptureList{Src, Dest});
      llvm::errs() << "calling after\n";
      C.Apply(Src.getAfterFn(), llvm::None);
    }
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
    assert(ApplyArgs[0] && "callee must not be null");
    return ApplyArgs[0];
  }
  ValueRefs getCaptures() {
    return cast<Lambda>(ApplyArgs[0])->getCaptures();
  }
  heavy::Value getCapture(unsigned I) {
    return cast<Lambda>(ApplyArgs[0])->getCapture(I);
  }
  heavy::Value getCurrentResult() {
    if (ApplyArgs.size() > 1) return ApplyArgs[1];
    return Undefined{};
  }

  // Yield
  //  - Breaks the run loop yielding a value to serve
  //    as the result.
  //    (ie for a possibly nested call to eval)
  void Yield(ValueRefs Results) {
    getDerived().Apply(Bottom, Results);
  }
  void Yield(Value Result) {
    getDerived().Yield(ValueRefs(Result));
  }

  // PushBreak
  //  - Schedules a yield to be called so any
  //    evaluation that occurs on top can finish
  void PushBreak() {
    Derived& C = getDerived();
    C.PushCont([](Derived& C, ValueRefs Args) {
      Yield(Args);
    });
  }

  bool isFinished() {
    return getCallee() == Bottom;
  }

  // Begins evaluation by calling what is set
  // in ApplyArgs
  void Resume() {
    Derived& Context = getDerived();

    while (Value Callee = getCallee()) {
      if (isFinished()) break;

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
        Context.RaiseError(Msg, Callee);
      }

      // this means a C++ function was not written correctly
      assert(DidCallContinuation && "function failed to call continuation");
    }
    DidCallContinuation = false;
  }

  // PushCont
  //    - Creates and pushes a temporary closure to the stack
  template <typename Fn>
  void PushCont(Fn const& F, llvm::ArrayRef<heavy::Value> Captures = {}) {
    auto FnData = heavy::Lambda::createFunctionDataView(F);
    size_t size = Lambda::sizeToAlloc(FnData, Captures.size());

    void* Mem = allocate(size);
    if (!Mem) {
      llvm_unreachable("TODO catastrophic failure or something");
    }

    Lambda* New = new (Mem) Lambda(FnData, Captures);
    Top = New;
    // cast to Value to check pointer alignment
    (void)Value(New);
  }

  void PushCont(heavy::Value Callable) {
    PushCont([](Derived& Context, ValueRefs Args) {
      heavy::Value Callable = Context.getCapture(0);
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
  void Cont() { Cont(Undefined()); }

  // ClearStack
  //  - Clear the stack effectively stopping execution after
  //    the current function call and preventing any resumption
  //    of the current continuation.
  void ClearStack() {
    Top = Bottom;
  }

  //  RestoreStack
  //    - Restores the stack from a String that was saved by CallCC
  void RestoreStack(heavy::String* Buffer) {
    llvm::StringRef BufferView = Buffer->getView();
    char* end = &(Storage.back());
    char* begin = end - BufferView.size();
    std::copy(BufferView.begin(), BufferView.end(), begin);
    Top = reinterpret_cast<Lambda*>(begin);
    (void)Value(Top);
  }

  //  CallCC
  //    - The lambda, its captures, and the entire stack buffer
  //      must be saved as an object on the heap as a new lambda
  //      that when invoked restores the stack buffer.
  void CallCC(Value InputProc) {
    Derived& C = getDerived();
    char* begin = reinterpret_cast<char*>(Top);
    char* end = &(Storage.back());
    size_t size = end - begin;
    Value SavedStack  = C.CreateString(llvm::StringRef(begin, size));
    Value SavedDW     = CurDW;

    Value Proc = C.CreateLambda([this](Derived& C, ValueRefs Args) {
      Value SavedStack  = C.getCapture(0);
      DWind SavedDW     = C.getCapture(1);
      Value SavedArgs   = C.CreateVector(Args);
      C.PushCont([](Derived& C, ValueRefs) {
        String* SavedStack  = cast<String>(C.getCapture(0));
        Vector* SavedArgs   = cast<Vector>(C.getCapture(1));
        C.RestoreStack(SavedStack);
        C.Cont(SavedArgs->getElements());
        return Value();
      }, CaptureList{SavedStack, SavedArgs});
      CurDW = SavedDW;
      C.TraverseWindings(this->CurDW, SavedDW);
      return Value();
    }, CaptureList{SavedStack, SavedDW});

    C.Apply(InputProc, Proc);
  }

  void DynamicWind(Value Before, Value Thunk, Value After) {
    Derived& C = getDerived();
    int Depth = CurDW.getDepth();
    Value NewDW = C.CreateVector(
        std::initializer_list<Value>{Int(Depth + 1), Before, After, CurDW});

    C.PushCont([this](Derived& C, ValueRefs) {
      Value Thunk = C.getCapture(0);
      Value After = C.getCapture(1);
      this->CurDW = C.getCapture(2);

      C.PushCont([](Derived& C, ValueRefs ThunkResults) {
        Value After = C.getCapture(0);

        C.PushCont([](Derived& C, ValueRefs) {
          ValueRefs ThunkResults = C.getCaptures();
          C.Cont(ThunkResults);
        }, /*Captures=*/ThunkResults); 
        C.Apply(After, {});
      }, CaptureList{After});
      C.Apply(Thunk, {});
    }, CaptureList{Thunk, After, NewDW});
    C.Apply(Before, {});
  }
};

}

#endif

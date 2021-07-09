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
  std::vector<Value> ApplyArgs; // includes the callee
  heavy::ValueBase* Top;
  heavy::ValueBase* Bottom; // not a valid ValueBase

  // Returns a pointer to an invalid ValueBase
  // used for Bottom and the initial Top
  heavy::ValueBase* getStartingPoint() {
    return reinterpret_cast<heavy::ValueBase*>(&Storage.back());
  }

  void EmitStackSpaceError();

  void* allocate(size) {
    char* CurPtr = reinterpret_cast<char*>(Top);
    char* NewPtr = CurPtr - size;
    if (NewPtr < &Storage.front()) {
      EmitStackSpaceError();
      return nullptr;
    }
    return NewPtr;
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

  void Run() {
    Value Callee = ApplyArgs[0]; 
    if (Callee == Bottom) return;
    ValueRefs Args = ValueRefs(ApplyArgs).drop_front()
    switch (Fn.getKind()) {
    case ValueKind::Lambda:
      Lambda* L = cast<Lambda>(Fn);
      L->call(Context, Args);
      if (L == Top) PopCont();
      break;
    case ValueKind::Builtin:
      Builtin* F = cast<Builtin>(Callee);
      F->call(Context, Args);
      break;
    default:
      llvm_unreachable(
          "TODO raise error for invalid operator to function call");
    }
  }

  // PushCont
  //    - Creates and pushes a temporary closure to the stack
  void PushCont(heavy::FunctionDataView FnData, ValueRefs Captures) {
    size_t size = Lambda::sizeToAlloc(FnData, Captures.size());

    void* Mem = allocate(size);
    if (!Mem) return nullptr;

    Lambda* New = new (Mem) Lambda(FnData, Captures);
    Top = New;
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

  //  Apply
  //    - Prepares a call without affecting the stack
  //    - This can be used for tail calls or, when used
  //      in conjunction with Push, non-tail calls
  void Apply(Value Fn, ValueRefs Args) {
    std::fill(ApplyArgs.begin(), ApplyArgs.end(), nullptr);
    ApplyArgs.resize(Args.size() + 1);
    ApplyArgs[0] = Fn;
    auto Itr = ApplyArgs.begin();
    ++Itr;
    std::copy(Args.begin(), Args.end(), Itr);
  }

  // Cont
  //    - Prepares a call to the topmost continuation
  void Cont(ValueRefs Args) {
    Apply(Top, Args);
  }

  //  RestoreStack
  //    - Restores the stack from a String that was saved by CallCC
  void RestoreStack(heavy::String Buffer) {
    // clear the current stack
    llvm_unreachable("TODO");
  }

  //  CallCC
  //    - The lambda, its captures, and the entire stack buffer
  //      must be saved as an object on the heap as a new lambda
  //      that when invoked restores the stack buffer.
  void CallCC(Value InputProc) {
    char* begin = reinterpret_cast<char*>(Top);
    char* end = &*(Storage.back());
    size_t end - begin;

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
};

}

#endif

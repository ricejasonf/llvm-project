//===--- CopyCollector.cpp - HeavyScheme GC Implementation -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implementations for heavy scheme CopyCollector.
//
//===----------------------------------------------------------------------===//

#include "heavy/Context.h"
#include "heavy/Heap.h"
#include "heavy/OpGen.h"
#include "heavy/Value.h"
#include "heavy/ValueVisitor.h"
#include "llvm/Support/Allocator.h"
#include <algorithm>

namespace heavy {
// CopyCollector
//  - Visit a Root node, copy the value to the current heap
//    and overwrite the heavy::Value with its replacement.
class CopyCollector : private ValueVisitor<CopyCollector, heavy::Value> {
  using AllocatorTy = llvm::BumpPtrAllocator;
  using Base = ValueVisitor<CopyCollector, heavy::Value>;
  friend class ValueVisitor<CopyCollector, heavy::Value>;

  AllocatorTy& NewHeap;
  AllocatorTy& OldHeap;
  llvm::AllocatorBase<AllocatorTy>& getAllocator() { return NewHeap; }
  // Track objects that are visited but not themselves collected.
  std::vector<void*> VisitedSpecials;

  // Return false if Special was already visited.
  bool markSpecialVisited(void* Ptr) {
    auto Itr = std::find(VisitedSpecials.begin(), VisitedSpecials.end(), Ptr);
    if (Itr != VisitedSpecials.end())
      return false;

    VisitedSpecials.push_back(Ptr);
    return true;
  }

  // Handle all of the types that are embedded in the pointer,
  // and, therefore do not require a heap allocation.
  // FIXME These are unreachable.
  heavy::Value VisitInt(heavy::Int V)               { return V; }
  heavy::Value VisitBool(heavy::Bool V)             { return V; }
  heavy::Value VisitChar(heavy::Char V)             { return V; }
  heavy::Value VisitEmpty(heavy::Empty V)           { return V; }
  heavy::Value VisitUndefined(heavy::Undefined V)   { return V; }
  heavy::Value VisitOperation(heavy::Operation* V)  { return V; }
  heavy::Value VisitContArg(heavy::ContArg* V)      { return V; }

  // String
  heavy::String* VisitString(heavy::String* String) {
    llvm::StringRef StringRef = String->getView();
    unsigned MemSize = String::sizeToAlloc(StringRef.size());
    void* Mem = heavy::allocate(getAllocator(), MemSize, alignof(heavy::String));
    return new (Mem) heavy::String(StringRef);
  }


  heavy::EnvEntry VisitEnvEntry(heavy::EnvEntry const& EnvEntry) {
    heavy::Value Value = Visit(EnvEntry.Value);
    heavy::String* MangledName
      = VisitString(EnvEntry.MangledName);
    return heavy::EnvEntry{Value, MangledName};
  }

  heavy::Value VisitBigInt(heavy::BigInt* B) {
    return new (NewHeap) heavy::BigInt(B->Val);
  }

  heavy::Value VisitBuiltin(heavy::Builtin* B) {
    return new (NewHeap) heavy::Builtin(B->Fn);
  }

  // Binding
  heavy::Binding* VisitBinding(heavy::Binding* Binding) {
    return new (NewHeap) heavy::Binding(
      cast<heavy::Symbol>(Visit(Binding->getName())),
      Visit(Binding->getValue()));
  }

  // BuiltinSyntax
  heavy::Value VisitBuiltinSyntax(heavy::BuiltinSyntax* BuiltinSyntax) {
    return new (NewHeap) heavy::BuiltinSyntax(*BuiltinSyntax);
  }

  // ByteVector
  heavy::Value VisitByteVector(heavy::ByteVector* ByteVector) {
    heavy::String* NewString = VisitString(ByteVector->getString());
    return new (NewHeap) heavy::ByteVector(NewString);
  }

  // EnvFrame
  heavy::Value VisitEnvFrame(heavy::EnvFrame* EnvFrame) {
    llvm::ArrayRef<heavy::Binding*> Bindings = EnvFrame->getBindings();
    unsigned MemSize = EnvFrame::sizeToAlloc(Bindings.size());

    void* Mem = NewHeap.Allocate(MemSize, alignof(heavy::EnvFrame));

    heavy::EnvFrame* NewE = new (Mem) heavy::EnvFrame(Bindings.size());
    auto NewBindings = NewE->getBindings();
    for (unsigned i = 0; i < Bindings.size(); i++) {
      NewBindings[i] = cast<heavy::Binding>(Visit(Bindings[i]));
    }
    return NewE;
  }

  // Error
  heavy::Value VisitError(heavy::Error* Error) {
    return new (NewHeap) heavy::Error(
      static_cast<ValueWithSource*>(Error)->getSourceLocation(),
      Visit(Error->getMessage()),
      Visit(Error->getIrritants()));
  }

  // Exception
  heavy::Value VisitException(heavy::Exception* Exception) {
    return new (NewHeap) heavy::Exception(Visit(Exception->Val));
  }

  // ExternName
  heavy::Value VisitExternName(heavy::ExternName* ExternName) {
    return new (NewHeap) heavy::ExternName(
      cast<String>(Visit(ExternName->getName())),
      static_cast<ValueWithSource*>(ExternName)->getSourceLocation()
      );
  }

  // Float
  heavy::Value VisitFloat(heavy::Float* Float) {
    return new (NewHeap) heavy::Float(Float->getVal());
  }

  // ForwardRef
  heavy::Value VisitForwardRef(heavy::ForwardRef* ForwardRef) {
    // This is a wrapper for an already copied value.
    // Unwrap it on the new heap.
    return ForwardRef->Val;
  }

  // ImportSet
  heavy::Value VisitImportSet(heavy::ImportSet* ImportSet) {
    return new (NewHeap) heavy::ImportSet(
      ImportSet->getImportKind(),
      cast_or_null<heavy::ImportSet>(Visit(ImportSet->getParent())),
      Visit(ImportSet->getSpecifier()));
  }

  // Lambda
  heavy::Value VisitLambda(heavy::Lambda* Old) {
    OpaqueFn FnData = Old->getFnData();
    llvm::ArrayRef<heavy::Value> Captures = Old->getCaptures();
    void* Mem = heavy::Lambda::allocate(getAllocator(), FnData, Captures);
    return new (Mem) Lambda(FnData, Captures);
  }

  // Pair
  heavy::Value VisitPair(heavy::Pair* Pair) {
    heavy::Value Car = Visit(Pair->Car);
    heavy::Pair* Bottom = isa<heavy::PairWithSource>(heavy::Value(Pair))
        ? new (NewHeap) heavy::PairWithSource(Car, heavy::Empty(),
                                              Pair->getSourceLocation())
        : new (NewHeap) heavy::Pair(Car, heavy::Empty());
    heavy::Value OldCdr = Pair->Cdr;

    heavy::Pair* Top = Bottom;
    while (heavy::Pair* P = dyn_cast<heavy::Pair>(OldCdr)) {
      heavy::Value Car = Visit(P->Car);
      Top->Cdr = isa<heavy::PairWithSource>(heavy::Value(P))
          ? new (NewHeap) heavy::PairWithSource(Car, heavy::Empty(),
                                                P->getSourceLocation())
          : new (NewHeap) heavy::Pair(Car, heavy::Empty());
      Top = cast<heavy::Pair>(Top->Cdr);
      OldCdr = P->Cdr;
    }

    // Handle improper list. (ie OldCdr was not a Pair or Empty)
    if (!isa<heavy::Empty>(OldCdr)) {
      Top->Cdr = Visit(OldCdr);
      // Do not update Top since we are done.
    }

    return Bottom;
  }

  // PairWithSource
  heavy::Value VisitPairWithSource(heavy::PairWithSource* PairWithSource) {
    return VisitPair(PairWithSource);
  }

  // Quote
  heavy::Value VisitQuote(heavy::Quote* Quote) {
    return new (NewHeap) heavy::Quote(Visit(Quote->Val));
  }

  // SourceValue
  heavy::Value VisitSourceValue(heavy::SourceValue* SourceValue) {
    return new (NewHeap) heavy::SourceValue(*SourceValue);
  }

  // Symbol
  heavy::Value VisitSymbol(heavy::Symbol* Symbol) {
    return new (NewHeap) heavy::Symbol(cast<String>(Visit(Symbol->getString())),
      static_cast<ValueWithSource*>(Symbol)->getSourceLocation());
  }

  // Syntax
  heavy::Value VisitSyntax(heavy::Syntax* Syntax) {
    OpaqueFn FnData = Syntax->getFnData();
    void* Mem = Syntax::allocate(getAllocator(), FnData);
    return new (Mem) heavy::Syntax(FnData);
  }

  // SyntaxClosure
  heavy::Value VisitSyntaxClosure(heavy::SyntaxClosure* SyntaxClosure) {
    return new (NewHeap) heavy::SyntaxClosure(SyntaxClosure->getSourceLocation(),
                                              Visit(SyntaxClosure->Env),
                                              Visit(SyntaxClosure->Node));
  }

  // Vector
  heavy::Value VisitVector(heavy::Vector* Vector) {
    heavy::ArrayRef<heavy::Value> Xs = Vector->getElements();
    return new (NewHeap, Xs) heavy::Vector(Xs);
  }

  heavy::Value VisitTagged(heavy::Tagged* Tagged) {
    heavy::Symbol* Tag = cast<Symbol>(Visit(Tagged->getTag()));
    llvm::StringRef ObjData = Tagged->getObjData();
    void* Mem = Tagged::allocate(getAllocator(), ObjData);
    return new (Mem) heavy::Tagged(Tag, ObjData);
  }

  template <typename ...Args>
  heavy::Value Visit(heavy::Value OldVal) {
    void* ValueBase = OldVal.get<ValueSumType::ValueBase>();
    if (ValueBase == nullptr)
      return OldVal;

    // This runs two comparisons to see if the pointer
    // is within the bounds of OldHeap.
    if (!OldHeap.identifyObject(ValueBase).has_value())
      return OldVal;
    // NOTE: The above check allows us to have special allocators
    //       for identifiers and other "static" objects that we
    //       can avoid unnecessary copying.

    heavy::Value NewVal = Base::Visit(OldVal);
    // Overwrite the OldVal with a ForwardRef.
    new (ValueBase) ForwardRef(NewVal);

    return NewVal;
  }

public:
  CopyCollector(AllocatorTy& New, AllocatorTy& Old)
    : NewHeap(New),
      OldHeap(Old)
  { }

  void VisitRootNode(heavy::Value& Val) {
    Val = Visit(Val);
  }

  // Module
  heavy::Value VisitModule(heavy::Module* Module) {
    if (!markSpecialVisited(Module))
      return Module;

    // The module itself is not garbage collected,
    // but it has contained objects that are.
    Module->Cleanup = cast_or_null<Lambda>(Visit(Module->Cleanup));
    for (auto& DensePair : Module->Map) {
      DensePair.getFirst() = VisitString(DensePair.getFirst());
      DensePair.getSecond() = VisitEnvEntry(DensePair.getSecond());
    }
    return Module;
  }

  // Environment
  heavy::Value VisitEnvironment(heavy::Environment* Env) {
    // Check if previously visited.
    if (!markSpecialVisited(Env))
      return Env;

    // Environment itself is not garbage collected, but it
    // contains objects that are.
    if (heavy::OpGen* OpGen = Env->OpGen.get()) {
      if (OpGen->TopLevelHandler)
        OpGen->TopLevelHandler = Visit(OpGen->TopLevelHandler);
      if (OpGen->LibraryEnvProc)
        OpGen->LibraryEnvProc = VisitBinding(OpGen->LibraryEnvProc);
      // Note that the BindingTable has no unique ownership of the Bindings
      // as they are all pushed to the Environment.
    }

    for (auto& DensePair : Env->EnvMap) {
      DensePair.getFirst() = VisitString(DensePair.getFirst());
      DensePair.getSecond() = VisitEnvEntry(DensePair.getSecond());
    }

    VisitedSpecials.push_back(Env);
    if (Env->Parent)
      VisitEnvironment(Env->Parent);
    return Env;
  }
};

void Context::CollectGarbage() {
  // Create NewHeap
  Heap::AllocatorTy NewHeap;
  CopyCollector GC(NewHeap, this->TrashHeap);

  // Note that Environments are captured in Lambdas.

  // Visit all Context.Modules
  for (auto& StringMapEntry : this->Modules) {
    GC.VisitModule(StringMapEntry.second.get());
  }

  GC.VisitRootNode(EnvStack);

  // EmbeddedEnvs
  for (auto& DensePair : EmbeddedEnvs) {
    heavy::Environment* Env = DensePair.second.get(); 
    GC.VisitEnvironment(Env);
  }

  // KnownAddresses
  for (auto& DensePair : KnownAddresses) {
    // The String* key is allocated with IdTable.
    GC.VisitRootNode(DensePair.second);
  }

  GC.VisitRootNode(Err);
  GC.VisitRootNode(ExceptionHandlers);

  // Visit ModuleOp (for contained LiteralOps, MatchOps)
  if (ModuleOp) {
    auto WalkerFn = [&GC](mlir::Operation* Op) {
      heavy::HeavyValueAttr ValAttr;
      if (auto LiteralOp = dyn_cast<heavy::LiteralOp>(Op))
        ValAttr = LiteralOp.getInputAttr();
      else if (auto MatchOp = dyn_cast<heavy::MatchOp>(Op))
        ValAttr = MatchOp.getValAttr();
      GC.VisitRootNode(ValAttr.getValue());
    };
    ModuleOp->walk(WalkerFn);
  }
}

String* IdTable::CreateIdTableEntry(llvm::StringRef Str) {
  String*& Entry = IdTableMap[Str];
  if (Entry)
    return Entry;

  unsigned MemSize = String::sizeToAlloc(Str.size());
  void* Mem = IdHeap.Allocate(MemSize, alignof(String));

  Entry = new (Mem) String(Str);
  return Entry;
}
String* IdTable::CreateIdTableEntry(llvm::StringRef Prefix,
                                    llvm::StringRef Str) {
  Buffer.clear();
  Buffer += Prefix;
  Buffer += Str;
  return CreateIdTableEntry(Buffer);
}

}  // namespace heavy


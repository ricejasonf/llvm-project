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

#include "heavy/Heap.h"
#include "heavy/Value.h"
#include "heavy/ValueVisitor.h"

namespace heavy {
// CopyCollector
//  - Visit a Root node, copy the value to the current heap
//    and overwrite the heavy::Value with its replacement.
class CopyCollector : public ValueVisitor<CopyCollector, heavy::Value> {
  friend class ValueVisitor<CopyCollector, heavy::Value>;
  AllocatorTy& NewHeap;
  AllocatorTy& OldHeap;
  llvm::AllocatorBase<AllocatorTy>& getAllocator() { return NewHeap; }

  // Handle all of the types that are embedded in the pointer,
  // and, therefore do not require a heap allocation.
  heavy::Value VisitInt(heavy::Int V)               { return V; }
  heavy::Value VisitBool(heavy::Bool V)             { return V; }
  heavy::Value VisitChar(heavy::Char V)             { return V; }
  heavy::Value VisitEmpty(heavy::Empty V)           { return V; }
  heavy::Value VisitUndefined(heavy::Undefined V)   { return V; }
  heavy::Value VisitOperation(heavy::Operation* V)  { return V; }
  heavy::Value VisitContArg(heavy::ContArg* V)      { return V; }

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
  heavy::Value VisitByteVector(heavy::ByteVector* ByteVector);

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

  // Environment
  heavy::Value VisitEnvironment(heavy::Environment* Environment);

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

  // Module
  heavy::Value VisitModule(heavy::Module* Module) {
    // The module itself is never garbage collected,
    // but it has a Cleanup object that is.
    Module->Cleanup = cast_or_null<Lambda>(Visit(Module->Cleanup));
    return Module;
  }

  // Pair
  heavy::Value VisitPair(heavy::Pair* Pair);

  // PairWithSource
  heavy::Value VisitPairWithSource(heavy::PairWithSource* PairWithSource);

  // Quote
  heavy::Value VisitQuote(heavy::Quote* Quote) {
    return new (NewHeap) heavy::Quote(Visit(Quote->Val));
  }

  // SourceValue
  heavy::Value VisitSourceValue(heavy::SourceValue* SourceValue) {
    return new (NewHeap) heavy::SourceValue(*SourceValue);
  }

  // String
  heavy::Value VisitString(heavy::String* String) {
    llvm::StringRef StringRef = String->getView();
    unsigned MemSize = String::sizeToAlloc(StringRef.size());
    void* Mem = heavy::allocate(getAllocator(), MemSize, alignof(heavy::String));
    return new (Mem) heavy::String(StringRef);
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
    size_t size = Vector::sizeToAlloc(Xs.size());
    void* Mem = NewHeap.Allocate(size, alignof(heavy::Vector));
    return new (Mem) heavy::Vector(Xs);
  }

public:
  CopyCollector(AllocatorTy& New, AllocatorTy& Old)
    : NewHeap(New),
      OldHeap(Old)
  { }

  void VisitRootNode(heavy::Value& Val) {
    Val = Visit(Val);
  }
};

// Create Functions

template <typename Allocator, typename ...StringRefs>
static String* CreateStringHelper(Allocator& Alloc, StringRefs ...S) {
  std::array<unsigned, sizeof...(S)> Sizes{static_cast<unsigned>(S.size())...};
  unsigned TotalLen = 0;
  for (unsigned Size : Sizes) {
    TotalLen += Size;
  }

  unsigned MemSize = String::sizeToAlloc(TotalLen);
  void* Mem = heavy::allocate(Alloc, MemSize, alignof(String));

  return new (Mem) String(TotalLen, S...);
}

Heap::Heap() = default;
Heap::~Heap() = default;

String* Heap::CreateString(unsigned Length, char InitChar) {
  unsigned MemSize = String::sizeToAlloc(Length);
  void* Mem = heavy::allocate(getAllocator(), MemSize, alignof(String));
  return new (Mem) String(Length, InitChar);
}

String* Heap::CreateString(llvm::StringRef S) {
  return CreateStringHelper(getAllocator(), S);
}

String* Heap::CreateString(llvm::StringRef S1, StringRef S2) {
  return CreateStringHelper(getAllocator(), S1, S2);
}

String* Heap::CreateString(llvm::StringRef S1,
                           llvm::StringRef S2,
                           llvm::StringRef S3) {
  return CreateStringHelper(getAllocator(), S1, S2, S3);
}

String* Heap::CreateString(llvm::StringRef S1,
                           llvm::StringRef S2,
                           llvm::StringRef S3,
                           llvm::StringRef S4) {
  return CreateStringHelper(getAllocator(), S1, S2, S3, S4);
}

Value Heap::CreateList(llvm::ArrayRef<Value> Vs) {
  // Returns a *newly allocated* list of its arguments.
  heavy::Value List = CreateEmpty();
  for (auto Itr = Vs.rbegin(); Itr != Vs.rend(); ++Itr) {
    List = CreatePair(*Itr, List);
  }
  return List;
}

Float* Heap::CreateFloat(llvm::APFloat Val) {
  return new (TrashHeap) Float(Val);
}

Vector* Heap::CreateVector(unsigned N) {
  size_t size = Vector::sizeToAlloc(N);
  void* Mem = TrashHeap.Allocate(size, alignof(Vector));
  return new (Mem) Vector(CreateUndefined(), N);
}

Vector* Heap::CreateVector(ArrayRef<Value> Xs) {
  // Copy the list of Value to our heap
  size_t size = Vector::sizeToAlloc(Xs.size());
  void* Mem = TrashHeap.Allocate(size, alignof(Vector));
  return new (Mem) Vector(Xs);
}

ByteVector* Heap::CreateByteVector(ArrayRef<Value> Xs) {
  // Convert the Values to bytes.
  // Invalid inputs will be set to zero.
  heavy::String* String = CreateString(Xs.size(), '\0');
  ByteVector* BV =  new (TrashHeap) ByteVector(String);
  llvm::MutableArrayRef<char> Data = String->getMutableView();
  for (unsigned I = 0; I < Xs.size(); ++I) {
    if (isa<heavy::Int>(Xs[I])) {
      auto Byte = cast<heavy::Int>(Xs[I]);
      Data[I] = int32_t(Byte);
    }
  }
  return BV;
}

EnvFrame* Heap::CreateEnvFrame(llvm::ArrayRef<Symbol*> Names) {
  unsigned MemSize = EnvFrame::sizeToAlloc(Names.size());

  void* Mem = TrashHeap.Allocate(MemSize, alignof(EnvFrame));

  EnvFrame* E = new (Mem) EnvFrame(Names.size());
  auto Bindings = E->getBindings();
  for (unsigned i = 0; i < Bindings.size(); i++) {
    Bindings[i] = CreateBinding(Names[i], CreateUndefined());
  }
  return E;
}
  
}  // namespace heavy


//===--- Context.cpp - HeavyScheme Context Implementation -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implementations for heavy scheme Context.
//
//===----------------------------------------------------------------------===//

#include "heavy/Builtins.h"
#include "heavy/Dialect.h"
#include "heavy/Context.h"
#include "heavy/OpGen.h"
#include "heavy/Source.h"
#include "heavy/Value.h"
#include "mlir/IR/Module.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cstring>

using namespace heavy;

void ValueBase::dump() {
  write(llvm::errs(), this);
  llvm::errs() << '\n';
}

void Value::dump() {
  write(llvm::errs(), *this);
  llvm::errs() << '\n';
}

// FIXME not sure if CreateEmbedded is even useful anymore
//       originally we were setting the IntWidth
std::unique_ptr<Context> Context::CreateEmbedded() {
  auto Cptr = std::make_unique<Context>();
  return Cptr;
}

Context::Context()
  : Context(builtin::eval)
{ }

Context::Context(ValueFn ParseResultHandler)
  : DialectRegisterer()
  , TrashHeap()
  , SystemModule(CreateModule())
  , SystemEnvironment(CreateEnvironment(CreatePair(SystemModule)))
  , HandleParseResult(ParseResultHandler)
  , EnvStack(SystemEnvironment)
  , EvalStack(*this)
  , MlirContext()
  , OpGen(std::make_unique<heavy::OpGen>(*this))
  , OpEval(*this)
{
  // Load Builtin Syntax
  AddBuiltinSyntax("define",      builtin_syntax::define);
  AddBuiltinSyntax("if",          builtin_syntax::if_);
  AddBuiltinSyntax("lambda",      builtin_syntax::lambda);
  AddBuiltinSyntax("quasiquote",  builtin_syntax::quasiquote);
  AddBuiltinSyntax("quote",       builtin_syntax::quote);
  AddBuiltinSyntax("set",         builtin_syntax::set);

  // Load Builtin Procedures
  AddBuiltin("+",                 builtin::operator_add);
  AddBuiltin("*",                 builtin::operator_mul);
  AddBuiltin("-",                 builtin::operator_sub);
  AddBuiltin("/",                 builtin::operator_div);
  AddBuiltin("list",              builtin::list);
  AddBuiltin("append",            builtin::append);
  AddBuiltin("dump",              builtin::dump);
  AddBuiltin("eq?",               builtin::eqv);
  AddBuiltin("equal?",            builtin::equal);
  AddBuiltin("eqv?",              builtin::eqv);
}

Context::~Context() = default;

template <typename A, typename ...StringRefs>
String* CreateStringHelper(A& TrashHeap, StringRefs ...S) {
  std::array<unsigned, sizeof...(S)> Sizes{static_cast<unsigned>(S.size())...};
  unsigned TotalLen = 0;
  for (unsigned Size : Sizes) {
    TotalLen += Size;
  }

  unsigned MemSize = String::sizeToAlloc(TotalLen);
  void* Mem = TrashHeap.Allocate(MemSize, alignof(String));

  return new (Mem) String(TotalLen, S...);
}

String* Context::CreateString(StringRef S) {
  // Allocate and copy the string data
  size_t MemSize = String::sizeToAlloc(S.size());
  void* Mem = TrashHeap.Allocate(MemSize, alignof(String));
  return new (Mem) String(S);
}

String* Context::CreateString(StringRef S1, StringRef S2) {
  return CreateStringHelper(TrashHeap, S1, S2);
}

String* Context::CreateString(StringRef S1,
                              StringRef S2,
                              StringRef S3) {
  return CreateStringHelper(TrashHeap, S1, S2, S3);
}

Float* Context::CreateFloat(llvm::APFloat Val) {
  return new (TrashHeap) Float(Val);
}

Vector* Context::CreateVector(ArrayRef<Value> Xs) {
  // Copy the list of Value to our heap
  size_t size = Vector::sizeToAlloc(Xs.size());
  void* Mem = TrashHeap.Allocate(size, alignof(Vector));
  return new (Mem) Vector(Xs);
}

Vector* Context::CreateVector(unsigned N) {
  size_t size = Vector::sizeToAlloc(N);
  void* Mem = TrashHeap.Allocate(size, alignof(Vector));
  return new (Mem) Vector(CreateUndefined(), N);
}

EnvFrame* Context::PushLambdaFormals(Value Formals,
                                     bool& HasRestParam) {
  llvm::SmallVector<Symbol*, 8> Names;
  HasRestParam = false;
  if (CheckLambdaFormals(Formals, Names,
                         HasRestParam)) return nullptr;

  llvm::SmallSet<llvm::StringRef, 8> NameSet;
  // ensure uniqueness of names
  for (Symbol* Name : Names) {
    auto Result = NameSet.insert(Name->getVal());
    if (!Result.second) {
      SetError("duplicate parameter name", Name);
    }
  }

  return PushEnvFrame(Names);
}

EnvFrame* Context::PushEnvFrame(llvm::ArrayRef<Symbol*> Names) {
  EnvFrame* E = CreateEnvFrame(Names);
  EnvStack = CreatePair(E, EnvStack);
  return E;
}

void Context::PopEnvFrame() {
  // We want to remove the current local scope
  // and assert that we aren't popping anything else
  // Walk through the local bindings
  Value Env = EnvStack;
  Pair* EnvPair;
  while ((EnvPair = dyn_cast<Pair>(Env))) {
    if (!isa<Binding>(EnvPair->Car)) break;
    Env = EnvPair->Cdr;
  }
  assert(isa<EnvFrame>(EnvPair->Car) &&
      "Scope must be in an EnvFrame");
  EnvStack = EnvPair->Cdr;
}

void Context::PushLocalBinding(Binding* B) {
  EnvStack = CreatePair(B, EnvStack);
}

EnvFrame* Context::CreateEnvFrame(llvm::ArrayRef<Symbol*> Names) {
  unsigned MemSize = EnvFrame::sizeToAlloc(Names.size());  

  void* Mem = TrashHeap.Allocate(MemSize, alignof(EnvFrame));

  EnvFrame* E = new (Mem) EnvFrame(Names.size());
  auto Bindings = E->getBindings();
  for (unsigned i = 0; i < Bindings.size(); i++) {
    Bindings[i] = CreateBinding(Names[i], CreateUndefined());
  }
  return E;
}


// CheckLambdaFormals - Returns true on error
bool Context::CheckLambdaFormals(Value Formals,
                               llvm::SmallVectorImpl<Symbol*>& Names,
                               bool& HasRestParam) {
  Value V = Formals;
  if (isa<Empty>(V)) return false;
  if (isa<Symbol>(V)) {
    // If the formals are just a Symbol
    // or an improper list ending with a
    // Symbol, then that Symbol is a "rest"
    // parameter that binds remaining
    // arguments as a list.
    Names.push_back(cast<Symbol>(V));
    HasRestParam = true;
    return false;
  }

  Pair* P = dyn_cast<Pair>(V);
  if (!P || !isa<Symbol>(P->Car)) {
    SetError("invalid formals syntax", V);
    return true;
  }
  Names.push_back(cast<Symbol>(P->Car));
  return CheckLambdaFormals(P->Cdr, Names, HasRestParam);
}
#if 0 // TODO implement creating a Procedure

Procedure* Context::CreateProcedure(Pair* P) {
  int Arity = 0;
  Value Formals = P->Car;
  BindingRegion* Region = CreateRegion();
  ProcessFormals(Formals, Region, Arity);

  // The rest are expressions considered as the body
  Procedure* New = new (TrashHeap) Procedure(/*stuff*/);
}
#endif

// NextStack supports tail recursion with nested Environments
// The Stack may be an improper list ending with an Environment
Binding* Context::Lookup(Symbol* Name, Value Stack, Value NextStack) {
  if (isa<Empty>(Stack) && !NextStack) return nullptr;
  if (isa<Empty>(Stack)) Stack = NextStack;
  if (isa<Environment>(Stack)) Stack = cast<Environment>(Stack)->EnvStack;
  Value Result = nullptr;
  Value V    = cast<Pair>(Stack)->Car;
  Value Next = cast<Pair>(Stack)->Cdr;
  switch (V.getKind()) {
    case ValueKind::Binding:
      Result = cast<Binding>(V)->Lookup(Name);
      break;
    case ValueKind::EnvFrame:
      Result = cast<EnvFrame>(V)->Lookup(Name);
      break;
    case ValueKind::Module:
      Result = cast<Module>(V)->Lookup(Name);
      break;
    case ValueKind::Environment:
      NextStack = Next;
      Next = cast<Environment>(V)->EnvStack;
      break;
    default:
      llvm_unreachable("Invalid Lookup Type");
  }
  if (Result && isa<Binding>(Result)) {
    return cast<Binding>(Result);
  }
  assert(!Result && "lookup result should be a binding or null");
  return Lookup(Name, Next, NextStack);
}

// Returns Binding or Undefined on error
Value Context::CreateGlobal(Symbol* S, Value V, Value OrigCall) {
  // A module at the top of the EnvStack is mutable
  Module* M = nullptr;
  Value EnvRest = nullptr;
  if (isa<Pair>(EnvStack)) {
    Value EnvTop  = cast<Pair>(EnvStack)->Car;
    EnvRest = cast<Pair>(EnvStack)->Cdr;
    M = dyn_cast<Module>(EnvTop);
  }
  if (!M) return SetError("define used in immutable environment", OrigCall);

  // If the name already exists in the current module
  // then it behaves like `set!`
  Binding* B = M->Lookup(S);
  if (B) {
    B->Val = V;
    return B;
  }

  // Top Level definitions may not shadow names in
  // the parent environment
  B = Lookup(S, EnvRest);
  if (B) return SetError("define overwrites immutable location", S);

  B = CreateBinding(S, V);
  M->Insert(B);
  return B;
}

void Context::AddBuiltin(StringRef Str, ValueFn Fn) {
  Symbol* S = CreateSymbol(Str);
  Module* M = SystemModule;
  mlir::Value V = OpGen->VisitTopLevel(CreateBuiltin(Fn));
  OpGen->createTopLevelDefine(S, V, M);
}

mlir::Operation* Context::getModuleOp() {
  return OpGen->getTopLevel();
}
void Context::dumpModuleOp() {
  OpGen->getTopLevel().dump();
}

void Context::PushTopLevel(heavy::Value V) {
  OpGen->VisitTopLevel(V);
}

namespace {
class Writer : public ValueVisitor<Writer>
{
  friend class ValueVisitor<Writer>;
  unsigned IndentationLevel = 0;
  llvm::raw_ostream &OS;

public:
  Writer(llvm::raw_ostream& OS)
    : OS(OS)
  { }

private:
  void PrintFormattedWhitespace() {
    // We could handle indentation and new lines
    // more dynamically here probably
    assert(IndentationLevel < 100 && "IndentationLevel overflow suspected");

    OS << '\n';
    for (unsigned i = 0; i < IndentationLevel; ++i) {
      OS << ' ';
    }
  }

  void VisitValue(Value V) {
    OS << "<Value of Kind:"
       << getKindName(V.getKind())
       << ">";
  }

  void VisitBool(Bool V) {
    if (V)
      OS << "#t";
    else
      OS << "#f";
  }

  void VisitEmpty(Empty) {
    OS << "()";
  }

  void VisitInt(Int V) { OS << int32_t{V}; }
  void VisitFloat(Float* V) {
    llvm::SmallVector<char, 16> Buffer;
    V->getVal().toString(Buffer);
    OS << Buffer;
  }

  void VisitPair(Pair* P) {
    // Iterate the whole list to print
    // in standard list notation
    ++IndentationLevel;
    OS << '(';
    Visit(P->Car);
    Value Cdr = P->Cdr;
    while (isa<Pair>(Cdr)) {
      OS << ' ';
      //PrintFormattedWhitespace();
      P = cast<Pair>(Cdr);
      Visit(P->Car);
      Cdr = P->Cdr;
    };

    if (!Empty::classof(Cdr)) {
      OS << " . ";
      Visit(Cdr);
    }
    OS << ')';
    --IndentationLevel;
  }

  void VisitQuote(Quote* Q) {
    OS << "(quote ";
    Visit(Q->Val);
    OS << ")";
  }

  void VisitVector(Vector* Vec) {
    OS << "#(";
    ArrayRef<Value> Xs = Vec->getElements();
    if (!Xs.empty()) {
      Visit(Xs[0]);
      Xs = Xs.drop_front(1);
      for (Value X : Xs) {
        OS << ' ';
        Visit(X);
      }
    }
    OS << ')';
  }

  void VisitSymbol(Symbol* S) {
    OS << S->getVal();
  }

  void VisitString(String* S) {
    // TODO we might want to escape special
    // characters other than newline
    OS << '"' << S->getView() << '"';
  }

  void VisitModule(Module* M) {
    OS << "Module() {\n";
    ++IndentationLevel;
    for (Binding* B : *M) {
      PrintFormattedWhitespace();
      OS << B->getName()->getVal() << ": ";
      Visit(B->getValue());
    }
    --IndentationLevel;
    PrintFormattedWhitespace();
    OS << "}";
    // TEMP dont print out the pointer value
    OS << " " << ((size_t) M);
    PrintFormattedWhitespace();
  }
};

} // end anon namespace

namespace heavy {
void write(llvm::raw_ostream& OS, Value V) {
  Writer W(OS);
  return W.Visit(V);
}

// this handles non-immediate values
// and assumes the values have the same kind
bool equal_slow(Value V1, Value V2) {
  assert(V1.getKind() == V2.getKind() &&
      "inputs are expected to have same kind");

  switch (V1.getKind()) {
  case ValueKind::String:
    return cast<String>(V1)->equals(cast<String>(V2));
  case ValueKind::Pair:
  case ValueKind::PairWithSource: {
    Pair* P1 = cast<Pair>(V1);
    Pair* P2 = cast<Pair>(V2);
    // FIXME this does not handle cyclic refs
    return equal(P1->Car, P2->Car) &&
           equal(P1->Cdr, P2->Cdr);
  }
  case ValueKind::Vector:
    llvm_unreachable("TODO");
    return false;
  default:
      return eqv(V1, V2);
  }
}

// this handles non-immediate values
// and assumes the values have the same kind
bool eqv_slow(Value V1, Value V2) {
  assert(V1.getKind() == V2.getKind() &&
      "inputs are expected to have same kind");
  switch (V1.getKind()) {
  case ValueKind::Symbol:
    return cast<Symbol>(V1)->equals(
              cast<Symbol>(V2));
  case ValueKind::Float:
    return cast<Float>(V1)->getVal() ==
              cast<Float>(V2)->getVal();
  default:
      return false;
  }
}

} // end namespace heavy

void EvaluationStack::EmitStackSpaceError() {
  Context.SetError("insufficient stack space");
}

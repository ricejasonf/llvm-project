//===--- HeavyScheme.cpp - HeavyScheme AST Node Implementation --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implementations for HeavyScheme AST and Context classes.
//
//===----------------------------------------------------------------------===//

#include "heavy/Builtins.h"
#include "heavy/Dialect.h"
#include "heavy/HeavyScheme.h"
#include "heavy/OpEval.h"
#include "heavy/OpGen.h"
#include "heavy/Source.h"
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

void Value::dump() {
  write(llvm::errs(), this);
  llvm::errs() << '\n';
}

// FIXME not sure if CreateEmbedded is even useful anymore
//       originally we were setting the IntWidth
std::unique_ptr<Context> Context::CreateEmbedded() {
  auto Cptr = std::make_unique<Context>();
  return Cptr;
}

Context::Context()
  : DialectRegisterer()
  , TrashHeap()
  , SystemModule(CreateModule())
  , SystemEnvironment(CreateEnvironment(CreatePair(SystemModule)))
  , EnvStack(SystemEnvironment)
  , EvalStack(*this)
  , MlirContext()
  , OpGen(std::make_unique<heavy::OpGen>(*this))
  , OpEval(std::make_unique<heavy::OpEval>(*this))
{
  // Load Builtin Syntax
  AddBuiltinSyntax("define",      builtin_syntax::define);
  AddBuiltinSyntax("lambda",      builtin_syntax::lambda);
  AddBuiltinSyntax("quasiquote",  builtin_syntax::quasiquote);
  AddBuiltinSyntax("quote",       builtin_syntax::quote);

  // Load Builtin Procedures
  AddBuiltin("+",                 builtin::operator_add);
  AddBuiltin("*",                 builtin::operator_mul);
  AddBuiltin("-",                 builtin::operator_sub);
  AddBuiltin("/",                 builtin::operator_div);
  AddBuiltin("list",              builtin::list);
  AddBuiltin("append",            builtin::append);
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

Integer* Context::CreateInteger(llvm::APInt Val) {
  return new (TrashHeap) Integer(Val);
}

Float* Context::CreateFloat(llvm::APFloat Val) {
  return new (TrashHeap) Float(Val);
}

Vector* Context::CreateVector(ArrayRef<Value*> Xs) {
  // Copy the list of Value* to our heap
  size_t size = Vector::sizeToAlloc(Xs.size());
  void* Mem = TrashHeap.Allocate(size, alignof(Vector));
  return new (Mem) Vector(Xs);
}

Vector* Context::CreateVector(unsigned N) {
  size_t size = Vector::sizeToAlloc(N);
  void* Mem = TrashHeap.Allocate(size, alignof(Vector));
  return new (Mem) Vector(CreateUndefined(), N);
}

Lambda* Context::CreateLambda(heavy::ValueFn Fn,
                              llvm::ArrayRef<heavy::Value*> Captures) {
  return new (TrashHeap) Lambda(Fn, /*NumCaptures=*/0);
}

LambdaIr* Context::CreateLambdaIr(LambdaOp Op,
                              llvm::ArrayRef<heavy::Value*> Captures) {
  return new (TrashHeap) LambdaIr(Op, /*NumCaptures=*/0);
}

EnvFrame* Context::PushLambdaFormals(Value* Formals,
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
  Value* Env = EnvStack;
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
bool Context::CheckLambdaFormals(Value* Formals,
                               llvm::SmallVectorImpl<Symbol*>& Names,
                               bool& HasRestParam) {
  Value* V = Formals;
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
  Value* Formals = P->Car;
  BindingRegion* Region = CreateRegion();
  ProcessFormals(Formals, Region, Arity);

  // The rest are expressions considered as the body
  Procedure* New = new (TrashHeap) Procedure(/*stuff*/);
}
#endif

// NextStack supports tail recursion with nested Environments
// The Stack may be an improper list ending with an Environment
Binding* Context::Lookup(Symbol* Name, Value* Stack, Value* NextStack) {
  if (isa<Empty>(Stack) && !NextStack) return nullptr;
  if (isa<Empty>(Stack)) Stack = NextStack;
  if (isa<Environment>(Stack)) Stack = cast<Environment>(Stack)->EnvStack;
  Value* Result = nullptr;
  Value* V    = cast<Pair>(Stack)->Car;
  Value* Next = cast<Pair>(Stack)->Cdr;
  switch (V->getKind()) {
    case Value::Kind::Binding:
      Result = cast<Binding>(V)->Lookup(Name);
      break;
    case Value::Kind::EnvFrame:
      Result = cast<EnvFrame>(V)->Lookup(Name);
      break;
    case Value::Kind::Module:
      Result = cast<Module>(V)->Lookup(Name);
      break;
    case Value::Kind::Environment:
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
Value* Context::CreateGlobal(Symbol* S, Value *V, Value* OrigCall) {
  // A module at the top of the EnvStack is mutable
  Module* M = nullptr;
  Value* EnvRest = nullptr;
  if (isa<Pair>(EnvStack)) {
    Value* EnvTop  = cast<Pair>(EnvStack)->Car;
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
  mlir::Value Op = OpGen->createTopLevelDefine(S, V, M);
  OpEval->Visit(Op);
}

void Context::dumpModuleOp() {
  OpGen->getTopLevel().dump();
}

void Context::PushTopLevel(heavy::Value* V) {
  OpGen->VisitTopLevel(V);
}

#if 0
// TODO remove I don't think it is used after OpGen was added
Builtin* Context::GetBuiltin(StringRef Name) {
  Binding* B = nullptr;
  Value* Result = SystemModule->Lookup(Name);
  if (Result) {
    if (ForwardRef* F = dyn_cast<ForwardRef>(Result)) {
      Result = F->Val;
    }
    B = cast<Binding>(Result);
  }
  assert(B && isa<Builtin>(B->getValue()) && "Internal builtin lookup failed");
  return cast<Builtin>(B->getValue());
}
#endif

namespace {
#if 0
class SyntaxExpander : public ValueVisitor<SyntaxExpander, Value*> {
  friend class ValueVisitor<SyntaxExpander, Value*>;
  heavy::Context& Context;
  // We use RAII to make the current call to `eval`
  // set the environment in Context
  Value* OldEnvStack;
  bool OldIsTopLevel;

public:
  SyntaxExpander(heavy::Context& C, Value* EnvStack = nullptr)
    : Context(C)
  {
    OldIsTopLevel = Context.IsTopLevel;
    if (EnvStack) {
      OldEnvStack = Context.EnvStack;
      Context.EnvStack = EnvStack;
    } else {
      OldEnvStack = Context.EnvStack;
    }
  }

  ~SyntaxExpander() {
    Context.EnvStack = OldEnvStack;
    Context.IsTopLevel = OldIsTopLevel;
  }

  Value* VisitTopLevel(Value* V) {
    Context.IsTopLevel = true;
    return Visit(V);
  }

private:
  Value* VisitValue(Value* V) {
    return V;
  }

  Value* VisitSymbol(Symbol* S) {
    if (Binding* B = Context.Lookup(S)) return B;

    String* Msg = Context.CreateString("unbound symbol: ", S->getVal());
    Context.SetError(Msg, S);
    return Context.CreateUndefined();
  }

  Value* HandleCallArgs(Value *V) {
    if (isa<Empty>(V)) return V;
    if (!isa<Pair>(V)) {
      return Context.SetError("improper list as call expression", V);
    }
    Pair* P = cast<Pair>(V);
    Value* CarResult = Visit(P->Car);
    Value* CdrResult = HandleCallArgs(P->Cdr);
    // TODO
    // Create a CallExpr AST node with a
    // known number of arguments instead of
    // recreating lists
    return Context.CreatePair(CarResult, CdrResult);
  }

  Value* VisitPair(Pair* P) {
    if (Context.CheckError()) return Context.CreateEmpty();
    Binding* B = Context.Lookup(P->Car);
    if (!B) return P;

    // Operator might be some kind of syntax transformer
    Value* Operator = B->getValue();

    switch (Operator->getKind()) {
      case Value::Kind::BuiltinSyntax: {
        BuiltinSyntax* BS = cast<BuiltinSyntax>(Operator);
        return BS->Fn(Context, P);
      }
      case Value::Kind::Syntax:
        llvm_unreachable("TODO");
        return Context.CreateEmpty();
      default:
        Context.IsTopLevel = false;
        return HandleCallArgs(P);
    }
  }

  Value* VisitVector(Vector* V) {
    llvm::ArrayRef<Value*> Xs = V->getElements();
    Vector* New = Context.CreateVector(Xs.size());
    llvm::MutableArrayRef<Value*> Ys = New->getElements();
    for (unsigned i = 0; i < Xs.size(); ++i) {
      Visit(Xs[i]);
      Ys[i] = Visit(Xs[i]);
    } 
    return New;
  }
};
#endif

#if 0 // migrate to OpGen
class Quasiquoter : private ValueVisitor<Quasiquoter, Value*> {
  friend class ValueVisitor<Quasiquoter, Value*>;
  heavy::Context& Context;
  // Values captured for hygiene purposes
  Value* Append;
  Value* ConsSource;

public:

  Quasiquoter(heavy::Context& C)
    : Context(C)
    , Append(C.GetBuiltin("append"))
    , ConsSource(C.GetBuiltin("cons-source"))
  { }

  // <quasiquotation>
  Value* Run(Pair* P) {
    bool Rebuilt = false;
    // <quasiquotation 1>
    return HandleQuasiquote(P, Rebuilt, /*Depth=*/1);
  }

private:

  Value* VisitValue(Value* V, bool& Rebuilt, int Depth) {
    return V;
  }

  // <qq template D>
  Value* HandleQQTemplate(Value* V, bool& Rebuilt, int Depth) {
    assert(Depth >= 0 && "Depth should not be negative");
    if (Depth < 1) {
      // Unquoting requires parents to be rebuilt
      Rebuilt = true;
      return V;
    }
    return Visit(V, Rebuilt, Depth);
  }

  // <quasiquotation D>
  Value* HandleQuasiquote(Pair* P, bool& Rebuilt, int Depth) {
    Value* Input = GetSingleSyntaxArg(P);
    if (!Input) {
      Context.SetError("invalid quasiquote syntax", P);
      return Context.CreateEmpty();
    }
    Value* Result = Visit(Input, Rebuilt, Depth);
    if (!Rebuilt) return Context.CreateQuote(Input);
    return Result;
  }

  // <unquotation D>
  Value* HandleUnquote(Pair* P, bool& Rebuilt, int Depth) {
    Value* Input = GetSingleSyntaxArg(P);
    if (!Input) {
      Context.SetError("invalid unquote syntax", P);
      return Context.CreateEmpty();
    }

    Value* Result = HandleQQTemplate(Input, Rebuilt, Depth - 1);
    if (!Rebuilt) return P;
    return Result;
  }

  Value* HandleUnquoteSplicing(Pair* P, Value* Next, bool& Rebuilt,
                               int Depth) {
    Value* Input = GetSingleSyntaxArg(P);
    if (!Input) {
      Context.SetError("invalid unquote-splicing syntax", P);
      return Context.CreateEmpty();
    }
    Value* Result = HandleQQTemplate(Input, Rebuilt, Depth - 1);
    if (!Rebuilt) {
      return P;
    }

    if (isa<Pair>(Result)) {
      // It is an error if unquote-splicing does not result in a list
      Context.SetError("unquote-splicing must evaluate to a list", P);
      return Context.CreateEmpty();
    }
    // append Next to Input (during evaluation)
    return Context.CreatePair(Append, Context.CreatePair(Result, Next));
  }

  Value* VisitPair(Pair* P, bool& Rebuilt, int Depth) {
    assert(Depth > 0 && "Depth cannot be zero here.");
    if (Context.CheckError()) return Context.CreateEmpty();
    if (P->Car->isSymbol("quasiquote")) {
      return HandleQuasiquote(P, Rebuilt, Depth + 1);
    } else if (P->Car->isSymbol("unquote")) {
      return HandleUnquote(P, Rebuilt, Depth);
    } else if (isa<Pair>(P->Car) &&
               cast<Pair>(P->Car)->Car->isSymbol("unquote-splicing")) {
      Pair* P2 = cast<Pair>(P->Car);
      return HandleUnquoteSplicing(P2, P->Cdr, Rebuilt, Depth);
    } else {
      // Just a regular old pair
      // <list qq template D>
      bool CarRebuilt = false;
      bool CdrRebuilt = false;
      Value* Car = Visit(P->Car, CarRebuilt, Depth);
      Value* Cdr = Visit(P->Cdr, CdrRebuilt, Depth);
      // Portions that are not rebuilt are always literal
      // '<qq template D>
      if (!CarRebuilt && CdrRebuilt) Car = Context.CreateQuote(Car);
      if (!CdrRebuilt && CarRebuilt) Cdr = Context.CreateQuote(Cdr);
      Rebuilt = CarRebuilt || CdrRebuilt;
      if (!Rebuilt) return P;
      return Context.CreatePair(ConsSource,
              Context.CreatePair(Car,
                Context.CreatePair(Cdr,
                  Context.CreatePair(
                    Context.CreateQuote(P)))));
    }
  }

  // TODO VisitVector
};
#endif


#if 0 // deprecated in favor of OpEvaluator
// Evaluator
//  - tree evaluator that uses the
//    evaluation stack
//  - to work with builtins and bytecode
//  - uses RAII to replace the Context.EnvStack
// TODO make this work on Ops instead of raw AST
class Evaluator : public ValueVisitor<Evaluator> {
  friend class ValueVisitor<Evaluator>;
  heavy::Context& Context;

public:
  Evaluator(heavy::Context& C)
    : Context(C)
  { }

private:
  void push(Value* V) {
    Context.EvalStack.push(V);
  }
  Value* pop() {
    return Context.EvalStack.pop();
  }
  Value* top() {
    return Context.EvalStack.top();
  }
  

  // Most objects simply evaluate to themselves
  void VisitValue(Value* V) {
    if (Context.CheckError()) return;
    push(V);
  }

  void VisitPair(Pair* P) {
    if (Context.CheckError()) return;
    // Visit each element in reverse order and evaluate
    // on the stack.
    int Len = 0;
    EvalArguments(P, Len);
    if (Context.CheckError()) return;

    Value* Operator = pop();
    --Len;
    // TODO Check arity here since we have
    //      the caller and callee
    // TODO We could check a "Contract"
    //      defined for builtins
    switch (Operator->getKind()) {
      case Value::Kind::Procedure:
        llvm_unreachable("TODO");
        break;
      case Value::Kind::Builtin: {
        Builtin* B = cast<Builtin>(Operator);
        B->Fn(Context, Len);
        break;
      }
      default: {
        String* Msg = Context.CreateString(
          "invalid operator for call expression: ",
          Operator->getKindName()
        );
        Context.SetError(P->getSourceLocation(), Msg, Operator);
        return;
      }
    }
  }

  void VisitQuote(Quote* Q) {
    // simply unwrap the quoted value
    push(Q->Val);
  }

  void VisitBinding(Binding* B) {
    push(B->Val);
  }

  void VisitVector(Vector* V) {
    llvm::ArrayRef<Value*> Xs = V->getElements();
    Vector* New = Context.CreateVector(Xs.size());
    llvm::MutableArrayRef<Value*> Ys = New->getElements();
    for (unsigned i = 0; i < Xs.size(); ++i) {
      Visit(Xs[i]);
      Ys[i] = pop();
    } 
    push(New);
  }

  void VisitSymbol(Symbol* S) {
    llvm_unreachable("Evaluation requires resolved syntax");
  }

  void EvalArguments(Value* Args, int& Len) {
    if (isa<Empty>(Args)) return;
    Pair* P = dyn_cast<Pair>(Args);
    if (!P) {
      Context.SetError("call expression must be a proper list", Args);
      return ;
    }
    // Arguments are evaluated right to left
    EvalArguments(P->Cdr, Len);
    if (Context.CheckError()) return;
    Len += 1;
    Visit(P->Car);
  }

#if 0
  // BindArguments
  // Args should be a list of evaluated inputs
  // TODO rewrite this to create an EnvFrame
  ValueResult BindArguments(Pair* Region,
                            Pair* Args,
                            Value* Formals) {
    Pair* P;
    switch (Formals->getKind()) {
    case Value::Empty: {
      llvm::errs() <<
        "\nTODO Diagnose arity mismatch (too many parameters)\n";
      return true;
    }
    case Value::Symbol:
      // Bind remaining args to "rest" parameter
      Symbol* Name = cast<Symbol>(Formals);
      AddBinding(Region, Formals, Args);
      return false;
    case Value::Pair:
      P = cast<Pair>(Formals);
      Symbol* Name = cast<Symbol>(P->Car);
      AddBinding(Region, Name, Args->Car);
      break;
    default:
      llvm_unreachable("Formals should already be checked");
    };

    Pair* NextArgs = dyn_cast<Pair>(Args->Cdr);
    if (!NextArgs) {
      llvm::errs() <<
        "\nTODO Diagnose arity mismatch (too few parameters)\n";
      return true;
    }

    return BindArguments(Region, NextArgs, P->Cdr);
  }
#endif
};
#endif

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

  void VisitValue(Value* V) {
    OS << "<Value of Kind:"
       << V->getKindName()
       << ">";
  }

  void VisitBoolean(Boolean* V) {
    if (V->getVal())
      OS << "#t";
    else
      OS << "#f";
  }

  void VisitEmpty(Empty*) {
    OS << "()";
  }

  void VisitInteger(Integer* V) { OS << V->getVal(); }
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
    Value* Cdr = P->Cdr;
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
    ArrayRef<Value*> Xs = Vec->getElements();
    if (!Xs.empty()) {
      Visit(Xs[0]);
      Xs = Xs.drop_front(1);
      for (Value* X : Xs) {
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
void write(llvm::raw_ostream& OS, Value* V) {
  Writer W(OS);
  return W.Visit(V);
}

} // end namespace heavy

void EvaluationStack::EmitStackSpaceError() {
  Context.SetError("insufficient stack space");
}

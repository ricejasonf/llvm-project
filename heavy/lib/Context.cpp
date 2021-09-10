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
#include "heavy/Mangle.h"
#include "heavy/Source.h"
#include "heavy/Value.h"
#include "heavy/ValueVisitor.h"
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

// NameForImportVar - Because we use String* for storing mangled
//                    names, we need to create one for "import"
//                    which is special. We also give it a relatively
//                    simple, readable symbol name.
heavy::ExternString<20> NameForImportVar;
heavy::ExternSyntax _HEAVY_import;

Context::Context()
  : DialectRegisterer()
  , ContinuationStack<Context>()
  , TrashHeap()
  , SystemModule(std::make_unique<Module>(*this))
  , SystemEnvironment(std::make_unique<Environment>())
  , EnvStack(SystemEnvironment.get())
  , MlirContext()
  , OpGen(std::make_unique<heavy::OpGen>(*this))
  , OpEval(*this)
{
  NameForImportVar = "_HEAVY_import";
  _HEAVY_import = heavy::base::import;
  RegisterModule(HEAVY_BASE_LIB_STR, HEAVY_BASE_LOAD_MODULE);
}

Context::~Context() = default;

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

String* Context::CreateString(llvm::StringRef S) {
  return CreateStringHelper(TrashHeap, S);
}

String* Context::CreateString(llvm::StringRef S1, StringRef S2) {
  return CreateStringHelper(TrashHeap, S1, S2);
}

String* Context::CreateString(llvm::StringRef S1,
                              llvm::StringRef S2,
                              llvm::StringRef S3) {
  return CreateStringHelper(TrashHeap, S1, S2, S3);
}

String* Context::CreateIdTableEntry(llvm::StringRef S) {
  String*& Str = IdTable[S];
  if (!Str) {
    Str = CreateString(S);
  }
  return Str;
}

String* Context::CreateIdTableEntry(llvm::StringRef Prefix,
                                    llvm::StringRef S) {
  // unfortunately we have to create a garbage string
  // just to check this
  String* Temp = CreateString(Prefix, S);
  return CreateIdTableEntry(Temp->getView());
}

Symbol* Context::CreateSymbol(llvm::StringRef S,
                         SourceLocation Loc) {
  String* Str = CreateIdTableEntry(S);
  return new (TrashHeap) Symbol(Str, Loc);
}

#if 0 // not sure if we want to create symbols for every possible import
// for import prefixes
Symbol* Context::CreateSymbol(llvm::StringRef S1, StringRef S2) {
  // settle for a std::string so we don't create a garbage String
  // every time we encounter this. (there has to be a better way)
  std::string Temp{};
  Temp.reserve(S1.size() + S2.size());
  Temp += S1;
  Temp += S2;
  return CreateSymbol(TrashHeap, Temp);
}
#endif

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

Module* Context::RegisterModule(llvm::StringRef MangledName,
                                heavy::ModuleLoadNamesFn* LoadNames) {
  auto Result = Modules.try_emplace(MangledName,
      std::make_unique<Module>(*this, LoadNames));
  auto Itr = Result.first;
  bool DidInsert = Result.second;

  assert(DidInsert && "module should be created only once");
  return Itr->second.get();
}

bool Context::Import(heavy::ImportSet* ImportSet) {
  heavy::Environment* Env = dyn_cast<Environment>(EnvStack);
  if (!Env) {
    // import doesn't work in local scope
    SetError("unexpected import", ImportSet);
    return true;
  }

  for (EnvBucket ImportVal : *ImportSet) {
    String* Name = ImportVal.first;
    // if there is no name just skip it (it was filtered out)
    if (!Name) continue;
    if (!Env->ImportValue(ImportVal)) {
      String* ErrMsg = CreateString("imported name already exists: ",
                                    Name->getView());
      SetError(ErrMsg, ImportSet);
      return true;
    }
  }

  return false;
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

// The Stack is an improper list ending with an Environment
EnvEntry Context::Lookup(Symbol* Name, Value Stack) {
  if (auto* E = dyn_cast<Environment>(Stack)) {
    EnvEntry Result = E->Lookup(Name);
    if (!Result && Name->equals("import")) {
      return EnvEntry{_HEAVY_import, NameForImportVar};
    }
    return Result;
  }
  EnvEntry Result = {};
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
    case ValueKind::ImportSet:
      Result = cast<ImportSet>(V)->Lookup(*this, Name);
      break;
    case ValueKind::Environment: {
      Result = cast<Environment>(V)->Lookup(Name);
      break;
    }
    default:
      llvm_unreachable("Invalid Lookup Type");
  }
  if (Result) return Result;
  return Lookup(Name, Next);
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

  void VisitError(Error* E) {
    OS << "(error ";
    Visit(E->getMessage());
    OS << " ";
    VisitCdr(E->getIrritants());
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
    VisitCdr(P->Cdr);
    --IndentationLevel;
  }

  // VisitCdr - Print the rest of a list
  void VisitCdr(Value Cdr) {
    while (Pair* P = dyn_cast<Pair>(Cdr)) {
      OS << ' ';
      //PrintFormattedWhitespace();
      Visit(P->Car);
      Cdr = P->Cdr;
    };

    if (!isa<Empty>(Cdr)) {
      OS << " . ";
      Visit(Cdr);
    }
    OS << ')';
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

void Context::EmitStackSpaceError() {
  // TODO We should probably clear the stack
  //      and run this as a continuation
  SetError("insufficient stack space");
}

EnvEntry ImportSet::Lookup(heavy::Context& C, Symbol* S) {
  switch(Kind) {
  case ImportKind::Library:
    return cast<Module>(Specifier)->Lookup(S);
  case ImportKind::Only:
    // Specifier is a list of Symbols
    return isInIdentiferList(S) ? Parent->Lookup(C, S) : EnvEntry{};
  case ImportKind::Except:
    // Specifier is a list of Symbols
    return !isInIdentiferList(S) ? Parent->Lookup(C, S) : EnvEntry{};
  case ImportKind::Prefix: {
    llvm::StringRef Str = S->getVal();
    // Specifier is just a Symbol
    llvm::StringRef Prefix = cast<Symbol>(Specifier)->getVal();
    if (!Str.consume_front(Prefix)) return {};
    S = C.CreateSymbol(Str, S->getSourceLocation());
    return Parent->Lookup(C, S);
  }
  case ImportKind::Rename:
    // Specifier is a list of pairs of symbols
    return LookupFromPairs(C, S);
  }
  llvm_unreachable("invalid import kind");
}

EnvEntry ImportSet::LookupFromPairs(heavy::Context& C, Symbol* S) {
  // The syntax of Specifier should be checked already
  assert(Kind == ImportKind::Rename && "expecting import rename");
  Value CurrentRow = Specifier;
  while (Pair* P = dyn_cast<Pair>(CurrentRow)) {
    Pair* Row = cast<Pair>(P->Car);
    Symbol* Key   = cast<Symbol>(Row->Car);
    Symbol* Value = cast<Symbol>(cast<Pair>(Row->Cdr)->Car);
    if (S->equals(Value)) return Parent->Lookup(C, Key);
    CurrentRow = P->Cdr;
  }
  return {};
}

String* ImportSet::FilterName(heavy::Context& C, String* S) {
  // recurse all the way to the module and
  // traverse back down removing or replacing the String
  if (Kind == ImportKind::Library) {
    Module* M = cast<Module>(Specifier);
    assert(M->Lookup(S).Value != nullptr && "filtered name not in library");
    return S;
  }
  S = Parent->FilterName(C, S);
  switch(Kind) {
  case ImportKind::Only:
    // filter if not in the list
    return isInIdentiferList(S) ? S : nullptr;
  case ImportKind::Except:
    // filter if its in the list
    return !isInIdentiferList(S) ? S : nullptr;
  case ImportKind::Prefix: {
    // add the prefix
    llvm::StringRef Str = S->getView();
    llvm::StringRef Prefix = cast<Symbol>(Specifier)->getVal();
    return C.CreateIdTableEntry(Prefix, Str);
  }
  case ImportKind::Rename:
    return FilterFromPairs(C, S);
  default:
    llvm_unreachable("invalid import kind");
  }
}

String* ImportSet::FilterFromPairs(heavy::Context& C, String* S) {
  assert(Kind == ImportKind::Rename && "expecting import rename");
  Value CurrentRow = Specifier;
  while (Pair* P = dyn_cast<Pair>(CurrentRow)) {
    Pair* Row = cast<Pair>(P->Car);
    String* Key   = cast<Symbol>(Row->Car)->getString();
    String* Value = cast<Symbol>(cast<Pair>(Row->Cdr)->Car)->getString();
    if (S->equals(Key)) return Value;
    CurrentRow = P->Cdr;
  }
  return S;
}

ImportSet* Context::CreateImportSet(Value Spec) {
  ImportSet::ImportKind Kind;
  Symbol* Keyword = dyn_cast_or_null<Symbol>(Spec.car());
  if (!Keyword) {
    SetError("expecting import set", Spec);
    return nullptr;
  }
  // TODO perhaps we could intern these keywords in the IdTable
  if (Keyword->equals("only")) {
    Kind = ImportSet::ImportKind::Only;
  } else if (Keyword->equals("except")) {
    Kind = ImportSet::ImportKind::Except;
  } else if (Keyword->equals("rename")) {
    Kind = ImportSet::ImportKind::Rename;
  } else if (Keyword->equals("prefix")) {
    Kind = ImportSet::ImportKind::Prefix;
  } else {
    // ImportSet::ImportKind::Library;
    Module* M = LoadModule(Spec);
    if (!M) return nullptr;
    return new (TrashHeap) ImportSet(M);
  }

  ImportSet* Parent = CreateImportSet(cadr(Spec));
  if (!Parent) return nullptr;
  Spec = Spec.cdr().cdr();

  if (Kind == ImportSet::ImportKind::Prefix) {
    Symbol* Prefix = dyn_cast_or_null<Symbol>(car(Spec));
    if (!Prefix) {
      SetError("expected identifier for prefix");
      return nullptr;
    }
    if (!isa_and_nonnull<Empty>(cdr(Spec))) {
      SetError("expected end of list");
      return nullptr;
    }
    return new (TrashHeap) ImportSet(Kind, Parent, Prefix);
  }

  // Identifier (pairs) list
  // Check the syntax and that each provided name
  // exists in the parent import set
  if (!isa_and_nonnull<Pair>(Spec)) {
    SetError("expecting list");
    return nullptr;
  }
  // Spec is now the list of ids to be used by ImportSet

  Value Current = Spec;
  Symbol* Name = nullptr;
  while (Pair* P = dyn_cast<Pair>(Current)) {
    switch (Kind) {
    case ImportSet::ImportKind::Only:
    case ImportSet::ImportKind::Except:
      Name = dyn_cast<Symbol>(P->Car);
      break;
    case ImportSet::ImportKind::Rename: {
      Pair* P2 = dyn_cast<Pair>(P->Car);
      if (!P2) {
       SetError("expected pair", P);
       return nullptr;
      }
      Name = dyn_cast<Symbol>(P2->Car);
      Symbol* Rename = dyn_cast_or_null<Symbol>(Value(P2).cadr());
      if (Name || Rename) {
        SetError("expected pair of identifiers", P2);
        return nullptr;
      }
      break;
    }
    default:
      llvm_unreachable("invalid import set kind");
    }

    EnvEntry LookupResult = Parent->Lookup(*this, Name);
    if (!LookupResult) {
      SetError("name does not exist in import set");
      return nullptr;
    }
    Current = P->Cdr;
  }
  return new (TrashHeap) ImportSet(Kind, Parent, Spec);
}

Module* Context::LoadModule(Value Spec) {
  heavy::Mangler Mangler(*this);
  std::string Name = Mangler.mangleModule(Spec);
  std::unique_ptr<Module>& M = Modules[Name];
  if (!M) {
    SetError("unable to load module", Spec);
    return nullptr;
#if 0
    // TODO
    // Load the file and compile the scheme code and check again
    // We might need the SourceManager to belong to Context
    std::string Filename = getModuleFilename(Spec);
    std::unique_ptr<Module>& M = Modules[Name];
    if (!M) {
      String* Msg = CreateString("loaded file does not contain library: ",
                                 Filename);
      SetError(Msg, Spec);
      return nullptr;
    }
#endif
  }
  M->LoadNames();
  return M.get();
}


void Context::AddKnownAddress(String* MangledName, heavy::Value Value) {
  assert(Value && "value at address must actually be known");
  KnownAddresses[MangledName] = Value;
}

Value Context::GetKnownValue(llvm::StringRef MangledName) {
  String* Id = IdTable[MangledName];
  assert(Id && "identifier should have been inserted into table for mangled name");
  // Could we possibly use dlsym or something here
  // for actual external values?
  return KnownAddresses.lookup(Id);
}

void heavy::initModule(heavy::Context& C, llvm::StringRef ModuleMangledName,
                  ModuleInitListTy InitList) {
  Module* M = C.Modules[ModuleMangledName].get();
  assert(M && "module must be registered");
  heavy::Mangler Mangler(C);
  for (ModuleInitListPairTy const& X : InitList) {
    String* Id = C.CreateIdTableEntry(X.first);
    Value Val = X.second;
    String* MangledName = C.CreateIdTableEntry(
        Mangler.mangleVariable(ModuleMangledName, Id));
    M->Insert(EnvBucket{Id, EnvEntry{Val, MangledName}});
    // Track valid values by their mangled names
    if (Val) {
      C.AddKnownAddress(MangledName, Val);
    }
  }
}

bool Context::CheckKind(ValueKind VK, Value V) {
  if (V.getKind() == VK) return false;
  String* S = CreateStringHelper(TrashHeap,
      llvm::StringRef("invalid type "),
      getKindName(V.getKind()),
      llvm::StringRef(", expecting "),
      getKindName(VK));
  RaiseError(S, V);
  return true;
}
bool Context::CheckNumber(Value V) {
  if (V.isNumber()) return false;
  String* S = CreateStringHelper(TrashHeap,
      llvm::StringRef("invalid type "), 
      getKindName(V.getKind()),
      llvm::StringRef(", expecting number"));
  RaiseError(S, V);
  return true;
}

void Context::SetError(Value E) {
  assert(isa<Error>(E) || isa<Exception>(E));
  Err = E;
  Raise(E);
}

void Context::SetErrorHandler(Value Handler) {
  assert((isa<Empty>(ExceptionHandlers) ||
          isa<Empty>(ExceptionHandlers.cdr())) &&
    "error handler may only be set at the bottom of the handler stack");

  ExceptionHandlers = Handler;
}

void Context::WithExceptionHandlers(Value NewHandlers, Value Thunk) {
  Value PrevHandlers = ExceptionHandlers;
  Value Before = CreateLambda([this](Context& C, ValueRefs) {
    Value NewHandlers = C.getCapture(0);
    this->ExceptionHandlers = NewHandlers;
    C.Cont(Undefined());
  }, {NewHandlers});
  Value After = CreateLambda([this](Context& C, ValueRefs) {
    Value PrevHandlers = C.getCapture(0);
    this->ExceptionHandlers = PrevHandlers;
    C.Cont(Undefined());
  }, {PrevHandlers});
  DynamicWind(Before, Thunk, After);
}

void Context::WithExceptionHandler(Value Handler, Value Thunk) {
  Value NewHandlers = CreatePair(Handler, ExceptionHandlers);
  WithExceptionHandlers(NewHandlers, Thunk); 
}

void Context::Raise(Value Obj) {
  if (isa<Empty>(ExceptionHandlers)) {
    ClearStack();
    return Cont(Undefined());
  }
  if (!isa<Pair>(ExceptionHandlers)) {
    if (!CheckError()) {
      std::string Msg;
      llvm::raw_string_ostream Stream(Msg);
      write(Stream << "uncaught object: ", Obj);
      return SetError(Msg, Obj);
    }
    return Apply(ExceptionHandlers, Obj);
  }
  Pair* P = cast<Pair>(ExceptionHandlers);
  Value Handler = P->Car;
  Value PrevHandlers = P->Cdr;
  WithExceptionHandlers(PrevHandlers,
                        CreateLambda([](Context& C, ValueRefs) {
    Value Handler = C.getCapture(0);
    Value Obj = C.getCapture(1);
    C.PushCont([](Context& C, ValueRefs Args) {
      assert(Args.size() == 1 && "expecting a single argument");
      Value Handler = C.getCapture(0);
      C.RaiseError("error handler returned", Handler);
    }, {Handler});
    C.Apply(Handler, Obj);
  }, {Handler, Obj}));
}

void Context::RaiseError(String* Msg, llvm::ArrayRef<Value> IrrArgs) {
  heavy::SourceLocation Loc = this->Loc;
  Value IrrList = Empty();
  while (!IrrArgs.empty()) {
    Value Irr = IrrArgs.front();
    if (!Loc.isValid()) {
      Loc = Irr.getSourceLocation();
    }
    IrrList = CreatePair(Irr, IrrList);
    IrrArgs = IrrArgs.drop_front();
  }
  Value Error = CreateError(Loc, Msg, IrrList);
  Raise(Error);
}

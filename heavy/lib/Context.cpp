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
#include "mlir/IR/Verifier.h" // TODO move to OpGen
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
heavy::ExternSyntax<> _HEAVY_import;

Context::Context()
  : ContinuationStack<Context>()
  , TrashHeap()
  , EnvStack(Empty())
  , MlirContext()
  , OpGen(nullptr)
{
  NameForImportVar = "_HEAVY_import";
  _HEAVY_import = heavy::base::import_;
  RegisterModule(HEAVY_BASE_LIB_STR, HEAVY_BASE_LOAD_MODULE);
}

Context::~Context() = default;

Environment::Environment(Environment* Parent)
  : ValueBase(ValueKind::Environment),
    OpGen(nullptr),
    Parent(Parent),
    EnvMap(0)
{ }
Environment::Environment(Context& C, std::string ModulePrefix)
  : ValueBase(ValueKind::Environment),
    OpGen(std::make_unique<heavy::OpGen>(C, std::move(ModulePrefix))),
    Parent(nullptr),
    EnvMap(0)
{ }
Environment::~Environment() = default;

Environment* Context::getTopLevelEnvironment() {
  // EnvStack is always an Environment or an improper
  // list ending with an Environment.
  if (auto* E = dyn_cast<Environment>(EnvStack)) {
    return E;
  }
  while (Pair* P = dyn_cast<Pair>(EnvStack)) {
    if (auto* E = dyn_cast<Environment>(P->Cdr)) {
      return E;
    }
  }
  llvm_unreachable("EnvStack should have an Environment.");
}

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
  return CreateStringHelper(getAllocator(), S);
}

String* Context::CreateString(llvm::StringRef S1, StringRef S2) {
  return CreateStringHelper(getAllocator(), S1, S2);
}

String* Context::CreateString(llvm::StringRef S1,
                              llvm::StringRef S2,
                              llvm::StringRef S3) {
  return CreateStringHelper(getAllocator(), S1, S2, S3);
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

Syntax* Context::CreateSyntaxWithOp(mlir::Operation* Op) {
  auto SyntaxOp = cast<heavy::SyntaxOp>(Op);
  auto Fn = [SyntaxOp](heavy::Context& C, ValueRefs Args) -> void {
    heavy::Value Input = Args[0];
    invokeSyntaxOp(C, SyntaxOp, Input);
    // The contained OpGenOp will call C.Apply(...).
  };
  return CreateSyntax(Fn);
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

#if 0 // TODO Make this create a module storing a function
              to initialize it from a ModuleOp
Module* Context::RegisterModule(mlir::Operation* ModuleOp) {
  // TODO Get the "load_module" FuncOp 
  mlir::ModuleOp Op = cast<mlir::ModuleOp>(ModuleOp);
  llvm::StringRef MangledName = Op.getName().getValueOr("");
  return RegisterModule(MangledName,
      std::make_unique<Module>(*this, ModuleOp));
}
#endif

void Context::Import(heavy::ImportSet* ImportSet) {
  Environment* Env = dyn_cast<Environment>(EnvStack);
  if (!Env) {
    // Import doesn't work in local scope.
    SetError("unexpected import", ImportSet);
    return;
  }

  // This should probably be recursive scheme calls.
  for (EnvBucket ImportVal : *ImportSet) {
    String* Name = ImportVal.first;
    // If there is no name just skip it (it was filtered out).
    if (!Name) continue;
    if (!Env->ImportValue(ImportVal)) {
      String* ErrMsg = CreateString("imported name already exists: ",
                                    Name->getView());
      SetError(ErrMsg, ImportSet);
      return;
    }
  }

  Cont();
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
  return ModuleOp;
}
void Context::dumpModuleOp() {
  return cast<mlir::ModuleOp>(ModuleOp).dump();
}

void Context::verifyModule() {
  mlir::ModuleOp M = cast<mlir::ModuleOp>(ModuleOp);
  if (mlir::failed(mlir::verify(M))) {
    llvm::errs() << "error: verfication failed\n";
  }
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

  void VisitExternName(ExternName* E) {
    OS << "#_" << E->getView();
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
    OS << S->getView();
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

void compile(Context& C, Value V, Value Env, Value Handler) {
  // TODO This should be the (heavy eval) module
  if (!HEAVY_BASE_IS_LOADED) {
    HEAVY_BASE_INIT(C);
  }
  heavy::Value Args[3] = {V, Env, Handler};
  C.Apply(HEAVY_BASE_VAR(compile), ValueRefs(Args));
}

void eval(Context& C, Value V, Value Env) {
  // TODO This should be the (heavy eval) module
  if (!HEAVY_BASE_IS_LOADED) {
    HEAVY_BASE_INIT(C);
  }
  heavy::Value Args[2] = {V, Env};
  C.Apply(HEAVY_BASE_VAR(eval), ValueRefs(Args));
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
    assert(M->Lookup(S) && "filtered name not in library");
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

// CreateImportSet - Return a created import set from an import-spec,
//                   or nullptr if a continuation was called
//                   (ie for an error or LoadModule).
//
void Context::CreateImportSet(Value Spec) {
  ImportSet::ImportKind Kind;
  Symbol* Keyword = dyn_cast_or_null<Symbol>(Spec.car());
  if (!Keyword) {
    SetError("expecting import set", Spec);
    return;
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
    PushCont([](Context& C, ValueRefs Args) {
      Module* M = cast<heavy::Module>(Args[0]);
      C.Cont(new (C.TrashHeap) ImportSet(M));
    });
    LoadModule(Spec);
    return;
  }

  Value ParentSpec = cadr(Spec);
  Spec = Spec.cdr().cdr();

  PushCont([Kind](Context& C, ValueRefs Args) {
    ImportSet* Parent = cast<ImportSet>(Args[0]);
    Value Spec = C.getCapture(0);

    if (Kind == ImportSet::ImportKind::Prefix) {
      Symbol* Prefix = dyn_cast_or_null<Symbol>(C.car(Spec));
      if (!Prefix) {
        C.SetError("expected identifier for prefix");
        return;
      }
      if (!isa_and_nonnull<Empty>(C.cdr(Spec))) {
        C.SetError("expected end of list");
        return;
      }
      C.Cont(new (C.TrashHeap) ImportSet(Kind, Parent, Prefix));
      return;
    }

    // Identifier (pairs) list
    // Check the syntax and that each provided name
    // exists in the parent import set
    if (!isa_and_nonnull<Pair>(Spec)) {
      C.SetError("expecting list");
      return;
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
         C.SetError("expected pair", P);
         return;
        }
        Name = dyn_cast<Symbol>(P2->Car);
        Symbol* Rename = dyn_cast_or_null<Symbol>(Value(P2).cadr());
        if (Name || Rename) {
          C.SetError("expected pair of identifiers", P2);
          return;
        }
        break;
      }
      default:
        llvm_unreachable("invalid import set kind");
      }

      EnvEntry LookupResult = Parent->Lookup(C, Name);
      if (!LookupResult) {
        C.SetError("name does not exist in import set");
        return;
      }
      Current = P->Cdr;
    }
    C.Cont(new (C.TrashHeap) ImportSet(Kind, Parent, Spec));
  }, CaptureList{Spec});
  CreateImportSet(ParentSpec);
}

void Context::LoadModule(Value Spec) {
  heavy::Mangler Mangler(*this);
  std::string Name = Mangler.mangleModule(Spec);
  std::unique_ptr<Module>& M = Modules[Name];
  if (M) {
    // Idempotently register the names of the globals.
    // (For external modules)
    M->LoadNames();
    Cont(M.get());
    return;
  }

  // Lookup the Module in ModuleOp
  mlir::ModuleOp TopOp = cast<mlir::ModuleOp>(ModuleOp);
  if (auto Op = dyn_cast_or_null<mlir::ModuleOp>(TopOp.lookupSymbol(Name))) {
    M = std::make_unique<Module>(*this);
    Value Args[] = {Op.getOperation()};
    PushCont([](Context& C, ValueRefs) {
      C.Cont(C.getCapture(0));
    }, CaptureList{M.get()});
    Apply(HEAVY_BASE_VAR(op_eval), Args);
    return;
  }

  // TODO
  // Load the file and compile the scheme code and try again.
  // We might need the SourceManager to accessible to Context.

  SetError("unable to load module (not supported)", Spec);
}

// Allow the user to add cleanup routines to unload a module/library.
// It is currently used for destroying the instance of OpEval.
// TODO expose this via run-time function accessible in scheme land.
void Context::PushModuleCleanup(llvm::StringRef MangledName, Value Fn) {
  std::unique_ptr<Module>& M = Modules[MangledName];
  if (!M) {
    return RaiseError("module not loaded",
        Value(CreateString(MangledName)));
  }
  if (heavy::Lambda* Lambda = dyn_cast<heavy::Lambda>(Fn)) {
    M->PushCleanup(Lambda); 
  } else {
    return RaiseError("expecting function", Fn);
  }
}

void Context::AddKnownAddress(llvm::StringRef MangledName, heavy::Value Value) {
  String* Name = CreateIdTableEntry(MangledName);
  KnownAddresses[Name] = Value;
}

Value Context::GetKnownValue(llvm::StringRef MangledName) {
  String* Name = CreateIdTableEntry(MangledName);
  // Could we possibly use dlsym or something here
  // for actual external values?
  return KnownAddresses.lookup(Name);
}

void heavy::initModule(heavy::Context& C, llvm::StringRef ModuleMangledName,
                  ModuleInitListTy InitList) {
  Module* M = C.Modules[ModuleMangledName].get();
  assert(M && "module must be registered");
  heavy::Mangler Mangler(C);
  for (ModuleInitListPairTy const& X : InitList) {
    Value Val = X.second;
    llvm::StringRef Id = X.first;
    std::string MangledName = Mangler.mangleVariable(ModuleMangledName, Id);
    registerModuleVar(C, M, MangledName, Id, Val);
#if 0
    String* Id = C.CreateIdTableEntry(X.first);
    Value Val = X.second;
    String* MangledName = C.CreateIdTableEntry(
        Mangler.mangleVariable(ModuleMangledName, Id));
    assert(Val && "value must not be nullptr");
    M->Insert(EnvBucket{Id, EnvEntry{Val, MangledName}});
#endif
    // Track valid values by their mangled names
    if (Val) {
      C.AddKnownAddress(MangledName, Val);
    }
  }
}

void heavy::registerModuleVar(heavy::Context& C,
                              heavy::Module* M,
                              llvm::StringRef VarSymbol,
                              llvm::StringRef VarId,
                              Value Val) {
  String* Id = C.CreateIdTableEntry(VarId);
  String* MangledName = C.CreateIdTableEntry(VarSymbol);
  M->Insert(EnvBucket{Id, EnvEntry{Val, MangledName}});
}

bool Context::CheckKind(ValueKind VK, Value V) {
  if (V.getKind() == VK) return false;
  String* S = CreateStringHelper(getAllocator(),
      llvm::StringRef("invalid type "),
      getKindName(V.getKind()),
      llvm::StringRef(", expecting "),
      getKindName(VK));
  RaiseError(S, V);
  return true;
}
bool Context::CheckNumber(Value V) {
  if (V.isNumber()) return false;
  String* S = CreateStringHelper(getAllocator(),
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
    return Cont();
  }
  if (!isa<Pair>(ExceptionHandlers)) {
    if (isa<heavy::Error>(Obj)) {
      Value BottomHandler = ExceptionHandlers;
      return Apply(BottomHandler, Obj);
    } else {
      std::string Msg;
      llvm::raw_string_ostream Stream(Msg);
      write(Stream << "uncaught exception: ", Obj);
      return SetError(Msg, Obj);
    }
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

namespace {
class LibraryEnv {
  // EnvPtr is meant to just keep the
  // object alive for the user when needed.
  std::unique_ptr<heavy::Environment> EnvPtr;
  heavy::Environment* Env;

public:
  LibraryEnv(std::unique_ptr<Environment> EP,
             heavy::Environment* E)
    : EnvPtr(std::move(EP)),
      Env(E)
  { }

  static void Wind(heavy::Context& Context,
                   std::unique_ptr<heavy::Environment> EnvPtr,
                   heavy::Environment* Env,
                   Value Thunk) {
    auto Ptr = std::make_unique<LibraryEnv>(std::move(EnvPtr), Env);
    heavy::OpGen* OpGen = Ptr->Env->GetOpGen();

    Value PrevEnv = Context.getEnvironment();
    heavy::OpGen* PrevOpGen = Context.OpGen;

    // Capture Env via scheme lambda so its references are checked
    // during garbage collection.
    Value Before = Context.CreateLambda([OpGen](heavy::Context& C,
                                                 ValueRefs) {
      Value Env = C.getCapture(0);
      C.setEnvironment(Env);
      C.OpGen = OpGen;
      C.Cont();
    }, CaptureList{Value((Ptr->Env))});

    // Since we don't own PrevEnv do not capture it in scheme land.
    Value After = Context.CreateLambda([PrevOpGen, PrevEnv, OpGen](heavy::Context& C,
                                                            ValueRefs) {
      C.setEnvironment(PrevEnv);
      C.OpGen = PrevOpGen;
      C.Cont();
    }, {});

    Context.DynamicWind(std::move(Ptr), Before, Thunk, After);
  }
};
} // namespace

void Context::WithEnv(std::unique_ptr<heavy::Environment> EnvPtr,
                      heavy::Environment* Env, Value Thunk) {
  LibraryEnv::Wind(*this, std::move(EnvPtr), Env, Thunk);
}

// Module

Module::~Module() {
  // Call cleanups pushed within the module
  // when the module is unloading.
  if (Cleanup) {
    Context.ApplyThunk(Cleanup);
  }
}

// Push a cleanup function to be called when the
// module is unloaded/destroyed.
void Module::PushCleanup(heavy::Lambda* Fn) {
  if (!Cleanup) {
    Cleanup = Fn;
    return;
  }
  Cleanup = Context.CreateLambda([](heavy::Context& C, ValueRefs) {
    heavy::Lambda* Next = cast<Lambda>(C.getCapture(0));
    heavy::Lambda* Fn   = cast<Lambda>(C.getCapture(1));
    C.PushCont(Value(Next));
    C.ApplyThunk(Fn);
  }, CaptureList{Cleanup, Fn});
}

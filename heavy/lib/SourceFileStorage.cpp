//===--- SourceFileStorage.cpp --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// System touching implementations for the ParseSourceFile hook in HeavyScheme
// and others.
//
//===----------------------------------------------------------------------===//

#include "heavy/Builtins.h"
#include "heavy/HeavyScheme.h"
#include "heavy/SourceFileStorage.h"
#include "heavy/SourceManager.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Path.h"

namespace heavy {

// HeavyScheme

void HeavyScheme::InitSourceFileStorage() {
  using PtrTy = decltype(SourceFileStoragePtr);
  SourceFileStoragePtr = PtrTy(new SourceFileStorage(),
                    [](SourceFileStorage* S) { delete S; });
  heavy::base::InitParseSourceFile(getContext(), 
                                   [this](heavy::Context& C,
                                          heavy::SourceLocation Loc,
                                          heavy::String* Filename) {
    heavy::Value Result = this->ParseSourceFile(Loc, Filename->getView());
    // The parser already raised an error.
    if (llvm::isa<heavy::Undefined>(Result))
      return;
    C.Cont(Result);
  });
}

void HeavyScheme::ProcessTopLevelCommands(llvm::StringRef Filename,
                          llvm::function_ref<ValueFnTy> ExprHandler,
                          llvm::function_ref<ErrorHandlerFn> ErrorHandler) {
  heavy::SourceManager& SM = getSourceManager();
  if (!SourceFileStoragePtr) {
    ErrorHandler("source file storage not initialized",
                 SM.getFullSourceLocation({}));
    return;
  }
  llvm::ErrorOr<heavy::SourceFile>
    FileResult = SourceFileStoragePtr->Open(SM, heavy::SourceLocation(),
                                            Filename);
  if (std::error_code ec = FileResult.getError()) {
    ErrorHandler((llvm::Twine("opening ", Filename) + 
                             llvm::Twine(": ", ec.message())).str(),
                 SM.getFullSourceLocation({}));
    return;
  }
  heavy::Lexer Lexer(FileResult.get());
  ProcessTopLevelCommands(Lexer, ExprHandler, ErrorHandler, tok::eof);
}

// This overload provides the default file system
// access for opening source files.
// The user must call InitSourceFileStorage.
heavy::Value HeavyScheme::ParseSourceFile(heavy::SourceLocation Loc,
                                          llvm::StringRef Filename) {
  assert(SourceFileStoragePtr &&
      "source file storage not initialized");
  llvm::ErrorOr<heavy::SourceFile>
    FileResult = SourceFileStoragePtr->Open(getSourceManager(), Loc, Filename);
  if (std::error_code ec = FileResult.getError()) {
    heavy::Context& C = getContext();
    C.RaiseError((llvm::Twine("\"", Filename) + 
                  llvm::Twine("\": ", ec.message())).str());
    return Undefined{};
  }
  heavy::Lexer Lexer(FileResult.get());
  return ParseSourceFile(Lexer);
}

llvm::ErrorOr<heavy::SourceFile>
SourceFileStorage::Open(heavy::SourceManager& SM,
                        heavy::SourceLocation IncludeLoc,
                        llvm::StringRef Filename) {
  // Include Loc may be invalid.
  heavy::SourceFile IncludeSF = SM.getFile(IncludeLoc);
  llvm::SmallString<128> FilePath;
  if (IncludeSF.isValid() && IncludeSF.Name != "-") {
    // Get the file path.
    FilePath = IncludeSF.Name;
    llvm::sys::path::remove_filename(FilePath);
  }

  FilePath += Filename;
  // Check LookupCache
  heavy::SourceLocation CachedLoc = LookupCache.lookup(FilePath); 
  if (CachedLoc.isValid()) {
    heavy::SourceFile CachedSF = SM.getFile(CachedLoc);
    assert(CachedSF.isValid() && "cached source file should be valid");
    return CachedSF;
  }

  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> File =
    llvm::MemoryBuffer::getFileOrSTDIN(FilePath);
  if (!File)
    return File.getError();

  // Ensure the name is stable in memory.
  NodeTy& Node = Storage.emplace_back(std::move(*File));
  return SM.createEntry(Node->getBuffer(), FilePath);
}

}

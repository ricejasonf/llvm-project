//===------- SourceManager.h - Heavy Scheme Source Manager -------- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file defines the interface for source files and locations.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_HEAVY_SOURCE_MANAGER_H
#define LLVM_HEAVY_SOURCE_MANAGER_H

#include "heavy/Source.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/MemoryBuffer.h"
#include <cassert>
#include <string>
#include <system_error>
#include <unordered_map>
#include <vector>

namespace heavy {
class SourceManager;

struct SourceLineStart {
  SourceLocation Loc = {};
  unsigned LineNumber = 1;

  // Compare by SourceLocation for searching
  // line numbers when printing diagnostics.
  // This only makes sense in the context of
  // a single SourceFile.
  bool operator<(SourceLineStart Other) {
    return Loc.getEncoding() < Other.Loc.getEncoding();
  }
};

class SourceFileStorage {
  friend class SourceManager;
  using StorageTy = std::unique_ptr<llvm::MemoryBuffer>;

  StorageTy Storage = {};
  std::string Name = {};

  llvm::StringRef getBuffer() const {
    return Storage->getBuffer();
  }

  std::error_code Open(llvm::StringRef Filename);

public:
  SourceFileStorage() = default;
};

class SourceManager {
  std::vector<std::unique_ptr<SourceFileStorage>> StoredEntries;

  std::vector<SourceFile> Entries;
  std::unordered_map<uintptr_t, SourceFile> ExternalLookup;

  SourceLocation getNextStartLoc();

  SourceFile createEntry(llvm::StringRef Buffer,
                         llvm::StringRef Name,
                         uintptr_t ExternalRawEncoding = 0);

public:

  SourceManager() = default;
  SourceManager(SourceManager const&) = delete;

  llvm::ErrorOr<SourceFile> Open(llvm::StringRef Filename);

  FullSourceLocation getFullSourceLocation(SourceLocation Loc) {
    return FullSourceLocation{*this, getFile(Loc), Loc};
  }

  SourceFile getFile(SourceLocation Loc) const;
  SourceFile getOrCreateExternal(uintptr_t RawEncoding,
                                 llvm::StringRef Buffer,
                                 llvm::StringRef Name);

  SourceLineStart getClosestLineStart(SourceFile File,
                                      SourceLocation Loc) const;
  char const* getBufferPos(SourceFile const& File,
                           SourceLocation Loc) const {
    return File.Buffer.data() + Loc.Loc;
  }

};

}

#endif

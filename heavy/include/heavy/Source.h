//===------------ Source.h - Heavy Scheme Source - --------------*- C++ -*-===//
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

#ifndef LLVM_HEAVY_SOURCE_H
#define LLVM_HEAVY_SOURCE_H

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/MemoryBuffer.h"
#include <cassert>
#include <string>
#include <system_error>
#include <unordered_map>
#include <vector>

namespace heavy {
class SourceManager;

// This exists for the reinterpret_cast<uintptr_t>
// stuff that mlir::OpaqueLoc does.
// (because they expect a pointer)
struct SourceLocationEncoding;

class SourceLocation {
  friend class SourceManager;
  unsigned Loc = 0;

  SourceLocation(unsigned Loc) : Loc(Loc) { }

public:
  SourceLocation() = default;
  SourceLocation(SourceLocation const&) = default;

  SourceLocation(SourceLocationEncoding* E)
    : Loc(reinterpret_cast<uintptr_t>(E))
  { }

  // getLocWithOffset - The caller is responsible for ensuring
  //                    that the offset does not go out of bounds
  //                    for the containing buffer which is null
  //                    terminated.
  SourceLocation getLocWithOffset(unsigned Offset) const {
    return SourceLocation(Loc + Offset);
  }

  unsigned getOffsetFrom(SourceLocation Start) {
    assert(Start.Loc < Loc && "location must exist after start location");
    return Loc - Start.Loc;
  }

  bool isValid() const {
    return Loc != 0;
  }

  unsigned getEncoding() const { return Loc; }
  SourceLocationEncoding* getOpaqueEncoding() const {
    uintptr_t E = Loc;
    return reinterpret_cast<SourceLocationEncoding*>(E);
  };
};

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

struct SourceLineContext {
  llvm::StringRef LineRange = {};
  // Column - The offset of the location of interest in LineBuffer.
  unsigned Column = 0;
  unsigned LineNumber = 0;
};

class SourceFileStorage {
  friend class SourceManager;
  using StorageTy = std::unique_ptr<llvm::MemoryBuffer>;

  StorageTy Storage = {};
  std::string Name = {};

  llvm::StringRef getBuffer() const {
    return Storage->getBuffer();
  }

  std::error_code Open(llvm::StringRef Filename) {
    llvm::ErrorOr<StorageTy> File =
      llvm::MemoryBuffer::getFileOrSTDIN(Filename);
    if (!File) return File.getError();
    Name = Filename.str();
    Storage = std::move(File.get());
    return std::error_code();
  }

public:
  SourceFileStorage() = default;
};

struct SourceFile {
  llvm::StringRef Buffer;
  llvm::StringRef Name;
  SourceLocation StartLoc;
  // ExternalLocRawEncoding - Used to store clang::SourceLocation
  uintptr_t ExternalRawEncoding = 0;

  // Store LineStart positions for like every 50 lines
  // to make searching faster without using too much
  // memory.
  // TODO Maybe remove this since we don't populate it.
  std::vector<SourceLineStart> LineStarts = {};

  bool isValid() const { return StartLoc.isValid(); }

  bool hasLoc(SourceLocation Loc) {
    unsigned L = Loc.getEncoding();
    unsigned Start = StartLoc.getEncoding();
    unsigned End = Start + Buffer.size();
    return L >= Start && L < End;
  }

  SourceLocation getStartLocation() const {
    return StartLoc;
  }

  bool isExternal() const { return ExternalRawEncoding > 0; }
};

class SourceFileId {
  unsigned IdValue = 0;
public:
  SourceFileId(unsigned IndexOffset) {
    IdValue = IndexOffset + 1;
  }

  bool isValid() const {
    return IdValue > 0;
  }

  unsigned getOffset() const {
    return IdValue - 1;
  }
};

class FullSourceLocation {
  friend SourceManager;
  SourceManager& Manager;
  SourceFileId FileId;
  SourceLocation Loc;

  FullSourceLocation(SourceManager& SM, SourceFileId FileId, SourceLocation Loc)
    : Manager(SM),
      FileId(FileId),
      Loc(Loc)
  { }

public:
  SourceLocation getLocation() const {
    return Loc;
  }

  uintptr_t getExternalRawEncoding() const {
    return Manager.getFile(FileId).ExternalRawEncoding;
  }

  // getOffset - returns offset of location from
  //             the start of the file
  unsigned getOffset() {
    return Loc.getOffsetFrom(Manager.getFile(FileId).StartLoc);
  }

  bool isValid() const {
    return Loc.isValid();
  }

  // FIXME: This is definitely big enough to go in a source file.
  SourceLineContext getLineContext() const {
    SourceFile& File = Manager.getFile(FileId);
    char const* const FileStartPos = File.Buffer.begin();
    char const* const FileEndPos = File.Buffer.eegin();
    // Find the closest LineStart
    auto& LineStarts = File.LineStarts;
    auto Itr = std::upper_bound(LineStarts.begin(),
                                LineStarts.end());
    Closest = (Itr == LineStarts.end()) ? *Itr : SourceLineStart{};
    // Now scan lines until we get to the desired location.
    char const* const TargetPos = Manager.getBufferPos(FileId, Loc);
    char const* CurPos = Manager.getBufferPos(FileId, Closest.Loc);
    unsigned CurrentLineNumber = Closest.LineNumber;
    char const* LineStartPos = CurPos;
    if (CurPos <= TargetPos) {
      // Traverse forward saving the last noted LineStart.
      while (TargetPos != CurPos) {
        if (CurPos == FileEndPos) break;
        // If we see a newline character
        if (CurPos == '\n') {
          ++CurrentLineNumber;
          LineStartPos = CurPos + 1;
        }
        ++CurPos;
      }
    } else {
      // Traverse backward going to one LineStart beyond the TargetPos.
      while (TargetPos != CurPos) {
        if (CurPos == FileStartPos) break;
        // If we see a newline character
        if (CurPos == '\n') {
          --CurrentLineNumber;
        }
        --CurPos;
      }
      while (*CurPos != '\n') {
        if (CurPos == FileStartPos) break;
        --CurPos;
      }
      LineStartPos = (CurPos == FileStartPos) ? FileStartPos : CurPos + 1;
    }
    return SourceLineContext{llvm::StringRef(LineStartPos, LineEndPos),
                             TargetPos - LineStartPos,
                             CurrentLineNumber};
  }
};

class SourceManager {
  std::vector<std::unique_ptr<SourceFileStorage>> StoredEntries;
  std::vector<SourceFile> Entries;
  std::unordered_map<uintptr_t, SourceFile> ExternalLookup;

  SourceLocation getNextStartLoc() {
    // Start Loc at 1 as 0 represents
    // an invalid location
    if (Entries.size() == 0) return 1;
    SourceFile LastFile = Entries.back();
    SourceLocation Start = LastFile.StartLoc;
    unsigned Len = LastFile.Buffer.size();
    return SourceLocation(Start.Loc + Len);
  }

  SourceFile createEntry(llvm::StringRef Buffer,
                         llvm::StringRef Name,
                         uintptr_t ExternalRawEncoding = 0) {
    SourceLocation StartLoc = getNextStartLoc();
    Entries.push_back(SourceFile{Buffer, Name, StartLoc,
                                 ExternalRawEncoding});
    return Entries.back();
  }


public:

  SourceManager() = default;
  SourceManager(SourceManager const&) = delete;

  llvm::ErrorOr<SourceFile> Open(llvm::StringRef Filename) {
    auto File = std::make_unique<SourceFileStorage>();
    if (std::error_code ec = File->Open(Filename)) {
      return ec;
    }
    SourceFile Entry = createEntry(File->getBuffer(), File->Name);
    StoredEntries.push_back(std::move(File));
    return Entry;
  }

  FullSourceLocation getFullSourceLocation(SourceLocation Loc) {
    return FullSourceLocation{*this, getFileId(Loc), Loc};
  }

  // Note that the references to SourceFile should not be stored.
  SourceFile& getFile(SourceFileId Id) const {
    return Id.isValid() ? Entries[Id.getOffset()] : SourceFile{};
  }

  SourceFile& getFile(SourceLocation Loc) const {
    for (SourceFile const& X : Entries) {
      if (X.hasLoc(Loc)) return X;
    }
    return SourceFile{};
  }

  SourceFile& getFileId(SourceLocation Loc) const {
    for (SourceFile const& X : Entries) {
      if (X.hasLoc(Loc)) return X;
    }
    return SourceFile{};
  }

  SourceFile getOrCreateExternal(uintptr_t RawEncoding,
                                 llvm::StringRef Buffer,
                                 llvm::StringRef Name) {
    auto itr = ExternalLookup.find(RawEncoding);
    if (itr == ExternalLookup.end()) {
      SourceFile File = createEntry(Buffer, Name, RawEncoding);
      ExternalLookup.insert({RawEncoding, File});
      return File;
    }
    return (*itr).second;
  }

  char const* getBufferPos(SourceFileId Id, SourceLocation Loc) const {
    return getFile(Id).Buffer[Loc.Loc];
  }
};

} // end namespace heavy

#endif

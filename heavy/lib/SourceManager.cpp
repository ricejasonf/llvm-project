//===--------- SourceManager.cpp - HeavyScheme SourceManager --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file implements the SourceManager and related classes.
//
//===----------------------------------------------------------------------===//

#include "heavy/Source.h"
#include "heavy/SourceManager.h"
#include "llvm/ADT/StringRef.h"
#include <cassert>

namespace heavy {

SourceLocation
SourceManager::getNextStartLoc() {
  // Start Loc at 1 as 0 represents
  // an invalid location
  if (Entries.empty()) return 1;
  SourceFile LastFile = Entries.back();
  SourceLocation Start = LastFile.StartLoc;
  unsigned Len = LastFile.Buffer.size();
  return SourceLocation(Start.Loc + Len);
}

SourceFile
SourceManager::createEntry(llvm::StringRef Buffer,
                           llvm::StringRef Name,
                           uintptr_t ExternalRawEncoding) {
  SourceLocation StartLoc = getNextStartLoc();
  Entries.push_back(SourceFile{Buffer, Name, StartLoc,
                               ExternalRawEncoding});
  return Entries.back();
}

llvm::ErrorOr<SourceFile>
SourceManager::Open(llvm::StringRef Filename) {
  auto File = std::make_unique<SourceFileStorage>();
  if (std::error_code ec = File->Open(Filename)) {
    return ec;
  }
  SourceFile Entry = createEntry(File->getBuffer(), File->Name);
  StoredEntries.push_back(std::move(File));
  return Entry;
}

SourceFile
SourceManager::getFile(SourceLocation Loc) const {
  for (SourceFile const& X : Entries) {
    if (X.hasLoc(Loc)) return X;
  }
  return SourceFile{};
}

#if 0 // TODO remove
SourceFileId
SourceManager::getFileId(SourceLocation Loc) const {
  unsigned I = 0;
  for (SourceFile const& X : Entries) {
    ++I;
    if (X.hasLoc(Loc)) return SourceFileId{I};
  }
  return SourceFileId{};
}
#endif

SourceFile
SourceManager::getOrCreateExternal(uintptr_t RawEncoding,
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

SourceLineStart
SourceManager::getClosestLineStart(SourceFile File,
                                   SourceLocation Loc) const {
  // Just get the beginning to the File at line 0.
  // NOTE: We could possibly "optimize" this by
  //       storing a few SourceLineStarts in a
  //       vector and using std::upper_bound.
  //       (like every 50 lines)
  //       Right now we don't expect large files.
  return SourceLineStart{File.StartLoc, /*Line=*/0};
}

// FullSourceLocation

SourceLineContext
FullSourceLocation::getLineContext() const {
  char const* const FileStartPos = File.Buffer.begin();
  char const* const FileEndPos = File.Buffer.end();
  // We could store SourceLineStarts to help with speeding
  // up searching instead of scanning the entire file every
  // time we need line numbers.
  SourceLineStart Closest = Manager.getClosestLineStart(File, Loc);
  // Now scan lines until we get to the desired location.
  char const* const TargetPos = Manager.getBufferPos(File, Loc);
  char const* CurPos = Manager.getBufferPos(File, Closest.Loc);
  unsigned CurrentLineNumber = Closest.LineNumber;
  char const* LineStartPos = CurPos;
  if (CurPos <= TargetPos) {
    // Traverse forward saving the last noted LineStart.
    while (TargetPos != CurPos) {
      if (CurPos == FileEndPos) break;
      // If we see a newline character
      if (*CurPos == '\n') {
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
      if (*CurPos == '\n') {
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
  // Traverse forwarrd to get LineEndPos.
  char const* LineEndPos = LineStartPos;
  while (*LineEndPos != '\n' ||
         *LineEndPos != '\r' ||
         LineEndPos != FileEndPos) {
    ++LineEndPos;
  }
  unsigned LineLen = LineEndPos - LineStartPos;

  return SourceLineContext{File.Name,
                           llvm::StringRef(LineStartPos, LineLen),
                           static_cast<unsigned>(TargetPos - LineStartPos),
                           CurrentLineNumber};
}

// SourceFileStorage

std::error_code
SourceFileStorage::Open(llvm::StringRef Filename) {
  llvm::ErrorOr<StorageTy> File =
    llvm::MemoryBuffer::getFileOrSTDIN(Filename);
  if (!File) return File.getError();
  Name = Filename.str();
  Storage = std::move(File.get());
  return std::error_code();
}

}

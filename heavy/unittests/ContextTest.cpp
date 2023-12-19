//=== - ContextTest.cpp - Tests for heavy::Context --*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===--------------------------------------------------===//

#include "heavy/Context.h"
#include "heavy/Value.h"

#include <gtest/gtest.h>

namespace {

struct LifetimeTracker {
  bool& IsAlive;

  LifetimeTracker(bool& IsAlive_)
    : IsAlive(IsAlive_)
  {
    IsAlive = true;
  }

  LifetimeTracker(LifetimeTracker const&) = delete;

  ~LifetimeTracker() {
    IsAlive = false;
  }
};

TEST(ContextTest, DynamicWindNormalExit) {
  auto Context = heavy::Context();
  bool IsAlive = false;
  bool Check_1 = false;
  auto TrackerPtr = std::make_unique<LifetimeTracker>(IsAlive);

  auto ThunkFn = [&](heavy::Context& C, heavy::ValueRefs) {
    Check_1 = IsAlive;
    // Continue normally.
    C.Cont();
  };

  heavy::ExternLambda<0, sizeof(ThunkFn)> Thunk;
  Thunk = ThunkFn;

  // Test the lifetime of managed objects
  // in a dynamic-wind environment.
  Context.DynamicWind(std::move(TrackerPtr), Thunk);
  EXPECT_TRUE(IsAlive);
  Context.Resume();
  EXPECT_TRUE(Check_1);
  EXPECT_FALSE(IsAlive);
}

TEST(ContextTest, DynamicWindExceptionExit) {
  auto Context = heavy::Context();
  bool IsAlive = false;
  bool Check_1 = false;
  auto TrackerPtr = std::make_unique<LifetimeTracker>(IsAlive);

  auto ThunkFn = [&](heavy::Context& C, heavy::ValueRefs) {
    Check_1 = IsAlive;
    // Fail gloriously!
    C.Raise(heavy::Undefined());
  };

  heavy::ExternLambda<0, sizeof(ThunkFn)> Thunk;
  Thunk = ThunkFn;

  // Test the lifetime of managed objects
  // in a dynamic-wind environment.
  Context.DynamicWind(std::move(TrackerPtr), Thunk);
  EXPECT_TRUE(IsAlive);
  Context.Resume();
  EXPECT_TRUE(Check_1);
  EXPECT_FALSE(IsAlive);
}

}

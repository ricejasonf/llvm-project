// Nbdl.td - MLIR Dialect Definitions for Nbdl
//
// Copyright Jason Rice 2025
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)
//
#ifndef NBDL_DIALECT_OPS_TD
#define NBDL_DIALECT_OPS_TD

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/OpBase.td"

def Nbdl_Dialect : Dialect {
  let name = "nbdl_gen";
  let useDefaultTypePrinterParser = 1;
}

// We keep cpp_typename around for when
// we want to go back to c++.
class Nbdl_TypeBase<string name, string type_mnemonic,
                    list<Trait> traits = []>
  : TypeDef<Nbdl_Dialect, name, traits> {
  let mnemonic = type_mnemonic;
  let parameters = (ins StringRefParameter<>:$cpp_typename);
  let assemblyFormat = "`<` $cpp_typename `>`";
}

def Nbdl_OpaqueType : Nbdl_TypeBase<"OpaqueType", "opaque_type"> {
  let description = [{
    If a C++ type is denoted without any additional semantic
    information, then we call it opaque. This can be useful
    for visitor functions, keys etc.
  }];
}

def Nbdl_State : Nbdl_TypeBase<"State", "state"> {
  let summary = "Nbdl State type";
  let description = [{
    In Nbdl, a State object uses a static key to access
    elements where paths are resolved at compile-time.
  }];
}

def Nbdl_Store : Nbdl_TypeBase<"Store", "store"> {
  let summary = "Nbdl Store type";
  let description = [{
    In Nbdl, a Store object has state that may have multiple
    alternative paths that are determined at run-time.
    A State object is also a Store.
  }];
}

def Nbdl_Tag : Nbdl_TypeBase<"Tag", "tag_type"> {
  let summary = "Nbdl tag type";
  let description = [{
    In Nbdl, an empty type that only carries information
    at compile-time is called a tag type.
  }];
}

def Nbdl_Type : AnyTypeOf<[Nbdl_OpaqueType,
                          Nbdl_Store,
                          Nbdl_State,
                          Nbdl_Tag]>;

// Keys are optional, but we are using variadic arguments
// so we resort to using this sum type.
def Nbdl_Key : AnyTypeOf<[Nbdl_OpaqueType, NoneType]>;

class Nbdl_Op<string mnemonic, list<Trait> traits = []> :
    Op<Nbdl_Dialect, mnemonic, traits>;

def Nbdl_TagOp : Nbdl_Op<"tag", []> {
  let summary = "tag";
  let description = [{
    Create an instance of a tag_type.
    These are intended primarily for use as
    keys and allow avoiding capture of stateless
    values.
  }];

  let results   = (outs Nbdl_Tag:$result);
}

def Nbdl_GetOp : Nbdl_Op<"get", []> {
  let summary = "get";
  let description = [{
  }];
  let arguments = (ins Nbdl_State:$state,
                       Nbdl_Key:$key);
  let results   = (outs Nbdl_OpaqueType:$result);
}

def Nbdl_MatchOp : Nbdl_Op<"match", [Terminator, NoTerminator]> {
  let summary = "match";
  let description = [{
    Given a continuation function, a nbdl::Store and, optionally, a key,
    `match` visits an element contained within that Store using a
    continuation for each of the possible typed alternatives.

    Stores match values of different types, so a region is
    used to provide the continuation for each possible
    alternative which is checked linearly.

    It is an error for the matched object to not have a matching
    alternative.

    For an example, a `std::unordered_map<int, std::string>` is a store
    that can provide access to a contained element with a key `5`. If the
    element does not exist some `not_found` alternative is matched.
    Alternatively, a `std::variant<Foo, Bar>` is a store that
    also provides access to its contained element without a key
    (ie it is just unwrapped). This requires a visitor to be overloaded
    with every possible alternative.

    Each region receives fn, store, and the captures as their
    arguments.
  }];
  let arguments = (ins Nbdl_OpaqueType:$fn,
                       AnyTypeOf<[Nbdl_Store, Nbdl_State]>:$store,
                       Nbdl_Key:$key,
                       Variadic<Nbdl_Type>:$captures);
  let results = (outs);
  let regions = (region SizedRegion<1>:$overloads);
}

def Nbdl_OverloadOp : Nbdl_Op<"overload", []> {
  let description = [{
    In the region of a MatchOp we specify a function overload of the
    given Nbdl_OpaqueType using OverloadOp.
    Each overload is checked linearly (sequentially) and a valid
    overload serves as a continuation.
    It is an error if no overload matches the matched alternative.
  }];
  let arguments = (ins Nbdl_Type:$arg,
                       Variadic<Nbdl_Type>:$captures);
  let results = (outs);
}

def Nbdl_ContOp : Nbdl_Op<"cont", [Terminator]> {
  let summary = "cont";
  let description = [{
    ContOp represents the call to an opaque continuation object
    passing the resolved object and any additional captures
    as arguments. The `fn_id` is the name used to lookup the function
    within the scope of the code being generated.
  }];
  let arguments = (ins StrAttr:$fn_id,
                       Variadic<Nbdl_Type>:$args);
  let results = (outs);
}

#endif

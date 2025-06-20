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
  let name = "nbdl";
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
    Represent an unknown C++ type.
  }];
}

def Nbdl_Struct : Nbdl_TypeBase<"Struct", "struct"> {
  let summary = "C++ semiregular aggregate type";
  let description = [{
    Represent C++ semiregular types whose members
    are accessed by their name.

    Struct is equivalent to State except that there is no
    way to access the members without reflection or generated
    code.
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

def Nbdl_Variant : Nbdl_TypeBase<"Variant", "variant"> {
  let summary = "Nbdl Variant type";
  let description = [{
    Variant is a store that is a sum type with a type index.
  }];
}

def Nbdl_Tag : Nbdl_TypeBase<"Tag", "tag_type"> {
  let summary = "Nbdl tag type";
  let description = [{
    In Nbdl, an empty type that only carries information
    at compile-time is called a tag type.
  }];
}

def Nbdl_Symbol : Nbdl_TypeBase<"Symbol", "symbol_type"> {
  let summary = "Nbdl symbol type";
  let description = [{
    Represent a string that is a valid C++ identifier.
    The intended primary use case is to specify a member of a Struct.
  }];
}

// Keys are optional, but we are using variadic arguments
// so we resort to using this sum type.
def Nbdl_KeyType : AnyTypeOf<[Nbdl_OpaqueType, Nbdl_Tag,
                              Nbdl_Symbol, NoneType]>;

def Nbdl_Type : AnyTypeOf<[Nbdl_OpaqueType,
                           Nbdl_Store,
                           Nbdl_Struct,
                           Nbdl_State,
                           Nbdl_Tag]>;

def Nbdl_TagAttr : AttrDef<Nbdl_Dialect, "tag_attr"> {
  let attrName = "nbdl.tag_attr";
  let parameters = (ins "::mlir::StringAttr":$cppTypeName);
}

def Nbdl_TypenameAttr : AttrDef<Nbdl_Dialect, "typename_attr"> {
  let attrName = "nbdl.typename_attr";
  let parameters = (ins "::mlir::StringAttr":$cppTypeName);
}

def Nbdl_SymbolAttr : AttrDef<Nbdl_Dialect, "symbol_attr"> {
  let attrName = "nbdl.symbol_attr";
  let parameters = (ins "::mlir::StringAttr":$cppIdentifier);
}

def Nbdl_KeyAttr : AnyAttrOf<[Nbdl_TagAttr,
                              Nbdl_TypenameAttr,
                              Nbdl_SymbolAttr]>;

class Nbdl_Op<string mnemonic, list<Trait> traits = []> :
    Op<Nbdl_Dialect, mnemonic, traits>;

// FIXME I think KeyOp is redundant with KeyAttr
/*
def Nbdl_KeyOp : Nbdl_Op<"key", []> {
  let summary = "key";
  let description = [{
    Create an instance of a stateless key object.
    These are intended primarily to allow avoiding
    capture of stateless values.
  }];

  let arguments = (ins Nbdl_KeyAttr:$key);
  let results   = (outs Nbdl_Keytype:$result);
}
*/

def Nbdl_GetOp : Nbdl_Op<"get", []> {
  let summary = "get";
  let description = [{
  }];
  let arguments = (ins Nbdl_State:$state,
                       Nbdl_KeyType:$key);
  let results   = (outs Nbdl_OpaqueType:$result);
}

def Nbdl_MatchOp : Nbdl_Op<"match", [Terminator, NoTerminator]> {
  let summary = "match";
  let description = [{
    Given a nbdl::Store and, optionally, a key,
    `match` visits an element contained within that Store using a
    continuation for each of the possible specified overloads.

    It is an error if there exists an alternative for which there
    exists no matching overload.

    For an example, a `std::unordered_map<int, std::string>` is a store
    that can provide access to a contained element with a key `5`. If the
    element does not exist some `not_found` alternative is matched.
    Alternatively, a `std::variant<Foo, Bar>` is a store that
    also provides access to its contained element without a key
    (ie it is just unwrapped). This requires a visitor to be overloaded
    with every possible alternative.
  }];
  // FIXME $store should only be a Nbdl_Store right?
  let arguments = (ins Nbdl_Store:$store,
                       Nbdl_KeyType:$key);
  let results = (outs);
  let regions = (region SizedRegion<1>:$overloads);
}

def Nbdl_OverloadOp : Nbdl_Op<"overload", []> {
  let description = [{
    In the region of a MatchOp we specify a function overload of the
    given Nbdl_TypenameAttr using OverloadOp.
    The body of the overload takes an argument that is the matched
    object and the rest of the arguments are the captures.
  }];
  let arguments = (ins Nbdl_TypenameAttr:$typeName,
                       Variadic<Nbdl_Type>:$captures);
  let results = (outs);
  let regions = (region SizedRegion<1>:$body);
}

def Nbdl_MatchIfOp : Nbdl_Op<"match_if", [Terminator]> {
  let description = [{
    Apply a predicate to an object and branch on the result
    like an if/else statement.
  }];

  let arguments = (ins Nbdl_Type:$obj, Nbdl_OpaqueType:$pred);
  let regions   = (region SizedRegion<1>:$thenRegion,
                          SizedRegion<1>:$elseRegion);
}

def Nbdl_ContOp : Nbdl_Op<"cont", [Terminator]> {
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

def Nbdl_ConstexprOp : Nbdl_Op<"constexpr", []> {
  let description = [{
    The result of a constant expression in C++.
  }];
  let arguments = (ins StrAttr:$expr);
  let results = (outs Nbdl_OpaqueType:$result);
}

def Nbdl_StoreComposeOp : NbdlOp<"store_compose", []> {
  let description = [{
    Add a store $rhs as a member to another store $lhs using a key $key
    to access the member. The set of keys in $lhs are shadowed by their
    intersection with $key.
  }];

  let arguments = (ins Nbdl_Type:$lhs, Nbdl_Type:$rhs, Nbdl_KeyAttr:$key);
  let results = (outs Nbdl_Type:$result);
}

def Nbdl_VariantOp : NbdlOp<"variant", []> {
  let description = [{
    Create a sum type provided stores.

    VariantOps can be folded so the type index
    is unspecified.
  }];
  let arguments = (ins Variadic<Nbdl_Type>:$operands);
  let results = (outs Nbdl_Variant:$result);
}

def Nbdl_CreateStoreOp : NbdlOp<"create_store", [Symbol, IsolatedFromAbove]> {
  let description = [{
    If no region exists, then the operation is considered external
    and should refer to an existing C++ type that satisfies nbdl::Store.
    Otherwise, a new C++ struct type is created. The symbol should be a
    fully namespace qualified C++ type name.

    Use ContOp to terminate and finalize the object.
    If ContOp receives the body argument directly then the
    created type is a Tag.
  }];
  let regions = (region AnyRegion:$body);
  let arguments = (ins FlatSymbolRefAttr:$name);
  let results = (outs);
}

#endif

set(LLVM_TARGET_DEFINITIONS Ops.td)
mlir_tablegen(Ops.h.inc -gen-op-decls)
mlir_tablegen(Ops.cpp.inc -gen-op-defs)
mlir_tablegen(Attrs.h.inc -gen-attrdef-decls) # Not used
mlir_tablegen(Attrs.cpp.inc -gen-attrdef-defs)
add_public_tablegen_target(HeavyOpsIncGen)

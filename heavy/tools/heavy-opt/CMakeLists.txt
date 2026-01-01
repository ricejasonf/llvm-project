cmake_minimum_required(VERSION 3.13.4)

add_llvm_executable(heavy-opt heavy_opt_main.cpp)
llvm_map_components_to_libnames(LLVM_LIBS core support)

target_link_libraries(heavy-opt
  PRIVATE
  MLIRBytecodeWriter
  MLIRDebug
  MLIROptLib
  MLIRParser
  MLIRPass
  MLIRRemarkStreamer
  MLIRSupport
  heavy
  )

add_llvm_install_targets(install-heavy-opt
  DEPENDS heavy-opt
  COMPONENT heavy-opt)

install(TARGETS heavy-opt
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        COMPONENT heavy-opt)

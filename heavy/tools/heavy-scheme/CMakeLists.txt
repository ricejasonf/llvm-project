cmake_minimum_required(VERSION 3.13.4)

add_llvm_executable(heavy-scheme heavy_main.cpp)
llvm_map_components_to_libnames(LLVM_LIBS core support)

target_link_libraries(heavy-scheme PRIVATE heavy ${LLVM_LIBS})
set_target_properties(heavy-scheme PROPERTIES ENABLE_EXPORTS 1)
            
add_llvm_install_targets(install-heavy-scheme
  DEPENDS heavy-scheme
  COMPONENT heavy-scheme)

install(TARGETS heavy-scheme
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        COMPONENT heavy-scheme)

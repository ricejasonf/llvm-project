cmake_minimum_required(VERSION 3.4.3)

add_llvm_executable(heavy-scheme heavy_main.cpp)
llvm_map_components_to_libnames(LLVM_LIBS core support)

target_link_libraries(heavy-scheme PRIVATE heavy ${LLVM_LIBS})
            

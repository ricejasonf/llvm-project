cmake_minimum_required(VERSION 3.4.3)

if(POLICY CMP0075)
  cmake_policy(SET CMP0075 NEW)
endif()

add_llvm_executable(heavy heavy_main.cpp
             ${HEAVY_LIB}/HeavyScheme.cpp
             ${HEAVY_LIB}/Lexer.cpp
             ${HEAVY_LIB}/Parser.cpp
             ${HEAVY_LIB}/../../clang/lib/Basic/CharInfo.cpp)

llvm_map_components_to_libnames(LLVM_LIBS core support)
target_link_libraries(heavy PRIVATE ${LLVM_LIBS})
            

#include <heavy/Context.h>
#include <heavy/Value.h>

extern "C" {
heavy::ContextLocal heavy_hello_world_ultimate_answer;

// This is only used to test the binding.
void heavy_hello_world_get_ultimate_answer(heavy::Context& C,
                                           heavy::ValueRefs) {
  heavy::Value V = heavy_hello_world_ultimate_answer.get(C);
  C.Cont(V);
}

void heavy_hello_world_my_write(heavy::Context& C,
                                heavy::ValueRefs Args) {
  heavy::write(llvm::outs(), Args[0]);
  C.Cont();
}
}

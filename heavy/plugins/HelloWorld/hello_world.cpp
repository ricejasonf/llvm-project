#include <heavy/Context.h>
#include <heavy/Value.h>

extern "C" {
void heavy_hello_world_compute_answer(heavy::Context& C, heavy::ValueRefs) {
  C.Cont(heavy::Int(42));
}

void heavy_hello_world_my_write(heavy::Context& C, heavy::ValueRefs Args) {
  heavy::write(llvm::outs(), Args[0]);
  C.Cont();
}
}

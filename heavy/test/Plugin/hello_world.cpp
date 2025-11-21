// RUN: clang++ -I %S/Inputs -fsyntax-only -Xclang -fheavy -Xclang -verify %s
// expected-no-diagnostics

heavy_scheme {
(import (heavy builtins)
        (heavy clang))

(load-plugin "libheavyHelloWorld.so")

(define compute-answer
  (load-builtin "heavy_hello_world_compute_answer"))

(define forty-two 'forty-two)

(write-lexer forty-two "static constexpr int forty_two = ")
(write-lexer forty-two (number->string (compute-answer)))
(write-lexer 0 ";")
}

static_assert(forty_two == 42);
int main() { }

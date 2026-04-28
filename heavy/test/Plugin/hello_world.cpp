// RUN: clang++ -I %S/Inputs -fsyntax-only -fplugin=HeavyClang.so -Xclang -verify %s
// expected-no-diagnostics

#pragma heavy_scheme
{
(import (heavy builtins))

(define-library (test lib)
  (export my-write
          get-ultimate-answer)
  (import (heavy builtins)
          (heavy clang))
  (begin
    (load-plugin "libheavyHelloWorld.so")
    (define my-write
      (load-builtin "heavy_hello_world_my_write"))
    (define get-ultimate-answer
      (load-builtin "heavy_hello_world_get_ultimate_answer"))
    (define-binding ultimate-answer
                    heavy_hello_world_ultimate_answer)
    (set! ultimate-answer 42)
  ));

(import (test lib)
        (heavy clang))

(define forty-two 'forty-two)

(write-lexer forty-two "static constexpr int forty_two = ")
(write-lexer forty-two (number->string (get-ultimate-answer)))
(write-lexer 0 ";")
}

static_assert(forty_two == 42);
int main() { }

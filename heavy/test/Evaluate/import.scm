; RUN: heavy-scheme %s | FileCheck %s

(import (except
          (only (heavy builtins)
                define lambda write newline
                apply)
          apply)
        (rename (prefix (heavy builtins) builtins.)
                (builtins.apply woof.renamed-apply))
        (rename (heavy builtins)
                (apply my-apply)
                #;(apply moo-apply))
        )

; The name apply is not imported.
(define apply 0)

; The name builtins.apply is not imported.
(define builtins.apply 0)

; CHECK: (2 2)
(write (woof.renamed-apply list 2 '(2)))
(newline)

; CHECK: (3 2)
(write (my-apply list 3 '(2)))
(newline)

; FIXME Multiple renames should be allowed.
; COM-CHECK: (4 2)
;(write (moo-apply list 4 '(2)))
;(newline)

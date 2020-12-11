; RUN: heavy-scheme -fread-only %s
; XFAIL: *
""
"this is fine"
"this is fine with new line
"

"this string does not have an ending
; expected-error@-1 {{unterminated string literal}}


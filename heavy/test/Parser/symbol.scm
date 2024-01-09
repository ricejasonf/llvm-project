; RUN: heavy-scheme --mode=read %s 2>&1 | FileCheck  %s --strict-whitespace --match-full-lines
; Escaped symbols should behave as string literals.
; Invalid identitifiers must write as escaped symbols.
; Otherwise, they should write as identifiers.

; CHECK:||
||

; Valid identifiers should write as identifiers.
; CHECK:foo
|foo|
; CHECK:bar
|bar|
; CHECK:hello
|hello|

; Support relaxed identifiers and write as escaped symbols.
; CHECK:|\\lambda|
\lambda
; CHECK:|foo\\lambda|
foo\lambda

; Support printable UTF8 characters
; as relaxed identifiers.
; CHECK:|λ|
λ
; CHECK:|ΣλΣ|
ΣλΣ
; CHECK:|ΣλΣfooΣλΣ|
ΣλΣfooΣλΣ

; Invalid identifiers should write as escaped symbols.
; CHECK:|foo\\nbar|
|foo\\nbar|
; CHECK:|escape\nnewline|
|escape\nnewline|

; Support mnemonic escapes
; CHECK:|\a\b\t\n\r"\|\\|
|\a\b\t\n\r\"\|\\|

; All relevant hex codes should
; normalize to relevant escape codes
; or a hex code if there is no escape code.
; Normalize to uppercase escape code.
; CHECK:|\a\b\t\n\r\\|
|\x007;\x8;\x9;\xA;\xd;\x5C;|

; Normalize to uppercase escape code
; with no leading zeros.
; CHECK:|\xC;|
|\x00c;|

; Normalize printable UTF8 codepoints.
; TODO-CHECK:☕
|\x2615;|

; Omit escaped whitespace
; CHECK:|Hello, world!|
|Hello, \  
  world!|

; Just render characters that do no represent
; an escape character;
; CHECK:|ml\n|
|\m\l\n|

; Support actual newlines
; CHECK:|Hello,\n\n    World!|
|Hello,

    World!|

; Support peculiar identifiers starting with explicit sign +-.
; Check:+hello
+hello
; Check:..
..
; Check:+..
+..

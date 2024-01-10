; RUN: heavy-scheme --mode=read %s 2>&1 | FileCheck  %s --strict-whitespace --match-full-lines
; Escaped symbols should behave as string literals.
; Invalid identitifiers must write as escaped symbols.
; Otherwise, they should write as identifiers.

||
; CHECK:||

; Valid identifiers should write as identifiers.
|foo|
; CHECK:foo
|bar|
; CHECK:bar
|hello|
; CHECK:hello

; Support relaxed identifiers and write as escaped symbols.
\lambda
; CHECK:|\\lambda|
foo\lambda
; CHECK:|foo\\lambda|

; Support printable UTF8 characters
; as relaxed identifiers.
λ
; CHECK:|λ|
ΣλΣ
; CHECK:|ΣλΣ|
ΣλΣfooΣλΣ
; CHECK:|ΣλΣfooΣλΣ|

; Invalid identifiers should write as escaped symbols.
|foo\\nbar|
; CHECK:|foo\\nbar|
|escape\nnewline|
; CHECK:|escape\nnewline|

; Support mnemonic escapes
|\a\b\t\n\r\"\|\\|
; CHECK:|\a\b\t\n\r"\|\\|

; All relevant hex codes should
; normalize to relevant escape codes
; or a hex code if there is no escape code.
; Normalize to uppercase escape code.
|\x007;\x8;\x9;\xA;\xd;\x5C;|
; CHECK:|\a\b\t\n\r\\|

; Normalize to uppercase escape code
; with no leading zeros.
|\x00c;|
; CHECK:|\xC;|

; Normalize printable UTF8 codepoints.
|\x2615;|
; CHECK:|☕|

; Omit escaped whitespace
|Hello, \  
  world!|
; CHECK:|Hello, world!|

; Just render characters that do no represent
; an escape character;
|\m\l\n|
; CHECK:|ml\n|

; Support actual newlines
|Hello,

    World!|
; CHECK:|Hello,\n\n    World!|

; Support peculiar identifiers starting with explicit sign +-.
+hello
; Check:+hello
..
; Check:..
+..
; Check:+..

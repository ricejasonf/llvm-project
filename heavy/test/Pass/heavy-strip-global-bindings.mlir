// RUN: heavy-opt \
// RUN:   --pass-pipeline="builtin.module(builtin.module(heavy-strip-global-bindings))" %s \
// RUN:   | FileCheck %s
module {
  module @_HEAVY {
    "heavy.global"() <{sym_name = "_HEAVYL5SheavyL8SbuiltinsVpl", type = !heavy.procedure}> ({
    }) : () -> ()
    "heavy.load_module"() <{name = @_HEAVYL5SheavyL8Sbuiltins}> : () -> ()
// CHECK-LABEL: "heavy.global"() <{sym_name = "_HEAVYV3Sgetmi4Sfive", type = !heavy.procedure}>
    "heavy.global"() <{sym_name = "_HEAVYV3Sgetmi4Sfive", type = !heavy.binding}> ({
      %0 = "heavy.lambda"() <{name = "_HEAVYA0"}> : () -> !heavy.procedure
// CHECK-NOT: "heavy.binding"
      %1 = "heavy.binding"(%0) : (!heavy.procedure) -> !heavy.binding
// CHECK: "heavy.init_global"({{.*}}) <{name = @_HEAVYV3Sgetmi4Sfive}> : (!heavy.procedure) -> ()
      "heavy.init_global"(%1) <{name = @_HEAVYV3Sgetmi4Sfive}> : (!heavy.binding) -> ()
      %2 = "heavy.undefined"() : () -> !heavy.undefined
      "heavy.cont"(%2) : (!heavy.undefined) -> ()
    }) : () -> ()
    func.func @_HEAVYA0(%arg0: !heavy.context) {
      %0 = "heavy.literal"() <{input = #heavy<"5">}> : () -> !heavy.value
      "heavy.cont"(%0) : (!heavy.value) -> ()
    }
// CHECK-LABEL: "heavy.global"() <{sym_name = "_HEAVYV3Sgetmi4Sfivemi3Smut", type = !heavy.binding}>
    "heavy.global"() <{sym_name = "_HEAVYV3Sgetmi4Sfivemi3Smut", type = !heavy.binding}> ({
      %0 = "heavy.lambda"() <{name = "_HEAVYA1"}> : () -> !heavy.procedure
// CHECK: [[RES:%[0-9]]] = "heavy.binding"
      %1 = "heavy.binding"(%0) : (!heavy.procedure) -> !heavy.binding
// CHECK: "heavy.init_global"([[RES]]) <{name = @_HEAVYV3Sgetmi4Sfivemi3Smut}> : (!heavy.binding) -> ()
      "heavy.init_global"(%1) <{name = @_HEAVYV3Sgetmi4Sfivemi3Smut}> : (!heavy.binding) -> ()
      %2 = "heavy.undefined"() : () -> !heavy.undefined
      "heavy.cont"(%2) : (!heavy.undefined) -> ()
    }) : () -> ()
    func.func @_HEAVYA1(%arg0: !heavy.context) {
      %0 = "heavy.literal"() <{input = #heavy<"#f">}> : () -> !heavy.value
      "heavy.cont"(%0) : (!heavy.value) -> ()
    }
// CHECK-LABEL: "heavy.global"() <{sym_name = "_HEAVYV3Saddmi5Sfives", type = !heavy.procedure}>
    "heavy.global"() <{sym_name = "_HEAVYV3Saddmi5Sfives", type = !heavy.binding}> ({
      %0 = "heavy.lambda"() <{name = "_HEAVYA2"}> : () -> !heavy.procedure
      %1 = "heavy.binding"(%0) : (!heavy.procedure) -> !heavy.binding
// CHECK: "heavy.init_global"({{.*}}) <{name = @_HEAVYV3Saddmi5Sfives}> : (!heavy.procedure) -> ()
      "heavy.init_global"(%1) <{name = @_HEAVYV3Saddmi5Sfives}> : (!heavy.binding) -> ()
      %2 = "heavy.undefined"() : () -> !heavy.undefined
      "heavy.cont"(%2) : (!heavy.undefined) -> ()
    }) : () -> ()
// CHECK-LABEL: @_HEAVYA2
    func.func @_HEAVYA2(%arg0: !heavy.context) {
// CHECK: [[RES1:%[0-9]]] = "heavy.load_global"() <{name = @_HEAVYV3Sgetmi4Sfive}> : () -> !heavy.procedure
// CHECK-NOT: "heavy.unbox"
// CHECK: [[RES2:%[0-9]]] = "heavy.load_global"() <{name = @_HEAVYV3Sgetmi4Sfivemi3Smut}> : () -> !heavy.binding
// CHECK: "heavy.set"([[RES2]], [[RES1]]) : (!heavy.binding, !heavy.procedure) -> !heavy.value
// CHECK-NOT: "heavy.match_type"([[RES1]])
      %0 = "heavy.load_global"() <{name = @_HEAVYV3Sgetmi4Sfive}> : () -> !heavy.binding
      %1 = "heavy.unbox"(%0) : (!heavy.binding) -> !heavy.value
      %2 = "heavy.load_global"() <{name = @_HEAVYV3Sgetmi4Sfivemi3Smut}> : () -> !heavy.binding
      %3 = "heavy.set"(%2, %1) : (!heavy.binding, !heavy.value) -> !heavy.value
      %4 = "heavy.load_global"() <{name = @_HEAVYL5SheavyL8SbuiltinsVpl}> : () -> !heavy.procedure
      %5 = "heavy.match_type"(%4) : (!heavy.procedure) -> !heavy.procedure
      %6 = "heavy.match_type"(%0) : (!heavy.binding) -> !heavy.procedure
      "heavy.push_cont"(%2, %5) <{name = "_HEAVYA3"}> : (!heavy.binding, !heavy.procedure) -> ()
      "heavy.apply"(%6) : (!heavy.procedure) -> ()
    }
// CHECK-LABEL @_HEAVYA3
    func.func @_HEAVYA3(%arg0: !heavy.context, %arg1: !heavy.value_refs) {
      %0 = "heavy.load_ref"(%arg0) <{index = 1 : ui32}> : (!heavy.context) -> !heavy.procedure
      %1 = "heavy.load_ref"(%arg0) <{index = 0 : ui32}> : (!heavy.context) -> !heavy.binding
      %2 = "heavy.load_ref"(%arg1) <{index = 0 : ui32}> : (!heavy.value_refs) -> !heavy.value
      %3 = "heavy.match_type"(%1) : (!heavy.binding) -> !heavy.procedure
      "heavy.push_cont"(%0, %2) <{name = "_HEAVYA4"}> : (!heavy.procedure, !heavy.value) -> ()
      "heavy.apply"(%3) : (!heavy.procedure) -> ()
    }
// CHECK-LABEL @_HEAVYA4
    func.func @_HEAVYA4(%arg0: !heavy.context, %arg1: !heavy.value_refs) {
      %0 = "heavy.load_ref"(%arg0) <{index = 1 : ui32}> : (!heavy.context) -> !heavy.value
      %1 = "heavy.load_ref"(%arg0) <{index = 0 : ui32}> : (!heavy.context) -> !heavy.procedure
      %2 = "heavy.load_ref"(%arg1) <{index = 0 : ui32}> : (!heavy.value_refs) -> !heavy.value
      "heavy.apply"(%1, %0, %2) : (!heavy.procedure, !heavy.value, !heavy.value) -> ()
    }
    "heavy.command"() ({
// CHECK-LABEL: "heavy.command"
// CHECK: %0 = "heavy.load_global"() <{name = @_HEAVYV3Saddmi5Sfives}> : () -> !heavy.procedure
// CHECK-NOT: "heavy.match_type"
      %0 = "heavy.load_global"() <{name = @_HEAVYV3Saddmi5Sfives}> : () -> !heavy.binding
      %1 = "heavy.match_type"(%0) : (!heavy.binding) -> !heavy.procedure
      "heavy.apply"(%1) : (!heavy.procedure) -> ()
    }) : () -> ()
  }
}

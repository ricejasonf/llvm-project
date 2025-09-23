# -*- Python -*-

import lit.formats
from lit.llvm import llvm_config

config.name = 'heavy'
config.test_source_root = os.path.dirname(__file__)
config.test_format = lit.formats.ShTest()
config.suffixes = ['.scm', '.cpp']
config.excludes = [
    "Inputs",
    "CMakeLists.txt",
    "README.txt",
    "LICENSE.txt",
]
config.substitutions.append(('%heavy_module_path', config.heavy_module_path))


# Tweak the PATH to include the tools dir
llvm_config.with_environment('PATH', config.llvm_tools_dir, append_path=True)

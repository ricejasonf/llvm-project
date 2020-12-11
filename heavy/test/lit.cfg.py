# -*- Python -*-

import lit.formats
from lit.llvm import llvm_config

config.name = 'heavy'
config.test_source_root = os.path.dirname(__file__)
config.test_format = lit.formats.ShTest()
config.suffixes = ['.scm', '.cpp']

# Tweak the PATH to include the tools dir.
llvm_config.with_environment('PATH', config.llvm_tools_dir, append_path=True)

import importlib.util
import sys

path = "./tmp/f.py"
spec = importlib.util.spec_from_file_location("custom", path)
foo = importlib.util.module_from_spec(spec)
sys.modules["custom"] = foo
spec.loader.exec_module(foo)

# import all from foo

# get a handle on the module
mdl = foo

# is there an __all__?  if so respect it
if "__all__" in mdl.__dict__:
    names = mdl.__dict__["__all__"]
else:
    # otherwise we import all names that don't begin with _
    names = [x for x in mdl.__dict__ if not x.startswith("_")]

# now drag them in
globals().update({k: getattr(mdl, k) for k in names})

# Test
print(TEST)

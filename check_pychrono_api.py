#!/usr/bin/env python3
"""
Static checker that enforces use of only the installed pychrono API snapshot.

Usage:
  python check_pychrono_api.py --snapdir pychrono_api_repo --paths src tests

Checks:
 - Recognizes imports like 'import pychrono.core as chrono' and 'from pychrono.core import ChBody'
 - Tracks simple instantiation patterns: var = chrono.ChBody()
 - Tracks variable types from direct assignments of constructors and checks later attribute access:
     wheel.SetMass(...)  => verifies 'SetMass' exists on class 'ChBody' in snapshot
 - Checks module-level attribute usage: chrono.ChBody, chrono.ChVector3d, chrono.AddBoxGeometry, etc.
 - Reports any uses (attribute names, constructors, methods) not present in the installed snapshot.

Limitations:
 - Conservative: does not attempt full type inference (works for common patterns).
 - May emit false positives for advanced dynamic code. For those cases add an allowlist entry
   to the snapshot or extend the checker.

Exit codes:
 - 0: no problems
 - 2: violations found (non-zero return so CI/PR can fail)

"""
from __future__ import annotations
import argparse
import ast
import json
import os
import sys
from typing import Dict, Set, Tuple, Optional, List

# -------------------------
# Helpers to load snapshots
# -------------------------
def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def build_api_index(snapshot: dict) -> Tuple[Dict[str, Set[str]], Dict[str, Set[str]]]:
    """
    Return:
      - module_symbols: mapping module_name -> set(module-level names)
      - class_attrs: mapping class_name -> set(attribute names)
    """
    module_symbols: Dict[str, Set[str]] = {}
    class_attrs: Dict[str, Set[str]] = {}
    modules = snapshot.get("modules", {})
    for mod_name, mod_info in modules.items():
        mod_attrs = set(mod_info.get("module_attributes", {}).keys())
        module_symbols[mod_name] = mod_attrs
        # also populate by short module (e.g., pychrono.core -> top-level 'ChBody' etc.)
        for name, entry in (mod_info.get("module_attributes") or {}).items():
            if entry.get("kind") == "class":
                clsname = name
                class_analysis = entry.get("class_analysis") or {}
                cls_attrs = set((class_analysis.get("class_attributes") or {}).keys())
                class_attrs[clsname] = cls_attrs
    return module_symbols, class_attrs

# -------------------------
# AST analysis
# -------------------------
class PyChronoChecker(ast.NodeVisitor):
    def __init__(self, module_symbols: Dict[str, Set[str]], class_attrs: Dict[str, Set[str]],
                 allowed_vehicle_constructors: Optional[Dict[str, any]] = None):
        self.module_symbols = module_symbols
        self.class_attrs = class_attrs
        self.violations: List[str] = []
        # map alias -> module e.g. 'chrono' -> 'pychrono.core'
        self.alias_map: Dict[str, str] = {}
        # map variable name -> class name (when var assigned via constructor)
        self.var_types: Dict[str, str] = {}
        # simple when code does: from pychrono.core import ChBody
        self.from_imports: Dict[str, str] = {}
        # allowed vehicle constructors mapping (for checking 'ARTcar(...)' etc.)
        self.allowed_vehicle_constructors = allowed_vehicle_constructors or {}

    def visit_Import(self, node: ast.Import):
        for alias in node.names:
            name = alias.name  # e.g., 'pychrono.core'
            asname = alias.asname or name.split(".")[-1]
            # if it's a pychrono submodule, map alias to the exact module found in snapshot
            if name.startswith("pychrono"):
                # Prefer exact module name; try to find in module_symbols
                # The alias will be used like: chrono = pychrono.core
                # We map alias -> name (best-effort)
                self.alias_map[asname] = name
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        # from pychrono.core import ChBody as C
        module = node.module
        if not module:
            return
        for alias in node.names:
            asname = alias.asname or alias.name
            # map imported symbol to its full module
            if module.startswith("pychrono"):
                self.from_imports[asname] = module
        self.generic_visit(node)

    def _resolve_attr_module(self, value: ast.AST) -> Optional[str]:
        # if value is Name and known alias -> return module
        if isinstance(value, ast.Name):
            idn = value.id
            if idn in self.alias_map:
                return self.alias_map[idn]
            if idn in self.from_imports:
                return self.from_imports[idn]
        # For chain accesses, e.g., pychrono.core.ChBody (Attribute(Attribute(Name('pychrono'),'core'),'ChBody'))
        if isinstance(value, ast.Attribute):
            # walk down to root
            root = value
            attrs = []
            while isinstance(root, ast.Attribute):
                attrs.append(root.attr)
                root = root.value
            if isinstance(root, ast.Name):
                root_name = root.id
                # reconstruct module like 'pychrono.core'
                # e.g., root_name == 'pychrono' and attrs like ['core', 'ChBody']
                full = ".".join([root_name] + list(reversed(attrs[:-1])))
                # Only accept if this module exists in snapshot
                for mod in self.module_symbols.keys():
                    if mod.endswith(full) or mod == full:
                        return mod
        return None

    def visit_Assign(self, node: ast.Assign):
        # detect patterns: var = chrono.ChBody()
        if isinstance(node.value, ast.Call):
            func = node.value.func
            if isinstance(func, ast.Attribute):
                # func.value might be alias
                mod = self._resolve_attr_module(func.value)
                if mod:
                    clsname = func.attr
                    # check that module exposes that class
                    if clsname not in self.module_symbols.get(mod, set()):
                        # record violation: constructing class not in snapshot
                        self.violations.append(f"Constructor {mod}.{clsname} used but not present in snapshot (line {node.lineno})")
                    # record var type for simple Name targets
                    for t in node.targets:
                        if isinstance(t, ast.Name):
                            self.var_types[t.id] = clsname
            elif isinstance(func, ast.Name):
                # e.g., ChBody() if 'from pychrono.core import ChBody'
                fname = func.id
                if fname in self.from_imports:
                    mod = self.from_imports[fname]
                    if fname not in self.module_symbols.get(mod, set()):
                        self.violations.append(f"Constructor {mod}.{fname} used but not present in snapshot (line {node.lineno})")
                    for t in node.targets:
                        if isinstance(t, ast.Name):
                            self.var_types[t.id] = fname
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute):
        # check module attribute access like chrono.ChBody or chrono.AddBoxGeometry
        if isinstance(node.value, (ast.Name, ast.Attribute)):
            mod = self._resolve_attr_module(node.value)
            if mod:
                name = node.attr
                # If attribute is a class and used as an attribute (not call), it's still a module-level symbol
                if name not in self.module_symbols.get(mod, set()):
                    self.violations.append(f"Module attribute {mod}.{name} used but not present in snapshot (line {node.lineno})")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        # Check calls like wheel.SetMass(...), where wheel has known type
        func = node.func
        if isinstance(func, ast.Attribute):
            # attempt to get target name
            value = func.value
            if isinstance(value, ast.Name):
                var = value.id
                meth = func.attr
                if var in self.var_types:
                    cls = self.var_types[var]
                    allowed = self.class_attrs.get(cls, set())
                    if meth not in allowed:
                        self.violations.append(f"Method '{meth}' called on var '{var}' typed as '{cls}' but '{meth}' not in {cls} attributes (line {node.lineno})")
                else:
                    # unknown var type: if value is literal from module (e.g. chrono.ChBody().SetMass),
                    # we might handle construct-and-call patterns (Call(func=Attribute(...)))
                    # skip for now unless the value is a Call which we can inspect:
                    if isinstance(value, ast.Call) and isinstance(value.func, ast.Attribute):
                        # e.g. chrono.ChBody().SetMass(...)
                        mod = self._resolve_attr_module(value.func.value)
                        if mod:
                            clsname = value.func.attr
                            allowed = self.class_attrs.get(clsname, set())
                            if func.attr not in allowed:
                                self.violations.append(f"Method '{func.attr}' called on newly constructed '{clsname}' but '{func.attr}' not in {clsname} attributes (line {node.lineno})")
        elif isinstance(func, ast.Attribute) is False and isinstance(func, ast.Name):
            # direct function call: e.g., ChVector3d(...) if imported
            fname = func.id
            if fname in self.from_imports:
                mod = self.from_imports[fname]
                if fname not in self.module_symbols.get(mod, set()):
                    self.violations.append(f"Constructor/function {mod}.{fname} used but not present in snapshot (line {node.lineno})")
        self.generic_visit(node)

# -------------------------
# CLI
# -------------------------
def find_python_files(paths: List[str]) -> List[str]:
    pyfiles = []
    for p in paths:
        if os.path.isfile(p) and p.endswith(".py"):
            pyfiles.append(p)
        elif os.path.isdir(p):
            for root, _, files in os.walk(p):
                for f in files:
                    if f.endswith(".py"):
                        pyfiles.append(os.path.join(root, f))
    return pyfiles

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapdir", required=True, help="Directory with installed snapshot JSONs")
    ap.add_argument("--paths", nargs="+", required=True, help="Files or folders to check")
    ap.add_argument("--vehicle-ctors", default="chrono_vehicle_constructors_installed.json", help="vehicle ctors filename in snapshot dir")
    args = ap.parse_args(argv)

    snapdir = args.snapdir
    core_snap = os.path.join(snapdir, "pychrono_attributes_installed.json")
    vehicle_snap = os.path.join(snapdir, args.vehicle_ctors)
    if not os.path.isfile(core_snap):
        print("ERROR: snapshot not found at", core_snap); return 3
    snapshot = load_json(core_snap)
    module_symbols, class_attrs = build_api_index(snapshot)

    vehicle_ctors = {}
    vfile = os.path.join(snapdir, args.vehicle_ctors)
    if os.path.isfile(vfile):
        vehicle_ctors = load_json(vfile).get("classes", {})

    checker = PyChronoChecker(module_symbols, class_attrs, allowed_vehicle_constructors=vehicle_ctors)
    files = find_python_files(args.paths)
    if not files:
        print("No Python files found in", args.paths); return 0

    for fp in files:
        try:
            src = open(fp, "r", encoding="utf-8").read()
        except Exception as e:
            print("SKIP reading", fp, "-", e); continue
        try:
            tree = ast.parse(src, filename=fp)
        except SyntaxError as e:
            print(f"SYNTAX ERROR in {fp}: {e}"); continue
        # reset per-file temporary maps while keeping global aliases
        # Create a fresh checker per file to avoid cross-file taint
        file_checker = PyChronoChecker(module_symbols, class_attrs, allowed_vehicle_constructors=vehicle_ctors)
        # copy global alias map if needed (we don't share between files)
        file_checker.alias_map = dict(checker.alias_map)
        file_checker.from_imports = dict(checker.from_imports)
        file_checker.visit(tree)
        for v in file_checker.violations:
            print(f"{fp}: {v}")

        # accumulate violations
        checker.violations.extend(file_checker.violations)

    if checker.violations:
        print("\nViolations found:", len(checker.violations))
        return 2
    print("No violations found (checked {} files).".format(len(files)))
    return 0

if __name__ == "__main__":
    sys.exit(main())
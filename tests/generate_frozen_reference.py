"""Maintainer tool: (re)create the frozen state the unit tests run on.

It copies the CURRENT ../data folder and the CURRENT EXAMPLE_EVALUATE1/2.yaml
into tests/frozen/, evaluates the four reference chi2 values (example1/2, each
with NLA and TATT), and pins every frozen file in tests/manifest_sha256.json.

Running this REDEFINES what the tests protect: only run it when a change to
the data vectors, n(z), or examples is deliberate, and review the printed
chi2 shifts before committing. Usage (from the Cocoa/ folder, cocoa
environment active, start_cocoa.sh sourced):

    python ./projects/lsst_y1/tests/generate_frozen_reference.py --overwrite
"""

import json
import os
import shutil
import sys
import time

os.environ["OMP_NUM_THREADS"] = "2"  # references must match the test setup

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cocoa_test_utils as u

PROJECT_DIR = os.path.dirname(u.TESTS_DIR)


def main():
    if "--overwrite" not in sys.argv:
        print(__doc__)
        print("Refusing to run without --overwrite (this redefines the "
              "frozen state every test compares against).")
        return 1
    u.require_cocoa_environment()

    if os.path.isdir(u.FROZEN_DIR):
        shutil.rmtree(u.FROZEN_DIR)
    os.makedirs(u.FROZEN_DIR)

    print("freezing ../data ...", flush=True)
    shutil.copytree(os.path.join(PROJECT_DIR, "data"),
                    os.path.join(u.FROZEN_DIR, "data"))
    for cfg in u.EXAMPLES.values():
        shutil.copy2(os.path.join(PROJECT_DIR, cfg["yaml"]),
                     os.path.join(u.FROZEN_DIR, cfg["yaml"]))

    reference = {
        "_meta": {
            "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "omp_num_threads": os.environ["OMP_NUM_THREADS"],
            "chi2_tolerance": u.CHI2_TOLERANCE,
        }
    }
    for example in u.EXAMPLES:
        for tatt in (False, True):
            key = f"{example}_{'tatt' if tatt else 'nla'}"
            t0 = time.time()
            chi2 = u.single_model_chi2(example, tatt)
            print(f"{key}: chi2 = {chi2:.6f}  ({time.time() - t0:.1f}s)",
                  flush=True)
            reference[key] = chi2

    with open(u.REFERENCE_FILE, "w") as f:
        json.dump(reference, f, indent=2, sort_keys=True)
        f.write("\n")

    manifest = {
        "_comment": "SHA-256 of every file under tests/frozen/; verified by "
                    "every test before evaluating anything.",
        "files": u.compute_manifest(),
    }
    with open(u.MANIFEST_FILE, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"manifest: {len(manifest['files'])} files pinned")
    return 0


if __name__ == "__main__":
    sys.exit(main())

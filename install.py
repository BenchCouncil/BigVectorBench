""" This script builds the docker images for the algorithms in the ann_benchmarks/algorithms directory. """
import argparse
import os
import subprocess
import sys
from multiprocessing import Pool

from ann_benchmarks.main import positive_int


def build(library, build_args):
    """ Build the docker image for the given library. """
    print(f"Building {library}...")
    if build_args is not None and len(build_args) != 0:
        q = " ".join(["--build-arg " + x.replace(" ", "\\ ") for x in build_args])
    else:
        q = ""

    try:
        subprocess.check_call(
            f"docker build {q} --rm -t ann-benchmarks-{library} -f ann_benchmarks/algorithms/{library}/Dockerfile  .",
            shell=True,
        )
        return {library: "success"}
    except subprocess.CalledProcessError:
        return {library: "fail"}


def build_multiprocess(build_args):
    """ Wrapper for the build function to allow for multiprocessing. """
    return build(*build_args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--proc", default=1, type=positive_int, help="the number of process to build docker images")
    parser.add_argument("--algorithm", metavar="NAME", help="build only the named algorithm image", default=None)
    parser.add_argument("--build-arg", help="pass given args to all docker builds", nargs="+")
    args = parser.parse_args()

    print("Building base image...")
    subprocess.check_call(
        "docker build --rm -t ann-benchmarks -f ann_benchmarks/algorithms/base/Dockerfile .",
        shell=True,
     )
    print("Building base image done.")

    if args.algorithm:
        tags = [args.algorithm]
    elif os.getenv("LIBRARY"):
        tags = [os.getenv("LIBRARY")]
    else:
        tags = [fn for fn in os.listdir("ann_benchmarks/algorithms") if fn not in ["__init__.py", "__pycache__", "base"]]
    print(f"Building algorithms: {tags}...")

    print(f"Building algorithm images with {args.proc} processes")
    if args.proc == 1:
        install_status = [build(tag, args.build_arg) for tag in tags]
    else:
        pool = Pool(processes=args.proc)
        install_status = pool.map(build_multiprocess, [(tag, args.build_arg) for tag in tags])
        pool.close()
        pool.join()

    print("\n\nInstall Status:\n" + "\n".join(str(algo) for algo in install_status))

    # Exit 1 if any of the installations fail.
    for x in install_status:
        for (k, v) in x.items():
            if v == "fail":
                sys.exit(1)

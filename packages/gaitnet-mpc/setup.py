"""Builds the gaitnet_mpc._mpc_osqp extension (convex MPC QP, OSQP and qpOASES solvers).

Eigen comes from the extern/eigen3 git submodule (headers only); OSQP and qpOASES are
vendored sources under extern/.
"""

from pathlib import Path

from pybind11.setup_helpers import ParallelCompile, Pybind11Extension, build_ext
from setuptools import setup

ParallelCompile("NPY_NUM_BUILD_JOBS").install()

root = Path(__file__).parent
extern = Path("extern")

eigen = extern / "eigen3"
if not (root / eigen / "Eigen" / "Core").exists():
    raise RuntimeError(
        "Eigen headers not found. Run `git submodule update --init packages/gaitnet-mpc/extern/eigen3`."
    )

osqp_sources = [
    *(extern / "osqp" / "src").glob("*.c"),
    extern / "osqp/lin_sys/direct/qdldl/qdldl_interface.c",
    extern / "osqp/lin_sys/direct/qdldl/qdldl_sources/src/qdldl.c",
    *(extern / "osqp/lin_sys/direct/qdldl/amd/src").glob("*.c"),
]
qpoases_sources = sorted((extern / "qpoases" / "src").glob("*.cpp"))

ext = Pybind11Extension(
    "gaitnet_mpc._mpc_osqp",
    sources=sorted(str(s) for s in ["cpp/mpc_osqp.cc", *osqp_sources, *qpoases_sources]),
    include_dirs=[
        str(extern),
        str(eigen),
        str(extern / "osqp/include"),
        str(extern / "osqp/include/linux"),
        str(extern / "osqp/lin_sys"),
        str(extern / "osqp/lin_sys/direct/qdldl"),
        str(extern / "osqp/lin_sys/direct/qdldl/qdldl_sources/include"),
        str(extern / "osqp/lin_sys/direct/qdldl/amd/include"),
        str(extern / "qpoases/include"),
    ],
    extra_compile_args=[
        "-D__SUPPRESSANYOUTPUT__",
        "-D_LINUX",
        "-fpermissive",
        "-Wno-sign-compare",
        "-Wno-reorder",
        "-Wno-unused-variable",
        "-Wno-unused-but-set-variable",
        "-Wno-unused-local-typedefs",
    ],
    libraries=["dl", "pthread"],
    cxx_std=17,
)

setup(ext_modules=[ext], cmdclass={"build_ext": build_ext})

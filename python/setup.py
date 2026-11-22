# -*- coding: utf-8 -*-
#
# @fileoverview Copyright (c) 2019-2021, Stefano Gualandi,
#                via Ferrata, 1, I-27100, Pavia, Italy
#
#  @author stefano.gualandi@gmail.com (Stefano Gualandi)
#

from setuptools import setup, Extension, Command, find_packages
from setuptools.command.build_ext import build_ext
from Cython.Build import cythonize

import numpy as np
import os
import platform
import sys
import warnings

CC_ARGS = [
    '-Wno-unused-function', '-std=c++11', '-O2', '-ffast-math',
    '-DNDEBUG', '-fno-wrapv'
]

LD_ARGS = ['-O2', '-lm', '-pthread', '-fno-wrapv']

if platform.system() == 'Windows':
    # Compiler options: 
    # https://docs.microsoft.com/en-us/cpp/build/reference/openmp-enable-openmp-2-0-support
    # No /arch:AVX2: the module must run on every x86-64 CPU
    CC_ARGS = ['/std:c++14', '/O2', '/EHsc', '/DNDEBUG']
    LD_ARGS = []

if platform.system() == 'Darwin':
    CC_ARGS = [
        '-std=c++11', '-stdlib=libc++', '-O2', '-ffast-math', '-DNDEBUG'
    ]
    LD_ARGS = ['-O2', '-lm', '-pthread', '-fno-wrapv']


# Optional SIMD sorting (x86-simd-sort, AVX2 and AVX-512). Only ot1d_avx2.cpp and
# ot1d_avx512.cpp are compiled with SIMD flags; OT1D.hpp selects at runtime the
# best one supported by the CPU, or falls back to pdqsort, so the module
# runs on every x86-64 CPU. Set OT1D_NO_SIMD=1 to build without them.
SIMD_UNITS = [  # (source, GCC/Clang flags, MSVC flags, macro): AVX2 first, see the link order below
    ('ot1d_avx2.cpp', ['-mavx2'], ['/arch:AVX2'], 'OT1D_USE_AVX2'),
    ('ot1d_avx512.cpp', ['-mavx512f', '-mavx512dq', '-mavx512vl', '-mavx512bw', '-mavx512cd'],
     ['/arch:AVX512'], 'OT1D_USE_AVX512'),
]

# /arch:AVX512 enables F, CD, BW, DQ and VL, the subsets used by x86-simd-sort.
# C4067: x86-simd-sort writes "and" in two #if lines, which MSVC does not parse
# in its default mode; both conditions are false anyway (Intel compiler, GCC).
# C4244, C4267: conversions between integer types, by the hundreds in x86-simd-sort.
SIMD_ARGS = {
    'unix': ['-std=c++17', '-O2', '-DNDEBUG'],
    'msvc': ['/std:c++17', '/O2', '/EHsc', '/DNDEBUG', '/wd4067', '/wd4244', '/wd4267'],
}

WITH_SIMD = (platform.machine().lower() in ('x86_64', 'amd64')
             and platform.system() in ('Linux', 'Darwin', 'Windows')
             and not (platform.system() == 'Windows' and sys.maxsize < 2**32)  # 32-bit Python
             and not os.environ.get('OT1D_NO_SIMD'))


class BuildExtSIMD(build_ext):
    def build_extension(self, ext):
        ctype = self.compiler.compiler_type
        if WITH_SIMD and ctype in SIMD_ARGS:
            for src, gcc_flags, msvc_flags, macro in SIMD_UNITS:
                flags = msvc_flags if ctype == 'msvc' else gcc_flags
                try:
                    objs = self.compiler.compile([src], output_dir=self.build_temp, include_dirs=['./'],
                                                 extra_postargs=SIMD_ARGS[ctype] + flags)
                except Exception as e:
                    warnings.warn('building OT1D without {}: {}'.format(src, e))
                    continue
                # Linked after the main object (and AVX2 before AVX-512): for any inline
                # function compiled in several files, the linker keeps the first copy,
                # which is safe on every CPU that can reach it. The MSVC linker does
                # the same as the GNU one: it keeps the first copy on its command line.
                ext.extra_objects = list(ext.extra_objects or []) + objs
                ext.define_macros = list(ext.define_macros or []) + [(macro, None)]
        build_ext.build_extension(self, ext)


extensions = Extension(name="ot1d",
                       sources=["ot1d.pyx"],
                       include_dirs=['./'],
                       extra_compile_args=CC_ARGS,
                       extra_link_args=LD_ARGS)

with open('README.md', encoding="utf-8") as f:
    long_descr = f.read()

setup(
    name='OT1D',
    version='0.5.1',
    packages=find_packages(),
    description='Discrete Optimal Transport in 1D by Linear Programming',
    author='Stefano Gualandi',
    author_email='stefano.gualandi@gmail.com',
    url='https://github.com/stegua/ot1d',
    license='EUPL-1.2',
    license_files=['LICENSE'],
    project_urls={
        'Source': 'https://github.com/stegua/ot1d',
        'Changelog': 'https://github.com/stegua/ot1d/blob/main/CHANGELOG.md',
        'Issues': 'https://github.com/stegua/ot1d/issues',
    },
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research',
        'Operating System :: POSIX :: Linux',
        'Operating System :: MacOS',
        'Operating System :: Microsoft :: Windows',
        'Programming Language :: C++',
        'Programming Language :: Cython',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Programming Language :: Python :: 3.13',
        'Programming Language :: Python :: 3.14',
        'Topic :: Scientific/Engineering :: Mathematics',
    ],
    platforms=['linux', 'macosx', 'windows'],
    python_requires='>=3.8',
    install_requires=['numpy'],
    include_dirs=np.get_include(),
    long_description=long_descr,
    long_description_content_type='text/markdown',
    cmdclass={'build_ext': BuildExtSIMD},
    ext_modules = cythonize(extensions, 
        compiler_directives={'language_level': "3"})
)

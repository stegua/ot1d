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
import warnings

CC_ARGS = [
    '-Wno-unused-function', '-std=c++11', '-O2', '-ffast-math',
    '-DNDEBUG', '-fno-wrapv'
]

LD_ARGS = ['-O2', '-lm', '-pthread', '-fno-wrapv']

if platform.system() == 'Windows':
    # Compiler options: 
    # https://docs.microsoft.com/en-us/cpp/build/reference/openmp-enable-openmp-2-0-support
    CC_ARGS = ['/std:c++14', '/arch:AVX2', '/O2']
    LD_ARGS = []

if platform.system() == 'Darwin':
    CC_ARGS = [
        '-std=c++11', '-stdlib=libc++', '-O2', '-ffast-math', '-DNDEBUG'
    ]
    LD_ARGS = ['-O2', '-lm', '-pthread', '-fno-wrapv']


# Optional SIMD sorting (x86-simd-sort, AVX2 and AVX-512). Only ot1d_avx2.cpp and
# ot1d_avx512.cpp are compiled with SIMD flags; OT1D.hpp selects at runtime the
# best one supported by the CPU, or falls back to pdqsort/parasort, so the module
# runs on every x86-64 CPU. Set OT1D_NO_SIMD=1 to build without them.
SIMD_UNITS = [  # (source, flags, macro): AVX2 first, see the link order below
    ('ot1d_avx2.cpp', ['-mavx2'], 'OT1D_USE_AVX2'),
    ('ot1d_avx512.cpp', ['-mavx512f', '-mavx512dq', '-mavx512vl', '-mavx512bw', '-mavx512cd'],
     'OT1D_USE_AVX512'),
]

WITH_SIMD = (platform.machine().lower() in ('x86_64', 'amd64')
             and platform.system() in ('Linux', 'Darwin')
             and not os.environ.get('OT1D_NO_SIMD'))


class BuildExtSIMD(build_ext):
    def build_extension(self, ext):
        if WITH_SIMD and self.compiler.compiler_type == 'unix':
            for src, flags, macro in SIMD_UNITS:
                try:
                    objs = self.compiler.compile([src], output_dir=self.build_temp, include_dirs=['./'],
                                                 extra_postargs=['-std=c++17', '-O2', '-DNDEBUG'] + flags)
                except Exception as e:
                    warnings.warn('building OT1D without {}: {}'.format(src, e))
                    continue
                # Linked after the main object (and AVX2 before AVX-512): for any inline
                # function compiled in several files, the linker keeps the first copy,
                # which is safe on every CPU that can reach it.
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
    version='0.5.1.dev0',
    packages=find_packages(),
    description='Discrete Optimal Transport in 1D by Linear Programming',
    author='Stefano Gualandi',
    author_email='stefano.gualandi@gmail.com',
    url='https://github.com/stegua/ot1d',
    platforms=['linux', 'macosx', 'windows'],
    setup_requires=['numpy', 'cython'],
    install_requires=['numpy', 'cython'],
    include_dirs=np.get_include(),
    long_description=long_descr,
    long_description_content_type='text/markdown',
    cmdclass={'build_ext': BuildExtSIMD},
    ext_modules = cythonize(extensions, 
        compiler_directives={'language_level': "3"})
)

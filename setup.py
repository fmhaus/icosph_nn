from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name='icosph_nn',
    ext_modules=[
        CppExtension(
            name='icosph_nn._backend',
            sources=[
                'src/backend/utils.cpp',
                'src/backend/conv.cpp'
            ],
            extra_compile_args={
                'cxx': ['-O3', '-march=native'],
            }
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)
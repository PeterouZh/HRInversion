import setuptools

setuptools.setup(
    name="hrinversion",
    version="0.0.1",
    author="Peng Zhou",
    description="VGG conv-based perceptual loss",
    packages=setuptools.find_packages(),
    python_requires='>=3.6',
    install_requires=[
        'tl2>=0.0.9',
        'streamlit',
        'timm',
        'ninja',
        'lpips',
    ],
)
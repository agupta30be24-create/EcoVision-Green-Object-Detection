from setuptools import setup, find_packages

with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='ecovision',
    version='0.1.0',
    description='Green AI-powered object detection for sustainable automotive systems',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='EcoVision Contributors',
    author_email='your-email@example.com',
    url='https://github.com/agupta30be24-create/EcoVision-Green-Object-Detection',
    license='MIT',
    packages=find_packages(),
    python_requires='>=3.8',
    install_requires=[
        'torch>=2.0.0',
        'torchvision>=0.15.0',
        'ultralytics>=8.0.0',
        'opencv-python>=4.8.0',
        'numpy>=1.24.0',
        'Pillow>=9.5.0',
        'matplotlib>=3.7.0',
        'scipy>=1.10.0',
        'pandas>=2.0.0',
        'scikit-learn>=1.3.0',
        'albumentations>=1.3.0',
        'pyyaml>=6.0',
        'onnx>=1.14.0',
        'onnxruntime>=1.15.0',
        'tensorboard>=2.13.0',
        'tqdm>=4.65.0',
    ],
    extras_require={
        'dev': [
            'pytest>=7.0.0',
            'black>=23.0.0',
            'flake8>=6.0.0',
            'isort>=5.12.0',
        ],
        'optimization': [
            'tensorflow>=2.13.0',
            'onnxruntime-gpu>=1.15.0',
        ],
        'edge': [
            'pycoral>=2.0.0',
            'tflite-runtime>=2.13.0',
        ],
        'monitoring': [
            'wandb>=0.15.0',
            'codecarbonwandb>=2.3.0',
        ],
    },
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Image Recognition',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
    keywords='object-detection yolo automotive green-ai sustainability deep-learning',
    project_urls={
        'Documentation': 'https://github.com/agupta30be24-create/EcoVision-Green-Object-Detection/wiki',
        'Source Code': 'https://github.com/agupta30be24-create/EcoVision-Green-Object-Detection',
        'Bug Tracker': 'https://github.com/agupta30be24-create/EcoVision-Green-Object-Detection/issues',
    },
)

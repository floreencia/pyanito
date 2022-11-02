from setuptools import setup

if __name__ == "__main__":
    setup()


setup(
    name='pyanito',
    version='0.1.0',    
    description='A example Python package',
    url = "https://github.com/floreencia/pyanito",
    author='Florencia Noriega',
    author_email='floreencia@gmail.com',
    license='MIT License',
    packages=['pyanito',
              'pyanito.pyanito'],
    install_requires=[
                      'numpy',                     
                      ],
    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
    ],

)
import setuptools 

#with open("README.md", "r", encoding="utf-8") as fh:
 #   long_description = fh.read()

setuptools.setup(
    name="nn13framework", # Replace with your own username
    version="0.0.6",
    author="Example Author",
    author_email="author@example.com",
    description="nn13framework",
    #long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/KarimSalahSadek/NN13Framework",
    packages=setuptools.find_packages(),
 install_requires=[
      'numpy','texttable','matplotlib',
   ],
  ## dependancy_links=['https://pypi.org/project/matplotlib/', 'https://pypi.org/project/numpy/', 'https://pypi.org/project/texttable/'],
    include_package_data=True,
    zip_safe=False,
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        "Operating System :: OS Independent",
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',


    ],
      keywords='Deep_learning_frame_work ',
      python_requires='>=3.6',

)
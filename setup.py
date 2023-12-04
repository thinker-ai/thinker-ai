import yaml
from setuptools import setup, find_packages


def parse_environment_yml():
    with open("environment.yml", 'r') as stream:
        data = yaml.safe_load(stream)

        pip_deps = []
        conda_deps = []

        for dep in data.get('dependencies', []):
            if isinstance(dep, str):
                conda_deps.append(dep)
            elif isinstance(dep, dict) and 'pip' in dep:
                pip_deps.extend(dep['pip'])

    return pip_deps + conda_deps


install_reqs = parse_environment_yml()


setup(
    name='thinker-ai',
    version='0.1.0', # 这是一个初始版本
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=install_reqs,
    extras_require={
        "playwright": ["playwright>=1.26", "beautifulsoup4"],
        "selenium": ["selenium>4", "webdriver_manager", "beautifulsoup4"],
        "search-google": ["google-api-python-client==2.94.0"],
        "search-ddg": ["duckduckgo-search==3.8.5"],
        'test': []
    },
    classifiers=[
        'Development Status :: 3 - Alpha', # Alpha, Beta or Production/Stable
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3.10',
    ],
    keywords='ai tasks',
    author='wunglee', # 添加你的名字
    author_email='wunglee@thinker-ai.net',
    description='base package for ai rpa',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/thinker-ai/thinker-ai.git',
    license='Apache 2.0',
    include_package_data=True,
    python_requires=">=3.10",
    zip_safe=True
)

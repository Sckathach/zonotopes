# Multi-norm Zonotope implementation 

- [**Theory**](./paper/main.pdf)

## Installation
> [!IMPORTANT]
> The project needs **Python 3.13** 

1. **Install the environment**
    <details open>
    <summary>With miniconda (recommended way)</summary>
    Install miniconda: 

    ```shell
    mkdir -p ~/miniconda3
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
    bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
    rm -rf ~/miniconda3/miniconda.sh
    ~/miniconda3/bin/conda init bash
    ```

    Source or reload your shell:
    ```shell
    source ~/.bashrc
    ```

    Create the environment with Python 3.13:
    ```shell
    conda create -n zonotopes python=3.13 -y
    conda activate zonotopes
    ```
    </details>

    <details>
    <summary>With python venv</summary>
    
    ```shell 
    python -m venv .venv  
    source .venv/bin/activate
    ```
    </details>


2. **Install dependencies**
    <details open>
    <summary>Main dependencies</summary>

    This project uses Poetry to manage its dependencies (<https://python-poetry.org/>). If you get an error with the lock file (out-dated), you can remove it: `rm poetry.lock`. 

    ```shell
    pip install poetry 
    poetry install
    ```

    </details>
    
    <details>
    <summary>Developpement (Optional)</summary>

    Ruff and Mypy libraries for developpement.

    ```shell
    poetry install --with dev
    ```
    </details>

    <details>
    <summary>Notebooks and plotting (Optional)</summary>
    Necessary when working with Jupyter.

    ```shell
    poetry install --with plot
    ```
    </details>

    <details>
    <summary>Documentation (Optional)</summary>
    Used to generate the Sphinx documentation.

    ```shell
    poetry install --with docs
    ```
    </details>



## Build
```shell
make build
```

## Documentation
Generate HTML if needed: 
```shell
make generate_doc
```

Run: 
```shell
firefox docs/build/html/index.html
```

A live version can be found here: <https://sckathach.github.io/zonotopes/>, but styling is currently broken (yes I'm bad). 


## Papers
- <https://engineering.purdue.edu/JainResearchLab/pdf/PhD_Trevor_Bird_2022.pdf>
- <https://arxiv.org/pdf/2210.03244> <https://arxiv.org/pdf/2303.10513>
- OVERT <https://arxiv.org/pdf/2108.01220>

- ADVANCING NEURAL NETWORK VERIFICATION THROUGH HIERARCHICAL SAFETY ABSTRACT INTERPRETATION: <https://arxiv.org/pdf/2505.05235> pas safe/unsafe mais degrés de safe &rarr; pour trouver adv inputs
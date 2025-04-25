test: 
	poetry run python -m pytest

build: 
	poetry build

generate_docs: 
	cd docs
	make html
	firefox buid/html/index.html

install:
	pip install -r requirements.txt
	npm install

format:
	ruff format .


lint:
	ruff check .


check:
	make format
	make lint


run:
	python app.py
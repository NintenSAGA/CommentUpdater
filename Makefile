
run:
	@echo Now running...
	@poetry run python3 ./src/main.py > ./logs/$(shell date +'%Y%m%d%H%M%S').log
	@echo Finished
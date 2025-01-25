#!/bin/bash



LOG_FILE="run_all_files_$(date +%Y%m%d_%H%M%S).log"

touch "${LOG_FILE}"



for FILE in *.py; do

	printf "\n\n --- processing ${FILE} ---\n\n" 2>&1 | tee -a "${LOG_FILE}"

	python "${FILE}" 2>&1 | tee -a "${LOG_FILE}"

done

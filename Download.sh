#!/bin/bash

# Capture the input arguments
INPUT_FILE="$1"
DOWNLOAD_DIR="$2"
QUERY="$3"

echo "Shell script started"
echo "Input file: ${INPUT_FILE}"
echo "Download directory: ${DOWNLOAD_DIR}"
echo "Query: ${QUERY}"

# Ensure the download directory exists
if [ ! -d "${DOWNLOAD_DIR}" ]; then
    echo "Creating download directory: ${DOWNLOAD_DIR}"
    mkdir -p "${DOWNLOAD_DIR}"
fi

if [ -e "${INPUT_FILE}" ]; then
    # Trigger the application using nohup to keep it running after logout
    echo "Triggering application for file ${INPUT_FILE} with query ${QUERY}..."
    nohup /usr/bin/python3 -m PyPaperBot --doi-file="${INPUT_FILE}" --dwn-dir="${DOWNLOAD_DIR}" --query="${QUERY}" > output.log 2>&1 &
    
    # Check if the application has completed running
    while pgrep -f "/usr/bin/python3 -m PyPaperBot --doi-file=${INPUT_FILE}" > /dev/null; do
        echo "Application is still running..."
        sleep 6200  # Sleep for 1 hour (adjust as needed)
    done
    
    # Once the application has completed running, continue with the next file
    echo "Application completed for file ${INPUT_FILE}"
else
    echo "No file found: ${INPUT_FILE}"
fi

echo "All files processed."

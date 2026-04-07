#!/usr/bin/env bash
set -e

if [ ! -f requirements.txt ]; then
  echo "requirements.txt not found. Run this script from the project root."
  exit 1
fi

python3 -m venv venv
source venv/bin/activate
python3 -m pip install --upgrade pip
pip install -r requirements.txt

echo "Dependencies installed in ./venv. Starting Streamlit app..."
streamlit run app.py

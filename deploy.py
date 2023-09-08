"""
deploy.py

DESCRIPTION: Run the app Yoga Pode Estimation in local web browser.

AUTHOR: Juan Ignacio Ribet
DATE: 08-Sep-2023
"""

import sys
from streamlit.web import cli as stcli

if __name__ == '__main__':
    sys.argv = ["streamlit", "run", "Inicio.py"]
    sys.exit(stcli.main())
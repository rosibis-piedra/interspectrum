"""
Louise - Memory System for InterSpectrum
Catalogues and remembers spectral signatures.
Named after Louise Banks (Arrival) - the interpreter.
Creator: Rosibis Piedra
"""

import gspread
from google.oauth2.service_account import Credentials
from datetime import datetime


SCOPES = [
    'https://www.googleapis.com/auth/spreadsheets',
    'https://www.googleapis.com/auth/drive'
]

SHEET_ID = '1axMmv6ncMyOXAyYwBoDB6Pz2VDTJ_w9Dl2NQ5G6blKM'


class Louise:
    """
    Louise remembers every spectrum InterSpectrum reads.
    She builds the dictionary of forms.
    """

    def __init__(self, credentials_path='louise_credentials.json'):
        creds = Credentials.from_service_account_file(
            credentials_path, scopes=SCOPES
        )
        client = gspread.authorize(creds)
        self.sheet = client.open_by_key(SHEET_ID).sheet1

    def remember(self, label: str, text: str, spectrum: dict):
        """
        Save a spectral signature to memory.
        """
        row = [
            datetime.now().strftime('%Y-%m-%d %H:%M'),
            label,
            text[:200],
            round(spectrum['size'], 4),
            round(spectrum['symmetry'], 4),
            round(spectrum['dispersion'], 4),
            round(spectrum['density'], 4),
            spectrum['shape']['dominant_dimensions']
        ]
        self.sheet.append_row(row)
        return True

    def recall(self):
        """
        Retrieve all remembered spectrums.
        """
        records = self.sheet.get_all_records()
        return records
import io
from pdfminer.high_level import extract_text
from pdfminer.high_level import extract_text_to_fp
from pdfminer.layout import LAParams
from io import StringIO
import os
import json
import re
from google.oauth2 import service_account
from googleapiclient.discovery import build
from dotenv import load_dotenv
from config import Config

class PDFReader:
    def read(self, file):
        try:
            pdf_content = file.read()
            # Create a StringIO object to capture the extracted text
            output_string = StringIO()
            
            # Set layout parameters to preserve formatting
            laparams = LAParams(
                char_margin=2.0,  # Character margin to preserve spaces
                line_margin=0.5,  # Line margin to separate lines
                word_margin=0.1,  # Word margin to detect words
                all_texts=True    # Extract all texts (even from figures)
            )
            extract_text_to_fp(io.BytesIO(pdf_content), output_string, laparams=laparams)

            file_content = output_string.getvalue()
            output_string.close()

            return file_content
        except Exception as e:
            raise ValueError(f"Failed to read PDF file: {str(e)}")
        
class TXTReader:
    def read(self, file):
        try:
            return file.read().decode('utf-8')
        except Exception as e:
            raise ValueError(f"Failed to read TXT file: {str(e)}")
        
class MarkdownReader:
    def read(self, file):
        try:
            return file.read().decode('utf-8')
        except Exception as e:
            raise ValueError(f"Failed to read Markdown file: {str(e)}")
        

class GoogleDocReader:
    def __init__(self):
        """Initialize the downloader by loading environment variables and authenticating."""
        self._load_env_variables()
        self._authenticate_google_api()

    def _load_env_variables(self):
        """Load environment variables and check for required keys."""
        load_dotenv()
        # service_account_file = os.getenv('GOOGLE_DOC_CREDS')
        with open(Config.GOOGLE_DOC_CREDS) as jsonfile:
            service_account_file = jsonfile.read()

        if not service_account_file:
            raise ValueError('The environment variable GOOGLE_APPLICATION_CREDENTIALS is not set')
        self.service_account_file = service_account_file

    def _authenticate_google_api(self):
        """Authenticate with Google APIs using the service account."""
        scopes = ['https://www.googleapis.com/auth/drive.readonly', 'https://www.googleapis.com/auth/documents.readonly']
        credentials = service_account.Credentials.from_service_account_info(
            json.loads(self.service_account_file), scopes=scopes
        )
        self.drive_service = build('drive', 'v3', credentials=credentials)

    def _extract_google_doc_id(self, doc_identifier):
        """
        Extract the Google Doc ID from a URL or return the ID if already provided.
        :param doc_identifier: Google Doc URL or ID
        :return: Google Doc ID
        """
        if re.match(r'^[a-zA-Z0-9-_]+$', doc_identifier):  # Check if it's a valid ID
            print('doc_identifier  1:', doc_identifier)
            return doc_identifier
        match = re.search(r'/d/([a-zA-Z0-9-_]+)', doc_identifier)
        if not match:
            print('match :', match)
            raise ValueError('Invalid Google Doc URL or ID')
        return match.group(1)

    def download_as_markdown(self, doc_identifier):
        """
        Download a Google Doc as markdown.
        :param doc_identifier: Google Doc URL or ID
        :return: Markdown content of the Google Doc
        """
        print('doc_identifier :', doc_identifier)

        doc_id = self._extract_google_doc_id(doc_identifier)

        print('doc_id :', doc_id)
        export_mime_type = 'text/markdown'

        print('doc_id 1 :', doc_id)

        # Export the Google Doc to a markdown file
        result = self.drive_service.files().export(fileId=doc_id, mimeType=export_mime_type).execute()
        if result:
            return result.decode('utf-8')
        else:
            raise Exception('Failed to download Google Doc. No content returned.')

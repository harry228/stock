from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from google.oauth2 import service_account
import os

def upload_file(file_path, folder_id):
    creds = service_account.Credentials.from_service_account_file(
        '~/stock-analysis/service-account.json',
        scopes=['https://www.googleapis.com/auth/drive.file']
    )
    service = build('drive', 'v3', credentials=creds)

    file_metadata = {
        'name': os.path.basename(file_path),
        'parents': [folder_id]
    }
    media = MediaFileUpload(file_path, resumable=True)
    file = service.files().create(
        body=file_metadata,
        media_body=media,
        fields='id'
    ).execute()
    print(f"File uploaded: {file.get('id')}")

if __name__ == "__main__":
    folder_id = '1CeMVaUXvJddF_mbVk54-Y8WP1VM_8feC'
    # Test uploading existing files
    files_to_upload = [
        '/data/data/com.termux/files/home/storage/downloads/gold_momentum_analysis.png',
        '/data/data/com.termux/files/home/storage/downloads/gold_momentum_backtest.csv'
    ]
    for f in files_to_upload:
        if os.path.exists(f):
            upload_file(f, folder_id)
        else:
            print(f"File not found: {f}")

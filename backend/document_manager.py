import os
import json
from typing import List, Dict, Optional

class DocumentManager:
    def __init__(self, base_dir: str = "data/documents"):
        self.base_dir = base_dir
        self._ensure_directory_structure()

    def _ensure_directory_structure(self):
        """必要なディレクトリ構造を作成"""
        directories = [
            "guidelines",
            "examples/violations",
            "examples/compliant"
        ]
        for dir_path in directories:
            os.makedirs(os.path.join(self.base_dir, dir_path), exist_ok=True)

    def get_document_list(self) -> List[Dict]:
        """全てのドキュメントのリストを取得"""
        documents = []
        for root, _, files in os.walk(self.base_dir):
            for file in files:
                if file.endswith('.txt'):
                    rel_path = os.path.relpath(root, self.base_dir)
                    category = rel_path.replace('\\', '/').split('/')[0]
                    subcategory = '/'.join(rel_path.replace('\\', '/').split('/')[1:])
                    documents.append({
                        'id': os.path.join(rel_path, file).replace('\\', '/'),
                        'name': file,
                        'category': category,
                        'subcategory': subcategory if subcategory else None,
                        'path': os.path.join(rel_path, file).replace('\\', '/')
                    })
        return documents

    def get_document_content(self, doc_path: str) -> Optional[str]:
        """ドキュメントの内容を取得"""
        full_path = os.path.join(self.base_dir, doc_path)
        try:
            with open(full_path, 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            return None

    def save_document(self, doc_path: str, content: str) -> bool:
        """ドキュメントの内容を保存"""
        full_path = os.path.join(self.base_dir, doc_path)
        try:
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            with open(full_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        except Exception as e:
            print(f"Error saving document: {e}")
            return False

    def create_document(self, category: str, subcategory: Optional[str], name: str, content: str = "") -> Optional[str]:
        """新規ドキュメントを作成"""
        if not name.endswith('.txt'):
            name = f"{name}.txt"

        path_parts = [category]
        if subcategory:
            path_parts.extend(subcategory.split('/'))
        path_parts.append(name)
        
        doc_path = '/'.join(path_parts)
        if self.save_document(doc_path, content):
            return doc_path
        return None

    def delete_document(self, doc_path: str) -> bool:
        """ドキュメントを削除"""
        full_path = os.path.join(self.base_dir, doc_path)
        try:
            os.remove(full_path)
            return True
        except Exception as e:
            print(f"Error deleting document: {e}")
            return False 
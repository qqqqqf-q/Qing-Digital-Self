#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sqlite3
import csv
import os
import argparse
from datetime import datetime
from pathlib import Path
import glob
import sys
from typing import List, Dict, Any

# Add project root to path to import project modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from utils.logger.logger import get_logger

logger = get_logger('WXParser')

class WXParser:
    """
    WeChat Chat Data Parser
    Converts WeChat SQLite database format to a unified CSV format.
    """

    def __init__(self, input_dir: str, output_dir: str):
        """
        Initializes the WeChat Parser.

        Args:
            input_dir: Directory containing WeChat MSG*.db files.
            output_dir: CSV output directory.
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.db_files = []

        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def find_db_files(self):
        """Finds all MSG*.db files in the input directory."""
        pattern = os.path.join(self.input_dir, 'MSG*.db')
        self.db_files = glob.glob(pattern)
        logger.info(f"Found {len(self.db_files)} WeChat database files in '{self.input_dir}'.")

    def format_timestamp(self, timestamp: int) -> str:
        """Formats a Unix timestamp."""
        try:
            return datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
        except (ValueError, OSError):
            return ""

    def parse_messages(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Parses all messages from all found database files and groups them by talker.
        """
        all_messages_by_talker: Dict[str, List[Dict[str, Any]]] = {}

        for db_path in self.db_files:
            logger.info(f"Processing database: {db_path}")
            try:
                with sqlite3.connect(db_path) as conn:
                    cursor = conn.cursor()
                    # Query to get all text messages (Type=1)
                    cursor.execute("""
                        SELECT MsgSvrID, IsSender, CreateTime, StrTalker, StrContent
                        FROM MSG
                        WHERE Type = 1 AND StrContent IS NOT NULL AND StrContent != ''
                        ORDER BY CreateTime ASC
                    """)
                    rows = cursor.fetchall()

                    for row in rows:
                        msg_svr_id, is_sender, create_time, talker, content = row
                        
                        if not talker or not content:
                            continue

                        # Skip group chats (which often end in '@chatroom')
                        if talker.endswith('@chatroom'):
                            continue
                        
                        # Clean content
                        content = content.strip()
                        if not content:
                            continue

                        message = {
                            'MsgSvrID': msg_svr_id,
                            'type_name': 'text',
                            'is_sender': is_sender,
                            'talker': talker,
                            'msg': content,
                            'src': '',
                            'CreateTime': self.format_timestamp(create_time),
                            'is_forward': 0
                        }

                        if talker not in all_messages_by_talker:
                            all_messages_by_talker[talker] = []
                        all_messages_by_talker[talker].append(message)

            except sqlite3.Error as e:
                logger.error(f"Failed to read or process database {db_path}: {e}")
        
        return all_messages_by_talker

    def save_to_csv(self, messages_by_talker: Dict[str, List[Dict[str, Any]]]):
        """Saves messages to CSV files, one for each talker."""
        if not messages_by_talker:
            logger.warning("No messages found to save.")
            return

        total_messages_saved = 0
        for talker, messages in messages_by_talker.items():
            if not messages:
                continue

            # Create a directory for the talker
            talker_id_safe = "".join(c for c in talker if c.isalnum() or c in ('_', '-')).rstrip()
            room_name = f"WX_{talker_id_safe}"
            peer_dir = self.output_dir / room_name
            peer_dir.mkdir(parents=True, exist_ok=True)

            # CSV file path
            csv_file = peer_dir / f"{room_name}_chat.csv"

            # Add id and room_name to each message
            for i, msg in enumerate(messages):
                msg['id'] = i + 1
                msg['room_name'] = room_name

            # CSV fields
            fieldnames = [
                'id', 'MsgSvrID', 'type_name', 'is_sender', 'talker',
                'msg', 'src', 'CreateTime', 'room_name', 'is_forward'
            ]

            try:
                with open(csv_file, 'w', newline='', encoding='utf-8') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(messages)
                
                logger.info(f"Successfully saved {len(messages)} messages for talker '{talker}' to {csv_file}")
                total_messages_saved += len(messages)
            
            except IOError as e:
                logger.error(f"Failed to write CSV file for talker {talker}: {e}")
        
        logger.info(f"Finished saving. Total messages saved: {total_messages_saved}")

    def run(self):
        """Main execution flow."""
        logger.info("Starting WeChat chat data parsing...")
        self.find_db_files()
        if not self.db_files:
            logger.warning("No database files found. Exiting.")
            return
        
        messages = self.parse_messages()
        self.save_to_csv(messages)
        logger.info("WeChat parsing complete.")


def create_parser():
    """Creates the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description='WeChat Chat History Parser - Converts WeChat SQLite DB to CSV format.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Usage example:
  python wx_parser.py --input-dir "dataset/original/wechat" --output-dir "dataset/csv"
"""
    )
    
    parser.add_argument(
        '--input-dir',
        type=str,
        required=True,
        help='Directory containing the WeChat MSG*.db files.'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./dataset/csv/',
        help='Directory to save the output CSV files (default: "./dataset/csv/").'
    )
    
    return parser

def main():
    """Main function."""
    arg_parser = create_parser()
    args = arg_parser.parse_args()

    if not os.path.isdir(args.input_dir):
        logger.error(f"Input directory not found: {args.input_dir}")
        return 1

    parser = WXParser(input_dir=args.input_dir, output_dir=args.output_dir)
    parser.run()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

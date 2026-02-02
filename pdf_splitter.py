"""
PDF Splitter Utility

Simple utility to split large PDFs into smaller, manageable chunks.
Useful for processing very large PDFs that exceed system capabilities.

"""

import os
import argparse
from pathlib import Path
from PyPDF2 import PdfReader, PdfWriter


class PDFSplitter:
    """Split large PDFs into smaller files"""
    
    @staticmethod
    def split_by_pages(input_path: str, pages_per_file: int = 100, output_dir: str = None):
        """
        Split PDF by number of pages
        
        Args:
            input_path: Path to input PDF
            pages_per_file: Number of pages per output file
            output_dir: Output directory (default: same as input)
        """
        input_path = Path(input_path)
        
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
        
        # Setup output directory
        if output_dir is None:
            output_dir = input_path.parent
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Read PDF
        print(f"Reading PDF: {input_path.name}")
        reader = PdfReader(str(input_path))
        total_pages = len(reader.pages)
        
        print(f"Total pages: {total_pages}")
        print(f"Pages per file: {pages_per_file}")
        
        num_files = (total_pages + pages_per_file - 1) // pages_per_file
        print(f"Will create {num_files} file(s)\n")
        
        # Split into files
        base_name = input_path.stem
        created_files = []
        
        for i in range(0, total_pages, pages_per_file):
            writer = PdfWriter()
            
            # Calculate page range
            start_page = i
            end_page = min(i + pages_per_file, total_pages)
            
            # Add pages to writer
            for page_num in range(start_page, end_page):
                writer.add_page(reader.pages[page_num])
            
            # Save output file
            part_num = (i // pages_per_file) + 1
            output_file = output_dir / f"{base_name}_part{part_num:03d}.pdf"
            
            with open(output_file, 'wb') as f:
                writer.write(f)
            
            file_size_mb = output_file.stat().st_size / (1024 * 1024)
            
            print(f"✅ Created: {output_file.name}")
            print(f"   Pages: {start_page + 1}-{end_page} ({end_page - start_page} pages)")
            print(f"   Size: {file_size_mb:.2f} MB\n")
            
            created_files.append(output_file)
        
        print(f"✅ Successfully split into {len(created_files)} files")
        print(f"📁 Output directory: {output_dir}")
        
        return created_files
    
    @staticmethod
    def split_by_size(input_path: str, max_size_mb: int = 50, output_dir: str = None):
        """
        Split PDF by target file size (approximate)
        
        Args:
            input_path: Path to input PDF
            max_size_mb: Target maximum size per file in MB
            output_dir: Output directory (default: same as input)
        """
        input_path = Path(input_path)
        
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
        
        # Setup output directory
        if output_dir is None:
            output_dir = input_path.parent
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Read PDF
        print(f"Reading PDF: {input_path.name}")
        reader = PdfReader(str(input_path))
        total_pages = len(reader.pages)
        
        input_size_mb = input_path.stat().st_size / (1024 * 1024)
        
        print(f"Total pages: {total_pages}")
        print(f"Total size: {input_size_mb:.2f} MB")
        print(f"Target size per file: {max_size_mb} MB\n")
        
        # Estimate pages per file based on size
        avg_mb_per_page = input_size_mb / total_pages
        estimated_pages_per_file = int(max_size_mb / avg_mb_per_page)
        
        print(f"Estimated pages per file: ~{estimated_pages_per_file}\n")
        
        # Split
        base_name = input_path.stem
        created_files = []
        current_writer = PdfWriter()
        current_pages = 0
        file_counter = 1
        
        for page_num, page in enumerate(reader.pages):
            current_writer.add_page(page)
            current_pages += 1
            
            # Check if we should start a new file
            # (either reached estimated pages or this is the last page)
            should_split = (
                current_pages >= estimated_pages_per_file or 
                page_num == total_pages - 1
            )
            
            if should_split and current_pages > 0:
                # Save current file
                output_file = output_dir / f"{base_name}_part{file_counter:03d}.pdf"
                
                with open(output_file, 'wb') as f:
                    current_writer.write(f)
                
                file_size_mb = output_file.stat().st_size / (1024 * 1024)
                
                print(f"✅ Created: {output_file.name}")
                print(f"   Pages: {current_pages}")
                print(f"   Size: {file_size_mb:.2f} MB\n")
                
                created_files.append(output_file)
                
                # Reset for next file
                current_writer = PdfWriter()
                current_pages = 0
                file_counter += 1
        
        print(f"✅ Successfully split into {len(created_files)} files")
        print(f"📁 Output directory: {output_dir}")
        
        return created_files


def main():
    """CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Split large PDFs into smaller files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Split by page count (100 pages per file)
  python pdf_splitter.py large_document.pdf --pages 100
  
  # Split by size (50 MB per file)
  python pdf_splitter.py large_document.pdf --size 50
  
  # Specify output directory
  python pdf_splitter.py large_document.pdf --pages 100 --output data/splits
        """
    )
    
    parser.add_argument(
        'input_file',
        type=str,
        help='Path to input PDF file'
    )
    
    parser.add_argument(
        '--pages',
        type=int,
        help='Number of pages per output file'
    )
    
    parser.add_argument(
        '--size',
        type=int,
        help='Target size per output file in MB (approximate)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        help='Output directory (default: same as input file)'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.pages and not args.size:
        parser.error("Must specify either --pages or --size")
    
    if args.pages and args.size:
        parser.error("Cannot specify both --pages and --size")
    
    # Execute split
    try:
        splitter = PDFSplitter()
        
        print("=" * 70)
        print("PDF SPLITTER")
        print("=" * 70)
        print()
        
        if args.pages:
            created_files = splitter.split_by_pages(
                args.input_file,
                pages_per_file=args.pages,
                output_dir=args.output
            )
        else:  # args.size
            created_files = splitter.split_by_size(
                args.input_file,
                max_size_mb=args.size,
                output_dir=args.output
            )
        
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print(f"Input file: {args.input_file}")
        print(f"Created {len(created_files)} output file(s):")
        for f in created_files:
            print(f"  - {f.name}")
        print("=" * 70)
    
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
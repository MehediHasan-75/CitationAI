import os
from datetime import datetime
from pathlib import Path

def read_file_content(file_path):
    """
    Safely read file content with encoding detection.
    Returns content if readable, otherwise returns error message.
    """
    try:
        # Try UTF-8 first
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return content
    except UnicodeDecodeError:
        # Try with error handling
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            return content
        except Exception as e:
            return f"[Binary/Unreadable file: {str(e)}]"
    except Exception as e:
        return f"[Error reading file: {str(e)}]"

def list_all_contents_with_file_content(output_filename="directory_contents.txt"):
    """
    Recursively reads all files and folders from the research-paper-rag-assessment directory
    and includes the CONTENT of each file (not just names).
    
    Args:
        output_filename: Name of the output text file (default: directory_contents.txt)
    """
    try:
        # Get current directory (research-paper-rag-assessment)
        directory_path = os.getcwd()
        
        # Set output file path in the same directory
        output_path = os.path.join(directory_path, output_filename)
        
        # File extensions to skip (binary files)
        skip_extensions = {'.jpg', '.png', '.gif', '.ico', '.svg', '.pack', '.idx', '.DS_Store',
                          '.mp4', '.mp3', '.mov', '.zip', '.tar', '.gz', '.pyc', '.o', '.so',
                          '.dll', '.exe', '.bin', '.woff', '.woff2', '.ttf', '.otf'}
        
        skip_files = {'.DS_Store', 'package-lock.json', 'node_modules'}
        skip_dirs = {'node_modules', '.git', '.next', 'venv', '__pycache__', '.env', 'dist', 'build'}
        
        total_files = 0
        total_folders = 0
        total_size = 0
        processed_content = 0
        skipped_binary = 0
        
        # Write to text file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("=" * 120 + "\n")
            f.write("Citation AI PROJECT - COMPLETE LISTING WITH FILE CONTENTS (RECURSIVE)\n")
            f.write("=" * 120 + "\n\n")
            
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Location: {os.path.abspath(directory_path)}\n")
            f.write(f"Scan Type: Recursive (All subdirectories with file contents)\n")
            f.write("\n" + "=" * 120 + "\n\n")
            
            # Walk through all directories recursively
            for root, dirs, files in os.walk(directory_path):
                # Skip directories
                dirs[:] = [d for d in dirs if d not in skip_dirs]
                
                # Calculate depth
                depth = root.replace(directory_path, '').count(os.sep)
                
                # Create relative path for cleaner display
                relative_path = os.path.relpath(root, directory_path)
                if relative_path == '.':
                    relative_path = 'src Root)'
                else:
                    relative_path = f"src/{relative_path}"
                
                # Write directory header
                indent = "  " * depth
                
                if relative_path == 'src (Root)':
                    f.write(f"\n{'=' * 120}\n")
                    f.write(f"📁 {relative_path}\n")
                    f.write(f"{'=' * 120}\n\n")
                else:
                    f.write(f"\n{indent}{'─' * (120 - len(indent))}\n")
                    f.write(f"{indent}📁 {os.path.basename(relative_path)}\n")
                    f.write(f"{indent}{'─' * (120 - len(indent))}\n\n")
                
                total_folders += 1
                
                # Process files
                for file in sorted(files):
                    # Skip certain files
                    if file in skip_files:
                        continue
                    
                    file_path = os.path.join(root, file)
                    file_ext = os.path.splitext(file)[1]
                    
                    try:
                        file_size = os.path.getsize(file_path)
                        total_size += file_size
                        
                        if file_size / (1024 * 1024) > 0.01:
                            size_display = f"{file_size / (1024 * 1024):.2f} MB"
                        else:
                            size_display = f"{file_size / 1024:.2f} KB"
                        
                        total_files += 1
                        
                        # Check if file should be read
                        if file_ext in skip_extensions or file_size > 5 * 1024 * 1024:  # Skip files > 5MB
                            skipped_binary += 1
                            f.write(f"{indent}📄 {file} ({size_display}) [Binary/Large file - skipped]\n\n")
                        else:
                            # Read and display file content
                            content = read_file_content(file_path)
                            
                            f.write(f"{indent}📄 {file} ({size_display})\n")
                            f.write(f"{indent}{'-' * 110}\n")
                            f.write(f"{indent}CONTENT:\n")
                            f.write(f"{indent}{'-' * 110}\n")
                            
                            # Add content with indentation
                            for line in content.split('\n'):
                                if line.strip():  # Only write non-empty lines
                                    f.write(f"{indent}  {line}\n")
                                else:
                                    f.write(f"\n")
                            
                            f.write(f"{indent}{'-' * 110}\n\n")
                            processed_content += 1
                    
                    except Exception as e:
                        total_files += 1
                        f.write(f"{indent}📄 {file} [Error: {str(e)}]\n\n")
            
            # Write summary
            f.write("\n" + "=" * 120 + "\n")
            f.write("SUMMARY REPORT\n")
            f.write("-" * 120 + "\n")
            f.write(f"Total Directories: {total_folders}\n")
            f.write(f"Total Files: {total_files}\n")
            f.write(f"Files with Content Displayed: {processed_content}\n")
            f.write(f"Binary/Large Files Skipped: {skipped_binary}\n")
            f.write(f"Total Size: {total_size / (1024 * 1024):.2f} MB ({total_size / 1024:.2f} KB)\n")
            f.write(f"\nNote: Binary files and files larger than 5MB are listed but content is skipped.\n")
            f.write(f"Directories excluded: {', '.join(skip_dirs)}\n")
            
            f.write("\n" + "=" * 120 + "\n")
            f.write("Report generated by FinBot Complete Content Analyzer\n")
            f.write("=" * 120 + "\n")
        
        print(f"✓ Success! Complete directory listing with file contents saved!")
        print(f"✓ File: {output_filename}")
        print(f"✓ Location: {output_path}")
        print(f"✓ Total Directories: {total_folders}")
        print(f"✓ Total Files: {total_files}")
        print(f"✓ Files with Content: {processed_content}")
        print(f"✓ Binary/Large Files (skipped): {skipped_binary}")
        print(f"✓ Total Size: {total_size / (1024 * 1024):.2f} MB")
        
    except Exception as e:
        print(f"✗ Error: {str(e)}")

# Usage - Run this script in the research-paper-rag-assessment directory
if __name__ == "__main__":
    list_all_contents_with_file_content("directory_contents.txt")

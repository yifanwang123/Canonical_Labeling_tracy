import hashlib

file1 = '/home/cds/Yifan/canonical_labeling/1'
file2 = '/home/cds/Yifan/canonical_labeling/2'
def files_are_equal(file1, file2):
    """Check if two files have the same contents."""
    # Compare file contents by calculating their hashes (this method is efficient for large files)
    hash1 = hashlib.md5()
    hash2 = hashlib.md5()
    
    # Read the files in binary mode to avoid encoding issues
    with open(file1, 'rb') as f1, open(file2, 'rb') as f2:
        while chunk := f1.read(8192):
            hash1.update(chunk)
        while chunk := f2.read(8192):
            hash2.update(chunk)
    
    # Compare the hashes
    return hash1.digest() == hash2.digest()


def compare_files(file1, file2):
    """Compare two files line by line and print the differences."""
    with open(file1, 'r') as f1, open(file2, 'r') as f2:
        line_num = 1
        while True:
            line1 = f1.readline()
            line2 = f2.readline()
            
            # If both lines are empty, end of file reached for both
            if not line1 and not line2:
                break
            
            # If the lines differ, print the difference
            if line1 != line2:
                print(f"Difference found at line {line_num}:")
                print(f"{file1}: {line1.strip()}")
                print(f"{file2}: {line2.strip()}")
                print("-" * 50)
            
            line_num += 1


compare_files(file1, file2)
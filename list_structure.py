import os


def list_directory_structure(startpath, output_file):
    """
    Recursively lists directory structure and saves to a text file,
    excluding any folders named 'myenv' or 'node_modules'.
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        for root, dirs, files in os.walk(startpath):
            # Remove excluded folders from directories to traverse
            if 'myenv' in dirs:
                dirs.remove('myenv')
            if 'node_modules' in dirs:
                dirs.remove('node_modules')

            level = root.replace(startpath, '').count(os.sep)
            indent = ' ' * 4 * (level)

            # Skip excluded folders if they're the current root
            if os.path.basename(root) in ('myenv', 'node_modules'):
                continue

            f.write('{}{}/\n'.format(indent, os.path.basename(root)))
            subindent = ' ' * 4 * (level + 1)
            for file in files:
                f.write('{}{}\n'.format(subindent, file))


if __name__ == "__main__":
    # Get the directory where this script is located
    projects_dir = os.path.dirname(os.path.abspath(__file__))
    output_filename = "directory_structure.txt"

    print(f"Listing directory structure of: {projects_dir}")
    print(f"Excluding folders named 'myenv' and 'node_modules'")
    print(f"Saving to: {output_filename}")

    list_directory_structure(projects_dir, output_filename)

    print("Done! Directory structure saved.")
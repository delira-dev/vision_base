import os


def get_files_from_dir(path, extensions, to_lower=True):
    files = []

    for item in sorted(os.listdir(path)):

        name, ext = item.split(".", 1)

        # convert extension to lowercase
        if to_lower:
            ext = ext.lower()

        # check if file extension is valid
        if ("." + ext) not in extensions:
            continue

        # check if file exists
        whole_path = os.path.join(path, item)
        if os.path.isfile(whole_path):
            files.append(whole_path)

    return files

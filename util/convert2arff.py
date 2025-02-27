#  Method for converting files to .arff files to use in MOA.
import os


def out2arff(filepath, classes="{0, 1}"):
    #  Write new .arff file based on .out files
    #  @param filepath: String representing filepath of .out file to convert
    #  @param classes: set of classes existing in data, for TSB-UAD {0,1}
    filename = filepath.split('.out')[0]
    data = []
    with open(filepath, 'r') as input_file:
        for line in input_file:
            data.append(f"{line.strip()},\n")
    header = f"@relation '{filename}'\n\n"
    header += "@attribute att1 numeric\n"
    header += f"@attribute class {classes}\n\n"
    header += "@data\n\n"
    with open(f"{filename}.arff", 'w') as output_file:
        output_file.writelines([header] + data)


def main():
    #  Converts all files within specified directory from .out to .arff
    #  Deletes original .out file in directory
    directory = "../data/benchmark/ECG"
    for path in os.listdir(directory):
        if os.path.isfile(os.path.join(directory, path)) and path.split(".")[-1] == "out":
            try:
                print(f"Converting {os.path.join(directory, path)}")
                out2arff(os.path.join(directory, path))
            except FileNotFoundError:
                print(f"Error converting {os.path.join(directory, path)}")
            else:
                os.remove(os.path.join(directory, path))


main()

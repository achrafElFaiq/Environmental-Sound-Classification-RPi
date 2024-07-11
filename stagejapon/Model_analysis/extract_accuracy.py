import subprocess

def extract_learning_info(command, output_file):
    # Run the command and capture the output
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    output = result.stdout.split('\n')
    
    with open(output_file, 'w') as outfile:
        for line in output:
            if '546' in line and '[==============================]' in line and 'ETA' not in line:
                outfile.write(line + '\n')

if __name__ == "__main__":
    command = 'python learnmodel_ps2024.py'  # Replace with your command
    output_file = 'learning.txt'
    extract_learning_info(command, output_file)

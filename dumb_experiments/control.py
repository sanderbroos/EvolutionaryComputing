import subprocess

program_list = ['1.py', '2.py', '3.py', '4.py', '5.py', '6.py', '7.py']

for program in program_list:
    subprocess.call(['python', f'{program}'])
    print("Finished:" + program)
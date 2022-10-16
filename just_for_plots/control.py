import subprocess

program_list = ['group_1.py', 'group_2.py']

for program in program_list:
    subprocess.call(['python', f'{program}'])
    print("Finished:" + program)
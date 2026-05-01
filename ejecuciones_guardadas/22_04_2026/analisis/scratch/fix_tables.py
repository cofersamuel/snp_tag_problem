import re

file_path = '/home/cofer/Documents/University/TFG/snp_tag_tfg/ejecuciones_guardadas/22_04_2026/analisis/analisis.md'

with open(file_path, 'r') as f:
    lines = f.readlines()

new_lines = []
for i, line in enumerate(lines):
    if line.startswith('| ') and i > 0:
        prev_line = lines[i-1].strip()
        # If the previous line is not empty and not another table line
        if prev_line and not prev_line.startswith('|'):
            new_lines.append('\n')
    new_lines.append(line)

with open(file_path, 'w') as f:
    f.writelines(new_lines)

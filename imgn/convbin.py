
output_lines = []

def to_base_8(n, width=4):
    digits = []
    for _ in range(width):
        digits.append(str(n % 8))
        n //= 8
    return digits[::-1]  # Most significant digit first

def to_base_4(n, width=5):
    digits = []
    for _ in range(width):
        digits.append(str(n % 4))
        n //= 4
    return digits[::-1]  # Most significant digit first
'''
with open("imgn.train.data", "r") as file:
    for line in file:
        numbers = map(int, line.strip().split(','))
        bits = [",".join(bin(n)[2:].zfill(9)) for n in numbers]
        output_lines.append(",".join(bits))
'''

with open("imgn.val.data", "r") as file:
    for line in file:
        numbers = map(int, line.strip().split(','))
        base4_digits = [",".join(to_base_4(n)) for n in numbers]
        output_lines.append(",".join(base4_digits))

result = "\n".join(output_lines)
print(result)

# Ensure the output directory exists
import os

# Correct relative path based on where you're running this
out_dir = "../imgn_bin"
os.makedirs(out_dir, exist_ok=True)

out_path = os.path.join(out_dir, "imgn_bin.val.data")
print("Writing to:", os.path.abspath(out_path))

with open(out_path, "w") as outfile:
    outfile.write(result)




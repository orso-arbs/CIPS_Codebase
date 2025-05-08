import numpy as np
import re

def BinGB(inBites):
  return inBites / 1024**3



def KBinGB(inBites):
  return inBites / 1024**2

# print(inTB(1528252600))
# print(inTB(1528252608))
# print(inTB(2046519140352))
# print(inTB(2499670966272))


# print(2112084930560 - 2046519140352)
# print(BinGB(2112084930560 - 2046519140352))

# print(1592281700 -1528252608)
# print(KBinGB(1592281700 -1528252608))

# print(1592281692 -1528252600)
# print(KBinGB(1592281692 -1528252600))


print(KBinGB(2598439698432))
print(KBinGB(2598439698432)/1024**2)

print(2598439698432 / 1024**4)

print(2.5*0.8)
print(2.5*1.1)

print(52.392304 * 10 /1024 - 332923980 /1024**3)


import re

# Paste your `ll` output as a multi-line string here
ll_output = """
-rwxrwxrwx 1 doancea doancea-group 56255808136 May  5 16:39 po_s912k_post0.f00001
-rwxrwxrwx 1 doancea doancea-group 56255808136 May  5 16:39 po_s912k_post0.f00002
-rwxrwxrwx 1 doancea doancea-group 56255808136 May  5 16:39 po_s912k_post0.f00003
-rwxrwxrwx 1 doancea doancea-group 56255808136 May  5 16:39 po_s912k_post0.f00004
-rwxrwxrwx 1 doancea doancea-group 56255808136 May  5 16:39 po_s912k_post0.f00005
-rwxrwxrwx 1 doancea doancea-group 56255808136 May  5 16:39 po_s912k_post0.f00006
-rwxrwxrwx 1 doancea doancea-group 56255808136 May  5 16:39 po_s912k_post0.f00007
-rwxrwxrwx 1 doancea doancea-group 56255808136 May  5 16:39 po_s912k_post0.f00008
-rwxrwxrwx 1 doancea doancea-group 56255808136 May  5 16:39 po_s912k_post0.f00009
-rwxrwxrwx 1 doancea doancea-group 56255808136 May  5 16:39 po_s912k_post0.f00010
-rwxrwxrwx 1 doancea doancea-group 56255808136 May  5 16:39 po_s912k_post0.f00011
-rwxrwxrwx 1 doancea doancea-group 56255808136 May  5 16:39 po_s912k_post0.f00012
-rwxrwxrwx 1 doancea doancea-group 56255808136 May  5 16:39 po_s912k_post0.f00013
-rwxrwxrwx 1 doancea doancea-group 56255808136 May  5 16:39 po_s912k_post0.f00014
-rwxrwxrwx 1 doancea doancea-group 56255808136 May  5 16:39 po_s912k_post0.f00015
-rwxrwxrwx 1 doancea doancea-group 56255808136 May  5 16:39 po_s912k_post0.f00016
-rwxrwxrwx 1 doancea doancea-group 56255808136 May  5 16:39 po_s912k_post0.f00017
-rwxrwxrwx 1 doancea doancea-group 56255808136 May  5 16:39 po_s912k_post0.f00018
-rwxrwxrwx 1 doancea doancea-group 56255808136 May  5 16:39 po_s912k_post0.f00019
-rwxrwxrwx 1 doancea doancea-group 56255808136 May  5 16:39 po_s912k_post0.f00020
-rwxrwxrwx 1 doancea doancea-group 56255808136 May  5 16:39 po_s912k_post0.f00021
-rwxrwxrwx 1 doancea doancea-group 56255808136 May  5 16:39 po_s912k_post0.f00022
-rwxrwxrwx 1 doancea doancea-group 56255808136 May  5 16:39 po_s912k_post0.f00023
-rwxrwxrwx 1 doancea doancea-group 56255808136 May  5 16:39 po_s912k_post0.f00024
-rwxrwxrwx 1 doancea doancea-group 56255808136 May  5 16:39 po_s912k_post0.f00025
-rwxrwxrwx 1 doancea doancea-group 56255808136 May  5 16:39 po_s912k_post0.f00026
-rwxrwxrwx 1 doancea doancea-group 56255808136 May  5 16:39 po_s912k_post0.f00027
-rwxrwxrwx 1 doancea doancea-group 56255808136 May  5 16:39 po_s912k_post0.f00028
-rwxrwxrwx 1 doancea doancea-group 56255808136 May  5 16:39 po_s912k_post0.f00029
-rwxrwxrwx 1 doancea doancea-group 56255808136 May  5 16:39 po_s912k_post0.f00030
-rwxr-xr-x 1 doancea doancea-group 56255808136 May  5 14:24 po_s912k_post0.f00031
-rwxr-xr-x 1 doancea doancea-group 56255808136 May  5 14:28 po_s912k_post0.f00032
-rwxr-xr-x 1 doancea doancea-group 56255808136 May  5 14:33 po_s912k_post0.f00033
-rwxr-xr-x 1 doancea doancea-group 56255808136 May  5 14:38 po_s912k_post0.f00034
-rwxr-xr-x 1 doancea doancea-group 56255808136 May  5 14:42 po_s912k_post0.f00035
-rwxr-xr-x 1 doancea doancea-group 56255808136 May  5 14:47 po_s912k_post0.f00036
-rwxr-xr-x 1 doancea doancea-group 56255808136 May  5 14:52 po_s912k_post0.f00037
-rwxr-xr-x 1 doancea doancea-group 56255808136 May  5 14:57 po_s912k_post0.f00038
-rwxr-xr-x 1 doancea doancea-group 56255808136 May  5 15:02 po_s912k_post0.f00039
-rwxr-xr-x 1 doancea doancea-group 56255808136 May  5 15:33 po_s912k_post0.f00040
-rwxr-xr-x 1 doancea doancea-group 56255808136 May  5 15:12 po_s912k_post0.f00041
-rwxr-xr-x 1 doancea doancea-group 56255808136 May  5 15:18 po_s912k_post0.f00042
-rwxr-xr-x 1 doancea doancea-group 56255808136 May  5 15:23 po_s912k_post0.f00043
-rwxr-xr-x 1 doancea doancea-group 56255808136 May  5 15:28 po_s912k_post0.f00044
-rwxrwxrwx 1 doancea doancea-group 56255808136 May  5 16:39 po_s912k_post0.f00045
-rwxrwxrwx 1 doancea doancea-group 56255808136 May  5 16:39 po_s912k_post0.f00046
-rwxrwxrwx 1 scleon  scleon-group  56255808136 May  5 16:39 po_s912k_post0.f00047
-rwxrwxrwx 1 scleon  scleon-group  56255808136 May  5 16:39 po_s912k_post0.f00048
-rwxrwxrwx 1 scleon  scleon-group  56255808136 May  5 16:39 po_s912k_post0.f00049
-rwxrwxrwx 1 scleon  scleon-group  56255808136 May  5 16:39 po_s912k_post0.f00050
"""

# Extract numbers from filenames using regex
matches = re.findall(r'po_s912k_post0\.f000(\d{2})*', ll_output)

# Convert to integers and filter for range 01 to 44
found = sorted(int(num) for num in matches if 1 <= int(num) <= 44)

# Determine missing numbers
expected = set(range(1, 45))
found_set = set(found)
missing = sorted(expected - found_set)

# Print results
print("Found file numbers (01 to 44):")
print(" ".join(f"{n:02d}" for n in found))

if missing:
    print("\nMissing file numbers:")
    print(" ".join(f"{n:02d}" for n in missing))
else:
    print("\nNo missing file numbers in range 01–44.")







import re



print("\n\n")

# Paste your `ll` output below as a multiline string
ll_output = """
-rwxrwxrwx 1 orsob    orsob-group    72818423567 May  5 16:39 po_Sphere_mid0.f00001.gz
-rwxr-xr-x 1 cfrouzak cfrouzak-group 73107665928 May  6 10:16 po_Sphere_mid0.f00002.gz
-rwxr-xr-x 1 cfrouzak cfrouzak-group 73390334605 May  6 10:29 po_Sphere_mid0.f00003.gz
-rwxr-xr-x 1 cfrouzak cfrouzak-group 73711360437 May  6 10:41 po_Sphere_mid0.f00004.gz
-rwxr-xr-x 1 cfrouzak cfrouzak-group 73983736536 May  6 12:04 po_Sphere_mid0.f00005.gz
-rwxr-xr-x 1 cfrouzak cfrouzak-group 74258825716 May  6 12:16 po_Sphere_mid0.f00006.gz
-rwxr-xr-x 1 cfrouzak cfrouzak-group 74505300895 May  6 12:29 po_Sphere_mid0.f00007.gz
-rwxr-xr-x 1 cfrouzak cfrouzak-group 74723810520 May  6 12:43 po_Sphere_mid0.f00008.gz
-rwxr-xr-x 1 cfrouzak cfrouzak-group 74904826535 May  6 12:57 po_Sphere_mid0.f00009.gz
-rwxr-xr-x 1 cfrouzak cfrouzak-group 75088494532 May  6 13:28 po_Sphere_mid0.f00010.gz
-rwxr-xr-x 1 cfrouzak cfrouzak-group 75254556187 May  6 13:41 po_Sphere_mid0.f00011.gz
-rwxr-xr-x 1 cfrouzak cfrouzak-group 75410979661 May  6 13:55 po_Sphere_mid0.f00012.gz
-rwxr-xr-x 1 cfrouzak cfrouzak-group 75567721020 May  6 14:09 po_Sphere_mid0.f00013.gz
-rwxr-xr-x 1 cfrouzak cfrouzak-group 75721098198 May  6 14:23 po_Sphere_mid0.f00014.gz
-rwxr-xr-x 1 cfrouzak cfrouzak-group 75809166498 May  6 14:38 po_Sphere_mid0.f00015.gz
-rwxr-xr-x 1 cfrouzak cfrouzak-group 75963580190 May  6 14:52 po_Sphere_mid0.f00016.gz
-rwxr-xr-x 1 cfrouzak cfrouzak-group 76104686713 May  6 15:06 po_Sphere_mid0.f00017.gz
-rwxr-xr-x 1 cfrouzak cfrouzak-group 76243050275 May  6 15:21 po_Sphere_mid0.f00018.gz
-rwxr-xr-x 1 cfrouzak cfrouzak-group 13312720896 May  6 15:23 po_Sphere_mid0.f00019.gz
"""

# Extract file sizes using a regular expression
sizes = [int(m.group(1)) for m in re.finditer(r"\s(\d+)\s+May", ll_output)]

print(len(sizes))
for i in range(len(sizes)+1):

    print(i, "\t", 1.47+ sum(sizes[0:i]) / 1024**4)


# Compute total size
total_bytes = sum(sizes)
total_gb = total_bytes / (1024**3)
total_tb = total_bytes / (1024**4)

print(f"Total size: {total_bytes} bytes")
print(f"Total size: {total_gb:.2f} GB")
print(f"Total size: {total_tb:.2f} TB")


print("\n\n")

s = """
24
25
26
"""
# 1) split into lines and convert to integers
nums = [int(line) for line in s.split()]

# 2) compute min, max, count
lo = min(nums)
hi = max(nums)
cnt = len(nums)

# 3) find any missing in the full range
missing = [i for i in range(lo, hi+1) if i not in nums]

print("min =", lo)
print("max =", hi)
print("count =", cnt)
print("missing =", missing)




print("\n", )